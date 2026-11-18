#pragma once
// Host side of the Q14 GPU kernels (kernels/q14.metal), as engine/q6_gpu.hpp: the 4 lineitem columns
// and the part hash table (built on the CPU, engine/q14.hpp) wrapped no-copy once, a morsel a row
// range bound by buffer offset, one Slot per morsel in flight. The grid follows filter_count's and
// Q6's: 256 threads per threadgroup, ~16 rows per thread, at most 4096 threadgroups.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <stdexcept>

#include "common/metal_ctx.hpp"
#include "engine/q14.hpp"

namespace mcm::q14 {

struct GpuParams {
  uint32_t n, date_lo, date_span, bits, mask;  // must match Q14Params in kernels/q14.metal
};

class Gpu {
 public:
  static constexpr NS::UInteger kThreads = 256;
  static constexpr NS::UInteger kRowsPerThread = 16;
  static NS::UInteger default_max_groups(int /*gpu_cores*/) { return 4096; }
  static constexpr NS::UInteger kFinalThreads = 256;

  struct Slot {
    MTL::Buffer* partials = nullptr;  // max_groups() x kVals uint64
    MTL::Buffer* out = nullptr;       // kVals uint64
  };

  Gpu(MetalCtx& ctx, const Columns& c, NS::UInteger max_groups) : ctx_(ctx), max_groups_(max_groups), bits_(c.table.bits), mask_(c.table.mask) {
    if (max_groups_ == 0) throw std::runtime_error("q14 gpu: max_groups must be at least 1");
    partial_ = ctx.pipeline("q14_partial");
    final_ = ctx.pipeline("q14_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads || final_->maxTotalThreadsPerThreadgroup() < kFinalThreads)
      throw std::runtime_error("q14 kernels: fewer than 256 threads per threadgroup available");
    for (int i = 0; i < 4; ++i) cols_[i] = wrap_no_copy(ctx.device(), *c.mem[i]);
    table_ = wrap_no_copy(ctx.device(), *c.table.mem);
  }
  ~Gpu() {
    table_->release();
    for (MTL::Buffer* b : cols_) b->release();
    final_->release();
    partial_->release();
  }
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;

  // The buffer wrapping lineitem column i (bench/val_query.hpp: the cache-state producer step).
  MTL::Buffer* column_buffer(int i) const { return cols_[i]; }

  Slot new_slot() const {
    Slot s;
    s.partials = ctx_.device()->newBuffer(max_groups_ * kVals * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    s.out = ctx_.device()->newBuffer(kVals * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    return s;
  }
  static void free_slot(Slot& s) {
    s.partials->release();
    s.out->release();
    s = Slot{};
  }
  static Result read(const Slot& s) {
    const uint64_t* v = static_cast<const uint64_t*>(s.out->contents());
    Result r;
    r.total = v[0];
    r.promo = v[1];
    return r;
  }

  MTL::ComputePipelineState* partial_pso() const { return partial_; }
  NS::UInteger max_groups() const { return max_groups_; }
  NS::UInteger groups_for(uint32_t n) const {
    const NS::UInteger per_group = kThreads * kRowsPerThread;
    return std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, max_groups_);
  }
  void set_final_pass(bool on) { final_pass_ = on; }
  bool final_pass() const { return final_pass_; }
  void final_on_cpu(const Slot& s, uint32_t n) const {
    const auto* p = static_cast<const uint64_t*>(s.partials->contents());
    auto* out = static_cast<uint64_t*>(s.out->contents());
    const NS::UInteger rows = groups_for(n);
    for (int v = 0; v < kVals; ++v) {
      uint64_t sum = 0;
      for (NS::UInteger g = 0; g < rows; ++g) sum += p[g * kVals + v];
      out[v] = sum;
    }
  }

  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger groups = groups_for(n);
    const GpuParams p{n, kDateLo, kDateHi - kDateLo, static_cast<uint32_t>(bits_), mask_};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 4; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(table_, 0, 4);
    enc->setBuffer(s.partials, 0, 5);
    enc->setBytes(&p, sizeof(p), 6);
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    if (!final_pass_) return;
    const uint32_t rows = static_cast<uint32_t>(groups);
    enc->setComputePipelineState(final_);
    enc->setBuffer(s.partials, 0, 0);
    enc->setBuffer(s.out, 0, 1);
    enc->setBytes(&rows, sizeof(rows), 2);
    enc->dispatchThreadgroups(MTL::Size(kVals, 1, 1), MTL::Size(kFinalThreads, 1, 1));
  }

 private:
  MetalCtx& ctx_;
  NS::UInteger max_groups_;
  int bits_;
  uint32_t mask_;
  bool final_pass_ = true;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* final_ = nullptr;
  MTL::Buffer* cols_[4] = {};
  MTL::Buffer* table_ = nullptr;
};

}  // namespace mcm::q14
