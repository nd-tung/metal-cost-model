#pragma once
// Host side of the Q6 GPU kernels (kernels/q6.metal), as engine/q1_gpu.hpp: the 4 columns are
// wrapped no-copy once, a morsel is a row range bound by buffer offset, and each morsel in flight
// needs its own Slot. The grid follows filter_count's (bench/calib_filter.cpp): 256 threads per
// threadgroup, ~16 rows per thread, at most 4096 threadgroups, so Q6 runs with the occupancy the
// filter was calibrated with.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <stdexcept>

#include "common/metal_ctx.hpp"
#include "engine/q6.hpp"

namespace mcm::q6 {

struct GpuParams {
  uint32_t n, date_lo, date_span, disc_lo, disc_span, qty_hi;  // must match Q6Params in kernels/q6.metal
};

class Gpu {
 public:
  static constexpr NS::UInteger kThreads = 256;  // per q6_partial threadgroup (kernel: <= 8 simdgroups)
  static constexpr NS::UInteger kRowsPerThread = 16;
  // q6_partial threadgroups per morsel at most (grid-stride beyond), as calib_filter's grid; Q6 does
  // not depend on it (on the M3 Ultra 128 and 4096 stream at the same rate). val_q6 --gpu-groups
  // overrides it.
  static NS::UInteger default_max_groups(int /*gpu_cores*/) { return 4096; }
  static constexpr NS::UInteger kFinalThreads = 256;

  struct Slot {
    MTL::Buffer* partials = nullptr;  // max_groups() x kVals uint64
    MTL::Buffer* out = nullptr;       // kVals uint64
  };

  Gpu(MetalCtx& ctx, const Columns& c, NS::UInteger max_groups)
      : ctx_(ctx), max_groups_(max_groups) {
    if (max_groups_ == 0) throw std::runtime_error("q6 gpu: max_groups must be at least 1");
    partial_ = ctx.pipeline("q6_partial");
    final_ = ctx.pipeline("q6_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads || final_->maxTotalThreadsPerThreadgroup() < kFinalThreads)
      throw std::runtime_error("q6 kernels: fewer than 256 threads per threadgroup available");
    for (int i = 0; i < 4; ++i) cols_[i] = wrap_no_copy(ctx.device(), *c.mem[i]);
  }
  ~Gpu() {
    for (MTL::Buffer* b : cols_) b->release();
    final_->release();
    partial_->release();
  }
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;

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
    r.count = v[0];
    r.revenue = v[1];
    return r;
  }

  MTL::ComputePipelineState* partial_pso() const { return partial_; }
  NS::UInteger max_groups() const { return max_groups_; }

  // q6_partial threadgroups for a morsel of n rows: one row of partials each.
  NS::UInteger groups_for(uint32_t n) const {
    const NS::UInteger per_group = kThreads * kRowsPerThread;
    return std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, max_groups_);
  }

  // Diagnostic (val_q6 --gpu-final 0), as in q1_gpu.hpp: q6_final left out of encode(), its sum
  // done on the CPU after the timed span.
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

  // Dispatches morsel [begin, begin + n) into `enc` (sets its own pipeline states): q6_partial,
  // then q6_final, which a serial compute encoder runs after q6_partial has finished.
  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger groups = groups_for(n);
    const GpuParams p{n, kDateLo, kDateHi - kDateLo, kDiscLo, kDiscHi - kDiscLo, kQtyHi};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 4; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(s.partials, 0, 4);
    enc->setBytes(&p, sizeof(p), 5);
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
  bool final_pass_ = true;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* final_ = nullptr;
  MTL::Buffer* cols_[4] = {};
};

}  // namespace mcm::q6
