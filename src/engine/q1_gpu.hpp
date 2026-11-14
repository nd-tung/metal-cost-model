#pragma once
// Host side of the Q1 GPU kernels (kernels/q1.metal). The 7 columns are wrapped no-copy once;
// a morsel is a row range bound by buffer offset, so nothing is copied per morsel (UMA).
// Each morsel in flight needs its own Slot (partials + result buffer): the kernels use no
// atomics, so morsels never share an output.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/metal_ctx.hpp"
#include "engine/q1.hpp"

namespace mcm::q1 {

struct GpuParams {
  uint32_t n, cutoff;  // must match Q1Params in kernels/q1.metal
};

class Gpu {
 public:
  static constexpr NS::UInteger kThreads = 256;  // per q1_partial threadgroup (kernel: <= 8 simdgroups)
  static constexpr NS::UInteger kRowsPerThread = 16;
  // q1_partial threadgroups per morsel at most (grid-stride beyond): 16 per GPU core, at least 128
  // (M1, 7 cores: 128; M3 Ultra, 80 cores: 1280). With too few threads a large GPU cannot keep up
  // with Q1's per-row work, 7 loads and 36 64-bit updates (128 ran large morsels 16-31% slower than
  // 1024 on the M3 Ultra); from 4096 on, every threadgroup's 36-value reduction costs more than it
  // gains (README, GPU grid size). val_q1 --gpu-groups overrides it.
  static NS::UInteger default_max_groups(int gpu_cores) {
    return std::max<NS::UInteger>(128, 16 * static_cast<NS::UInteger>(std::max(gpu_cores, 0)));
  }
  static constexpr NS::UInteger kFinalThreads = 64;

  struct Slot {
    MTL::Buffer* partials = nullptr;  // max_groups() x kVals uint64
    MTL::Buffer* out = nullptr;       // kVals uint64
  };

  // partial_kernel: q1_partial (the query), or q1_partial_dependent, the kernel before 13 Nov 2026: a
  // diagnostic with the same result that loads l_returnflag and l_linestatus only after the shipdate
  // test (kernels/q1.metal; val_q1 --gpu-loads dependent).
  Gpu(MetalCtx& ctx, const Columns& c, NS::UInteger max_groups, const char* partial_kernel = "q1_partial")
      : ctx_(ctx), max_groups_(max_groups) {
    if (max_groups_ == 0) throw std::runtime_error("q1 gpu: max_groups must be at least 1");
    partial_ = ctx.pipeline(partial_kernel);
    final_ = ctx.pipeline("q1_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads)
      throw std::runtime_error(std::string(partial_kernel) + ": fewer than 256 threads per threadgroup available");
    for (int i = 0; i < 7; ++i) cols_[i] = wrap_no_copy(ctx.device(), *c.mem[i]);
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
    Result r;
    std::memcpy(r.v, s.out->contents(), sizeof(r.v));
    return r;
  }

  MTL::ComputePipelineState* partial_pso() const { return partial_; }
  NS::UInteger max_groups() const { return max_groups_; }

  // q1_partial threadgroups for a morsel of n rows: one row of partials each.
  NS::UInteger groups_for(uint32_t n) const {
    const NS::UInteger per_group = kThreads * kRowsPerThread;
    return std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, max_groups_);
  }

  // Diagnostic (val_q1 --gpu-final 0): leave q1_final out of encode(), so a morsel is q1_partial
  // alone, and do its sum on the CPU (final_on_cpu, after the timed span) so the result can still
  // be checked. Not a way to run Q1: it times only part of the query.
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

  // Dispatches morsel [begin, begin + n) into `enc` (sets its own pipeline states): q1_partial,
  // then q1_final, which a serial compute encoder runs after q1_partial has finished.
  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger groups = groups_for(n);
    const GpuParams p{n, kCutoff};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 7; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(s.partials, 0, 7);
    enc->setBytes(&p, sizeof(p), 8);
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
  MTL::Buffer* cols_[7] = {};
};

}  // namespace mcm::q1
