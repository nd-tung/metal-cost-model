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
  static constexpr NS::UInteger kMaxGroups = 128;  // q1_partial threadgroups per morsel (grid-stride beyond)
  static constexpr NS::UInteger kFinalThreads = 64;

  struct Slot {
    MTL::Buffer* partials = nullptr;  // kMaxGroups x kVals uint64
    MTL::Buffer* out = nullptr;       // kVals uint64
  };

  Gpu(MetalCtx& ctx, const Columns& c) : ctx_(ctx) {
    partial_ = ctx.pipeline("q1_partial");
    final_ = ctx.pipeline("q1_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads)
      throw std::runtime_error("q1_partial: fewer than 256 threads per threadgroup available");
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
    s.partials = ctx_.device()->newBuffer(kMaxGroups * kVals * sizeof(uint64_t), MTL::ResourceStorageModeShared);
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

  // Dispatches morsel [begin, begin + n) into `enc` (sets its own pipeline states): q1_partial,
  // then q1_final, which a serial compute encoder runs after q1_partial has finished.
  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger per_group = kThreads * kRowsPerThread;
    const NS::UInteger groups = std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, kMaxGroups);
    const GpuParams p{n, kCutoff};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 7; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(s.partials, 0, 7);
    enc->setBytes(&p, sizeof(p), 8);
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    const uint32_t rows = static_cast<uint32_t>(groups);
    enc->setComputePipelineState(final_);
    enc->setBuffer(s.partials, 0, 0);
    enc->setBuffer(s.out, 0, 1);
    enc->setBytes(&rows, sizeof(rows), 2);
    enc->dispatchThreadgroups(MTL::Size(kVals, 1, 1), MTL::Size(kFinalThreads, 1, 1));
  }

 private:
  MetalCtx& ctx_;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* final_ = nullptr;
  MTL::Buffer* cols_[7] = {};
};

}  // namespace mcm::q1
