#pragma once
// The GPU build of an engine hash table (engine/hash_join.hpp) with join_build of kernels/join.metal,
// as bench/calib_join.cpp runs it: one 32-bit CAS per inserted row, 256 threads per threadgroup,
// ~16 rows per thread, at most 4096 threadgroups. Unlike the calibration (which clears the table
// untimed), build_sync() also clears it, as a blit fill (every byte 0xFF) in the same command buffer,
// so a timed build is the whole build phase of a query. The key and payload columns and the table are
// wrapped no-copy: the table the GPU builds is the memory every device then probes.
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "common/metal_ctx.hpp"
#include "engine/hash_join.hpp"

namespace mcm::join {

struct GpuBuildParams {
  uint32_t n, bits, mask, use_filter;  // must match JoinParams in kernels/join.metal
};

class GpuBuilder {
 public:
  static constexpr NS::UInteger kThreads = 256;
  static constexpr NS::UInteger kRowsPerThread = 16;
  static constexpr NS::UInteger kMaxGroups = 4096;

  // ctx must have compiled kernels/join.metal. rows: the build rows (keys and payload, page-aligned).
  GpuBuilder(MetalCtx& ctx, const Table& t, const PageAlloc& keys, const PageAlloc& payload, size_t rows)
      : ctx_(ctx), bits_(t.bits), mask_(t.mask), bytes_(t.bytes()), rows_(rows) {
    if (rows_ > UINT32_MAX) throw std::runtime_error("gpu build: more than 2^32 rows");
    pso_ = ctx.pipeline("join_build");
    table_ = wrap_no_copy(ctx.device(), *t.mem);
    keys_ = wrap_no_copy(ctx.device(), keys);
    payload_ = wrap_no_copy(ctx.device(), payload);
    stats_ = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
  }
  ~GpuBuilder() {
    stats_->release();
    payload_->release();
    keys_->release();
    table_->release();
    pso_->release();
  }
  GpuBuilder(const GpuBuilder&) = delete;
  GpuBuilder& operator=(const GpuBuilder&) = delete;

  // Clears the table and inserts every build row, in one command buffer, and waits for it. Returns the
  // rows the kernel inserted.
  uint32_t build_sync() {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    std::memset(stats_->contents(), 0, sizeof(uint32_t));
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::BlitCommandEncoder* blit = cb->blitCommandEncoder();
    blit->fillBuffer(table_, NS::Range(0, bytes_), 0xFF);
    blit->endEncoding();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    const GpuBuildParams p{static_cast<uint32_t>(rows_), static_cast<uint32_t>(bits_), mask_, 0};
    enc->setComputePipelineState(pso_);
    enc->setBuffer(keys_, 0, 0);
    enc->setBuffer(payload_, 0, 1);
    enc->setBuffer(keys_, 0, 2);  // the filter: unused (use_filter = 0)
    enc->setBuffer(table_, 0, 3);
    enc->setBuffer(stats_, 0, 4);
    enc->setBytes(&p, sizeof(p), 5);
    const NS::UInteger groups =
        std::clamp<NS::UInteger>((rows_ + kThreads * kRowsPerThread - 1) / (kThreads * kRowsPerThread), 1, kMaxGroups);
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    const bool ok = cb->status() == MTL::CommandBufferStatusCompleted;
    pool->release();
    if (!ok) throw std::runtime_error("GPU build command buffer failed");
    return *static_cast<const uint32_t*>(stats_->contents());
  }

 private:
  MetalCtx& ctx_;
  int bits_;
  uint32_t mask_;
  size_t bytes_, rows_;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::Buffer* table_ = nullptr;
  MTL::Buffer* keys_ = nullptr;
  MTL::Buffer* payload_ = nullptr;
  MTL::Buffer* stats_ = nullptr;
};

}  // namespace mcm::join
