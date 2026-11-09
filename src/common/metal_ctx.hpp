#pragma once
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>

namespace mcm {

// Owns the device, one command queue, and the compiled library.
// Kernels are compiled from source at startup; compile time is never part of a measurement.
class MetalCtx {
 public:
  explicit MetalCtx(const std::string& kernel_file);
  ~MetalCtx();
  MetalCtx(const MetalCtx&) = delete;
  MetalCtx& operator=(const MetalCtx&) = delete;

  MTL::Device* device() const { return dev_; }
  MTL::CommandQueue* queue() const { return queue_; }
  MTL::ComputePipelineState* pipeline(const char* fn_name);  // caller releases

 private:
  MTL::Device* dev_ = nullptr;
  MTL::CommandQueue* queue_ = nullptr;
  MTL::Library* lib_ = nullptr;
};

// Page-aligned host allocation whose size is rounded up to whole pages,
// as required by newBuffer(bytesNoCopy...). Not copyable.
class PageAlloc {
 public:
  explicit PageAlloc(size_t bytes);
  ~PageAlloc();
  PageAlloc(const PageAlloc&) = delete;
  PageAlloc& operator=(const PageAlloc&) = delete;
  void* data() const { return p_; }
  size_t size() const { return size_; }  // rounded size

 private:
  void* p_ = nullptr;
  size_t size_ = 0;
};

// Wrap host memory as a shared MTLBuffer without copying (UMA zero-copy path).
MTL::Buffer* wrap_no_copy(MTL::Device* dev, const PageAlloc& a);

std::string read_file(const std::string& path);

}  // namespace mcm
