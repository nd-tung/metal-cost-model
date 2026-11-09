#include "common/metal_ctx.hpp"

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace mcm {

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

MetalCtx::MetalCtx(const std::string& kernel_file) {
  dev_ = MTL::CreateSystemDefaultDevice();
  if (!dev_) throw std::runtime_error("no Metal device");
  queue_ = dev_->newCommandQueue();

  std::string src = read_file(kernel_file);
  NS::Error* err = nullptr;
  auto* opts = MTL::CompileOptions::alloc()->init();
  lib_ = dev_->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opts, &err);
  opts->release();
  if (!lib_) {
    std::string msg = err ? err->localizedDescription()->utf8String() : "unknown error";
    throw std::runtime_error("Metal compile failed for " + kernel_file + ": " + msg);
  }
}

MetalCtx::~MetalCtx() {
  if (lib_) lib_->release();
  if (queue_) queue_->release();
  if (dev_) dev_->release();
}

MTL::ComputePipelineState* MetalCtx::pipeline(const char* fn_name) {
  MTL::Function* fn = lib_->newFunction(NS::String::string(fn_name, NS::UTF8StringEncoding));
  if (!fn) throw std::runtime_error(std::string("kernel not found: ") + fn_name);
  NS::Error* err = nullptr;
  MTL::ComputePipelineState* pso = dev_->newComputePipelineState(fn, &err);
  fn->release();
  if (!pso) {
    std::string msg = err ? err->localizedDescription()->utf8String() : "unknown error";
    throw std::runtime_error(std::string("pipeline failed for ") + fn_name + ": " + msg);
  }
  return pso;
}

PageAlloc::PageAlloc(size_t bytes) {
  const size_t page = static_cast<size_t>(getpagesize());
  size_ = ((bytes + page - 1) / page) * page;
  if (size_ == 0) size_ = page;
  if (posix_memalign(&p_, page, size_) != 0) throw std::bad_alloc();
}

PageAlloc::~PageAlloc() { std::free(p_); }

MTL::Buffer* wrap_no_copy(MTL::Device* dev, const PageAlloc& a) {
  MTL::Buffer* b = dev->newBuffer(a.data(), a.size(), MTL::ResourceStorageModeShared, nullptr);
  if (!b) throw std::runtime_error("newBuffer(bytesNoCopy) failed");
  return b;
}

}  // namespace mcm
