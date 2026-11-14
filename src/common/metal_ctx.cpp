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

MetalCtx::MetalCtx(const std::vector<std::string>& kernel_files) {
  dev_ = MTL::CreateSystemDefaultDevice();
  if (!dev_) throw std::runtime_error("no Metal device");
  queue_ = dev_->newCommandQueue();

  std::string src, names;
  for (const std::string& f : kernel_files) {
    src += read_file(f) + "\n";
    names += (names.empty() ? "" : ", ") + f;
  }
  NS::Error* err = nullptr;
  auto* opts = MTL::CompileOptions::alloc()->init();
  lib_ = dev_->newLibrary(NS::String::string(src.c_str(), NS::UTF8StringEncoding), opts, &err);
  opts->release();
  if (!lib_) {
    std::string msg = err ? err->localizedDescription()->utf8String() : "unknown error";
    throw std::runtime_error("Metal compile failed for " + names + ": " + msg);
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
  return build(fn, fn_name);
}

MTL::ComputePipelineState* MetalCtx::pipeline(const char* fn_name, const std::vector<uint32_t>& constants) {
  MTL::FunctionConstantValues* values = MTL::FunctionConstantValues::alloc()->init();
  for (size_t i = 0; i < constants.size(); ++i) values->setConstantValue(&constants[i], MTL::DataTypeUInt, i);
  NS::Error* err = nullptr;
  MTL::Function* fn = lib_->newFunction(NS::String::string(fn_name, NS::UTF8StringEncoding), values, &err);
  values->release();
  if (!fn) {
    std::string msg = err ? err->localizedDescription()->utf8String() : "not found";
    throw std::runtime_error(std::string("kernel ") + fn_name + " with function constants: " + msg);
  }
  return build(fn, fn_name);
}

MTL::ComputePipelineState* MetalCtx::build(MTL::Function* fn, const char* fn_name) {
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
