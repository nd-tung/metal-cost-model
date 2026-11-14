#pragma once
// uint32 columns written by scripts/gen_tpch.py (<dir>/<name>.u32), each loaded into its own
// page-aligned allocation, so the GPU can wrap it no-copy.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "common/metal_ctx.hpp"

namespace mcm {

// Loads <dir>/<names[i]>.u32 into mem[i]; returns the row count (equal for every column).
template <int N>
size_t load_u32_columns(const std::string& dir, const char* const (&names)[N], std::unique_ptr<PageAlloc> (&mem)[N]) {
  size_t rows = 0;
  for (int i = 0; i < N; ++i) {
    const std::string path = dir + "/" + names[i] + ".u32";
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open column " + path);
    const size_t bytes = static_cast<size_t>(f.tellg());
    const size_t n = bytes / sizeof(uint32_t);
    if (i == 0) rows = n;
    if (n != rows) throw std::runtime_error(path + ": row count differs from " + names[0]);
    mem[i] = std::make_unique<PageAlloc>(bytes);
    f.seekg(0);
    f.read(static_cast<char*>(mem[i]->data()), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("short read on " + path);
  }
  return rows;
}

}  // namespace mcm
