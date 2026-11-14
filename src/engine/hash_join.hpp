#pragma once
// Hash join for calib_join, CPU side and table layout (the GPU side is kernels/join.metal).
// Open addressing with linear probing; 32-bit keys and 32-bit payloads in interleaved
// (key, payload) slots, one page-aligned allocation that the CPU and the GPU share: on UMA one
// device builds the table and all three probe it, with no copy.
//
//   capacity = next power of two >= 2 * build rows (load factor <= 0.5)
//   slot     = (key * 0x9E3779B1) >> (32 - log2 capacity)   (Fibonacci hashing)
//   empty    = key 0xFFFFFFFF (TPC-H keys are smaller)
// Build keys are primary keys (unique), so inserts do not check for duplicates.
// Probe output is fused, not materialized: match count and payload sum, both uint32 and wrapping
// mod 2^32 on every device, so results are checked exactly.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/metal_ctx.hpp"

namespace mcm::join {

inline constexpr uint32_t kEmpty = 0xFFFFFFFFu;
inline constexpr uint32_t kHashMul = 0x9E3779B1u;

struct Table {
  int bits = 1;
  uint32_t mask = 1;
  size_t cap = 2;
  std::unique_ptr<PageAlloc> mem;  // cap x {key, payload} uint32
  uint32_t* slots() const { return static_cast<uint32_t*>(mem->data()); }
  size_t bytes() const { return cap * 2 * sizeof(uint32_t); }
  void clear() { std::memset(mem->data(), 0xFF, bytes()); }  // every key (and payload) 0xFFFFFFFF
  uint32_t slot_of(uint32_t key) const { return (key * kHashMul) >> (32 - bits); }
};

inline Table make_table(size_t build_rows) {
  Table t;
  while ((size_t(1) << t.bits) < 2 * std::max<size_t>(build_rows, 1)) ++t.bits;
  if (t.bits > 31) throw std::runtime_error("hash table too large");
  t.cap = size_t(1) << t.bits;
  t.mask = static_cast<uint32_t>(t.cap - 1);
  t.mem = std::make_unique<PageAlloc>(t.bytes());
  t.clear();
  return t;
}

struct ProbeResult {
  uint32_t matches = 0, sum = 0;
  bool operator==(const ProbeResult& o) const { return matches == o.matches && sum == o.sum; }
};

// Inserts the build rows [begin, begin + n) whose filter value is non-zero (filter may be null).
inline uint32_t cpu_build(Table& t, const uint32_t* keys, const uint32_t* payload, const uint32_t* filter,
                          size_t begin, size_t n) {
  uint32_t* s = t.slots();
  uint32_t inserted = 0;
  for (size_t i = begin; i < begin + n; ++i) {
    if (filter && !filter[i]) continue;
    const uint32_t k = keys[i];
    uint32_t h = t.slot_of(k);
    while (s[2 * h] != kEmpty) h = (h + 1) & t.mask;
    s[2 * h] = k;
    s[2 * h + 1] = payload[i];
    ++inserted;
  }
  return inserted;
}

// Probes [begin, begin + n). Slots are prefetched a batch of kBatch probes ahead, so a
// DRAM-resident table is not probed one cache miss at a time.
inline constexpr int kBatch = 16;

inline ProbeResult cpu_probe(const Table& t, const uint32_t* keys, size_t begin, size_t n) {
  const uint32_t* s = t.slots();
  uint32_t cnt = 0, sum = 0;
  auto probe = [&](uint32_t k, uint32_t h) {
    for (;;) {
      const uint32_t sk = s[2 * h];
      if (sk == k) {
        ++cnt;
        sum += s[2 * h + 1];
        return;
      }
      if (sk == kEmpty) return;
      h = (h + 1) & t.mask;
    }
  };
  const size_t end = begin + n;
  size_t i = begin;
  uint32_t h[kBatch];
  for (; i + kBatch <= end; i += kBatch) {
    for (int j = 0; j < kBatch; ++j) {
      h[j] = t.slot_of(keys[i + j]);
      __builtin_prefetch(s + 2 * h[j]);
    }
    for (int j = 0; j < kBatch; ++j) probe(keys[i + j], h[j]);
  }
  for (; i < end; ++i) probe(keys[i], t.slot_of(keys[i]));
  return {cnt, sum};
}

// Independent reference: dense arrays over the key domain (TPC-H keys are dense enough).
class Reference {
 public:
  Reference(const uint32_t* keys, const uint32_t* payload, const uint32_t* filter, size_t rows) {
    uint32_t mx = 0;
    for (size_t i = 0; i < rows; ++i) mx = std::max(mx, keys[i]);
    payload_.assign(size_t(mx) + 1, 0);
    row_.assign(size_t(mx) + 1, kNone);
    for (size_t i = 0; i < rows; ++i) {
      if (filter && !filter[i]) continue;
      if (row_[keys[i]] != kNone) throw std::runtime_error("duplicate build key " + std::to_string(keys[i]));
      payload_[keys[i]] = payload[i];
      row_[keys[i]] = static_cast<uint32_t>(i);
      ++built_;
    }
  }
  size_t build_rows() const { return built_; }  // rows passing the filter

  ProbeResult probe(const uint32_t* keys, size_t begin, size_t n) const {
    ProbeResult r;
    for (size_t i = begin; i < begin + n; ++i) {
      const uint32_t k = keys[i];
      if (k < row_.size() && row_[k] != kNone) {
        ++r.matches;
        r.sum += payload_[k];
      }
    }
    return r;
  }

  // Exact check of a table built from build rows [0, n): it holds exactly the rows of that range
  // that pass the filter, each with its payload.
  bool check_build(const Table& t, size_t n, size_t expected_rows) const {
    const uint32_t* s = t.slots();
    size_t found = 0;
    for (size_t h = 0; h < t.cap; ++h) {
      const uint32_t k = s[2 * h];
      if (k == kEmpty) continue;
      if (k >= row_.size() || row_[k] == kNone || row_[k] >= n || s[2 * h + 1] != payload_[k]) return false;
      ++found;
    }
    return found == expected_rows;
  }

 private:
  static constexpr uint32_t kNone = 0xFFFFFFFFu;
  std::vector<uint32_t> payload_, row_;
  size_t built_ = 0;
};

inline std::vector<uint32_t> load_u32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open column " + path);
  std::vector<uint32_t> v(static_cast<size_t>(f.tellg()) / sizeof(uint32_t));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(uint32_t)));
  return v;
}

// DuckDB's answer (scripts/tpch_reference.py): lines "name build_rows matches sum_mod_2^32".
inline bool load_expected(const std::string& path, const std::string& name, size_t& build_rows, ProbeResult& r) {
  std::ifstream f(path);
  std::string n;
  size_t b, m, s;
  while (f >> n >> b >> m >> s)
    if (n == name) {
      build_rows = b;
      r.matches = static_cast<uint32_t>(m);
      r.sum = static_cast<uint32_t>(s);
      return true;
    }
  return false;
}

}  // namespace mcm::join
