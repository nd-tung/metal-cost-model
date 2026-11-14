#pragma once
// TPC-H Q6 on the uint32 lineitem columns of scripts/gen_tpch.py: the micro-engine's fused
// filter + sum kernel, CPU side (the GPU side is kernels/q6.metal + engine/q6_gpu.hpp).
//
//   SELECT sum(l_extendedprice * l_discount) AS revenue
//   FROM lineitem
//   WHERE l_shipdate >= date '1994-01-01' AND l_shipdate < date '1994-01-01' + interval '1' year
//     AND l_discount BETWEEN 0.06 - 0.01 AND 0.06 + 0.01 AND l_quantity < 24
//
// Exact integer arithmetic on the encodings (dates: days since 1970-01-01; decimals: x100):
// revenue = sum(price * disc) is x10^4, uint64 (one row's price * disc <= 10494950 * 7 fits 32 bits).
// count, the number of rows that pass, is not part of Q6: a second check.
//
// The kernels are branchless and read all 4 columns of every row (16 B), whatever the predicate
// says, so the memory traffic does not depend on the data: bytes = 16 n, as the cost model assumes.
// A morsel is a row range [begin, begin + n) of the columns.
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/metal_ctx.hpp"
#include "engine/columns.hpp"

namespace mcm::q6 {

inline constexpr uint32_t kDateLo = 8766;  // DATE '1994-01-01', days since 1970-01-01
inline constexpr uint32_t kDateHi = 9131;  // DATE '1995-01-01' (exclusive)
inline constexpr uint32_t kDiscLo = 5;     // l_discount >= 0.05 (x100)
inline constexpr uint32_t kDiscHi = 7;     // l_discount <= 0.07 (x100)
inline constexpr uint32_t kQtyHi = 2400;   // l_quantity < 24 (x100)
inline constexpr int kVals = 2;            // count, revenue (must match kernels/q6.metal)
inline constexpr size_t kBytesPerRow = 4 * sizeof(uint32_t);
inline const char* const kColumnNames[4] = {"l_shipdate", "l_discount", "l_quantity", "l_extendedprice"};

struct Result {
  uint64_t count = 0, revenue = 0;  // revenue x10^4
  void merge(const Result& o) {
    count += o.count;
    revenue += o.revenue;
  }
  bool operator==(const Result& o) const { return count == o.count && revenue == o.revenue; }
};

// The predicate, two-sided ranges as one unsigned compare each (values below the low bound wrap).
inline bool pass(uint32_t shipdate, uint32_t disc, uint32_t qty) {
  return (shipdate - kDateLo < kDateHi - kDateLo) & (disc - kDiscLo <= kDiscHi - kDiscLo) & (qty < kQtyHi);
}

struct Columns {
  size_t rows = 0;
  std::unique_ptr<PageAlloc> mem[4];
  const uint32_t* col(int i) const { return static_cast<const uint32_t*>(mem[i]->data()); }
  const uint32_t* shipdate() const { return col(0); }
  const uint32_t* discount() const { return col(1); }
  const uint32_t* quantity() const { return col(2); }
  const uint32_t* price() const { return col(3); }
};

inline Columns load_columns(const std::string& dir) {
  Columns c;
  c.rows = load_u32_columns(dir, kColumnNames, c.mem);
  return c;
}

// Plain scalar loop with a branch, kept independent of the timed kernels: the correctness reference.
inline Result reference(const Columns& c, size_t begin, size_t n) {
  Result r;
  for (size_t i = begin; i < begin + n; ++i) {
    const uint32_t sd = c.shipdate()[i], disc = c.discount()[i], qty = c.quantity()[i];
    if (sd < kDateLo || sd >= kDateHi || disc < kDiscLo || disc > kDiscHi || qty >= kQtyHi) continue;
    r.count += 1;
    r.revenue += static_cast<uint64_t>(c.price()[i]) * disc;
  }
  return r;
}

// The timed CPU kernel (one worker): branchless, so every row's four columns are read and the
// compiler vectorizes it. A uint64 accumulator would keep it scalar (clang: vectorizing is "not
// beneficial" with 2 lanes), so rows go in blocks of 32 summed in uint32 lanes and widened once per
// block: 32 * 10494950 * 7 < 2^32 (l_extendedprice <= 104949.50 at every scale factor).
inline constexpr size_t kBlock = 32;

inline void cpu(const Columns& c, size_t begin, size_t n, Result& out) {
  const uint32_t *sd = c.shipdate() + begin, *dc = c.discount() + begin, *qt = c.quantity() + begin,
                 *pr = c.price() + begin;
  uint64_t cnt = 0, rev = 0;
  size_t i = 0;
  for (; i + kBlock <= n; i += kBlock) {
    uint32_t bc = 0, br = 0;
    for (size_t j = i; j < i + kBlock; ++j) {
      const uint32_t d = dc[j];
      const uint32_t hit = pass(sd[j], d, qt[j]);
      bc += hit;
      br += (pr[j] * d) & (0u - hit);  // pr * d <= 10494950 * 10: fits 32 bits
    }
    cnt += bc;
    rev += br;
  }
  for (; i < n; ++i) {
    const uint32_t d = dc[i];
    const uint32_t hit = pass(sd[i], d, qt[i]);
    cnt += hit;
    rev += static_cast<uint64_t>((pr[i] * d) & (0u - hit));
  }
  out.count += cnt;
  out.revenue += rev;
}

// DuckDB's answer (scripts/tpch_reference.py): one line "count revenue". Returns false if the file
// does not exist.
inline bool load_expected(const std::string& path, Result& r) {
  std::ifstream f(path);
  if (!f) return false;
  if (!(f >> r.count >> r.revenue)) throw std::runtime_error(path + ": expected one line 'count revenue'");
  return true;
}

inline std::string describe(const Result& r) {
  std::ostringstream o;
  o << "  count=" << r.count << " revenue=" << r.revenue << " (x10^4)\n";
  return o.str();
}

}  // namespace mcm::q6
