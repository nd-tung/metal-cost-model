#pragma once
// TPC-H Q1 on the uint32 lineitem columns of scripts/gen_tpch.py: the micro-engine's fused
// filter + group-by kernel, CPU side (the GPU side is kernels/q1.metal + engine/q1_gpu.hpp).
//
//   SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice),
//          sum(l_extendedprice * (1 - l_discount)), sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)),
//          avg(...), count(*)
//   FROM lineitem WHERE l_shipdate <= date '1998-12-01' - interval '90' day
//   GROUP BY l_returnflag, l_linestatus
//
// Exact integer arithmetic on the x100 encodings: per group, count, sum_qty, sum_base_price and
// sum_disc (x100), sum_disc_price = sum(price * (100 - disc)) (x10^4) and
// sum_charge = sum(price * (100 - disc) * (100 + tax)) (x10^6), all uint64. Averages follow from
// these. uint64 holds SF10 (sum_charge ~1.1e18 for the largest group); SF100 would overflow.
//
// Groups: returnflag in {A, N, R} x linestatus in {F, O} -> 6 dense slots, slot = 2 * r + l.
// A morsel is a row range [begin, begin + n) of the columns.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/metal_ctx.hpp"
#include "engine/columns.hpp"

namespace mcm::q1 {

inline constexpr uint32_t kCutoff = 10471;  // DATE '1998-12-01' - INTERVAL 90 DAY, days since 1970-01-01
inline constexpr int kSlots = 6;
inline constexpr int kFields = 6;  // per slot, in this order:
enum Field { COUNT = 0, SUM_QTY, SUM_BASE_PRICE, SUM_DISC_PRICE, SUM_CHARGE, SUM_DISC };
inline constexpr int kVals = kSlots * kFields;     // 36 uint64, slot-major (must match kernels/q1.metal)
inline constexpr size_t kBytesPerRow = 7 * sizeof(uint32_t);
inline const char* const kColumnNames[7] = {"l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
                                            "l_extendedprice", "l_discount", "l_tax"};

struct Result {
  uint64_t v[kVals] = {};
  uint64_t& at(int slot, Field f) { return v[slot * kFields + f]; }
  uint64_t at(int slot, Field f) const { return v[slot * kFields + f]; }
  void merge(const Result& o) {
    for (int i = 0; i < kVals; ++i) v[i] += o.v[i];
  }
  bool operator==(const Result& o) const { return std::memcmp(v, o.v, sizeof(v)) == 0; }
};

inline constexpr char kFlag[3] = {'A', 'N', 'R'};
inline constexpr char kStatus[2] = {'F', 'O'};

inline uint32_t slot_of(uint32_t rf, uint32_t ls) {
  const uint32_t r = (rf == 'N') ? 1u : (rf == 'R') ? 2u : 0u;
  return r * 2u + (ls == 'O' ? 1u : 0u);
}

// The 7 input columns, each one page-aligned allocation (wrappable no-copy for the GPU).
struct Columns {
  size_t rows = 0;
  std::unique_ptr<PageAlloc> mem[7];
  const uint32_t* col(int i) const { return static_cast<const uint32_t*>(mem[i]->data()); }
  const uint32_t* shipdate() const { return col(0); }
  const uint32_t* returnflag() const { return col(1); }
  const uint32_t* linestatus() const { return col(2); }
  const uint32_t* quantity() const { return col(3); }
  const uint32_t* price() const { return col(4); }
  const uint32_t* discount() const { return col(5); }
  const uint32_t* tax() const { return col(6); }
};

inline Columns load_columns(const std::string& dir) {
  Columns c;
  c.rows = load_u32_columns(dir, kColumnNames, c.mem);
  return c;
}

// Plain scalar loop, kept independent of the timed kernels: the correctness reference.
// Throws on flag values outside {A, N, R} x {F, O} (they would be grouped wrongly).
inline Result reference(const Columns& c, size_t begin, size_t n) {
  Result r;
  for (size_t i = begin; i < begin + n; ++i) {
    const uint32_t rf = c.returnflag()[i], ls = c.linestatus()[i];
    if ((rf != 'A' && rf != 'N' && rf != 'R') || (ls != 'F' && ls != 'O'))
      throw std::runtime_error("unexpected returnflag/linestatus at row " + std::to_string(i));
    if (c.shipdate()[i] > kCutoff) continue;
    const int s = static_cast<int>(slot_of(rf, ls));
    const uint64_t price = c.price()[i], disc = c.discount()[i], tax = c.tax()[i];
    r.at(s, COUNT) += 1;
    r.at(s, SUM_QTY) += c.quantity()[i];
    r.at(s, SUM_BASE_PRICE) += price;
    r.at(s, SUM_DISC_PRICE) += price * (100 - disc);
    r.at(s, SUM_CHARGE) += price * (100 - disc) * (100 + tax);
    r.at(s, SUM_DISC) += disc;
  }
  return r;
}

// The timed CPU kernel (one worker). Branchless: a row that fails the filter is added to a
// discard slot (slot kSlots) that is never merged, so no row needs masking. Consecutive rows
// often fall in the same group, and one accumulator table would chain every row's
// load-add-store on the previous one; rows go round-robin to kCopies private tables instead
// (merged at the end, timed), as in calib_agg.
inline constexpr int kCopies = 8;

inline void cpu(const Columns& c, size_t begin, size_t n, Result& out) {
  constexpr int kTable = (kSlots + 1) * kFields;  // + the discard slot
  uint64_t acc[kCopies][kTable] = {};
  const uint32_t *sd = c.shipdate(), *rf = c.returnflag(), *ls = c.linestatus(), *qt = c.quantity(),
                 *pr = c.price(), *dc = c.discount(), *tx = c.tax();
  auto row = [&](uint64_t* a, size_t i) {
    const uint32_t slot = sd[i] <= kCutoff ? slot_of(rf[i], ls[i]) : kSlots;
    uint64_t* s = a + slot * kFields;
    const uint32_t dp = pr[i] * (100u - dc[i]);  // <= 10494950 * 100: fits uint32
    s[COUNT] += 1;
    s[SUM_QTY] += qt[i];
    s[SUM_BASE_PRICE] += pr[i];
    s[SUM_DISC_PRICE] += dp;
    s[SUM_CHARGE] += static_cast<uint64_t>(dp) * (100u + tx[i]);
    s[SUM_DISC] += dc[i];
  };
  const size_t end = begin + n;
  size_t i = begin;
  for (; i + kCopies <= end; i += kCopies)
    for (int k = 0; k < kCopies; ++k) row(acc[k], i + k);
  for (; i < end; ++i) row(acc[0], i);
  for (int k = 0; k < kCopies; ++k)
    for (int j = 0; j < kVals; ++j) out.v[j] += acc[k][j];
}

// DuckDB's answer (scripts/tpch_reference.py): lines "rf ls count sum_qty sum_base sum_disc_price
// sum_charge sum_disc". Returns false if the file does not exist.
inline bool load_expected(const std::string& path, Result& r) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream s(line);
    uint32_t rf, ls;
    uint64_t v[kFields];
    if (!(s >> rf >> ls >> v[0] >> v[1] >> v[2] >> v[3] >> v[4] >> v[5]))
      throw std::runtime_error(path + ": malformed line: " + line);
    for (int k = 0; k < kFields; ++k) r.at(static_cast<int>(slot_of(rf, ls)), static_cast<Field>(k)) = v[k];
  }
  return true;
}

inline std::string describe(const Result& r) {
  std::ostringstream o;
  for (int s = 0; s < kSlots; ++s) {
    if (!r.at(s, COUNT)) continue;
    o << "  " << kFlag[s / 2] << kStatus[s % 2] << " count=" << r.at(s, COUNT) << " sum_qty=" << r.at(s, SUM_QTY)
      << " sum_charge=" << r.at(s, SUM_CHARGE) << "\n";
  }
  return o.str();
}

}  // namespace mcm::q1
