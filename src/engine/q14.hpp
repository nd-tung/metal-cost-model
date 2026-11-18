#pragma once
// TPC-H Q14 on the uint32 columns of scripts/gen_tpch.py: the micro-engine's fused filter + hash
// probe + sum kernel, CPU side (the GPU side is kernels/q14.metal + engine/q14_gpu.hpp).
//
//   SELECT 100 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
//          / sum(l_extendedprice * (1 - l_discount)) AS promo_revenue
//   FROM lineitem, part
//   WHERE l_partkey = p_partkey AND l_shipdate >= date '1995-09-01' AND l_shipdate < date '1995-10-01'
//
// Exact integer arithmetic on the encodings (dates: days since 1970-01-01; decimals: x100): the two
// sums of price * (100 - disc) are x10^4, uint64; the quotient is a presentation step. The part side
// is the hash table of engine/hash_join.hpp (key p_partkey, payload p_promo = 1 for PROMO types),
// built on the CPU when the columns are loaded (bench/run_query rebuilds it every rep, the query's
// build phase: clear_share, build_share below, or the GPU) and shared by every device; every lineitem row
// of the month matches exactly one part (foreign key), the others do not probe. A morsel is a row
// range [begin, begin + n) of the four lineitem columns; the kernels branch on the date, so the rows
// of the month (1.2% of lineitem) load three more columns and probe, the rest read 4 B. kBytesPerRow
// is the upper bound, 16 B: the model's rules decide what to charge (analysis/predict_q14.py).
#include <arm_neon.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/metal_ctx.hpp"
#include "engine/columns.hpp"
#include "engine/hash_join.hpp"

namespace mcm::q14 {

inline constexpr uint32_t kDateLo = 9374;  // DATE '1995-09-01', days since 1970-01-01 (scripts/tpch_reference.py checks)
inline constexpr uint32_t kDateHi = 9404;  // DATE '1995-10-01' (exclusive)
inline constexpr int kVals = 2;            // total, promo (must match kernels/q14.metal)
inline constexpr size_t kBytesPerRow = 4 * sizeof(uint32_t);
inline const char* const kColumnNames[4] = {"l_shipdate", "l_partkey", "l_extendedprice", "l_discount"};
inline const char* const kPartColumnNames[2] = {"p_partkey", "p_promo"};  // the build side

struct Result {
  uint64_t total = 0, promo = 0;  // x10^4
  void merge(const Result& o) {
    total += o.total;
    promo += o.promo;
  }
  bool operator==(const Result& o) const { return total == o.total && promo == o.promo; }
};

struct Columns {
  size_t rows = 0;
  std::unique_ptr<PageAlloc> mem[4];  // the lineitem columns a morsel ranges over
  const uint32_t* col(int i) const { return static_cast<const uint32_t*>(mem[i]->data()); }
  const uint32_t* shipdate() const { return col(0); }
  const uint32_t* partkey() const { return col(1); }
  const uint32_t* price() const { return col(2); }
  const uint32_t* discount() const { return col(3); }
  // The part side: its two columns (page-aligned, so the GPU can build from them no-copy), the hash
  // table (shared by every device) and, for the reference, promo by key.
  std::unique_ptr<PageAlloc> part[2];  // p_partkey, p_promo
  const uint32_t* part_keys() const { return static_cast<const uint32_t*>(part[0]->data()); }
  const uint32_t* part_promo() const { return static_cast<const uint32_t*>(part[1]->data()); }
  join::Table table;
  std::vector<uint8_t> promo_of;  // indexed by p_partkey; 2 = no such part
  size_t parts = 0;
};

inline Columns load_columns(const std::string& dir) {
  Columns c;
  c.rows = load_u32_columns(dir, kColumnNames, c.mem);
  c.parts = load_u32_columns(dir, kPartColumnNames, c.part);
  const uint32_t *keys = c.part_keys(), *promo = c.part_promo();
  c.table = join::make_table(c.parts);
  join::cpu_build(c.table, keys, promo, nullptr, 0, c.parts);
  uint32_t max_key = 0;
  for (size_t i = 0; i < c.parts; ++i) max_key = std::max(max_key, keys[i]);
  c.promo_of.assign(size_t(max_key) + 1, 2);
  for (size_t i = 0; i < c.parts; ++i) c.promo_of[keys[i]] = static_cast<uint8_t>(promo[i] ? 1 : 0);
  return c;
}

// The build side on k CPU threads, as bench/run_query runs it on its P workers (the GPU build is
// engine/hash_join_gpu.hpp). Thread w of k first clears its share of the table (every slot 0xFF);
// once all k have cleared (the caller's barrier: an insert may land in any share), it inserts its
// share of the parts with join::cpu_build_concurrent and returns how many.
inline std::pair<size_t, size_t> share(size_t total, int w, int k) {
  return {total * static_cast<size_t>(w) / static_cast<size_t>(k), total * static_cast<size_t>(w + 1) / static_cast<size_t>(k)};
}
inline void clear_share(Columns& c, int w, int k) {
  const auto [b, e] = share(c.table.cap, w, k);  // in slots, so a share never splits a slot
  std::memset(c.table.slots() + 2 * b, 0xFF, (e - b) * 2 * sizeof(uint32_t));
}
inline uint32_t build_share(Columns& c, int w, int k) {
  const auto [b, e] = share(c.parts, w, k);
  return join::cpu_build_concurrent(c.table, c.part_keys(), c.part_promo(), b, e - b);
}

// Plain scalar loop with the dense part lookup, independent of the hash table: the correctness reference.
inline Result reference(const Columns& c, size_t begin, size_t n) {
  Result r;
  for (size_t i = begin; i < begin + n; ++i) {
    const uint32_t sd = c.shipdate()[i];
    if (sd < kDateLo || sd >= kDateHi) continue;
    const uint32_t k = c.partkey()[i];
    const uint8_t flag = k < c.promo_of.size() ? c.promo_of[k] : 2;
    if (flag == 2) continue;
    const uint64_t rev = static_cast<uint64_t>(c.price()[i]) * (100 - c.discount()[i]);
    r.total += rev;
    if (flag) r.promo += rev;
  }
  return r;
}

// The timed CPU kernel (one worker), as a vectorized engine runs it, in three steps per chunk of up
// to kChunk rows:
//  1. the date test, kBlock = 16 rows at a time in NEON (block_mask): a block with no row of the
//     month (82% of them at 1.2% selectivity) costs four loads, compares and one horizontal max;
//  2. the rows that pass go into a selection vector, and their lines of l_partkey, l_extendedprice
//     and l_discount are prefetched (each selected row is ~80 rows from the next: a random line);
//  3. only the selected rows probe, the hash slots prefetched kAhead probes ahead as in
//     join::cpu_probe, and add to the two sums.
// Until 17 Nov 2026 this was one scalar loop with a branch per row, which clang cannot vectorize: on
// the M3 Ultra it took ~1 ns per row on a P-core, 10x the streaming rate of its 4 B
// (results/Apple-M3-Ultra_20261117-090820_q14 and the two runs after it). On the M1 (one P-core,
// SF10, 2^25 rows) the scalar loop takes 1.35 ns per row, this kernel 0.33-0.40, its steps 1-2 alone
// 0.17: building the selection is not free (branches on the 18% of blocks with a hit), and each
// selected row costs ~4 random cache misses (three column lines and the slot).
inline constexpr size_t kBlock = 16;
inline constexpr size_t kChunk = 4096;
inline constexpr size_t kAhead = 16;

// Bit t is set iff row d[t] of the block is in the month; 0 (the common case) after the any-test.
inline uint32_t block_mask(const uint32_t* d) {
  const uint32x4_t lo = vdupq_n_u32(kDateLo), span = vdupq_n_u32(kDateHi - kDateLo);
  const uint32x4_t a = vcltq_u32(vsubq_u32(vld1q_u32(d), lo), span), b = vcltq_u32(vsubq_u32(vld1q_u32(d + 4), lo), span);
  const uint32x4_t e = vcltq_u32(vsubq_u32(vld1q_u32(d + 8), lo), span), f = vcltq_u32(vsubq_u32(vld1q_u32(d + 12), lo), span);
  if (vmaxvq_u32(vorrq_u32(vorrq_u32(a, b), vorrq_u32(e, f))) == 0) return 0;
  const uint32x4_t w = {1, 2, 4, 8};  // lane weights; the four vectors' bits are disjoint, so add = or
  const uint32x4_t m = vorrq_u32(vorrq_u32(vandq_u32(a, w), vshlq_n_u32(vandq_u32(b, w), 4)),
                                 vorrq_u32(vshlq_n_u32(vandq_u32(e, w), 8), vshlq_n_u32(vandq_u32(f, w), 12)));
  return vaddvq_u32(m);
}

inline void cpu(const Columns& c, size_t begin, size_t n, Result& out) {
  static_assert(kBlock == 16, "block_mask tests 16 rows");
  const uint32_t *sd = c.shipdate() + begin, *pk = c.partkey() + begin, *pr = c.price() + begin, *dc = c.discount() + begin;
  const uint32_t* s = c.table.slots();
  const uint32_t mask = c.table.mask;
  uint64_t total = 0, promo = 0;
  uint32_t sel[kChunk];  // row offsets within the morsel (n < 2^32)
  for (size_t base = 0; base < n; base += kChunk) {
    const size_t m = std::min(kChunk, n - base);
    size_t cnt = 0, j = 0;
    for (; j + kBlock <= m; j += kBlock) {
      for (uint32_t bits = block_mask(sd + base + j); bits; bits &= bits - 1) {
        const uint32_t r = static_cast<uint32_t>(base + j + __builtin_ctz(bits));
        sel[cnt++] = r;
        __builtin_prefetch(pk + r);
        __builtin_prefetch(pr + r);
        __builtin_prefetch(dc + r);
      }
    }
    for (; j < m; ++j)
      if (sd[base + j] - kDateLo < kDateHi - kDateLo) sel[cnt++] = static_cast<uint32_t>(base + j);
    for (size_t t = 0; t < cnt; ++t) {
      if (t + kAhead < cnt) __builtin_prefetch(s + 2 * c.table.slot_of(pk[sel[t + kAhead]]));
      const uint32_t i = sel[t], k = pk[i];
      for (uint32_t h = c.table.slot_of(k);; h = (h + 1) & mask) {
        const uint32_t sk = s[2 * h];
        if (sk == k) {
          const uint64_t rev = static_cast<uint64_t>(pr[i]) * (100 - dc[i]);
          total += rev;
          promo += rev * s[2 * h + 1];
          break;
        }
        if (sk == join::kEmpty) break;
      }
    }
  }
  out.total += total;
  out.promo += promo;
}

// DuckDB's answer (scripts/tpch_reference.py): one line "count promo total" (count: the rows of the
// month). Returns false if the file does not exist.
inline bool load_expected(const std::string& path, Result& r) {
  std::ifstream f(path);
  if (!f) return false;
  uint64_t count = 0;
  if (!(f >> count >> r.promo >> r.total)) throw std::runtime_error(path + ": expected one line 'count promo total'");
  return true;
}

inline std::string describe(const Result& r) {
  std::ostringstream o;
  o << "  total=" << r.total << " promo=" << r.promo << " (x10^4); promo_revenue = "
    << (r.total ? 100.0 * static_cast<double>(r.promo) / static_cast<double>(r.total) : 0.0) << "\n";
  return o.str();
}

}  // namespace mcm::q14
