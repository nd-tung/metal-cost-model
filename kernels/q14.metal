#include <metal_stdlib>
using namespace metal;

// TPC-H Q14 on the uint32 lineitem columns and the part hash table of src/engine/hash_join.hpp (see
// src/engine/q14.hpp for the encodings): a fused shipdate filter, hash probe of part (payload: the
// PROMO flag) and two sums. As in kernels/q6.metal there are no atomics: each thread keeps its two
// uint64 sums in registers, the threadgroup reduces them into one row of `partials`, and q14_final
// sums the rows. Unlike Q6 the kernel branches: only the rows of the month (1.2% of lineitem) load
// l_partkey, l_extendedprice and l_discount and probe the table, so the traffic per row is 4 B for
// the rest of the rows plus what the selected rows pull in.

namespace q14 {

constant constexpr uint VALS = 2;            // total revenue, promo revenue
constant constexpr uint MAX_SIMDGROUPS = 8;  // threadgroups of at most 256 threads
constant constexpr uint EMPTY = 0xFFFFFFFFu;
constant constexpr uint HASH_MUL = 0x9E3779B1u;

inline ulong simd_sum_u64(ulong v) {
  for (ushort d = 16; d > 0; d >>= 1) {
    uint2 h = as_type<uint2>(v);
    uint2 o = uint2(simd_shuffle_down(h.x, d), simd_shuffle_down(h.y, d));
    v += as_type<ulong>(o);
  }
  return v;
}

}  // namespace q14

struct Q14Params {
  uint n;                   // rows in the morsel (buffers are bound at the morsel's first row)
  uint date_lo, date_span;  // date_lo <= l_shipdate < date_lo + date_span
  uint bits, mask;          // the hash table: capacity 2^bits (hash_join.hpp)
};

kernel void q14_partial(device const uint*   shipdate [[buffer(0)]],
                        device const uint*   partkey  [[buffer(1)]],
                        device const uint*   price    [[buffer(2)]],
                        device const uint*   discount [[buffer(3)]],
                        device const uint2*  slots    [[buffer(4)]],  // (key, promo flag)
                        device ulong*        partials [[buffer(5)]],  // [threadgroups][VALS]
                        constant Q14Params&  p        [[buffer(6)]],
                        uint gid  [[thread_position_in_grid]],
                        uint grid [[threads_per_grid]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint nsg  [[simdgroups_per_threadgroup]])
{
  ulong total = 0, promo = 0;
  for (uint i = gid; i < p.n; i += grid) {
    if (shipdate[i] - p.date_lo >= p.date_span) continue;
    const uint k = partkey[i];
    uint h = (k * q14::HASH_MUL) >> (32 - p.bits);
    uint flag = 0, found = 0;
    for (;;) {
      const uint2 s = slots[h];
      if (s.x == k) { flag = s.y; found = 1; break; }
      if (s.x == q14::EMPTY) break;
      h = (h + 1) & p.mask;
    }
    const ulong rev = ulong(price[i]) * ulong(100 - discount[i]) * ulong(found);  // x10^4
    total += rev;
    promo += rev * ulong(flag);
  }

  threadgroup ulong tg_part[q14::MAX_SIMDGROUPS][q14::VALS];
  const ulong t = q14::simd_sum_u64(total);
  const ulong r = q14::simd_sum_u64(promo);
  if (lane == 0) {
    tg_part[sg][0] = t;
    tg_part[sg][1] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < q14::VALS) {
    ulong s = 0;
    for (uint g = 0; g < nsg; ++g) s += tg_part[g][tid];
    partials[tgid * q14::VALS + tid] = s;
  }
}

// One threadgroup per output value: out[v] = sum over rows g of partials[g][v].
kernel void q14_final(device const ulong* partials [[buffer(0)]],
                      device ulong*       out      [[buffer(1)]],
                      constant uint&      rows     [[buffer(2)]],
                      uint v    [[threadgroup_position_in_grid]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint tgsz [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]])
{
  ulong s = 0;
  for (uint g = tid; g < rows; g += tgsz) s += partials[g * q14::VALS + v];
  s = q14::simd_sum_u64(s);
  threadgroup ulong part[q14::MAX_SIMDGROUPS];
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong t = 0;
    for (uint g = 0; g < nsg; ++g) t += part[g];
    out[v] = t;
  }
}
