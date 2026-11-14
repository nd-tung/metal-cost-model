#include <metal_stdlib>
using namespace metal;

// TPC-H Q6 on the uint32 lineitem columns (see src/engine/q6.hpp for the encodings). As in
// kernels/q1.metal there are no atomics (Apple7 has no 64-bit ones): each thread keeps its count
// and uint64 revenue in registers, the threadgroup reduces them into one row of `partials`, and
// q6_final sums the rows. Both kernels run in one serial compute encoder. Branchless: every row
// loads all four columns, so the traffic is 16 B per row whatever the predicate selects.

namespace q6 {

constant constexpr uint VALS = 2;            // count, revenue
constant constexpr uint MAX_SIMDGROUPS = 8;  // threadgroups of at most 256 threads

// Sum of a uint64 over the simdgroup (result in lane 0), with two 32-bit shuffles per step.
inline ulong simd_sum_u64(ulong v) {
  for (ushort d = 16; d > 0; d >>= 1) {
    uint2 h = as_type<uint2>(v);
    uint2 o = uint2(simd_shuffle_down(h.x, d), simd_shuffle_down(h.y, d));
    v += as_type<ulong>(o);
  }
  return v;
}

}  // namespace q6

struct Q6Params {
  uint n;                   // rows in the morsel (buffers are bound at the morsel's first row)
  uint date_lo, date_span;  // date_lo <= l_shipdate < date_lo + date_span
  uint disc_lo, disc_span;  // disc_lo <= l_discount <= disc_lo + disc_span
  uint qty_hi;              // l_quantity < qty_hi
};

kernel void q6_partial(device const uint* shipdate [[buffer(0)]],
                       device const uint* discount [[buffer(1)]],
                       device const uint* quantity [[buffer(2)]],
                       device const uint* price    [[buffer(3)]],
                       device ulong*      partials [[buffer(4)]],  // [threadgroups][VALS]
                       constant Q6Params& p        [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
  uint cnt = 0;
  ulong rev = 0;
  for (uint i = gid; i < p.n; i += grid) {
    const uint sd = shipdate[i], d = discount[i], q = quantity[i], e = price[i];
    // Two-sided ranges as one unsigned compare each; & (not &&): no branch, all four loads happen.
    const uint hit = uint(sd - p.date_lo < p.date_span) & uint(d - p.disc_lo <= p.disc_span) & uint(q < p.qty_hi);
    cnt += hit;
    rev += ulong(e * d) * ulong(hit);  // e * d <= 10494950 * 7: fits 32 bits
  }

  threadgroup ulong tg_part[q6::MAX_SIMDGROUPS][q6::VALS];
  const ulong c = ulong(simd_sum(cnt));  // <= 32 lanes x rows per thread: fits 32 bits
  const ulong r = q6::simd_sum_u64(rev);
  if (lane == 0) {
    tg_part[sg][0] = c;
    tg_part[sg][1] = r;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < q6::VALS) {
    ulong s = 0;
    for (uint g = 0; g < nsg; ++g) s += tg_part[g][tid];
    partials[tgid * q6::VALS + tid] = s;
  }
}

// One threadgroup per output value: out[v] = sum over rows g of partials[g][v].
kernel void q6_final(device const ulong* partials [[buffer(0)]],
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
  for (uint g = tid; g < rows; g += tgsz) s += partials[g * q6::VALS + v];
  s = q6::simd_sum_u64(s);
  threadgroup ulong part[q6::MAX_SIMDGROUPS];
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong t = 0;
    for (uint g = 0; g < nsg; ++g) t += part[g];
    out[v] = t;
  }
}
