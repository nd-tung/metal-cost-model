#include <metal_stdlib>
using namespace metal;

// Diagnostic kernels (bench/val_stream.cpp), part of no query. On the M3 Ultra the fused Q1 kernel
// streams at 81% of the filter's pipelined rate and Q6's at 96% (README, Q1). These run on Q1's
// columns with Q1's grid and Q1's two dispatches per morsel (partial, then final), and each keeps
// one half of q1_partial:
//   stream_partial: loads the first COLS of Q1's 7 columns (function constant 0) and sums them into
//                   one uint64 per thread: Q1's loads without its aggregation.
//   agg_partial:    loads l_shipdate alone and runs q1_partial's aggregation (36 uint64 accumulators
//                   per thread, branchless) on values derived from it: Q1's arithmetic and registers
//                   on one stream.
// As in kernels/q1.metal there are no atomics: each threadgroup writes one row of `partials`, and
// stream_final sums the rows.

namespace stream {

constant constexpr uint SLOTS = 6;
constant constexpr uint FIELDS = 6;
constant constexpr uint VALS = SLOTS * FIELDS;  // agg_partial's values per row of partials (stream_partial: 1)
constant constexpr uint MAX_SIMDGROUPS = 8;     // partial kernels: threadgroups of at most 256 threads
constant constexpr uint MAX_FINAL_SIMDGROUPS = 32;

// As kernels/q1.metal.
inline uint slot_of(uint rf, uint ls) {
  uint r = (rf == 'N') ? 1u : ((rf == 'R') ? 2u : 0u);
  return r * 2u + (ls == 'O' ? 1u : 0u);
}

inline ulong simd_sum_u64(ulong v) {
  for (ushort d = 16; d > 0; d >>= 1) {
    uint2 h = as_type<uint2>(v);
    uint2 o = uint2(simd_shuffle_down(h.x, d), simd_shuffle_down(h.y, d));
    v += as_type<ulong>(o);
  }
  return v;
}

}  // namespace stream

constant uint COLS [[function_constant(0)]];  // stream_partial: columns loaded per row, 1..7

struct StreamParams {
  uint n;       // rows in the morsel (buffers are bound at the morsel's first row)
  uint cutoff;  // agg_partial: l_shipdate <= cutoff, as Q1
};

kernel void stream_partial(device const uint* c0       [[buffer(0)]],
                           device const uint* c1       [[buffer(1)]],
                           device const uint* c2       [[buffer(2)]],
                           device const uint* c3       [[buffer(3)]],
                           device const uint* c4       [[buffer(4)]],
                           device const uint* c5       [[buffer(5)]],
                           device const uint* c6       [[buffer(6)]],
                           device ulong*      partials [[buffer(7)]],  // [threadgroups]
                           constant StreamParams& p    [[buffer(8)]],
                           uint gid  [[thread_position_in_grid]],
                           uint grid [[threads_per_grid]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint tid  [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg   [[simdgroup_index_in_threadgroup]],
                           uint nsg  [[simdgroups_per_threadgroup]])
{
  // COLS is fixed when the pipeline is built, so the branches fold away: COLS loads per row in
  // straight-line code, as q1_partial's 7.
  ulong acc = 0;
  for (uint i = gid; i < p.n; i += grid) {
    uint s = c0[i];
    if (COLS > 1) s += c1[i];
    if (COLS > 2) s += c2[i];
    if (COLS > 3) s += c3[i];
    if (COLS > 4) s += c4[i];
    if (COLS > 5) s += c5[i];
    if (COLS > 6) s += c6[i];
    acc += ulong(s);  // every column is below 2^24: a row's sum fits 32 bits
  }

  threadgroup ulong tg_part[stream::MAX_SIMDGROUPS];
  const ulong v = stream::simd_sum_u64(acc);
  if (lane == 0) tg_part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong s = 0;
    for (uint g = 0; g < nsg; ++g) s += tg_part[g];
    partials[tgid] = s;
  }
}

kernel void agg_partial(device const uint* shipdate [[buffer(0)]],  // buffers 1-6 are bound, not read
                        device ulong*      partials [[buffer(7)]],  // [threadgroups][VALS]
                        constant StreamParams& p    [[buffer(8)]],
                        uint gid  [[thread_position_in_grid]],
                        uint grid [[threads_per_grid]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tid  [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint sg   [[simdgroup_index_in_threadgroup]],
                        uint nsg  [[simdgroups_per_threadgroup]])
{
  ulong acc[stream::VALS];
#pragma unroll
  for (uint k = 0; k < stream::VALS; ++k) acc[k] = 0;

  for (uint i = gid; i < p.n; i += grid) {
    // Q1's other six values derived from l_shipdate (bench/val_stream.cpp derive() is the same):
    // a few bit operations in place of six loads. The flags hit all six groups.
    const uint x = shipdate[i];
    const uint rf = 'A' + (x & 31u), ls = 'F' + ((x >> 5) & 15u);
    const uint q = x & 63u, e = x, d = x & 7u, t = (x >> 3) & 7u;
    // From here on q1_partial's loop body.
    const uint s = x <= p.cutoff ? stream::slot_of(rf, ls) : stream::SLOTS;
    const uint dp = e * (100u - d);
    const ulong ch = ulong(dp) * ulong(100u + t);
#pragma unroll
    for (uint k = 0; k < stream::SLOTS; ++k) {
      const bool hit = s == k;
      acc[k * stream::FIELDS + 0] += hit ? 1ul : 0ul;
      acc[k * stream::FIELDS + 1] += hit ? ulong(q) : 0ul;
      acc[k * stream::FIELDS + 2] += hit ? ulong(e) : 0ul;
      acc[k * stream::FIELDS + 3] += hit ? ulong(dp) : 0ul;
      acc[k * stream::FIELDS + 4] += hit ? ch : 0ul;
      acc[k * stream::FIELDS + 5] += hit ? ulong(d) : 0ul;
    }
  }

  threadgroup ulong tg_part[stream::MAX_SIMDGROUPS][stream::VALS];
#pragma unroll
  for (uint k = 0; k < stream::VALS; ++k) {
    const ulong v = stream::simd_sum_u64(acc[k]);
    if (lane == 0) tg_part[sg][k] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < stream::VALS) {
    ulong s = 0;
    for (uint g = 0; g < nsg; ++g) s += tg_part[g][tid];
    partials[tgid * stream::VALS + tid] = s;
  }
}

struct StreamFinalParams {
  uint rows;  // rows of partials (the partial kernel's threadgroups)
  uint vals;  // values per row: 1 (stream_partial) or VALS (agg_partial)
};

// One threadgroup per value, as q1_final: out[v] = sum over rows g of partials[g][v].
kernel void stream_final(device const ulong* partials   [[buffer(0)]],
                         device ulong*       out        [[buffer(1)]],
                         constant StreamFinalParams& f  [[buffer(2)]],
                         uint v    [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tgsz [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]],
                         uint nsg  [[simdgroups_per_threadgroup]])
{
  ulong s = 0;
  for (uint g = tid; g < f.rows; g += tgsz) s += partials[g * f.vals + v];
  s = stream::simd_sum_u64(s);
  threadgroup ulong part[stream::MAX_FINAL_SIMDGROUPS];
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong t = 0;
    for (uint g = 0; g < nsg; ++g) t += part[g];
    out[v] = t;
  }
}
