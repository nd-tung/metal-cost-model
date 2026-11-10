#include <metal_stdlib>
using namespace metal;

// TPC-H Q1 on the uint32 lineitem columns (see src/engine/q1.hpp for the encodings and layout).
// Apple7 (M1) has no 64-bit atomics, so there are none here: each thread keeps its 36 uint64
// accumulators (6 groups x 6 fields) in registers, the threadgroup reduces them without atomics into
// one row of `partials`, and q1_final sums the rows of `partials` into `out`. Both kernels run in one
// serial compute encoder, so q1_final sees every q1_partial write.

constant constexpr uint SLOTS = 6;
constant constexpr uint FIELDS = 6;  // count, sum_qty, sum_base_price, sum_disc_price, sum_charge, sum_disc
constant constexpr uint VALS = SLOTS * FIELDS;
constant constexpr uint MAX_SIMDGROUPS = 8;  // threadgroups of at most 256 threads

struct Q1Params {
  uint n;       // rows in the morsel (buffers are bound at the morsel's first row)
  uint cutoff;  // l_shipdate <= cutoff
};

inline uint slot_of(uint rf, uint ls) {
  uint r = (rf == 'N') ? 1u : ((rf == 'R') ? 2u : 0u);
  return r * 2u + (ls == 'O' ? 1u : 0u);
}

// Sum of a uint64 over the simdgroup (result in lane 0), with two 32-bit shuffles per step.
inline ulong simd_sum_u64(ulong v) {
  for (ushort d = 16; d > 0; d >>= 1) {
    uint2 h = as_type<uint2>(v);
    uint2 o = uint2(simd_shuffle_down(h.x, d), simd_shuffle_down(h.y, d));
    v += as_type<ulong>(o);
  }
  return v;
}

kernel void q1_partial(device const uint* shipdate   [[buffer(0)]],
                       device const uint* returnflag [[buffer(1)]],
                       device const uint* linestatus [[buffer(2)]],
                       device const uint* quantity   [[buffer(3)]],
                       device const uint* price      [[buffer(4)]],
                       device const uint* discount   [[buffer(5)]],
                       device const uint* tax        [[buffer(6)]],
                       device ulong*      partials   [[buffer(7)]],  // [threadgroups][VALS]
                       constant Q1Params& p          [[buffer(8)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint tid  [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
  // Every loop over acc is fully unrolled: constant indices keep the accumulators in registers
  // (a dynamically indexed array would live in device memory).
  ulong acc[VALS];
#pragma unroll
  for (uint k = 0; k < VALS; ++k) acc[k] = 0;

  for (uint i = gid; i < p.n; i += grid) {
    // A row that fails the filter gets slot SLOTS, which matches no accumulator: no masking needed.
    const uint s = shipdate[i] <= p.cutoff ? slot_of(returnflag[i], linestatus[i]) : SLOTS;
    const uint q = quantity[i], e = price[i], d = discount[i], t = tax[i];
    const uint dp = e * (100u - d);  // <= 10494950 * 100: fits 32 bits
    const ulong ch = ulong(dp) * ulong(100u + t);
#pragma unroll
    for (uint k = 0; k < SLOTS; ++k) {
      const bool hit = s == k;
      acc[k * FIELDS + 0] += hit ? 1ul : 0ul;
      acc[k * FIELDS + 1] += hit ? ulong(q) : 0ul;
      acc[k * FIELDS + 2] += hit ? ulong(e) : 0ul;
      acc[k * FIELDS + 3] += hit ? ulong(dp) : 0ul;
      acc[k * FIELDS + 4] += hit ? ch : 0ul;
      acc[k * FIELDS + 5] += hit ? ulong(d) : 0ul;
    }
  }

  threadgroup ulong tg_part[MAX_SIMDGROUPS][VALS];
#pragma unroll
  for (uint k = 0; k < VALS; ++k) {
    const ulong v = simd_sum_u64(acc[k]);
    if (lane == 0) tg_part[sg][k] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < VALS) {
    ulong s = 0;
    for (uint g = 0; g < nsg; ++g) s += tg_part[g][tid];
    partials[tgid * VALS + tid] = s;
  }
}

// ALU-only busy loop, no memory traffic: run right before a timed morsel to bring the GPU clock up
// without refilling the caches (val_q1 --gpu-spin-ms).
kernel void gpu_spin(device uint*   out   [[buffer(0)]],
                     constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
  uint x = gid;
  for (uint i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
  if (x == 0xFFFFFFFFu) out[0] = x;  // data-dependent store: keeps the loop from being removed
}

// One threadgroup per output value: out[v] = sum over rows g of partials[g][v].
kernel void q1_final(device const ulong* partials [[buffer(0)]],
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
  for (uint g = tid; g < rows; g += tgsz) s += partials[g * VALS + v];
  s = simd_sum_u64(s);
  threadgroup ulong part[MAX_SIMDGROUPS];
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong t = 0;
    for (uint g = 0; g < nsg; ++g) t += part[g];
    out[v] = t;
  }
}
