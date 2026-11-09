#include <metal_stdlib>
using namespace metal;

// Aggregation kernels for calib_agg: SUM(val) and SUM(val) GROUP BY key over dense keys.
// Keys are dense after subtracting `base`: group = key - base, 0 <= group < G.
// Sums are uint32 and wrap mod 2^32 (Apple7/M1 has no 64-bit atomic add); the host
// reference wraps identically, so the correctness check is exact.

struct AggParams {
  uint n;       // number of rows
  uint base;    // smallest key
  uint groups;  // G
  uint copies;  // agg_group_tg: private bin copies per threadgroup (1 or simdgroups per tg)
};

constant constexpr uint TG_BINS = 4096;  // 16 KB of threadgroup memory (M1 has 32 KB)

// G = 1: grid-stride sum, SIMD reduction, one atomic per threadgroup (same shape as filter_count).
kernel void agg_sum(device const uint*    vals [[buffer(1)]],
                    device atomic_uint*   out  [[buffer(2)]],
                    constant AggParams&   p    [[buffer(3)]],
                    uint gid  [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint sg   [[simdgroup_index_in_threadgroup]],
                    uint nsg  [[simdgroups_per_threadgroup]])
{
  threadgroup uint partial[32];
  uint s = 0;
  for (uint i = gid; i < p.n; i += grid) s += vals[i];
  s = simd_sum(s);
  if (lane == 0) partial[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint v = (lane < nsg) ? partial[lane] : 0u;
    v = simd_sum(v);
    if (lane == 0) atomic_fetch_add_explicit(out, v, memory_order_relaxed);
  }
}

// Small G (copies * G <= TG_BINS): bins in threadgroup memory, merged into `out` at the end.
// With copies > 1 each simdgroup owns a private copy of the bins, so atomic contention for
// tiny G (e.g. 7 groups) is limited to the 32 lanes of one simdgroup.
kernel void agg_group_tg(device const uint*  keys [[buffer(0)]],
                         device const uint*  vals [[buffer(1)]],
                         device atomic_uint* out  [[buffer(2)]],
                         constant AggParams& p    [[buffer(3)]],
                         uint gid  [[thread_position_in_grid]],
                         uint grid [[threads_per_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint tgsz [[threads_per_threadgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]])
{
  threadgroup atomic_uint bins[TG_BINS];
  const uint total = p.copies * p.groups;
  for (uint b = tid; b < total; b += tgsz) atomic_store_explicit(&bins[b], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint off = (sg % p.copies) * p.groups;
  for (uint i = gid; i < p.n; i += grid)
    atomic_fetch_add_explicit(&bins[off + (keys[i] - p.base)], vals[i], memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint b = tid; b < p.groups; b += tgsz) {
    uint s = 0;
    for (uint c = 0; c < p.copies; ++c) s += atomic_load_explicit(&bins[c * p.groups + b], memory_order_relaxed);
    if (s) atomic_fetch_add_explicit(&out[b], s, memory_order_relaxed);
  }
}

// Large G: one device-memory atomic per row.
kernel void agg_group_global(device const uint*  keys [[buffer(0)]],
                             device const uint*  vals [[buffer(1)]],
                             device atomic_uint* out  [[buffer(2)]],
                             constant AggParams& p    [[buffer(3)]],
                             uint gid  [[thread_position_in_grid]],
                             uint grid [[threads_per_grid]])
{
  for (uint i = gid; i < p.n; i += grid)
    atomic_fetch_add_explicit(&out[keys[i] - p.base], vals[i], memory_order_relaxed);
}
