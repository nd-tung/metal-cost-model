#include <metal_stdlib>
using namespace metal;

struct FilterParams {
  uint n;          // number of elements
  uint threshold;  // predicate: v < threshold
};

// Filter + count: the simplest streaming operator. Grid-stride loop, SIMD
// reduction, one atomic per threadgroup. Used to calibrate launch, residency,
// sync and bandwidth terms with minimal compute.
kernel void filter_count(device const uint*    in    [[buffer(0)]],
                         device atomic_uint*   out   [[buffer(1)]],
                         constant FilterParams& p    [[buffer(2)]],
                         uint gid   [[thread_position_in_grid]],
                         uint grid  [[threads_per_grid]],
                         uint lane  [[thread_index_in_simdgroup]],
                         uint sg    [[simdgroup_index_in_threadgroup]],
                         uint nsg   [[simdgroups_per_threadgroup]])
{
  threadgroup uint partial[32];

  uint c = 0;
  for (uint i = gid; i < p.n; i += grid) {
    c += (in[i] < p.threshold) ? 1u : 0u;
  }
  c = simd_sum(c);
  if (lane == 0) partial[sg] = c;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0) {
    uint v = (lane < nsg) ? partial[lane] : 0u;
    v = simd_sum(v);
    if (lane == 0) atomic_fetch_add_explicit(out, v, memory_order_relaxed);
  }
}
