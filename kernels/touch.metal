#include <metal_stdlib>
using namespace metal;

// Cache-state producers (Toucher in src/common/bench.hpp; calib_filter --cache gpu-write and
// gpu-read): one GPU pass over a morsel's rows right before the timed morsel, untimed, so the morsel
// finds its input where a GPU operator that had just produced it (touch_write) or scanned it
// (touch_read) would leave it. touch_write adds p.zero, 0 at run time, to every row in place: the
// data is unchanged and every cache line is written. Compiled together with the benchmark's kernels.
struct TouchParams {
  uint n;     // rows (buffers are bound at the morsel's first row)
  uint zero;  // 0; a runtime value, so the compiler keeps the stores
};

kernel void touch_write(device uint*           x   [[buffer(0)]],
                        constant TouchParams&  p   [[buffer(2)]],
                        uint gid  [[thread_position_in_grid]],
                        uint grid [[threads_per_grid]])
{
  for (uint i = gid; i < p.n; i += grid) x[i] += p.zero;
}

kernel void touch_read(device const uint*     x   [[buffer(0)]],
                       device atomic_uint*    out [[buffer(1)]],
                       constant TouchParams&  p   [[buffer(2)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
  threadgroup uint partial[32];
  uint s = 0;
  for (uint i = gid; i < p.n; i += grid) s += x[i];
  s = simd_sum(s);
  if (lane == 0) partial[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint v = (lane < nsg) ? partial[lane] : 0u;
    v = simd_sum(v);
    if (lane == 0) atomic_fetch_add_explicit(out, v, memory_order_relaxed);  // keeps the loads
  }
}
