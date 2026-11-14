#include <metal_stdlib>
using namespace metal;

// ALU-only busy loop, no memory traffic: run right before a timed morsel to bring the GPU clock up
// without refilling the caches (GpuSpin in src/common/bench.hpp, --gpu-spin-ms). Compiled together
// with each benchmark's own kernel file.
kernel void gpu_spin(device uint*   out   [[buffer(0)]],
                     constant uint& iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]])
{
  uint x = gid;
  for (uint i = 0; i < iters; ++i) x = x * 1664525u + 1013904223u;
  if (x == 0xFFFFFFFFu) out[0] = x;  // data-dependent store: keeps the loop from being removed
}
