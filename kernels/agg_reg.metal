#include <metal_stdlib>
using namespace metal;

// Register aggregation, the calibration kernel of bench/calib_reg.cpp: SUM GROUP BY over G dense
// keys with F sums per group, every accumulator in a register. Each thread keeps U = G x F uint64
// accumulators and updates all U for every row, branchless: a row adds to its own group's F sums
// and adds 0 to the others. A GPU group-by over a handful of groups works this way to avoid
// atomics (kernels/q1.metal does, with 6 groups x 6 fields); the cost model takes its cost per
// update from this kernel. The f-th sum of a group adds val >> f, so no two accumulators can be
// merged. As in kernels/q1.metal there are no atomics (Apple7 has no 64-bit ones): each threadgroup
// writes one row of U partials, and agg_reg_final sums the rows.
//
// COLS (calib_reg --cols, 2..7) is the number of columns a row loads: the key, the value, and
// COLS - 2 extra columns added to the value (x0..x4), so a variant reads 4 COLS bytes per row and
// makes the same updates. Q1 loads 7 columns. COLS is a function constant: the compiler drops the
// loads a variant does not use, as in kernels/stream.metal.
constant uint COLS [[function_constant(0)]];

namespace reg {

constant constexpr uint MAX_SIMDGROUPS = 8;  // agg_reg_partial: threadgroups of at most 256 threads
constant constexpr uint MAX_FINAL_SIMDGROUPS = 32;

// Sum of a uint64 over the simdgroup (result in lane 0), with two 32-bit shuffles per step.
inline ulong simd_sum_u64(ulong v) {
  for (ushort d = 16; d > 0; d >>= 1) {
    uint2 h = as_type<uint2>(v);
    uint2 o = uint2(simd_shuffle_down(h.x, d), simd_shuffle_down(h.y, d));
    v += as_type<ulong>(o);
  }
  return v;
}

}  // namespace reg

struct RegParams {
  uint n;     // rows in the morsel (buffers are bound at the morsel's first row)
  uint base;  // smallest key
};

// G and F are template parameters, so every loop over the accumulators unrolls with constant
// indices and the accumulators stay in registers, as q1_partial's. (With G and F as function
// constants the M1's compiler kept them in memory: at U = 42 the whole table took 256 ms against
// 20 ms.) One kernel per (G, F), named agg_reg_g<G>_f<F>, below.
template <uint G, uint F>
inline void agg_reg_rows(device const uint* keys, device const uint* vals, device const uint* x0,
                         device const uint* x1, device const uint* x2, device const uint* x3,
                         device const uint* x4, device ulong* partials,
                         threadgroup ulong* tg_part,  // [MAX_SIMDGROUPS][G * F]
                         constant RegParams& p, uint gid, uint grid, uint tgid, uint tid, uint lane,
                         uint sg, uint nsg)
{
  constexpr uint U = G * F;
  ulong acc[U];
#pragma unroll
  for (uint u = 0; u < U; ++u) acc[u] = 0;

  for (uint i = gid; i < p.n; i += grid) {
    const uint g = keys[i] - p.base;
    uint v = vals[i];
    if (COLS > 2) v += x0[i];
    if (COLS > 3) v += x1[i];
    if (COLS > 4) v += x2[i];
    if (COLS > 5) v += x3[i];
    if (COLS > 6) v += x4[i];
#pragma unroll
    for (uint j = 0; j < G; ++j) {
      const bool hit = g == j;
#pragma unroll
      for (uint f = 0; f < F; ++f) acc[j * F + f] += hit ? ulong(v >> f) : 0ul;
    }
  }

#pragma unroll
  for (uint u = 0; u < U; ++u) {
    const ulong s = reg::simd_sum_u64(acc[u]);
    if (lane == 0) tg_part[sg * U + u] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < U) {
    ulong s = 0;
    for (uint k = 0; k < nsg; ++k) s += tg_part[k * U + tid];
    partials[tgid * U + tid] = s;
  }
}

#define AGG_REG_KERNEL(G, F)                                                                        \
  kernel void agg_reg_g##G##_f##F(device const uint* keys     [[buffer(0)]],                        \
                                  device const uint* vals     [[buffer(1)]],                        \
                                  device ulong*      partials [[buffer(2)]],                        \
                                  constant RegParams& p       [[buffer(3)]],                        \
                                  device const uint* x0       [[buffer(4)]],                        \
                                  device const uint* x1       [[buffer(5)]],                        \
                                  device const uint* x2       [[buffer(6)]],                        \
                                  device const uint* x3       [[buffer(7)]],                        \
                                  device const uint* x4       [[buffer(8)]],                        \
                                  uint gid  [[thread_position_in_grid]],                            \
                                  uint grid [[threads_per_grid]],                                   \
                                  uint tgid [[threadgroup_position_in_grid]],                       \
                                  uint tid  [[thread_index_in_threadgroup]],                        \
                                  uint lane [[thread_index_in_simdgroup]],                          \
                                  uint sg   [[simdgroup_index_in_threadgroup]],                     \
                                  uint nsg  [[simdgroups_per_threadgroup]]) {                       \
    threadgroup ulong tg_part[reg::MAX_SIMDGROUPS * G * F];                                         \
    agg_reg_rows<G, F>(keys, vals, x0, x1, x2, x3, x4, partials, tg_part, p, gid, grid, tgid, tid,  \
                       lane, sg, nsg);                                                              \
  }

// G = 7: l_shipmode (the calibration's key column); F = 1..8 sums per group.
AGG_REG_KERNEL(7, 1)
AGG_REG_KERNEL(7, 2)
AGG_REG_KERNEL(7, 3)
AGG_REG_KERNEL(7, 4)
AGG_REG_KERNEL(7, 5)
AGG_REG_KERNEL(7, 6)
AGG_REG_KERNEL(7, 7)
AGG_REG_KERNEL(7, 8)

struct RegFinalParams {
  uint rows;  // rows of partials (agg_reg_partial's threadgroups)
  uint vals;  // values per row: G * F
};

// One threadgroup per value, as q1_final: out[v] = sum over rows r of partials[r][v].
kernel void agg_reg_final(device const ulong* partials  [[buffer(0)]],
                          device ulong*       out       [[buffer(1)]],
                          constant RegFinalParams& f    [[buffer(2)]],
                          uint v    [[threadgroup_position_in_grid]],
                          uint tid  [[thread_index_in_threadgroup]],
                          uint tgsz [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]],
                          uint nsg  [[simdgroups_per_threadgroup]])
{
  ulong s = 0;
  for (uint r = tid; r < f.rows; r += tgsz) s += partials[r * f.vals + v];
  s = reg::simd_sum_u64(s);
  threadgroup ulong part[reg::MAX_FINAL_SIMDGROUPS];
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    ulong t = 0;
    for (uint k = 0; k < nsg; ++k) t += part[k];
    out[v] = t;
  }
}
