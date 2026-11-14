#include <metal_stdlib>
using namespace metal;

// LSD radix sort of (uint32 key, uint32 row id) pairs with 8-bit digits, for calib_sort (the CPU side
// and the pass count are in src/engine/radix_sort.hpp). Per pass, three dispatches in one serial
// compute encoder:
//   sort_hist    : each of G threadgroups counts the digits of its contiguous chunk -> hist[d * G + g]
//   sort_scan    : one threadgroup turns hist into exclusive offsets in digit-major order (all chunks
//                  of digit 0, then digit 1, ...): chunk g's digit-d elements start at offset[d * G + g]
//   sort_scatter : each threadgroup walks its chunk tile by tile, in order; it sorts each tile stably by
//                  the digit in threadgroup memory (8 one-bit split rounds on packed (digit, index)
//                  words), writes every element to its chunk's running offset for the digit plus its
//                  rank within the digit, and advances the offsets. Stable across tiles and chunks.
// The first pass reads the raw keys and generates the row ids (row id = position in the morsel).

constant constexpr uint THREADS = 256;  // per sort_hist / sort_scatter threadgroup; == BINS
constant constexpr uint ITEMS = 8;
constant constexpr uint TILE = THREADS * ITEMS;  // 2048 elements per tile
constant constexpr uint BINS = 256;
constant constexpr uint SIMDGROUPS = THREADS / 32;

struct SortParams {
  uint n;       // elements in the morsel
  uint shift;   // digit = (key >> shift) & 255
  uint groups;  // G
  uint chunk;   // elements per threadgroup, a multiple of TILE (the last chunk may be shorter)
  uint first;   // 1: input is the raw keys; 0: input is (key, row id) pairs
};

inline uint key_at(device const uint* keys, device const uint2* src, uint i, uint first) {
  return first != 0 ? keys[i] : src[i].x;
}

// Exclusive prefix sum over the 256 threads of a threadgroup. Every thread must call it.
inline uint block_exclusive_scan(uint v, threadgroup uint* sums, uint lane, uint sg) {
  const uint x = simd_prefix_exclusive_sum(v);
  if (lane == 31) sums[sg] = x + v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint off = 0;
  for (uint s = 0; s < sg; ++s) off += sums[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);  // sums is reused by the next call
  return x + off;
}

kernel void sort_hist(device const uint*   keys [[buffer(0)]],
                      device const uint2*  src  [[buffer(1)]],
                      device uint*         hist [[buffer(3)]],
                      constant SortParams& p    [[buffer(4)]],
                      uint g   [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint sg  [[simdgroup_index_in_threadgroup]])
{
  threadgroup atomic_uint h[SIMDGROUPS][BINS];  // one copy per simdgroup: less atomic contention
  for (uint b = tid; b < SIMDGROUPS * BINS; b += THREADS)
    atomic_store_explicit(&h[b / BINS][b % BINS], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint begin = g * p.chunk, end = min(begin + p.chunk, p.n);
  for (uint i = begin + tid; i < end; i += THREADS)
    atomic_fetch_add_explicit(&h[sg][(key_at(keys, src, i, p.first) >> p.shift) & 255u], 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint c = 0;
  for (uint s = 0; s < SIMDGROUPS; ++s) c += atomic_load_explicit(&h[s][tid], memory_order_relaxed);
  hist[tid * p.groups + g] = c;  // tid is the digit (THREADS == BINS)
}

kernel void sort_scan(device uint*         hist [[buffer(3)]],
                      constant SortParams& p    [[buffer(4)]],
                      uint tid  [[thread_index_in_threadgroup]],
                      uint tgsz [[threads_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint sg   [[simdgroup_index_in_threadgroup]],
                      uint nsg  [[simdgroups_per_threadgroup]])
{
  threadgroup uint sums[32];
  threadgroup uint offs[32];
  const uint total = BINS * p.groups;
  const uint per = (total + tgsz - 1) / tgsz;
  const uint begin = min(tid * per, total), end = min(begin + per, total);
  uint s = 0;
  for (uint i = begin; i < end; ++i) s += hist[i];
  const uint x = simd_prefix_exclusive_sum(s);
  if (lane == 31) sums[sg] = x + s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) offs[lane] = simd_prefix_exclusive_sum(lane < nsg ? sums[lane] : 0u);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint run = x + offs[sg];
  for (uint i = begin; i < end; ++i) {
    const uint v = hist[i];
    hist[i] = run;
    run += v;
  }
}

kernel void sort_scatter(device const uint*   keys    [[buffer(0)]],
                         device const uint2*  src     [[buffer(1)]],
                         device uint2*        dst     [[buffer(2)]],
                         device const uint*   offsets [[buffer(3)]],
                         constant SortParams& p       [[buffer(4)]],
                         uint g    [[threadgroup_position_in_grid]],
                         uint tid  [[thread_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg   [[simdgroup_index_in_threadgroup]])
{
  threadgroup uint A[TILE], B[TILE];  // packed words: digit << 16 | index in the tile
  threadgroup uint run[BINS];         // the chunk's next output position per digit
  threadgroup uint start[BINS];       // first position of each digit in the sorted tile
  threadgroup atomic_uint cnt[BINS];  // digit counts of the tile
  threadgroup uint sums[SIMDGROUPS];
  threadgroup uint total_zeros;

  run[tid] = offsets[tid * p.groups + g];
  const uint cbegin = g * p.chunk, cend = min(cbegin + p.chunk, p.n);
  for (uint base = cbegin; base < cend; base += TILE) {
    const uint valid = min(TILE, cend - base);
    atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < ITEMS; ++j) {
      const uint i = j * THREADS + tid;  // strided: coalesced loads
      uint d = 255u;                     // padding: sorts after the real digit-255 elements, never written
      if (i < valid) {
        d = (key_at(keys, src, base + i, p.first) >> p.shift) & 255u;
        atomic_fetch_add_explicit(&cnt[d], 1u, memory_order_relaxed);
      }
      A[i] = (d << 16) | i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    start[tid] = block_exclusive_scan(atomic_load_explicit(&cnt[tid], memory_order_relaxed), sums, lane, sg);

    // Stable sort of the tile by digit: 8 one-bit split rounds; thread t owns positions [t*ITEMS, (t+1)*ITEMS).
    threadgroup uint* in = A;
    threadgroup uint* out = B;
    for (uint bit = 16; bit < 24; ++bit) {
      uint v[ITEMS];
      uint zeros = 0;
      for (uint j = 0; j < ITEMS; ++j) {
        v[j] = in[tid * ITEMS + j];
        zeros += ((v[j] >> bit) & 1u) ^ 1u;
      }
      const uint zb = block_exclusive_scan(zeros, sums, lane, sg);
      if (tid == THREADS - 1) total_zeros = zb + zeros;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint Z = total_zeros;
      uint z = zb;  // zeros before the current position
      for (uint j = 0; j < ITEMS; ++j) {
        const uint pos = tid * ITEMS + j;
        const uint one = (v[j] >> bit) & 1u;
        out[one != 0 ? Z + (pos - z) : z] = v[j];
        z += one ^ 1u;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      threadgroup uint* t = in;
      in = out;
      out = t;
    }

    // Write in sorted order (strided positions: consecutive threads write consecutive addresses of a digit run).
    for (uint j = 0; j < ITEMS; ++j) {
      const uint q = j * THREADS + tid;
      const uint w = in[q];
      const uint i = w & 0xFFFFu, d = w >> 16;
      if (i < valid) {
        const uint gi = base + i;
        dst[run[d] + (q - start[d])] = (p.first != 0) ? uint2(keys[gi], gi) : src[gi];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run[tid] += atomic_load_explicit(&cnt[tid], memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
