#include <metal_stdlib>
using namespace metal;

// Hash join kernels for calib_join (layout in src/engine/hash_join.hpp): open addressing, linear
// probing, interleaved (key, payload) uint32 slots, empty key 0xFFFFFFFF.
// Build: one 32-bit CAS on the key slot per inserted row (the M1 GPU has no 64-bit atomics), then a
// plain payload store; probes run in later command buffers, so they only see complete slots.
// Probe: match count and payload sum per morsel, fused (not materialized), uint32, wrapping.

constant constexpr uint EMPTY = 0xFFFFFFFFu;
constant constexpr uint HASH_MUL = 0x9E3779B1u;

struct JoinParams {
  uint n;           // rows in the morsel (buffers are bound at its first row)
  uint bits;        // log2 of the table capacity
  uint mask;        // capacity - 1
  uint use_filter;  // build: skip rows whose filter value is 0
};

inline uint slot_of(uint key, uint bits) { return (key * HASH_MUL) >> (32u - bits); }

// Sums v over the threadgroup and adds it to *dst with one atomic (lane 0 of simdgroup 0).
inline void tg_add(uint v, device atomic_uint* dst, threadgroup uint* part, uint lane, uint sg, uint nsg) {
  v = simd_sum(v);
  if (lane == 0) part[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint w = (lane < nsg) ? part[lane] : 0u;
    w = simd_sum(w);
    if (lane == 0) atomic_fetch_add_explicit(dst, w, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void join_build(device const uint*   keys    [[buffer(0)]],
                       device const uint*   payload [[buffer(1)]],
                       device const uint*   filter  [[buffer(2)]],
                       device atomic_uint*  slots   [[buffer(3)]],  // 2 per slot: key, payload
                       device atomic_uint*  stats   [[buffer(4)]],  // [0] rows inserted
                       constant JoinParams& p       [[buffer(5)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
  threadgroup uint part[32];
  uint inserted = 0;
  for (uint i = gid; i < p.n; i += grid) {
    if (p.use_filter != 0 && filter[i] == 0) continue;
    const uint k = keys[i];
    uint h = slot_of(k, p.bits);
    for (;;) {
      uint expected = EMPTY;
      if (atomic_compare_exchange_weak_explicit(&slots[2 * h], &expected, k, memory_order_relaxed,
                                                memory_order_relaxed)) {
        atomic_store_explicit(&slots[2 * h + 1], payload[i], memory_order_relaxed);
        ++inserted;
        break;
      }
      // Taken by another key: next slot. A weak CAS that failed spuriously (slot still empty)
      // retries the same slot.
      if (expected != EMPTY) h = (h + 1) & p.mask;
    }
  }
  tg_add(inserted, &stats[0], part, lane, sg, nsg);
}

kernel void join_probe(device const uint*   keys  [[buffer(0)]],
                       device const uint2*  slots [[buffer(1)]],
                       device atomic_uint*  out   [[buffer(2)]],  // [0] matches, [1] payload sum
                       constant JoinParams& p     [[buffer(3)]],
                       uint gid  [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint nsg  [[simdgroups_per_threadgroup]])
{
  threadgroup uint part[32];
  uint cnt = 0, sum = 0;
  for (uint i = gid; i < p.n; i += grid) {
    const uint k = keys[i];
    uint h = slot_of(k, p.bits);
    for (;;) {
      const uint2 s = slots[h];
      if (s.x == k) {
        ++cnt;
        sum += s.y;
        break;
      }
      if (s.x == EMPTY) break;
      h = (h + 1) & p.mask;
    }
  }
  tg_add(cnt, &out[0], part, lane, sg, nsg);
  tg_add(sum, &out[1], part, lane, sg, nsg);
}
