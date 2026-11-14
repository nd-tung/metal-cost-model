#pragma once
// Sort for calib_sort, CPU side (the GPU side is kernels/sort.metal): LSD radix sort of
// (uint32 key, uint32 row id) pairs with 8-bit digits, the std::sort reference, and the k-way merge
// of sorted runs.
//
// passes = ceil(bits of the largest key / 8), from the column's statistics (as a database keeps
// them): 4 for full 32-bit keys, 3 for l_partkey, 2 for l_shipdate. Pass 1 reads the raw keys and
// generates the row ids (row id = row_base + position). LSD radix sort is stable, so the output is
// sorted by (key, row id): one exact answer, which every device must produce.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcm::sort {

struct Pair {
  uint32_t key, row;
};

inline bool pair_less(const Pair& a, const Pair& b) { return a.key < b.key || (a.key == b.key && a.row < b.row); }

inline int passes_for(uint32_t max_key) {
  const int bits = 32 - __builtin_clz(max_key | 1u);
  return (bits + 7) / 8;
}

// Sorts keys[0, n) into out; tmp is n pairs of scratch. The histograms of all passes are counted in
// one read of the keys; the pass parity is chosen so the last pass writes `out`.
inline void cpu_radix(const uint32_t* keys, size_t n, int passes, Pair* out, Pair* tmp, uint32_t row_base = 0) {
  std::array<std::array<uint32_t, 256>, 4> hist{};
  for (size_t i = 0; i < n; ++i) {
    const uint32_t k = keys[i];
    for (int p = 0; p < passes; ++p) ++hist[p][(k >> (8 * p)) & 255u];
  }
  for (int p = 0; p < passes; ++p) {
    std::array<uint32_t, 256> off;
    uint32_t s = 0;
    for (int d = 0; d < 256; ++d) {
      off[d] = s;
      s += hist[p][d];
    }
    Pair* dst = ((passes - 1 - p) % 2 == 0) ? out : tmp;
    const int shift = 8 * p;
    if (p == 0) {
      for (size_t i = 0; i < n; ++i) {
        const uint32_t k = keys[i];
        dst[off[k & 255u]++] = Pair{k, row_base + static_cast<uint32_t>(i)};
      }
    } else {
      const Pair* src = (dst == out) ? tmp : out;
      for (size_t i = 0; i < n; ++i) {
        const Pair e = src[i];
        dst[off[(e.key >> shift) & 255u]++] = e;
      }
    }
  }
}

// The comparison-sort reference: pairs, then std::sort by (key, row id).
inline void cpu_std(const uint32_t* keys, size_t n, Pair* out) {
  for (size_t i = 0; i < n; ++i) out[i] = Pair{keys[i], static_cast<uint32_t>(i)};
  std::sort(out, out + n, pair_less);
}

// k-way merge of the sorted runs in[begin[r], begin[r + 1]) into out, with a binary min-heap of
// run heads (replace-top + sift-down: log2 k comparisons per element).
inline void cpu_merge(const Pair* in, const std::vector<size_t>& begin, Pair* out) {
  struct Head {
    Pair v;
    uint32_t run;
  };
  const size_t k = begin.size() - 1;
  std::vector<size_t> pos(begin.begin(), begin.end() - 1);
  std::vector<Head> heap;
  heap.reserve(k);
  for (size_t r = 0; r < k; ++r)
    if (pos[r] < begin[r + 1]) heap.push_back({in[pos[r]++], static_cast<uint32_t>(r)});
  auto before = [](const Head& a, const Head& b) { return pair_less(a.v, b.v); };
  auto sift_down = [&](size_t i) {
    const size_t m = heap.size();
    const Head x = heap[i];
    for (;;) {
      size_t c = 2 * i + 1;
      if (c >= m) break;
      if (c + 1 < m && before(heap[c + 1], heap[c])) ++c;
      if (!before(heap[c], x)) break;
      heap[i] = heap[c];
      i = c;
    }
    heap[i] = x;
  };
  for (size_t i = heap.size() / 2; i-- > 0;) sift_down(i);
  size_t o = 0;
  while (!heap.empty()) {
    Head& top = heap[0];
    out[o++] = top.v;
    if (pos[top.run] < begin[top.run + 1]) {
      top.v = in[pos[top.run]++];
    } else {
      top = heap.back();
      heap.pop_back();
      if (heap.empty()) break;
    }
    sift_down(0);
  }
}

// Exact O(n) check: out is sorted by (key, row id), holds every row id in [0, n) once, and each
// element's key is keys[row id]. Together these mean out is the unique sorted answer.
inline bool verify(const uint32_t* keys, size_t n, const Pair* out, std::vector<uint8_t>& seen) {
  seen.assign(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const Pair e = out[i];
    if (e.row >= n || seen[e.row] || keys[e.row] != e.key) return false;
    seen[e.row] = 1;
    if (i > 0 && !pair_less(out[i - 1], e)) return false;
  }
  return true;
}

}  // namespace mcm::sort
