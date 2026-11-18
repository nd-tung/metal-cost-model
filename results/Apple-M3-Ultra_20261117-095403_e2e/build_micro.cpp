// Parallel hash build of part (default all 2M keys, 2^22 slots; argv[2]: the first N keys only, the table
// sized for N): clear and insert phases apart, k threads, kernels:
//   cas     relaxed CAS per key (join::cpu_build_concurrent)
//   cas-pf  the same, the slot of key i+16 prefetched (write) before key i
//   plain   plain stores, k=1 only (join::cpu_build, the calibrated kernel)
//   plain-pf plain stores with the prefetch, k=1 only
// Build: clang++ -std=c++20 -O3 -Isrc -isystem third_party/metal-cpp build_micro.cpp build/libmcm_common.a \
//          -framework Metal -framework Foundation -o build_micro   (from the project root)
// usage: build_micro [data/tpch-sf10] [N keys] [kinds, comma-separated]
#include <pthread/qos.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include "engine/columns.hpp"
#include "engine/hash_join.hpp"
using namespace mcm;
using Clk = std::chrono::steady_clock;
static double us(Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
static constexpr size_t A = 16;

static void insert(join::Table& t, const uint32_t* keys, const uint32_t* pay, size_t b, size_t e, const std::string& kind) {
  uint32_t* s = t.slots();
  const bool pf = kind.find("-pf") != std::string::npos, cas = kind.rfind("cas", 0) == 0;
  for (size_t i = b; i < e; ++i) {
    if (pf && i + A < e) __builtin_prefetch(s + 2 * t.slot_of(keys[i + A]), 1, 3);
    const uint32_t k = keys[i];
    uint32_t h = t.slot_of(k);
    if (cas) {
      for (;; h = (h + 1) & t.mask) {
        std::atomic_ref<uint32_t> key(s[2 * h]);
        uint32_t ex = join::kEmpty;
        if (key.load(std::memory_order_relaxed) == join::kEmpty &&
            key.compare_exchange_strong(ex, k, std::memory_order_relaxed, std::memory_order_relaxed)) break;
      }
    } else {
      while (s[2 * h] != join::kEmpty) h = (h + 1) & t.mask;
      s[2 * h] = k;
    }
    s[2 * h + 1] = pay[i];
  }
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "data/tpch-sf10";
  const char* const names[2] = {"p_partkey", "p_promo"};
  std::unique_ptr<PageAlloc> cols[2];
  const size_t all = load_u32_columns(dir, names, cols);
  const size_t N = argc > 2 ? std::min<size_t>(std::stoull(argv[2]), all) : all;
  std::string kinds = argc > 3 ? argv[3] : "plain,plain-pf,cas,cas-pf";
  const uint32_t *keys = static_cast<const uint32_t*>(cols[0]->data()), *pay = static_cast<const uint32_t*>(cols[1]->data());
  join::Table t = join::make_table(N);
  std::printf("keys,kind,k,clear_us,insert_us,total_us\n");
  std::vector<std::string> kl;
  for (size_t a = 0, b; a <= kinds.size(); a = b + 1) { b = kinds.find(',', a); if (b == std::string::npos) b = kinds.size(); kl.push_back(kinds.substr(a, b - a)); }
  for (const std::string& kind : kl) {
    for (int k : {1, 2, 4, 8, 12, 16, 24}) {
      if (kind.rfind("plain", 0) == 0 && k != 1) continue;
      std::vector<double> c_us, i_us, t_us;
      for (int rep = 0; rep < 7; ++rep) {
        std::atomic<int> ready{0}, go{0}, cleared{0}, done{0};
        Clk::time_point t0, tc, te;
        std::vector<std::thread> th;
        for (int w = 0; w < k; ++w)
          th.emplace_back([&, w] {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            ready.fetch_add(1);
            while (!go.load(std::memory_order_acquire)) {}
            const size_t cb = t.cap * w / k, ce = t.cap * (w + 1) / k;
            std::memset(t.slots() + 2 * cb, 0xFF, (ce - cb) * 8);
            if (cleared.fetch_add(1, std::memory_order_acq_rel) == k - 1) tc = Clk::now();
            while (cleared.load(std::memory_order_acquire) < k) std::this_thread::yield();
            insert(t, keys, pay, N * w / k, N * (w + 1) / k, kind);
            if (done.fetch_add(1, std::memory_order_acq_rel) == k - 1) te = Clk::now();
          });
        while (ready.load() < k) std::this_thread::yield();
        t0 = Clk::now();
        go.store(1, std::memory_order_release);
        for (auto& x : th) x.join();
        size_t found = 0;
        for (size_t i = 0; i < t.cap; ++i) found += t.slots()[2 * i] != join::kEmpty;
        if (found != N) { std::printf("ERROR %s k=%d: %zu of %zu\n", kind.c_str(), k, found, N); return 1; }
        if (rep == 0) continue;  // warm-up
        c_us.push_back(us(t0, tc)); i_us.push_back(us(tc, te)); t_us.push_back(us(t0, te));
      }
      auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
      std::printf("%zu,%s,%d,%.1f,%.1f,%.1f\n", N, kind.c_str(), k, med(c_us), med(i_us), med(t_us));
      std::fflush(stdout);
    }
  }
}
