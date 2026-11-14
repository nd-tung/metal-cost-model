// calib_sort — sort calibration microbenchmark: per-morsel run generation, i.e. sorting
// (uint32 key, row id) pairs, on CPU-P, CPU-E and the GPU, and the CPU merge of k sorted runs.
//
//   --mode sort  : a morsel is the first n keys of --keys (or uniform random keys with --random).
//                  Output: n pairs sorted by (key, row id), row id = position in the morsel.
//                  --algo radix : LSD radix sort, 8-bit digits, passes from the column's largest key
//                                 (src/engine/radix_sort.hpp; GPU: kernels/sort.metal)
//                  --algo std   : std::sort of the pairs, CPU only (the comparison-sort reference)
//   --mode merge : CPU only. The morsel's n keys are split into --runs k contiguous runs, each sorted
//                  untimed; timed: the k-way merge (binary heap) into one sorted output.
// Devices, states and cache as in calib_agg: gpu warm (keys wrapped no-copy once, used before), gpu
// cold (a fresh host copy of the morsel's keys wrapped right before the rep, t_wrap), cpu p|e (one
// worker). GPU: sync dispatch (all passes in one command buffer); --gpu-spin-ms M keeps the GPU busy
// ~M ms before each timed rep (busy regime). Output and scratch buffers are allocated once per n
// (a router's pool) and reused.
// Every output is checked exactly: sorted by (key, row id), each row id once, key = keys[row id].
// CSV columns as calib_agg's plus gpu_spin_ms: op = sort | sort_merge, data = key column (+ _std or
// _merge<k>), groups =
// radix passes (sort; 0 for std) or k (merge), path = radix | std | heap, bytes = 8 n (pairs written).

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "common/args.hpp"
#include "common/bench.hpp"
#include "common/machine_info.hpp"
#include "common/metal_ctx.hpp"
#include "common/timer.hpp"
#include "engine/radix_sort.hpp"

using namespace mcm;

namespace {

struct Row {
  std::string device, qos, state;
  uint64_t n = 0;
  int rep = 0;
  double t_wrap = 0, t_encode = 0, t_total = 0, gpu_exec = 0;
  bool ok = true;
};

struct SortParams {
  uint32_t n, shift, groups, chunk, first;  // must match kernels/sort.metal
};

class GpuSort {
 public:
  static constexpr uint32_t kTile = 2048;  // kernels/sort.metal TILE
  static constexpr uint32_t kThreads = 256;
  static constexpr uint32_t kMaxGroups = 256;

  explicit GpuSort(MetalCtx& ctx) : ctx_(ctx) {
    hist_ = ctx.pipeline("sort_hist");
    scan_ = ctx.pipeline("sort_scan");
    scatter_ = ctx.pipeline("sort_scatter");
    if (hist_->maxTotalThreadsPerThreadgroup() < kThreads || scatter_->maxTotalThreadsPerThreadgroup() < kThreads)
      throw std::runtime_error("sort kernels: fewer than 256 threads per threadgroup available");
    scan_threads_ = std::min<NS::UInteger>(1024, scan_->maxTotalThreadsPerThreadgroup() / 32 * 32);
    hist_buf_ = ctx.device()->newBuffer(256 * kMaxGroups * sizeof(uint32_t), MTL::ResourceStorageModeShared);
  }
  ~GpuSort() {
    hist_buf_->release();
    scatter_->release();
    scan_->release();
    hist_->release();
  }

  // All passes of one sort of keys[0, n) into `out`, with `tmp` as the other ping-pong buffer.
  void encode(MTL::ComputeCommandEncoder* enc, MTL::Buffer* keys, uint32_t n, int passes, MTL::Buffer* out,
              MTL::Buffer* tmp) const {
    uint32_t groups = std::clamp<uint32_t>((n + kTile - 1) / kTile, 1, kMaxGroups);
    const uint32_t chunk = ((n + groups - 1) / groups + kTile - 1) / kTile * kTile;
    groups = (n + chunk - 1) / chunk;
    for (int p = 0; p < passes; ++p) {
      MTL::Buffer* dst = ((passes - 1 - p) % 2 == 0) ? out : tmp;
      MTL::Buffer* src = dst == out ? tmp : out;  // unread in the first pass
      const SortParams sp{n, 8u * static_cast<uint32_t>(p), groups, chunk, p == 0 ? 1u : 0u};
      enc->setComputePipelineState(hist_);
      enc->setBuffer(keys, 0, 0);
      enc->setBuffer(src, 0, 1);
      enc->setBuffer(hist_buf_, 0, 3);
      enc->setBytes(&sp, sizeof(sp), 4);
      enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
      enc->setComputePipelineState(scan_);
      enc->setBuffer(hist_buf_, 0, 3);
      enc->setBytes(&sp, sizeof(sp), 4);
      enc->dispatchThreadgroups(MTL::Size(1, 1, 1), MTL::Size(scan_threads_, 1, 1));
      enc->setComputePipelineState(scatter_);
      enc->setBuffer(keys, 0, 0);
      enc->setBuffer(src, 0, 1);
      enc->setBuffer(dst, 0, 2);
      enc->setBuffer(hist_buf_, 0, 3);
      enc->setBytes(&sp, sizeof(sp), 4);
      enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    }
  }

  void run_sync(MTL::Buffer* keys, uint32_t n, int passes, MTL::Buffer* out, MTL::Buffer* tmp, Row& r) const {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto t0 = Clock::now();
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    encode(enc, keys, n, passes, out, tmp);
    enc->endEncoding();
    cb->commit();
    auto t1 = Clock::now();
    cb->waitUntilCompleted();
    auto t2 = Clock::now();
    r.t_encode = us_since(t0, t1);
    r.t_total = us_since(t0, t2);
    r.gpu_exec = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e6;
    if (cb->status() != MTL::CommandBufferStatusCompleted) r.ok = false;
    pool->release();
  }

 private:
  MetalCtx& ctx_;
  MTL::ComputePipelineState *hist_ = nullptr, *scan_ = nullptr, *scatter_ = nullptr;
  NS::UInteger scan_threads_ = 1024;
  MTL::Buffer* hist_buf_ = nullptr;
};

void write_rows(const std::string& path, const std::string& machine, const std::string& op, const std::string& data,
                const std::string& cache, int groups, const std::string& algo, double spin_ms, const std::vector<Row>& rows) {
  const bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!exists)
    f << "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,"
         "cache,groups,path,dispatch,batch,gpu_spacing_us,gpu_spin_ms\n";
  for (const Row& r : rows)
    f << machine << "," << op << "," << data << ",0," << r.device << "," << r.qos << "," << r.state << "," << r.n << ","
      << r.n * 8 << "," << r.rep << "," << r.t_wrap << "," << r.t_encode << "," << r.t_total << "," << r.gpu_exec << ","
      << (r.ok ? 1 : 0) << "," << cache << "," << groups << "," << algo << "," << (r.device == "gpu" ? "sync" : "-")
      << ",1,0," << (r.device == "gpu" ? spin_ms : 0.0) << "\n";
}

void usage() {
  std::cerr << "usage: calib_sort --device gpu|cpu [--qos p|e] [--state warm|cold]\n"
               "                  (--keys data/tpch-sf10/l_partkey.u32 | --random)\n"
               "                  [--mode sort|merge] [--algo radix|std] [--runs 8]\n"
               "                  [--min-log2 10] [--max-log2 25] [--reps 15] [--cache cold|hot] [--flush-mb 64]\n"
               "                  [--gpu-spin-ms 0] [--out results/calib_sort.csv] [--machine-json path]\n";
}

std::string stem(const std::string& p) {
  std::string s = p.substr(p.find_last_of('/') + 1);
  return s.substr(0, s.find('.'));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help") || !a.has("device") || (!a.has("keys") && !a.has("random"))) {
      usage();
      return a.has("help") ? 0 : 1;
    }
    const std::string device = a.str("device", "gpu");
    if (device != "gpu" && device != "cpu") throw std::runtime_error("--device must be gpu or cpu");
    const std::string qos = a.str("qos", "p");
    const std::string state = a.str("state", "warm");
    if (state != "warm" && state != "cold") throw std::runtime_error("--state must be warm or cold");
    const std::string mode = a.str("mode", "sort");
    if (mode != "sort" && mode != "merge") throw std::runtime_error("--mode must be sort or merge");
    const std::string algo = mode == "merge" ? "heap" : a.str("algo", "radix");
    if (mode == "sort" && algo != "radix" && algo != "std") throw std::runtime_error("--algo must be radix or std");
    if (device == "gpu" && (mode != "sort" || algo != "radix"))
      throw std::runtime_error("the GPU runs --mode sort --algo radix only");
    const int runs = static_cast<int>(a.i64("runs", 8));
    if (mode == "merge" && runs < 2) throw std::runtime_error("--runs must be >= 2");
    const int min_l2 = static_cast<int>(a.i64("min-log2", 10));
    int max_l2 = static_cast<int>(a.i64("max-log2", 25));
    const int reps = static_cast<int>(a.i64("reps", 15));
    const std::string cache = a.str("cache", "cold");
    if (cache != "cold" && cache != "hot") throw std::runtime_error("--cache must be cold or hot");
    const double spin_ms = a.f64("gpu-spin-ms", 0);
    if (spin_ms > 0 && device != "gpu") throw std::runtime_error("--gpu-spin-ms applies to --device gpu only");
    const std::string out_path = a.str("out", "results/calib_sort.csv");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    acquire_bench_lock();  // one benchmark process at a time, AC power (common/bench.hpp)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const std::string kdir = MCM_KERNEL_DIR;
    MetalCtx ctx(std::vector<std::string>{kdir + "/sort.metal", kdir + "/spin.metal"});
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

    // ---- keys: a column, or uniform random uint32
    std::string data = "random";
    size_t rows = size_t(1) << max_l2;
    std::unique_ptr<PageAlloc> kmem;
    if (a.has("keys")) {
      const std::string path = a.str("keys", "");
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("cannot open column " + path);
      const size_t bytes = static_cast<size_t>(f.tellg());
      rows = bytes / sizeof(uint32_t);
      kmem = std::make_unique<PageAlloc>(bytes);
      f.seekg(0);
      f.read(static_cast<char*>(kmem->data()), static_cast<std::streamsize>(bytes));
      data = stem(path);
    } else {
      kmem = std::make_unique<PageAlloc>(rows * sizeof(uint32_t));
      std::mt19937 rng(42);
      auto* k = static_cast<uint32_t*>(kmem->data());
      for (size_t i = 0; i < rows; ++i) k[i] = rng();
    }
    const auto* keys = static_cast<const uint32_t*>(kmem->data());
    int fit = 0;
    while ((2ull << fit) <= rows) ++fit;
    if (fit < max_l2) {
      std::cerr << "column has " << rows << " rows; capping max-log2 at " << fit << "\n";
      max_l2 = fit;
    }
    uint32_t max_key = 0;
    for (size_t i = 0; i < rows; ++i) max_key = std::max(max_key, keys[i]);
    const int passes = sort::passes_for(max_key);  // column statistics
    const int groups_col = mode == "merge" ? runs : algo == "radix" ? passes : 0;
    std::cerr << "keys: " << data << " (" << rows << " rows, max " << max_key << ") | " << mode << " " << algo
              << (algo == "radix" ? " " + std::to_string(passes) + " passes" : "")
              << (mode == "merge" ? " of " + std::to_string(runs) + " runs" : "") << " | " << device
              << (device == "cpu" ? "-" + qos : " " + state) << "\n";

    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", default_flush_mb())) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };
    std::unique_ptr<GpuSort> gpu;
    std::unique_ptr<GpuSpin> spin;
    MTL::Buffer* wkeys = nullptr;
    if (device == "gpu") {
      gpu = std::make_unique<GpuSort>(ctx);
      wkeys = wrap_no_copy(ctx.device(), *kmem);
      if (spin_ms > 0) spin = std::make_unique<GpuSpin>(ctx, spin_ms);
    } else {
      place_cpu_process(qos);
    }

    std::vector<Row> rows_out;
    std::vector<uint8_t> seen;
    auto sweep = [&] {
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        if (device == "gpu") {
          MTL::Buffer* out = ctx.device()->newBuffer(size_t(n) * 8, MTL::ResourceStorageModeShared);
          MTL::Buffer* tmp = ctx.device()->newBuffer(size_t(n) * 8, MTL::ResourceStorageModeShared);
          auto rep_once = [&](Row& row) {
            std::unique_ptr<PageAlloc> cold;
            if (state == "cold") {
              cold = std::make_unique<PageAlloc>(size_t(n) * 4);
              std::memcpy(cold->data(), keys, size_t(n) * 4);  // CPU-touched, GPU-unseen
            }
            flush();
            if (spin) (*spin)();
            MTL::Buffer* kb = wkeys;
            if (state == "cold") {
              auto t0 = Clock::now();
              kb = wrap_no_copy(ctx.device(), *cold);
              row.t_wrap = us_since(t0, Clock::now());
            }
            gpu->run_sync(kb, n, passes, out, tmp, row);
            if (state == "cold") kb->release();
            return sort::verify(keys, n, static_cast<const sort::Pair*>(out->contents()), seen) && row.ok;
          };
          Row warm{};
          for (int w = 0; w < 3; ++w) rep_once(warm);
          for (int r = 0; r < reps; ++r) {
            Row row{"gpu", "-", state, n, r};
            row.ok = rep_once(row);
            rows_out.push_back(row);
          }
          out->release();
          tmp->release();
        } else {
          std::vector<sort::Pair> out(n), tmp(n), in;
          std::vector<size_t> begin;
          if (mode == "merge") {  // k sorted runs with global row ids, untimed
            in.resize(n);
            for (int r = 0; r <= runs; ++r) begin.push_back(size_t(n) * r / runs);
            for (int r = 0; r < runs; ++r)
              sort::cpu_radix(keys + begin[r], begin[r + 1] - begin[r], passes, in.data() + begin[r], tmp.data(),
                              static_cast<uint32_t>(begin[r]));
          }
          for (int r = 0; r < reps; ++r) {
            Row row{"cpu", qos, r == 0 ? "first" : "warm", n, r};
            flush();
            asm volatile("" ::: "memory");
            auto t0 = Clock::now();
            if (mode == "merge") sort::cpu_merge(in.data(), begin, out.data());
            else if (algo == "radix") sort::cpu_radix(keys, n, passes, out.data(), tmp.data());
            else sort::cpu_std(keys, n, out.data());
            asm volatile("" ::: "memory");
            row.t_total = us_since(t0, Clock::now());
            row.ok = sort::verify(keys, n, out.data(), seen);
            rows_out.push_back(row);
          }
        }
        std::cerr << device << " " << mode << " " << algo << " n=2^" << l2 << " done\n";
      }
    };
    if (device == "cpu") {
      std::thread t([&] {
        place_cpu_thread(qos);
        sweep();
      });
      t.join();
    } else {
      sweep();
    }

    const size_t bad = std::count_if(rows_out.begin(), rows_out.end(), [](const Row& r) { return !r.ok; });
    // One data tag per configuration that shares a device: radix <keys>, std <keys>_std, merge <keys>_merge<k>.
    const std::string tag = mode == "merge" ? data + "_merge" + std::to_string(runs) : algo == "std" ? data + "_std" : data;
    write_rows(out_path, mi.tag(), mode == "merge" ? "sort_merge" : "sort", tag, cache, groups_col, algo, spin_ms, rows_out);
    std::cerr << rows_out.size() << " rows -> " << out_path
              << (bad ? " | WARNING: " + std::to_string(bad) + " rows failed the correctness check" : "") << "\n";
    if (wkeys) wkeys->release();
    spin.reset();
    gpu.reset();
    pool->release();
    return bad ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
