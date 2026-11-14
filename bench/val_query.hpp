#pragma once
// Shared harness of the held-out query benchmarks (bench/val_q1.cpp, bench/val_q6.cpp): one fused
// query per morsel on CPU-P, CPU-E and the GPU. Validation data for the cost model, never used for
// fitting (analysis/predict_q1.py and predict_q6.py compare it with the model's prediction).
//
// The micro-engine's fused kernels (src/engine/<query>.hpp, kernels/<query>.metal) run on morsels
// of n = 2^min .. 2^max rows, in the calibration conventions:
//   device=gpu, state=warm, dispatch=sync|pipelined : the columns wrapped no-copy once and used
//                                                     before (GPU-resident). Sync: one round trip
//                                                     on rows [0, n). Pipelined: one timed rep is a
//                                                     batch of K morsels on K disjoint row ranges,
//                                                     t_total = GPU span / K (common/bench.hpp).
//   device=cpu, qos=p|e                             : one worker thread on rows [0, n)
// --cache cold (default) flushes 64 MB (MCM_FLUSH_MB) before every timed rep. CSV columns as
// calib_filter's, with op = the query, thr = the query's main threshold and bytes = the bytes the
// kernel reads per row times n, plus gpu_spin_ms and gpu_groups.
//
// --gpu-spin-ms M (default 0): after the flush, keep the GPU busy for about M ms with an ALU-only
// kernel (no memory traffic, untimed) right before each timed GPU rep. Without it, short morsels
// separated by idle gaps run the GPU at a low clock; a router that keeps the GPU busy does not.
//
// --gpu-groups G (default: the query's Gpu::default_max_groups(GPU cores)): at most G threadgroups per GPU
// morsel; beyond G * 4096 rows the threads stride over the morsel. CSV column gpu_groups (0 on CPU
// rows).
//
// --gpu-final 0 (diagnostic; default 1): time the GPU morsels without their final pass (<query>_final,
// which sums the per-threadgroup partials), so a morsel is the partial kernel alone; the sum then
// runs on the CPU after the timed span, so every morsel is still checked. The full-table check
// keeps the final pass. CSV column gpu_final (0 on CPU rows); analysis/fit.py names these
// configurations -nofinal, so the predictors never take them for the query.
//
// CSV column gpu_kernel: the partial kernel of the GPU rows (Q::kPartialKernel; "-" on CPU rows).
// <query>_partial is the query; a diagnostic variant <query>_partial_<v> (val_q1 --gpu-loads
// dependent: q1_partial_dependent) gets configurations named -<v> by analysis/fit.py, as -nofinal.
//
// Before the sweep the whole table runs twice on the chosen device, in morsels of 2^24 rows, and
// is checked against the scalar reference and, if present, <data>/<query>_expected.txt (DuckDB's
// answer, scripts/tpch_reference.py). Every timed morsel is checked against the reference as well.
//
// A query Q is a struct with (see bench/val_q1.cpp):
//   kOp ("q1"), kKernelFile ("q1.metal"), kPartialKernel ("q1_partial"; may be set before run()), kBytesPerRow, kThr, what()
//   Columns (rows, mem), Result (merge, ==), Gpu (kThreads, default_max_groups(gpu_cores), Gpu(ctx, cols, max_groups),
//   Slot, new_slot, free_slot, read, encode, partial_pso, max_groups, set_final_pass, final_pass, final_on_cpu),
//   load(dir), reference(cols, begin, n), cpu(cols, begin, n, out), load_expected(path, out), describe(result)

#include <pthread/qos.h>
#include <sys/resource.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/args.hpp"
#include "common/bench.hpp"
#include "common/machine_info.hpp"
#include "common/metal_ctx.hpp"
#include "common/timer.hpp"

namespace mcm::valq {

struct Row {
  std::string device, qos, state;
  uint64_t n = 0;
  int rep = 0;
  double t_encode = 0, t_total = 0, gpu_exec = 0;
  bool ok = true;
  int batch = 1;
  double spacing = 0;
};

inline constexpr const char* kCsvHeader =
    "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,cache,"
    "dispatch,batch,gpu_spacing_us,gpu_spin_ms,gpu_groups,gpu_final,gpu_kernel";

// Refuses to append to a CSV with other columns (one written before a column was added).
inline void check_csv_header(const std::string& path) {
  std::ifstream in(path);
  std::string header;
  if (in.good() && std::getline(in, header) && header != kCsvHeader)
    throw std::runtime_error(path + " has other columns than this version writes; use a new --out");
}

inline void write_rows(const std::string& path, const std::string& machine, const std::string& op,
                       const std::string& data, uint32_t thr, size_t bytes_per_row, const std::string& cache,
                       const std::string& dispatch, double spin_ms, uint64_t gpu_groups, bool gpu_final,
                       const std::string& gpu_kernel, const std::vector<Row>& rows) {
  std::ifstream in(path);
  const bool empty = !in.good() || in.peek() == std::ifstream::traits_type::eof();  // missing or empty: header
  in.close();
  std::ofstream f(path, std::ios::app);
  if (empty) f << kCsvHeader << "\n";
  for (const Row& r : rows) {
    const bool g = r.device == "gpu";
    f << machine << "," << op << "," << data << "," << thr << "," << r.device << "," << r.qos << "," << r.state << ","
      << r.n << "," << r.n * bytes_per_row << "," << r.rep << ",0," << r.t_encode << "," << r.t_total << ","
      << r.gpu_exec << "," << (r.ok ? 1 : 0) << "," << cache << "," << (g ? dispatch : "-") << "," << r.batch << ","
      << r.spacing << "," << (g ? spin_ms : 0.0) << "," << (g ? gpu_groups : 0) << "," << (g && gpu_final ? 1 : 0)
      << "," << (g ? gpu_kernel : "-") << "\n";
  }
}

inline void usage(const std::string& op) {
  std::cerr << "usage: val_" << op << " --device gpu|cpu [--qos p|e] [--dispatch sync|pipelined] [--batch 16] [--batch-mb 1024]\n"
            << "              [--data data/tpch-sf10] [--min-log2 10] [--max-log2 25] [--reps 15]\n"
            << "              [--cache cold|hot] [--flush-mb 64] [--out results/val_" << op << ".csv] [--machine-json path]\n"
            << "              [--expect <data>/" << op << "_expected.txt] [--gpu-spin-ms 0] [--gpu-groups <query default>]\n"
            << "              [--gpu-final 1 (0: diagnostic, morsels without the final pass)]\n";
}

// One synchronous round trip of morsel [begin, begin + n), timed as in calib_filter.
template <class Gpu>
void run_sync(MetalCtx& ctx, const Gpu& g, const typename Gpu::Slot& s, size_t begin, uint32_t n, Row& r) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  auto t0 = Clock::now();
  MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  g.encode(enc, begin, n, s);
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

template <class Q>
int run(int argc, char** argv) {
  using Columns = typename Q::Columns;
  using Result = typename Q::Result;
  using Gpu = typename Q::Gpu;
  const std::string op = Q::kOp;
  try {
    Args a(argc, argv);
    if (a.has("help") || !a.has("device")) {
      usage(op);
      return a.has("help") ? 0 : 1;
    }
    const std::string device = a.str("device", "gpu");
    if (device != "gpu" && device != "cpu") throw std::runtime_error("--device must be gpu or cpu");
    const std::string qos = a.str("qos", "p");
    const std::string dispatch = a.str("dispatch", "sync");
    if (dispatch != "sync" && dispatch != "pipelined") throw std::runtime_error("--dispatch must be sync or pipelined");
    const bool pipelined = dispatch == "pipelined";
    if (pipelined && device != "gpu") throw std::runtime_error("--dispatch pipelined applies to --device gpu only");
    const int batch = static_cast<int>(a.i64("batch", 16));
    const size_t batch_cap = static_cast<size_t>(a.i64("batch-mb", 1024)) << 20;
    const std::string data_dir = a.str("data", "data/tpch-sf10");
    const int min_l2 = static_cast<int>(a.i64("min-log2", 10));
    int max_l2 = static_cast<int>(a.i64("max-log2", 25));
    const int reps = static_cast<int>(a.i64("reps", 15));
    const std::string cache = a.str("cache", "cold");
    if (cache != "cold" && cache != "hot") throw std::runtime_error("--cache must be cold or hot");
    const std::string out_path = a.str("out", "results/val_" + op + ".csv");
    const std::string expect_path = a.str("expect", data_dir + "/" + op + "_expected.txt");
    const double spin_ms = a.f64("gpu-spin-ms", 0);
    if (spin_ms > 0 && device != "gpu") throw std::runtime_error("--gpu-spin-ms applies to --device gpu only");
    const bool groups_set = a.has("gpu-groups");
    const int64_t groups_arg = a.i64("gpu-groups", 1);
    if (groups_set && (groups_arg < 1 || groups_arg > (int64_t(1) << 20)))
      throw std::runtime_error("--gpu-groups must be in [1, 2^20]");
    if (groups_set && device != "gpu") throw std::runtime_error("--gpu-groups applies to --device gpu only");
    const int64_t final_arg = a.i64("gpu-final", 1);
    if (final_arg != 0 && final_arg != 1) throw std::runtime_error("--gpu-final must be 0 or 1");
    const bool gpu_final = final_arg == 1;
    if (!gpu_final && device != "gpu") throw std::runtime_error("--gpu-final applies to --device gpu only");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    check_csv_header(out_path);  // before the sweep, not after it
    acquire_bench_lock();  // one benchmark process at a time (common/bench.hpp)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MetalCtx ctx(std::vector<std::string>{std::string(MCM_KERNEL_DIR) + "/" + Q::kKernelFile,
                                          std::string(MCM_KERNEL_DIR) + "/spin.metal"});
    MachineInfo mi = collect_machine_info(ctx.device());
    const NS::UInteger max_groups =
        groups_set ? static_cast<NS::UInteger>(groups_arg) : Gpu::default_max_groups(mi.gpu_cores);
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

    Columns cols = Q::load(data_dir);
    int fit = 0;
    while ((2ull << fit) <= cols.rows) ++fit;
    if (fit < max_l2) {
      std::cerr << "table has " << cols.rows << " rows; capping max-log2 at " << fit << "\n";
      max_l2 = fit;
    }
    std::string data_tag = data_dir.substr(data_dir.find_last_of('/') + 1);
    std::cerr << "data: " << data_dir << " (" << cols.rows << " rows) | " << Q::what() << "\n";

    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", default_flush_mb())) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };

    // ---- full table: correctness against the reference and DuckDB
    const Result ref_all = Q::reference(cols, 0, cols.rows);
    Result exp_all;
    const bool have_expect = Q::load_expected(expect_path, exp_all);
    std::unique_ptr<Gpu> gpu;
    if (device == "gpu") gpu = std::make_unique<Gpu>(ctx, cols, max_groups);
    if (device == "cpu") place_cpu_process(qos);  // see common/bench.hpp: E needs the process-wide policy
    if (device == "gpu")
      std::cerr << Q::kPartialKernel << ": up to " << gpu->partial_pso()->maxTotalThreadsPerThreadgroup()
                << " threads per threadgroup (uses " << Gpu::kThreads << "), at most " << gpu->max_groups()
                << " threadgroups per morsel"
                << (groups_set ? std::string(" (--gpu-groups)")
                               : " (default for " + std::to_string(mi.gpu_cores) + " GPU cores)")
                << "\n";
    bool full_ok = true;
    // Twice: the first run is the first use of the columns (GPU: buffer residency; informational).
    for (int pass = 0; pass < 2; ++pass) {
      constexpr size_t kMorsel = size_t(1) << 24;
      Result got;
      double us = 0;
      if (device == "gpu") {
        typename Gpu::Slot s = gpu->new_slot();
        for (size_t b = 0; b < cols.rows; b += kMorsel) {
          Row r;
          run_sync(ctx, *gpu, s, b, static_cast<uint32_t>(std::min(kMorsel, cols.rows - b)), r);
          us += r.t_total;
          full_ok = full_ok && r.ok;
          got.merge(Gpu::read(s));
        }
        Gpu::free_slot(s);
      } else {
        std::thread t([&] {
          place_cpu_thread(qos);
          auto t0 = Clock::now();
          for (size_t b = 0; b < cols.rows; b += kMorsel) Q::cpu(cols, b, std::min(kMorsel, cols.rows - b), got);
          us = us_since(t0, Clock::now());
        });
        t.join();
      }
      const bool ref_ok = got == ref_all, exp_ok = !have_expect || got == exp_all;
      full_ok = full_ok && ref_ok && exp_ok;
      std::cerr << "full table on " << device << (device == "cpu" ? "-" + qos : "") << ", "
                << (pass == 0 ? "first" : "second") << " run (sync morsels of 2^24): " << us / 1e3
                << " ms | reference " << (ref_ok ? "match" : "MISMATCH") << " | DuckDB "
                << (!have_expect ? "not checked (no " + expect_path + ")" : exp_ok ? "match" : "MISMATCH") << "\n";
      if (pass == 1) std::cerr << Q::describe(got);
    }
    if (have_expect && !(ref_all == exp_all)) std::cerr << "WARNING: the scalar reference disagrees with DuckDB\n";
    if (!gpu_final) {
      gpu->set_final_pass(false);
      std::cerr << "DIAGNOSTIC --gpu-final 0: timed morsels run " << Q::kPartialKernel
                << " alone; the final sum runs on the CPU after timing (not a Q" << op.substr(1) << " time)\n";
    }

    // ---- sweep
    std::vector<Row> rows;
    std::unique_ptr<GpuSpin> spin;
    if (spin_ms > 0) spin = std::make_unique<GpuSpin>(ctx, spin_ms);
    if (device == "gpu") {
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        const int K = pipelined ? batch_size(batch, size_t(n) * Q::kBytesPerRow, batch_cap) : 1;
        std::vector<size_t> begin(K);
        std::vector<Result> ref(K);
        std::vector<typename Gpu::Slot> slots(K);
        for (int k = 0; k < K; ++k) {
          begin[k] = (size_t(k) * n) % (cols.rows - n + 1);  // disjoint while K * n <= rows
          ref[k] = Q::reference(cols, begin[k], n);
          slots[k] = gpu->new_slot();
        }
        auto check = [&] {
          for (int k = 0; k < K; ++k) {
            if (!gpu->final_pass()) gpu->final_on_cpu(slots[k], n);
            if (!(Gpu::read(slots[k]) == ref[k])) return false;
          }
          return true;
        };
        auto timed = [&](Row& row) {
          if (!pipelined) {
            run_sync(ctx, *gpu, slots[0], begin[0], n, row);
          } else {
            BatchTiming b = run_pipelined(ctx.queue(), gpu->partial_pso(), K, [&](MTL::ComputeCommandEncoder* enc, int k) {
              gpu->encode(enc, begin[k], n, slots[k]);
            });
            row.t_encode = b.encode_us;
            row.t_total = b.per_morsel_us;
            row.spacing = b.spacing_us;
            row.gpu_exec = b.exec_us;
            row.batch = K;
            if (!b.ok) row.ok = false;
          }
          return check() && row.ok;
        };
        Row warm{};
        for (int w = 0; w < 3; ++w) timed(warm);  // resident + pipeline warm, untimed
        for (int r = 0; r < reps; ++r) {
          Row row{"gpu", "-", "warm", n, r};
          flush();
          if (spin) (*spin)();  // ALU only: leaves the flushed caches alone
          row.ok = timed(row);
          rows.push_back(row);
        }
        for (auto& s : slots) Gpu::free_slot(s);
        std::cerr << "gpu warm " << dispatch << (pipelined ? " x" + std::to_string(K) : "") << " n=2^" << l2 << " done\n";
      }
    } else {
      std::thread worker([&] {
        place_cpu_thread(qos);
        for (int l2 = min_l2; l2 <= max_l2; ++l2) {
          const uint64_t n = 1ull << l2;
          const Result ref = Q::reference(cols, 0, n);
          for (int r = 0; r < reps; ++r) {
            Row row{"cpu", qos, r == 0 ? "first" : "warm", n, r};
            Result got;
            flush();
            asm volatile("" ::: "memory");
            auto t0 = Clock::now();
            Q::cpu(cols, 0, n, got);
            asm volatile("" ::: "memory");
            row.t_total = us_since(t0, Clock::now());
            row.ok = got == ref;
            rows.push_back(row);
          }
          std::cerr << "cpu " << qos << " n=2^" << l2 << " done\n";
        }
      });
      worker.join();
    }

    const size_t bad = std::count_if(rows.begin(), rows.end(), [](const Row& r) { return !r.ok; });
    spin.reset();
    write_rows(out_path, mi.tag(), op, data_tag, Q::kThr, Q::kBytesPerRow, cache, dispatch, spin_ms,
               device == "gpu" ? gpu->max_groups() : 0, gpu_final, Q::kPartialKernel, rows);
    std::cerr << rows.size() << " rows -> " << out_path << (bad ? " | WARNING: " + std::to_string(bad) +
                                                                      " rows failed the correctness check" : "")
              << (full_ok ? "" : " | WARNING: the full-table check failed") << "\n";
    gpu.reset();
    pool->release();
    return (bad || !full_ok) ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}

}  // namespace mcm::valq
