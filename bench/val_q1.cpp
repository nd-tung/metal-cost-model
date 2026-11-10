// val_q1 — TPC-H Q1 per morsel on CPU-P, CPU-E and the GPU: held-out validation data for the cost
// model (never used for fitting; analysis/predict_q1.py compares it with the model's prediction).
//
// The micro-engine's fused Q1 kernels (src/engine/q1.hpp, kernels/q1.metal) run on morsels of
// n = 2^min .. 2^max rows, in the calibration conventions:
//   device=gpu, state=warm, dispatch=sync|pipelined : the 7 columns wrapped no-copy once and used
//                                                     before (GPU-resident). Sync: one round trip
//                                                     on rows [0, n). Pipelined: one timed rep is a
//                                                     batch of K morsels on K disjoint row ranges,
//                                                     t_total = GPU span / K (common/bench.hpp).
//   device=cpu, qos=p|e                             : one worker thread on rows [0, n)
// --cache cold (default) flushes 64 MB before every timed rep. CSV columns as calib_filter's, with
// op = q1 and bytes = 28 n (7 uint32 columns), plus gpu_spin_ms.
//
// --gpu-spin-ms M (default 0): after the flush, keep the GPU busy for about M ms with an ALU-only
// kernel (no memory traffic, untimed) right before each timed GPU rep. Without it, short morsels
// separated by idle gaps run the GPU at a low clock; a router that keeps the GPU busy does not.
//
// Before the sweep the whole table runs once on the chosen device, in morsels of 2^24 rows, and is
// checked against the scalar reference and, if present, <data>/q1_expected.txt (DuckDB's answer,
// scripts/tpch_reference.py). Every timed morsel is checked against the reference as well.

#include <pthread/qos.h>
#include <sys/resource.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common/args.hpp"
#include "common/bench.hpp"
#include "common/machine_info.hpp"
#include "common/metal_ctx.hpp"
#include "common/timer.hpp"
#include "engine/q1.hpp"
#include "engine/q1_gpu.hpp"

using namespace mcm;

namespace {

struct Row {
  std::string device, qos, state;
  uint64_t n = 0;
  int rep = 0;
  double t_encode = 0, t_total = 0, gpu_exec = 0;
  bool ok = true;
  int batch = 1;
  double spacing = 0;
};

void write_rows(const std::string& path, const std::string& machine, const std::string& data,
                const std::string& cache, const std::string& dispatch, double spin_ms, const std::vector<Row>& rows) {
  const bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!exists)
    f << "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,cache,"
         "dispatch,batch,gpu_spacing_us,gpu_spin_ms\n";
  for (const Row& r : rows)
    f << machine << ",q1," << data << "," << q1::kCutoff << "," << r.device << "," << r.qos << "," << r.state << ","
      << r.n << "," << r.n * q1::kBytesPerRow << "," << r.rep << ",0," << r.t_encode << "," << r.t_total << ","
      << r.gpu_exec << "," << (r.ok ? 1 : 0) << "," << cache << "," << (r.device == "gpu" ? dispatch : "-") << ","
      << r.batch << "," << r.spacing << "," << (r.device == "gpu" ? spin_ms : 0.0) << "\n";
}

// ALU-only GPU busy loop of about `ms` milliseconds (kernels/q1.metal gpu_spin), sync, untimed.
class GpuSpin {
 public:
  GpuSpin(MetalCtx& ctx, double ms) : ctx_(ctx) {
    pso_ = ctx.pipeline("gpu_spin");
    out_ = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    for (int i = 0; i < 3; ++i) run(1u << 14);  // warm up, then scale the loop to the target time
    const double us = run(1u << 16);
    iters_ = static_cast<uint32_t>(std::clamp((1u << 16) * (ms * 1e3) / std::max(us, 1.0), 1.0, 4e9));
    std::cerr << "gpu spin: " << iters_ << " iterations ~ " << run(iters_) / 1e3 << " ms\n";
  }
  ~GpuSpin() {
    out_->release();
    pso_->release();
  }
  void operator()() { run(iters_); }

 private:
  double run(uint32_t iters) {  // returns GPU execution time in us
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso_);
    enc->setBuffer(out_, 0, 0);
    enc->setBytes(&iters, sizeof(iters), 1);
    enc->dispatchThreadgroups(MTL::Size(1024, 1, 1), MTL::Size(256, 1, 1));
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    const double us = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e6;
    pool->release();
    return us;
  }
  MetalCtx& ctx_;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::Buffer* out_ = nullptr;
  uint32_t iters_ = 1;
};

void usage() {
  std::cerr << "usage: val_q1 --device gpu|cpu [--qos p|e] [--dispatch sync|pipelined] [--batch 16] [--batch-mb 1024]\n"
               "              [--data data/tpch-sf10] [--min-log2 10] [--max-log2 25] [--reps 15]\n"
               "              [--cache cold|hot] [--flush-mb 64] [--out results/val_q1.csv] [--machine-json path]\n"
               "              [--expect <data>/q1_expected.txt] [--gpu-spin-ms 0]\n";
}

// One synchronous round trip of morsel [begin, begin + n), timed as in calib_filter.
void run_sync(MetalCtx& ctx, const q1::Gpu& g, const q1::Gpu::Slot& s, size_t begin, uint32_t n, Row& r) {
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

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help") || !a.has("device")) {
      usage();
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
    const std::string out_path = a.str("out", "results/val_q1.csv");
    const std::string expect_path = a.str("expect", data_dir + "/q1_expected.txt");
    const double spin_ms = a.f64("gpu-spin-ms", 0);
    if (spin_ms > 0 && device != "gpu") throw std::runtime_error("--gpu-spin-ms applies to --device gpu only");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    acquire_bench_lock();  // one benchmark process at a time (common/bench.hpp)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MetalCtx ctx(std::string(MCM_KERNEL_DIR) + "/q1.metal");
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

    q1::Columns cols = q1::load_columns(data_dir);
    int fit = 0;
    while ((2ull << fit) <= cols.rows) ++fit;
    if (fit < max_l2) {
      std::cerr << "table has " << cols.rows << " rows; capping max-log2 at " << fit << "\n";
      max_l2 = fit;
    }
    std::string data_tag = data_dir.substr(data_dir.find_last_of('/') + 1);
    std::cerr << "data: " << data_dir << " (" << cols.rows << " rows) | Q1, shipdate <= " << q1::kCutoff << "\n";

    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", 64)) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };

    // ---- full table once: correctness against the reference and DuckDB
    const q1::Result ref_all = q1::reference(cols, 0, cols.rows);
    q1::Result exp_all;
    const bool have_expect = q1::load_expected(expect_path, exp_all);
    std::unique_ptr<q1::Gpu> gpu;
    if (device == "gpu") gpu = std::make_unique<q1::Gpu>(ctx, cols);
    if (device == "cpu") place_cpu_process(qos);  // see common/bench.hpp: E needs the process-wide policy
    if (device == "gpu")
      std::cerr << "q1_partial: up to " << gpu->partial_pso()->maxTotalThreadsPerThreadgroup()
                << " threads per threadgroup (uses " << q1::Gpu::kThreads << ")\n";
    bool full_ok = true;
    // Twice: the first run is the first use of the columns (GPU: buffer residency; informational).
    for (int pass = 0; pass < 2; ++pass) {
      constexpr size_t kMorsel = size_t(1) << 24;
      q1::Result got;
      double us = 0;
      if (device == "gpu") {
        q1::Gpu::Slot s = gpu->new_slot();
        for (size_t b = 0; b < cols.rows; b += kMorsel) {
          Row r;
          run_sync(ctx, *gpu, s, b, static_cast<uint32_t>(std::min(kMorsel, cols.rows - b)), r);
          us += r.t_total;
          full_ok = full_ok && r.ok;
          got.merge(q1::Gpu::read(s));
        }
        q1::Gpu::free_slot(s);
      } else {
        std::thread t([&] {
          place_cpu_thread(qos);
          auto t0 = Clock::now();
          for (size_t b = 0; b < cols.rows; b += kMorsel) q1::cpu(cols, b, std::min(kMorsel, cols.rows - b), got);
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
      if (pass == 1) std::cerr << q1::describe(got);
    }
    if (have_expect && !(ref_all == exp_all)) std::cerr << "WARNING: the scalar reference disagrees with DuckDB\n";

    // ---- sweep
    std::vector<Row> rows;
    std::unique_ptr<GpuSpin> spin;
    if (spin_ms > 0) spin = std::make_unique<GpuSpin>(ctx, spin_ms);
    if (device == "gpu") {
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        const int K = pipelined ? batch_size(batch, size_t(n) * q1::kBytesPerRow, batch_cap) : 1;
        std::vector<size_t> begin(K);
        std::vector<q1::Result> ref(K);
        std::vector<q1::Gpu::Slot> slots(K);
        for (int k = 0; k < K; ++k) {
          begin[k] = (size_t(k) * n) % (cols.rows - n + 1);  // disjoint while K * n <= rows
          ref[k] = q1::reference(cols, begin[k], n);
          slots[k] = gpu->new_slot();
        }
        auto check = [&] {
          for (int k = 0; k < K; ++k)
            if (!(q1::Gpu::read(slots[k]) == ref[k])) return false;
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
        for (auto& s : slots) q1::Gpu::free_slot(s);
        std::cerr << "gpu warm " << dispatch << (pipelined ? " x" + std::to_string(K) : "") << " n=2^" << l2 << " done\n";
      }
    } else {
      std::thread worker([&] {
        place_cpu_thread(qos);
        for (int l2 = min_l2; l2 <= max_l2; ++l2) {
          const uint64_t n = 1ull << l2;
          const q1::Result ref = q1::reference(cols, 0, n);
          for (int r = 0; r < reps; ++r) {
            Row row{"cpu", qos, r == 0 ? "first" : "warm", n, r};
            q1::Result got;
            flush();
            asm volatile("" ::: "memory");
            auto t0 = Clock::now();
            q1::cpu(cols, 0, n, got);
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
    write_rows(out_path, mi.tag(), data_tag, cache, dispatch, spin_ms, rows);
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
