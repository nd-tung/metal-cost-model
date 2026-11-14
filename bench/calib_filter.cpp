// calib_filter — first calibration microbenchmark.
//
// Operator: filter + count (v < threshold) over n uint32 values.
// Data: a TPC-H column exported by scripts/gen_tpch.py (--column path/to/col.u32,
// --thr value in the column's encoding), or random uint32 if --column is omitted.
// A morsel of size n is the first n rows of the column.
// Sweeps morsel size n = 2^min .. 2^max and measures, per morsel:
//
//   device=gpu, state=warm : buffer already used by the GPU (resident)
//   device=gpu, state=cold : fresh host allocation wrapped no-copy right before the run
//
// GPU storage (--storage, CSV columns storage + t_blit_us):
//   shared (default) : the kernel reads the host buffer directly (no-copy wrap)
//   private          : every rep blits the host buffer into a MTLStorageModePrivate buffer (allocated
//                      once per n, as a router's pool would be), then runs the kernel on it. State then
//                      describes the blit's source: warm = GPU-resident, cold = fresh host memory
//                      wrapped right before the rep; resident = no blit, the private buffer already
//                      holds the data (the kernel-only cost of private memory). Sync dispatch only.
//   device=cpu, qos=p|e    : one worker thread; p = USER_INTERACTIVE QoS, e = whole process under
//                            the darwin background policy (thread QoS alone does not reach E-cores)
//
// --cache: the state of the morsel's input at dispatch, set by what last touched it (the router's
// cache policy; common/bench.hpp Toucher). cold (default) reads a 64 MB scratch buffer before every
// timed rep, so input comes from DRAM for CPU and GPU alike; hot skips it (input may be
// cache-resident). cpu-write, cpu-read, gpu-write and gpu-read flush, then let one CPU thread or a GPU
// kernel rewrite (x += 0) or read the morsel's rows, untimed, right before the timed morsel: a
// producer or a scan on that device has just left them in the caches it shares with the consumer.
// They need --state warm --storage shared (the GPU) or --device cpu. CSV column cache; analysis/fit.py
// names every state but cold <config>-<state>.
//
// GPU dispatch (--dispatch, CSV columns dispatch + batch):
//   sync (default) : one command buffer per timed rep: encode, commit, waitUntilCompleted.
//                    The full round trip a morsel pays when it finds the GPU queue empty.
//   pipelined      : one timed rep = a batch of K morsels on K distinct buffers (--batch, capped by
//                    --batch-mb), all encoded first, then committed back to back. t_total is the
//                    GPU time per morsel over the batch, (last GPU end - first GPU start) / K: what
//                    a router pays per morsel when it keeps the GPU queue full. Command buffers
//                    overlap on the GPU, so gpu_spacing_us (spacing of completions) is smaller and
//                    is only a diagnostic (see common/bench.hpp).
//
// --gpu-spin-ms M: after each flush, keep the GPU busy ~M ms with ALU-only work (untimed; common/bench.hpp
// GpuSpin) so the timed morsel runs at the clock of a busy GPU, as under a router that keeps it fed.
//
// One morsel = one GPU dispatch (one command buffer) or one CPU worker.
// Output: one CSV row per repetition (pipelined: per batch); analysis/fit.py takes medians.
//
// Columns of the GPU timing split (all microseconds, host clock unless noted):
//   t_wrap    : newBuffer(bytesNoCopy) time (cold only; residency term R); pipelined: per buffer
//   t_encode  : command buffer + encoder creation through commit(); pipelined: encode per buffer
//   t_total   : sync: commit start .. waitUntilCompleted() return   (L + exec + S)
//               pipelined: GPU time per morsel over the batch (span / K)
//   gpu_exec  : GPUEndTime - GPUStartTime  (GPU-side execution; pipelined: mean per buffer)
//   => sync launch+sync overhead ≈ t_total - gpu_exec

#include <pthread/qos.h>
#include <sys/resource.h>
#include <unistd.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstdio>
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

using namespace mcm;

namespace {

struct FilterParams {
  uint32_t n;
  uint32_t threshold;
};

struct Row {
  std::string device, qos, state;
  uint64_t n = 0;
  int rep = 0;
  double t_wrap = 0, t_encode = 0, t_total = 0, gpu_exec = 0;
  bool ok = true;
  int batch = 1;       // morsels per timed rep (pipelined GPU dispatch); 1 otherwise
  double spacing = 0;  // pipelined: spacing of GPU completion times (diagnostic); 0 otherwise
  double t_blit = 0;   // private storage: GPU time of the blit command buffer; 0 otherwise
};

uint32_t cpu_count(const uint32_t* v, uint64_t n, uint32_t thr) {
  uint32_t c = 0;
  for (uint64_t i = 0; i < n; ++i) c += v[i] < thr;
  return c;
}

// ---------------------------------------------------------------- GPU path
struct GpuRunner {
  MetalCtx& ctx;
  MTL::ComputePipelineState* pso;
  MTL::Buffer* out;       // single uint32 counter, shared (sync dispatch)
  MTL::Buffer* out_pipe;  // the same, hazard-untracked, for pipelined batches: the morsels only
                          // add to it atomically, so back-to-back command buffers may overlap
  NS::UInteger tg;        // threads per threadgroup

  GpuRunner(MetalCtx& c) : ctx(c) {
    pso = ctx.pipeline("filter_count");
    out = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    out_pipe = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared |
                                                             MTL::ResourceHazardTrackingModeUntracked);
    tg = std::min<NS::UInteger>(256, pso->maxTotalThreadsPerThreadgroup());
  }
  ~GpuRunner() {
    out_pipe->release();
    out->release();
    pso->release();
  }

  // Binds buffers and dispatches one morsel (the pipeline state is already set).
  void encode(MTL::ComputeCommandEncoder* enc, MTL::Buffer* in, MTL::Buffer* o, uint32_t n, uint32_t thr) const {
    // Each thread handles ~16 elements; capped grid. Occupancy is modeled later (M1/M2).
    NS::UInteger groups = (n + tg * 16 - 1) / (tg * 16);
    groups = std::clamp<NS::UInteger>(groups, 1, 4096);
    FilterParams p{n, thr};
    enc->setBuffer(in, 0, 0);
    enc->setBuffer(o, 0, 1);
    enc->setBytes(&p, sizeof(p), 2);
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(tg, 1, 1));
  }

  // Pipelined: one batch, morsel k on ins[k]. Returns the counter (K x count, mod 2^32).
  uint32_t run_batch(const std::vector<MTL::Buffer*>& ins, uint32_t n, uint32_t thr, Row& r) {
    *static_cast<uint32_t*>(out_pipe->contents()) = 0;
    BatchTiming b = run_pipelined(ctx.queue(), pso, static_cast<int>(ins.size()),
                                  [&](MTL::ComputeCommandEncoder* enc, int k) { encode(enc, ins[k], out_pipe, n, thr); });
    r.t_encode = b.encode_us;
    r.t_total = b.per_morsel_us;
    r.spacing = b.spacing_us;
    r.gpu_exec = b.exec_us;
    r.batch = static_cast<int>(ins.size());
    if (!b.ok) r.ok = false;
    return *static_cast<uint32_t*>(out_pipe->contents());
  }

  // Private storage, sync: blit src -> priv (skipped if src is null), then the kernel on priv.
  uint32_t run_private(MTL::Buffer* src, MTL::Buffer* priv, size_t bytes, uint32_t n, uint32_t thr, Row& r) {
    *static_cast<uint32_t*>(out->contents()) = 0;
    std::vector<BlitCopy> copies;
    if (src) copies.push_back({src, priv, bytes});
    BlitComputeTiming t = run_blit_then_compute(ctx.queue(), pso, copies,
                                                [&](MTL::ComputeCommandEncoder* enc) { encode(enc, priv, out, n, thr); });
    r.t_encode = t.encode_us;
    r.t_total = t.total_us;
    r.gpu_exec = t.exec_us;
    r.t_blit = t.blit_exec_us;
    if (!t.ok) r.ok = false;
    return *static_cast<uint32_t*>(out->contents());
  }

  // Sync: runs one morsel on `in`. Fills encode/total/gpu_exec; returns the count.
  uint32_t run(MTL::Buffer* in, uint32_t n, uint32_t thr, Row& r) {
    *static_cast<uint32_t*>(out->contents()) = 0;

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto t0 = Clock::now();
    MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    encode(enc, in, out, n, thr);
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
    return *static_cast<uint32_t*>(out->contents());
  }
};

// ---------------------------------------------------------------- CSV
void write_rows(const std::string& path, const std::string& machine, const std::string& data,
                uint32_t thr, const std::string& cache, const std::string& dispatch, const std::string& storage,
                double spin_ms, const std::vector<Row>& rows) {
  bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!exists)
    f << "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,cache,"
         "dispatch,batch,gpu_spacing_us,storage,t_blit_us,gpu_spin_ms\n";
  for (const Row& r : rows) {
    f << machine << ",filter_count," << data << "," << thr << "," << r.device << "," << r.qos << "," << r.state << "," << r.n << ","
      << r.n * sizeof(uint32_t) << "," << r.rep << "," << r.t_wrap << "," << r.t_encode << ","
      << r.t_total << "," << r.gpu_exec << "," << (r.ok ? 1 : 0) << "," << cache << ","
      << (r.device == "gpu" ? dispatch : "-") << "," << r.batch << "," << r.spacing << ","
      << (r.device == "gpu" ? storage : "-") << "," << r.t_blit << "," << (r.device == "gpu" ? spin_ms : 0.0) << "\n";
  }
}

void usage() {
  std::cerr << "usage: calib_filter --device gpu|cpu [--state warm|cold|resident] [--qos p|e]\n"
               "                    [--column data/tpch-sf1/l_quantity.u32 --thr 2400]   (TPC-H column)\n"
               "                    [--sel 0.5]                                       (random data)\n"
               "                    [--min-log2 10] [--max-log2 26] [--reps 15]\n"
               "                    [--cache cold|hot|cpu-write|cpu-read|gpu-write|gpu-read] [--flush-mb 64]\n"
               "                        (cold: flush caches before each timed rep; cpu-*/gpu-*: flush, then that device\n"
               "                         rewrites or reads the morsel's rows, untimed: its cache state at dispatch)\n"
               "                    [--dispatch sync|pipelined] [--batch 16] [--batch-mb 1024]   (GPU only)\n"
               "                    [--gpu-spin-ms 0]   (GPU only: keep the GPU busy ~M ms after each flush)\n"
               "                    [--storage shared|private]   (GPU only; private: blit, then kernel)\n"
               "                    [--out results/calib_filter.csv] [--machine-json path]\n";
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
    const std::string state = a.str("state", "warm");
    const std::string qos = a.str("qos", "p");
    const int min_l2 = static_cast<int>(a.i64("min-log2", 10));
    int max_l2 = static_cast<int>(a.i64("max-log2", 26));
    const int reps = static_cast<int>(a.i64("reps", 15));
    const double sel = a.f64("sel", 0.5);
    const std::string out_path = a.str("out", "results/calib_filter.csv");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    const std::string cache = a.str("cache", "cold");
    if (std::find(kCachePolicies.begin(), kCachePolicies.end(), cache) == kCachePolicies.end())
      throw std::runtime_error("--cache must be one of cold, hot, cpu-write, cpu-read, gpu-write, gpu-read");
    const std::string dispatch = a.str("dispatch", "sync");
    if (dispatch != "sync" && dispatch != "pipelined") throw std::runtime_error("--dispatch must be sync or pipelined");
    const bool pipelined = dispatch == "pipelined";
    if (pipelined && device != "gpu") throw std::runtime_error("--dispatch pipelined applies to --device gpu only");
    const std::string storage = a.str("storage", "shared");
    if (storage != "shared" && storage != "private") throw std::runtime_error("--storage must be shared or private");
    if (storage == "private" && device != "gpu") throw std::runtime_error("--storage applies to --device gpu only");
    if (storage == "private" && pipelined) throw std::runtime_error("--storage private supports --dispatch sync only");
    if (state == "resident" && storage != "private") throw std::runtime_error("--state resident needs --storage private");
    if (cache_policy_touches(cache) && device == "gpu" && (state != "warm" || storage != "shared"))
      throw std::runtime_error("--cache " + cache + " needs --state warm --storage shared (a GPU-resident buffer)");
    const int batch = static_cast<int>(a.i64("batch", 16));
    const double spin_ms = a.f64("gpu-spin-ms", 0);
    if (spin_ms > 0 && device != "gpu") throw std::runtime_error("--gpu-spin-ms applies to --device gpu only");
    const size_t batch_cap = static_cast<size_t>(a.i64("batch-mb", 1024)) << 20;
    acquire_bench_lock();  // one benchmark process at a time (common/bench.hpp)
    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", default_flush_mb())) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MetalCtx ctx(std::vector<std::string>{std::string(MCM_KERNEL_DIR) + "/filter.metal", std::string(MCM_KERNEL_DIR) + "/spin.metal",
                                          std::string(MCM_KERNEL_DIR) + "/touch.metal"});
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";
    std::unique_ptr<GpuSpin> gspin;  // --gpu-spin-ms: ALU-only GPU work after each flush (busy clock regime)
    if (spin_ms > 0) gspin = std::make_unique<GpuSpin>(ctx, spin_ms);
    auto spin = [&] { if (gspin) (*gspin)(); };
    Toucher touch(&ctx, cache);  // --cache cpu-*/gpu-*: the producer step after each flush (untimed)

    // Master data: a TPC-H column, or random uint32.
    const std::string column = a.str("column", "");
    std::string data_tag = "random";
    uint32_t thr = static_cast<uint32_t>(sel * 4294967295.0);
    std::vector<uint32_t> col;
    if (!column.empty()) {
      std::ifstream f(column, std::ios::binary | std::ios::ate);
      if (!f) throw std::runtime_error("cannot open column " + column);
      const size_t bytes = static_cast<size_t>(f.tellg());
      col.resize(bytes / sizeof(uint32_t));
      f.seekg(0);
      f.read(reinterpret_cast<char*>(col.data()), static_cast<std::streamsize>(col.size() * sizeof(uint32_t)));
      if (!a.has("thr")) throw std::runtime_error("--column needs --thr (predicate: v < thr)");
      thr = static_cast<uint32_t>(a.i64("thr", 0));
      int fit = 0;
      while ((2ull << fit) <= col.size()) ++fit;  // largest l2 with 2^l2 <= rows
      if (fit < max_l2) {
        std::cerr << "column has " << col.size() << " rows; capping max-log2 at " << fit << "\n";
        max_l2 = fit;
      }
      data_tag = column.substr(column.find_last_of('/') + 1);
      data_tag = data_tag.substr(0, data_tag.find('.'));
    }
    const uint64_t max_n = 1ull << max_l2;
    PageAlloc master(max_n * sizeof(uint32_t));
    {
      auto* v = static_cast<uint32_t*>(master.data());
      if (!col.empty()) {
        std::memcpy(v, col.data(), max_n * sizeof(uint32_t));
        std::vector<uint32_t>().swap(col);
      } else {
        std::mt19937 rng(42);
        for (uint64_t i = 0; i < max_n; ++i) v[i] = rng();
      }
    }
    const auto* mv = static_cast<const uint32_t*>(master.data());
    std::cerr << "data: " << data_tag << " | predicate v < " << thr << " | selectivity "
              << static_cast<double>(cpu_count(mv, max_n, thr)) / max_n << "\n";

    std::vector<Row> rows;

    if (device == "gpu") {
      GpuRunner g(ctx);
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        const size_t bytes = n * sizeof(uint32_t);
        const uint32_t expect = cpu_count(mv, n, thr);
        const int K = pipelined ? batch_size(batch, bytes, batch_cap) : 1;
        const uint32_t expect_k = expect * static_cast<uint32_t>(K);  // the GPU counter wraps mod 2^32 too
        // One timed rep: a sync round trip on bufs[0], or a pipelined batch over all K buffers.
        auto timed = [&](const std::vector<MTL::Buffer*>& bufs, Row& row) {
          return pipelined ? g.run_batch(bufs, n, thr, row) == expect_k : g.run(bufs[0], n, thr, row) == expect;
        };

        if (storage == "private") {
          MTL::Buffer* priv = new_private_buffer(ctx.device(), bytes);
          auto run_priv = [&](MTL::Buffer* src, Row& row) { return g.run_private(src, priv, bytes, n, thr, row) == expect; };
          Row warm{};
          if (state == "warm" || state == "resident") {
            PageAlloc mem(bytes);
            std::memcpy(mem.data(), mv, bytes);
            MTL::Buffer* src = wrap_no_copy(ctx.device(), mem);
            for (int w = 0; w < 3; ++w) run_priv(src, warm);  // fills priv, makes both resident; untimed
            for (int r = 0; r < reps; ++r) {
              Row row{"gpu", "-", state, n, r};
              flush();
              spin();
              row.ok = run_priv(state == "warm" ? src : nullptr, row) && row.ok;
              rows.push_back(row);
            }
            src->release();
          } else if (state == "cold") {
            {  // warm the pipeline, the blit path and priv itself, so we measure the source's residency
              PageAlloc tmp(bytes);
              std::memcpy(tmp.data(), mv, bytes);
              MTL::Buffer* b = wrap_no_copy(ctx.device(), tmp);
              run_priv(b, warm);
              b->release();
            }
            for (int r = 0; r < reps; ++r) {
              PageAlloc mem(bytes);
              std::memcpy(mem.data(), mv, bytes);  // CPU-touched, GPU-unseen
              Row row{"gpu", "-", "cold", n, r};
              flush();
              spin();
              auto t0 = Clock::now();
              MTL::Buffer* src = wrap_no_copy(ctx.device(), mem);
              row.t_wrap = us_since(t0, Clock::now());
              row.ok = run_priv(src, row) && row.ok;
              src->release();
              rows.push_back(row);
            }
          } else {
            throw std::runtime_error("--state must be warm, cold or resident");
          }
          priv->release();
        } else if (state == "warm") {
          std::vector<std::unique_ptr<PageAlloc>> mem;
          std::vector<MTL::Buffer*> bufs;
          for (int k = 0; k < K; ++k) {
            mem.push_back(std::make_unique<PageAlloc>(bytes));
            std::memcpy(mem.back()->data(), mv, bytes);
            bufs.push_back(wrap_no_copy(ctx.device(), *mem.back()));
          }
          std::vector<Toucher::Range> ranges;
          for (int k = 0; k < K; ++k) ranges.push_back({static_cast<uint32_t*>(mem[k]->data()), bufs[k], 0, n});
          Row warm{};
          for (int w = 0; w < 3; ++w) timed(bufs, warm);  // make resident, untimed
          for (int r = 0; r < reps; ++r) {
            Row row{"gpu", "-", "warm", n, r};
            flush();
            touch(ranges);  // --cache cpu-*/gpu-*: the producer's pass over the K morsels
            spin();
            row.ok = timed(bufs, row) && row.ok;
            rows.push_back(row);
          }
          for (MTL::Buffer* b : bufs) b->release();
        } else if (state == "cold") {
          Row warm{};
          {  // warm the pipeline itself so we measure buffer residency, not first-dispatch setup
            PageAlloc tmp(4096);
            std::memset(tmp.data(), 0, tmp.size());
            MTL::Buffer* b = wrap_no_copy(ctx.device(), tmp);
            g.run(b, 1, thr, warm);
            b->release();
          }
          for (int r = 0; r < reps; ++r) {
            std::vector<std::unique_ptr<PageAlloc>> mem;
            for (int k = 0; k < K; ++k) {
              mem.push_back(std::make_unique<PageAlloc>(bytes));
              std::memcpy(mem.back()->data(), mv, bytes);  // CPU-touched, GPU-unseen
            }
            Row row{"gpu", "-", "cold", n, r};
            std::vector<MTL::Buffer*> bufs;
            bufs.reserve(K);
            flush();  // the memcpy above would otherwise leave the input cache-hot
            spin();
            auto t0 = Clock::now();
            for (auto& m : mem) bufs.push_back(wrap_no_copy(ctx.device(), *m));
            row.t_wrap = us_since(t0, Clock::now()) / K;
            row.ok = timed(bufs, row) && row.ok;
            for (MTL::Buffer* b : bufs) b->release();
            rows.push_back(row);
          }
        } else {
          throw std::runtime_error("--state must be warm or cold");
        }
        std::cerr << "gpu " << state << " " << storage << " " << dispatch << (pipelined ? " x" + std::to_string(K) : "")
                  << " n=2^" << l2 << " done\n";
      }
    } else if (device == "cpu") {
      place_cpu_process(qos);  // see common/bench.hpp: E needs the process-wide policy
      // --cache gpu-*: the GPU touches the master column itself, wrapped no-copy (the CPU reads the same memory).
      MTL::Buffer* master_buf = cache_policy_on_gpu(cache) ? wrap_no_copy(ctx.device(), master) : nullptr;
      // Run the whole sweep inside one worker thread with the requested QoS.
      std::thread worker([&] {
        place_cpu_thread(qos);
        for (int l2 = min_l2; l2 <= max_l2; ++l2) {
          const uint64_t n = 1ull << l2;
          const uint32_t expect = cpu_count(mv, n, thr);
          const std::vector<Toucher::Range> ranges{{static_cast<uint32_t*>(master.data()), master_buf, 0, static_cast<uint32_t>(n)}};
          for (int r = 0; r < reps; ++r) {
            Row row{"cpu", qos, r == 0 ? "first" : "warm", n, r};
            flush();
            touch(ranges);  // --cache cpu-*: this thread's own pass (L2); gpu-*: a GPU kernel's
            asm volatile("" ::: "memory");  // stop the compiler reusing `expect`
            auto t0 = Clock::now();
            uint32_t c = cpu_count(mv, n, thr);
            row.t_total = us_since(t0, Clock::now());
            row.ok = (c == expect);
            rows.push_back(row);
          }
          std::cerr << "cpu " << qos << " n=2^" << l2 << " done\n";
        }
      });
      worker.join();
      if (master_buf) master_buf->release();
    } else {
      throw std::runtime_error("--device must be gpu or cpu");
    }

    size_t bad = std::count_if(rows.begin(), rows.end(), [](const Row& r) { return !r.ok; });
    write_rows(out_path, mi.tag(), data_tag, thr, cache, dispatch, storage, spin_ms, rows);
    std::cerr << rows.size() << " rows -> " << out_path << (bad ? " | WARNING: " : "")
              << (bad ? std::to_string(bad) + " rows failed the correctness check" : "") << "\n";
    pool->release();
    return bad ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
