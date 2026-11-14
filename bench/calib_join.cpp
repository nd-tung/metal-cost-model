// calib_join — hash-join calibration microbenchmark: build and probe on CPU-P, CPU-E and the GPU.
//
// One join configuration per run (scripts/run_calib_join.sh has the TPC-H ones): build rows from
// --build (key column), --payload and optionally --filter (rows whose filter value is 0 are not
// inserted, e.g. part filtered to PROMO types), probed with --probe. The table is sized for every
// build row that passes the filter (src/engine/hash_join.hpp: open addressing, linear probing).
//   --phase probe : the whole table is built once (CPU, untimed); a morsel is n probe rows (the first n
//                   rows of the probe column, or K disjoint ranges in a pipelined batch); output is the
//                   match count + payload sum (fused, not materialized).
//   --phase build : a morsel is the first n build rows, inserted into the empty full-size table
//                   (cleared, untimed, before every rep). Sync dispatch only.
// Devices, states, cache and dispatch as in calib_agg:
//   gpu warm  : input column(s) wrapped no-copy once and used before
//   gpu cold  : a fresh host copy of the morsel's input, wrapped right before the rep (t_wrap)
//   cpu p|e   : one worker thread
//   --dispatch pipelined (probe only): K morsels per timed rep, t_total = GPU span / K
//   --gpu-spin-ms M : keep the GPU busy ~M ms (ALU only) before each timed GPU rep (busy regime)
// The table is always GPU-resident: on UMA it is the same memory the CPU built or probes.
// Checks: every probe morsel against an independent dense reference, every build against the exact
// contents of the table, and the whole probe column against DuckDB (<probe dir>/join_expected.txt).
// CSV columns as calib_agg's plus gpu_spin_ms: op = join_probe | join_build, data = <join name>_<phase>,
// groups = build rows after the filter, path = linear-probe, bytes = 4 n (probe), 8 or 12 n (build).

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstdint>
#include <cstring>
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
#include "engine/hash_join.hpp"

using namespace mcm;

namespace {

struct Row {
  std::string device, qos, state;
  uint64_t n = 0;
  int rep = 0;
  double t_wrap = 0, t_encode = 0, t_total = 0, gpu_exec = 0;
  bool ok = true;
  int batch = 1;
  double spacing = 0;
};

struct JoinParams {
  uint32_t n, bits, mask, use_filter;  // must match kernels/join.metal
};

// A column in one page-aligned allocation (wrappable no-copy for the GPU).
struct Col {
  std::unique_ptr<PageAlloc> mem;
  size_t rows = 0;
  const uint32_t* data() const { return static_cast<const uint32_t*>(mem->data()); }
};

Col load_col(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open column " + path);
  Col c;
  const size_t bytes = static_cast<size_t>(f.tellg());
  c.rows = bytes / sizeof(uint32_t);
  c.mem = std::make_unique<PageAlloc>(bytes);
  f.seekg(0);
  f.read(static_cast<char*>(c.mem->data()), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("short read on " + path);
  return c;
}

class GpuJoin {
 public:
  static constexpr NS::UInteger kThreads = 256;

  GpuJoin(MetalCtx& ctx, join::Table& t) : ctx_(ctx), t_(t) {
    build_ = ctx.pipeline("join_build");
    probe_ = ctx.pipeline("join_probe");
    table_ = wrap_no_copy(ctx.device(), *t.mem);
    out_ = ctx.device()->newBuffer(2 * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    out_pipe_ = ctx.device()->newBuffer(2 * sizeof(uint32_t),
                                        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    stats_ = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
  }
  ~GpuJoin() {
    for (MTL::Resource* r : {static_cast<MTL::Resource*>(stats_), static_cast<MTL::Resource*>(out_pipe_),
                             static_cast<MTL::Resource*>(out_), static_cast<MTL::Resource*>(table_)})
      r->release();
    probe_->release();
    build_->release();
  }

  MTL::ComputePipelineState* probe_pso() const { return probe_; }
  MTL::Buffer* out(bool pipelined) const { return pipelined ? out_pipe_ : out_; }
  void zero_out(bool pipelined) { std::memset(out(pipelined)->contents(), 0, 2 * sizeof(uint32_t)); }
  join::ProbeResult result(bool pipelined) const {
    const auto* v = static_cast<const uint32_t*>(out(pipelined)->contents());
    return {v[0], v[1]};
  }
  uint32_t inserted() const { return *static_cast<const uint32_t*>(stats_->contents()); }

  void encode_probe(MTL::ComputeCommandEncoder* enc, MTL::Buffer* keys, size_t offset, uint32_t n, bool pipelined) const {
    const JoinParams p{n, static_cast<uint32_t>(t_.bits), t_.mask, 0};
    enc->setComputePipelineState(probe_);
    enc->setBuffer(keys, offset, 0);
    enc->setBuffer(table_, 0, 1);
    enc->setBuffer(out(pipelined), 0, 2);
    enc->setBytes(&p, sizeof(p), 3);
    enc->dispatchThreadgroups(MTL::Size(groups(n), 1, 1), MTL::Size(kThreads, 1, 1));
  }

  void encode_build(MTL::ComputeCommandEncoder* enc, MTL::Buffer* keys, MTL::Buffer* pay, MTL::Buffer* filt,
                    size_t offset, uint32_t n) const {
    const JoinParams p{n, static_cast<uint32_t>(t_.bits), t_.mask, filt ? 1u : 0u};
    std::memset(stats_->contents(), 0, sizeof(uint32_t));
    enc->setComputePipelineState(build_);
    enc->setBuffer(keys, offset, 0);
    enc->setBuffer(pay, offset, 1);
    enc->setBuffer(filt ? filt : keys, offset, 2);  // unused without a filter
    enc->setBuffer(table_, 0, 3);
    enc->setBuffer(stats_, 0, 4);
    enc->setBytes(&p, sizeof(p), 5);
    enc->dispatchThreadgroups(MTL::Size(groups(n), 1, 1), MTL::Size(kThreads, 1, 1));
  }

  // One synchronous round trip, timed as in calib_agg.
  template <class Enc>
  void run_sync(Row& r, Enc&& encode) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto t0 = Clock::now();
    MTL::CommandBuffer* cb = ctx_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    encode(enc);
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
  static NS::UInteger groups(uint32_t n) {
    return std::clamp<NS::UInteger>((n + kThreads * 16 - 1) / (kThreads * 16), 1, 4096);  // ~16 rows per thread
  }
  MetalCtx& ctx_;
  join::Table& t_;
  MTL::ComputePipelineState* build_ = nullptr;
  MTL::ComputePipelineState* probe_ = nullptr;
  MTL::Buffer* table_ = nullptr;
  MTL::Buffer* out_ = nullptr;
  MTL::Buffer* out_pipe_ = nullptr;
  MTL::Buffer* stats_ = nullptr;
};

void write_rows(const std::string& path, const std::string& machine, const std::string& op, const std::string& data,
                const std::string& cache, size_t build_rows, uint64_t bytes_per_row, const std::string& dispatch,
                double spin_ms, const std::vector<Row>& rows) {
  const bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!exists)
    f << "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,"
         "cache,groups,path,dispatch,batch,gpu_spacing_us,gpu_spin_ms\n";
  for (const Row& r : rows)
    f << machine << "," << op << "," << data << ",0," << r.device << "," << r.qos << "," << r.state << "," << r.n << ","
      << r.n * bytes_per_row << "," << r.rep << "," << r.t_wrap << "," << r.t_encode << "," << r.t_total << ","
      << r.gpu_exec << "," << (r.ok ? 1 : 0) << "," << cache << "," << build_rows << ",linear-probe,"
      << (r.device == "gpu" ? dispatch : "-") << "," << r.batch << "," << r.spacing << ","
      << (r.device == "gpu" ? spin_ms : 0.0) << "\n";
}

void usage() {
  std::cerr << "usage: calib_join --name supplier --build s_suppkey.u32 --payload s_nationkey.u32 [--filter p_promo.u32]\n"
               "                  --probe l_suppkey.u32 [--phase probe|build]\n"
               "                  --device gpu|cpu [--state warm|cold] [--qos p|e]\n"
               "                  [--dispatch sync|pipelined] [--batch 16] [--batch-mb 1024] [--gpu-spin-ms 0]\n"
               "                  [--min-log2 10] [--max-log2 25] [--reps 15] [--cache cold|hot] [--flush-mb 64]\n"
               "                  [--out results/calib_join.csv] [--machine-json path] [--expect join_expected.txt]\n";
}

std::string dir_of(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help") || !a.has("device") || !a.has("name") || !a.has("build") || !a.has("payload") ||
        !a.has("probe")) {
      usage();
      return a.has("help") ? 0 : 1;
    }
    const std::string name = a.str("name", "");
    const std::string phase = a.str("phase", "probe");
    if (phase != "probe" && phase != "build") throw std::runtime_error("--phase must be probe or build");
    const std::string device = a.str("device", "gpu");
    if (device != "gpu" && device != "cpu") throw std::runtime_error("--device must be gpu or cpu");
    const std::string state = a.str("state", "warm");
    if (state != "warm" && state != "cold") throw std::runtime_error("--state must be warm or cold");
    const std::string qos = a.str("qos", "p");
    const std::string dispatch = a.str("dispatch", "sync");
    if (dispatch != "sync" && dispatch != "pipelined") throw std::runtime_error("--dispatch must be sync or pipelined");
    const bool pipelined = dispatch == "pipelined";
    if (pipelined && (device != "gpu" || phase != "probe"))
      throw std::runtime_error("--dispatch pipelined applies to --device gpu --phase probe only");
    const int batch = static_cast<int>(a.i64("batch", 16));
    const size_t batch_cap = static_cast<size_t>(a.i64("batch-mb", 1024)) << 20;
    const int min_l2 = static_cast<int>(a.i64("min-log2", 10));
    int max_l2 = static_cast<int>(a.i64("max-log2", 25));
    const int reps = static_cast<int>(a.i64("reps", 15));
    const std::string cache = a.str("cache", "cold");
    if (cache != "cold" && cache != "hot") throw std::runtime_error("--cache must be cold or hot");
    const double spin_ms = a.f64("gpu-spin-ms", 0);
    if (spin_ms > 0 && device != "gpu") throw std::runtime_error("--gpu-spin-ms applies to --device gpu only");
    const std::string out_path = a.str("out", "results/calib_join.csv");
    const std::string probe_path = a.str("probe", "");
    const std::string expect_path = a.str("expect", dir_of(probe_path) + "/join_expected.txt");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    acquire_bench_lock();  // one benchmark process at a time (common/bench.hpp)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    const std::string kdir = MCM_KERNEL_DIR;
    MetalCtx ctx(std::vector<std::string>{kdir + "/join.metal", kdir + "/spin.metal"});
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

    // ---- data, reference, table
    Col bkeys = load_col(a.str("build", "")), bpay = load_col(a.str("payload", "")), probe = load_col(probe_path);
    Col bfilt;
    if (a.has("filter")) bfilt = load_col(a.str("filter", ""));
    if (bpay.rows != bkeys.rows || (bfilt.mem && bfilt.rows != bkeys.rows))
      throw std::runtime_error("--build, --payload and --filter have different row counts");
    const uint32_t* filt = bfilt.mem ? bfilt.data() : nullptr;
    join::Reference ref(bkeys.data(), bpay.data(), filt, bkeys.rows);
    const size_t build_rows = ref.build_rows();
    join::Table table = join::make_table(build_rows);
    const size_t morsel_rows = phase == "probe" ? probe.rows : bkeys.rows;
    int fit = 0;
    while ((2ull << fit) <= morsel_rows) ++fit;
    if (fit < max_l2) {
      std::cerr << (phase == "probe" ? "probe" : "build") << " side has " << morsel_rows << " rows; capping max-log2 at "
                << fit << "\n";
      max_l2 = fit;
    }
    std::cerr << "join " << name << ": build " << bkeys.rows << " rows, " << build_rows << " after the filter | table "
              << table.cap << " slots (" << (table.bytes() >> 10) << " KiB) | probe " << probe.rows << " rows | phase "
              << phase << "\n";

    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", default_flush_mb())) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };
    std::unique_ptr<GpuJoin> gpu;
    if (device == "gpu") gpu = std::make_unique<GpuJoin>(ctx, table);
    std::unique_ptr<GpuSpin> spin;
    if (spin_ms > 0) spin = std::make_unique<GpuSpin>(ctx, spin_ms);
    if (device == "cpu") place_cpu_process(qos);
    MTL::Buffer *wkeys = nullptr, *wpay = nullptr, *wfilt = nullptr, *wprobe = nullptr;  // warm inputs
    if (device == "gpu") {
      wkeys = wrap_no_copy(ctx.device(), *bkeys.mem);
      wpay = wrap_no_copy(ctx.device(), *bpay.mem);
      if (filt) wfilt = wrap_no_copy(ctx.device(), *bfilt.mem);
      wprobe = wrap_no_copy(ctx.device(), *probe.mem);
    }
    // Runs `fn` on the worker thread with the CPU placement (CPU runs) or inline (GPU runs).
    auto on_device = [&](auto&& fn) {
      if (device == "cpu") {
        std::thread t([&] {
          place_cpu_thread(qos);
          fn();
        });
        t.join();
      } else {
        fn();
      }
    };

    // ---- full build (on the device under test) and full probe column: correctness first
    bool full_ok = true;
    on_device([&] {
      Row r;
      if (device == "gpu") {
        gpu->run_sync(r, [&](MTL::ComputeCommandEncoder* enc) {
          gpu->encode_build(enc, wkeys, wpay, wfilt, 0, static_cast<uint32_t>(bkeys.rows));
        });
      } else {
        join::cpu_build(table, bkeys.data(), bpay.data(), filt, 0, bkeys.rows);
      }
      const bool build_ok = ref.check_build(table, bkeys.rows, build_rows) && r.ok &&
                            (device != "gpu" || gpu->inserted() == build_rows);
      join::ProbeResult got;
      constexpr size_t kMorsel = size_t(1) << 24;
      for (size_t b = 0; b < probe.rows; b += kMorsel) {
        const uint32_t n = static_cast<uint32_t>(std::min(kMorsel, probe.rows - b));
        if (device == "gpu") {
          Row pr;
          gpu->zero_out(false);
          gpu->run_sync(pr, [&](MTL::ComputeCommandEncoder* enc) { gpu->encode_probe(enc, wprobe, b * 4, n, false); });
          const join::ProbeResult m = gpu->result(false);
          got.matches += m.matches;
          got.sum += m.sum;
        } else {
          const join::ProbeResult m = join::cpu_probe(table, probe.data(), b, n);
          got.matches += m.matches;
          got.sum += m.sum;
        }
      }
      const join::ProbeResult want = ref.probe(probe.data(), 0, probe.rows);
      size_t exp_rows = 0;
      join::ProbeResult exp;
      const bool have = join::load_expected(expect_path, name, exp_rows, exp);
      const bool duck_ok = !have || (exp == got && exp_rows == build_rows);
      full_ok = build_ok && got == want && duck_ok;
      std::cerr << "full build + probe on " << device << (device == "cpu" ? "-" + qos : "") << ": build "
                << (build_ok ? "ok" : "WRONG") << " | " << got.matches << " matches ("
                << 100.0 * got.matches / std::max<size_t>(probe.rows, 1) << "%) | reference "
                << (got == want ? "match" : "MISMATCH") << " | DuckDB "
                << (!have ? "not checked (no " + name + " in " + expect_path + ")" : duck_ok ? "match" : "MISMATCH")
                << "\n";
    });

    // ---- sweep
    std::vector<Row> rows;
    std::vector<uint32_t> passing(bkeys.rows + 1, 0);  // build rows passing the filter in [0, i)
    for (size_t i = 0; i < bkeys.rows; ++i) passing[i + 1] = passing[i] + (!filt || filt[i] ? 1 : 0);

    on_device([&] {
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        if (phase == "probe") {
          const int K = pipelined ? batch_size(batch, size_t(n) * 4, batch_cap) : 1;
          std::vector<size_t> begin(K);
          join::ProbeResult expect;
          for (int k = 0; k < K; ++k) {
            begin[k] = (size_t(k) * n) % (probe.rows - n + 1);  // disjoint while K * n <= probe rows
            const join::ProbeResult m = ref.probe(probe.data(), begin[k], n);
            expect.matches += m.matches;
            expect.sum += m.sum;
          }
          auto gpu_rep = [&](Row& row) {  // one timed GPU rep (after flush + spin)
            std::vector<std::unique_ptr<PageAlloc>> mem;
            std::vector<MTL::Buffer*> bufs;
            std::vector<size_t> off(K);
            if (state == "cold") {
              for (int k = 0; k < K; ++k) {
                mem.push_back(std::make_unique<PageAlloc>(size_t(n) * 4));
                std::memcpy(mem.back()->data(), probe.data() + begin[k], size_t(n) * 4);  // CPU-touched, GPU-unseen
              }
            }
            gpu->zero_out(pipelined);
            flush();
            if (spin) (*spin)();
            auto t0 = Clock::now();
            for (int k = 0; k < K; ++k) {
              if (state == "cold") {
                bufs.push_back(wrap_no_copy(ctx.device(), *mem[k]));
                off[k] = 0;
              } else {
                bufs.push_back(wprobe);
                off[k] = begin[k] * 4;
              }
            }
            row.t_wrap = state == "cold" ? us_since(t0, Clock::now()) / K : 0;
            if (!pipelined) {
              gpu->run_sync(row, [&](MTL::ComputeCommandEncoder* enc) { gpu->encode_probe(enc, bufs[0], off[0], n, false); });
            } else {
              BatchTiming b = run_pipelined(ctx.queue(), gpu->probe_pso(), K, [&](MTL::ComputeCommandEncoder* enc, int k) {
                gpu->encode_probe(enc, bufs[k], off[k], n, true);
              });
              row.t_encode = b.encode_us;
              row.t_total = b.per_morsel_us;
              row.spacing = b.spacing_us;
              row.gpu_exec = b.exec_us;
              row.batch = K;
              if (!b.ok) row.ok = false;
            }
            if (state == "cold")
              for (MTL::Buffer* bb : bufs) bb->release();
            return gpu->result(pipelined) == expect && row.ok;
          };
          if (device == "gpu") {
            Row warm{};
            for (int w = 0; w < 3; ++w) gpu_rep(warm);  // untimed
          }
          for (int r = 0; r < reps; ++r) {
            Row row{device, device == "gpu" ? "-" : qos, device == "gpu" ? state : (r == 0 ? "first" : "warm"), n, r};
            if (device == "gpu") {
              row.ok = gpu_rep(row);
            } else {
              flush();
              asm volatile("" ::: "memory");
              auto t0 = Clock::now();
              const join::ProbeResult got = join::cpu_probe(table, probe.data(), 0, n);
              asm volatile("" ::: "memory");
              row.t_total = us_since(t0, Clock::now());
              row.ok = got == expect;
            }
            rows.push_back(row);
          }
        } else {  // build
          const size_t want_rows = passing[n];
          auto rep_once = [&](Row& row) {
            table.clear();
            std::unique_ptr<PageAlloc> mk, mp, mf;
            if (device == "gpu" && state == "cold") {
              mk = std::make_unique<PageAlloc>(size_t(n) * 4);
              mp = std::make_unique<PageAlloc>(size_t(n) * 4);
              std::memcpy(mk->data(), bkeys.data(), size_t(n) * 4);
              std::memcpy(mp->data(), bpay.data(), size_t(n) * 4);
              if (filt) {
                mf = std::make_unique<PageAlloc>(size_t(n) * 4);
                std::memcpy(mf->data(), filt, size_t(n) * 4);
              }
            }
            flush();
            if (spin) (*spin)();
            if (device == "gpu") {
              MTL::Buffer *k = wkeys, *p = wpay, *f = wfilt;
              auto t0 = Clock::now();
              if (state == "cold") {
                k = wrap_no_copy(ctx.device(), *mk);
                p = wrap_no_copy(ctx.device(), *mp);
                f = filt ? wrap_no_copy(ctx.device(), *mf) : nullptr;
                row.t_wrap = us_since(t0, Clock::now());
              }
              gpu->run_sync(row, [&](MTL::ComputeCommandEncoder* enc) { gpu->encode_build(enc, k, p, f, 0, n); });
              if (state == "cold") {
                k->release();
                p->release();
                if (f) f->release();
              }
              row.ok = row.ok && gpu->inserted() == want_rows;
            } else {
              asm volatile("" ::: "memory");
              auto t0 = Clock::now();
              const uint32_t ins = join::cpu_build(table, bkeys.data(), bpay.data(), filt, 0, n);
              asm volatile("" ::: "memory");
              row.t_total = us_since(t0, Clock::now());
              row.ok = ins == want_rows;
            }
            return ref.check_build(table, n, want_rows) && row.ok;
          };
          if (device == "gpu") {
            Row warm{};
            for (int w = 0; w < 3; ++w) rep_once(warm);
          }
          for (int r = 0; r < reps; ++r) {
            Row row{device, device == "gpu" ? "-" : qos, device == "gpu" ? state : (r == 0 ? "first" : "warm"), n, r};
            row.ok = rep_once(row);
            rows.push_back(row);
          }
        }
        std::cerr << device << (device == "gpu" ? " " + state + " " + dispatch : " " + qos) << " " << phase << " n=2^"
                  << l2 << " done\n";
      }
    });

    const size_t bad = std::count_if(rows.begin(), rows.end(), [](const Row& r) { return !r.ok; });
    const uint64_t bpr = phase == "probe" ? 4 : (filt ? 12 : 8);
    write_rows(out_path, mi.tag(), "join_" + phase, name + "_" + phase, cache, build_rows, bpr, dispatch, spin_ms, rows);
    std::cerr << rows.size() << " rows -> " << out_path
              << (bad ? " | WARNING: " + std::to_string(bad) + " rows failed the correctness check" : "")
              << (full_ok ? "" : " | WARNING: the full build/probe check failed") << "\n";
    for (MTL::Buffer* b : {wkeys, wpay, wfilt, wprobe})
      if (b) b->release();
    spin.reset();
    gpu.reset();
    pool->release();
    return (bad || !full_ok) ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
