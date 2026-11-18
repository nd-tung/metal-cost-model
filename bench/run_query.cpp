// run_query — TPC-H Q1, Q6 or Q14 end to end: the whole table in morsels of n rows, placed on the GPU,
// on CPU worker threads (P and E), or on both, under a placement policy. The evaluation of the router
// (PLAN.md M3): what the cost model's placement wins over the static placements and over a
// model-free dynamic scheduler, measured as the wall time of the whole query.
//
// Q14 has a pipeline breaker: the part hash table (2M rows at SF10) is built before any lineitem
// morsel probes it. Every rep starts with that build phase, timed: --build-device gpu clears and
// builds the table in one command buffer (engine/hash_join_gpu.hpp) while every CPU worker waits,
// blocked; --build-device cpu has the P workers build it together (engine/q14.hpp: each clears its
// share of the table, a barrier, each inserts its share of the parts with a CAS per row) while the
// GPU thread and the E workers wait. The probe morsels start when the whole table is built, the GPU's
// driver thread woken first; the CSV has the build phase's time (build_ms) apart.
//
//   --policy gpu      every morsel on the GPU, pipelined (--inflight command buffers in flight)
//   --policy cpu      every morsel on the CPU workers (--p-workers at USER_INTERACTIVE QoS, --e-workers at
//                     BACKGROUND QoS), which share one morsel counter
//   --policy split    the model's static split: the GPU takes the first share of the morsels, the CPU
//                     workers the rest, the share from the predicted per-morsel times (--t-gpu, --t-p,
//                     --t-e, in us, under contention; analysis/plan_e2e.py computes them from
//                     models/<machine>.json) so that both sides finish together; --gpu-share overrides it
//   --policy dynamic  one shared counter: every device takes the next morsel when it is free (no model)
//   --policy router   the shared counter, guided by the model: a device takes the next morsel only if its
//                     predicted latency for it (the GPU's: its per-morsel time times the morsels it has in
//                     flight plus one) is no longer than the time the other devices still taking morsels
//                     would need for everything left (for a CPU worker, the GPU's queue included), so a
//                     slow worker does not start a morsel that would end after the rest of the query; a
//                     CPU worker the rule turns away stops and is no longer counted; the device with the
//                     shortest latency always takes
//   --policy cpu-router  the router's rule on the CPU workers alone (GPU off): the model keeps the E-cores,
//                     and at the end the P-cores, from starting a morsel that would be the tail
// Every rep runs the whole table once and is checked against the scalar reference and, if present,
// <data>/<query>_expected.txt (DuckDB). One warm-up run per policy, untimed. Before every rep the GPU is
// kept busy for --gpu-spin-ms (default 20, ALU only, untimed), the clock regime the model was calibrated
// in, so a rep never starts on a GPU that idled while the CPU workers were set up. No cache flush: the
// table (SF10: 0.96 GB for Q6 and Q14, 1.7 GB for Q1) is far larger than the caches, and a query starts cold anyway.
// CSV (one row per rep): machine, query, policy, n, morsels, workers, inflight, the predicted times, the
// measured wall time, the morsels and busy time per device class and when each class finished, the build
// phase (build_device, build_ms) and the GPU's command queues (gpu_queues).
//
// usage: run_query --query q1|q6|q14 --policy gpu|cpu|split|dynamic|router|cpu-router [--morsel-log2 20] [--data data/tpch-sf10]
//                  [--p-workers <P cores>] [--e-workers <E cores>] [--inflight 16] [--t-gpu us] [--t-p us] [--t-e us]
//                  [--gpu-share f] [--pred-ms t] [--gpu-spin-ms 20] [--reps 3] [--out results/run_query.csv]
//                  [--machine-json path] [--expect <data>/<query>_expected.txt] [--build-device gpu|cpu (q14)]
//                  [--gpu-timeline path]   (diagnostic: one CSV row per GPU command buffer of every timed rep: the
//                                           morsel, when the driver thread began encoding it, committed it and saw
//                                           it complete, and its GPU start and end; us from the rep's start, host clock)
//                  [--queues 1]            (diagnostic: the GPU's command buffers in turn on this many command queues,
//                                           the first the context's. More queues do not shorten the GPU's idle gaps
//                                           between the driver's batches: results/Apple-M3-Ultra_20261117-150333_queues)
#include <mach/mach_time.h>
#include <pthread/qos.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/args.hpp"
#include "common/bench.hpp"
#include "common/machine_info.hpp"
#include "common/metal_ctx.hpp"
#include "common/timer.hpp"
#include "engine/hash_join_gpu.hpp"
#include "engine/q1.hpp"
#include "engine/q14.hpp"
#include "engine/q14_gpu.hpp"
#include "engine/q1_gpu.hpp"
#include "engine/q6.hpp"
#include "engine/q6_gpu.hpp"

namespace {

using mcm::Clock;
using mcm::us_since;

struct Q1 {
  static constexpr const char* kOp = "q1";
  static constexpr const char* kKernelFile = "q1.metal";
  static constexpr bool kHasBuild = false;
  static constexpr size_t kBytesPerRow = mcm::q1::kBytesPerRow;
  using Columns = mcm::q1::Columns;
  using Result = mcm::q1::Result;
  using Gpu = mcm::q1::Gpu;
  static Columns load(const std::string& dir) { return mcm::q1::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q1::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q1::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q1::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q1::describe(r); }
};

struct Q6 {
  static constexpr const char* kOp = "q6";
  static constexpr const char* kKernelFile = "q6.metal";
  static constexpr bool kHasBuild = false;
  static constexpr size_t kBytesPerRow = mcm::q6::kBytesPerRow;
  using Columns = mcm::q6::Columns;
  using Result = mcm::q6::Result;
  using Gpu = mcm::q6::Gpu;
  static Columns load(const std::string& dir) { return mcm::q6::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q6::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q6::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q6::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q6::describe(r); }
};

struct Q14 {
  static constexpr const char* kOp = "q14";
  static constexpr const char* kKernelFile = "q14.metal";
  static constexpr bool kHasBuild = true;  // the part hash table, before the probe morsels
  static constexpr size_t kBytesPerRow = mcm::q14::kBytesPerRow;
  using Columns = mcm::q14::Columns;
  using Result = mcm::q14::Result;
  using Gpu = mcm::q14::Gpu;
  static Columns load(const std::string& dir) { return mcm::q14::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q14::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q14::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q14::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q14::describe(r); }
  static void clear_share(Columns& c, int w, int k) { mcm::q14::clear_share(c, w, k); }
  static uint32_t build_share(Columns& c, int w, int k) { return mcm::q14::build_share(c, w, k); }
};

// The rows the build phase inserts (0: the query has none).
template <class C>
size_t cols_rows_built(const C& c) {
  if constexpr (requires { c.parts; }) return c.parts; else return 0;
}

struct Morsel {
  size_t begin;
  uint32_t n;
};

enum class Dev { gpu, p, e };
constexpr size_t kNone = static_cast<size_t>(-1);

// Who takes the next morsel, per policy (file header).
struct Dispatcher {
  std::string policy;
  size_t M = 0;
  std::atomic<size_t> next{0};      // shared counter (cpu, dynamic, router), or the CPU side of split
  std::atomic<size_t> gpu_next{0};  // split: the GPU's own counter over [0, gpu_end)
  size_t gpu_end = 0;
  double t_gpu = 0, t_p = 0, t_e = 0;  // predicted us per morsel under contention (split, router)
  int n_p = 0, n_e = 0, inflight = 1;
  bool gpu_on = false;

  double rate(Dev d) const { return d == Dev::gpu ? 1.0 / t_gpu : d == Dev::p ? 1.0 / t_p : 1.0 / t_e; }
  double total_rate() const { return (gpu_on ? rate(Dev::gpu) : 0) + n_p * rate(Dev::p) + n_e * rate(Dev::e); }
  double latency(Dev d, int in_flight) const {
    return d == Dev::gpu ? t_gpu * (in_flight + 1) : d == Dev::p ? t_p : t_e;
  }
  std::atomic<int> gpu_in_flight{0};  // the GPU's queue depth right now (its latency for one more morsel)
  // Router: the CPU workers of each class still taking morsels (reset every rep). A worker the rule
  // turns away stops and leaves its count, so the rule counts only the devices that will still work.
  std::atomic<int> active_p{0}, active_e{0};
  // The shortest latency any device class offers now: that class always takes, so the work never stalls.
  double shortest_latency() const {
    double best = 1e300;
    if (gpu_on) best = std::min(best, latency(Dev::gpu, gpu_in_flight.load()));
    if (n_p > 0) best = std::min(best, t_p);
    if (n_e > 0) best = std::min(best, t_e);
    return best;
  }
  size_t take(Dev d, int in_flight) {
    if (d == Dev::gpu) gpu_in_flight.store(in_flight);
    if (policy == "split" && d == Dev::gpu) {
      const size_t i = gpu_next.fetch_add(1);
      return i < gpu_end ? i : kNone;
    }
    if (policy == "router" || policy == "cpu-router") {
      const double lat = latency(d, in_flight);
      if (lat > shortest_latency()) {
        // The others: the GPU and the CPU workers still taking morsels, this one excluded. Everything
        // left: the morsels nobody has taken and, for a CPU worker, the GPU's queue (taken, not done).
        // Until 17 Nov every worker was counted, also those the rule had turned away, and the GPU's
        // queue was not, so at Q14 2^20 no P worker took a morsel (README "Router: end to end").
        std::atomic<int>* mine = d == Dev::p ? &active_p : d == Dev::e ? &active_e : nullptr;
        int a = mine ? mine->load() : 0;
        while (true) {
          const size_t queued = gpu_on && d != Dev::gpu ? static_cast<size_t>(gpu_in_flight.load()) : 0;
          const size_t rem = M - std::min(M, next.load()) + queued;
          const int ap = d == Dev::p ? a - 1 : active_p.load(), ae = d == Dev::e ? a - 1 : active_e.load();
          const double others = (gpu_on && d != Dev::gpu ? rate(Dev::gpu) : 0) + (ap > 0 ? ap * rate(Dev::p) : 0) +
                                (ae > 0 ? ae * rate(Dev::e) : 0);
          if (!(others > 0 && lat > static_cast<double>(rem) / others)) break;  // it takes
          if (!mine) return kNone;  // the GPU: not now (its thread retires a command buffer and asks again)
          if (mine->compare_exchange_weak(a, a - 1)) return kNone;  // the worker stops and leaves the count
          // the count changed meanwhile (a holds the new one): decide again
        }
      }
    }
    const size_t i = next.fetch_add(1);
    return i < M ? i : kNone;
  }
};

struct DevStats {
  size_t morsels = 0;
  double busy_us = 0, finished_us = 0;  // finished: when the last morsel of this device completed, from t0
};

// Host time in seconds on the clock of MTLCommandBuffer GPUStartTime / GPUEndTime (mach_absolute_time).
inline double host_s() {
  static const double tick = [] {
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    return static_cast<double>(tb.numer) / static_cast<double>(tb.denom) * 1e-9;
  }();
  return static_cast<double>(mach_absolute_time()) * tick;
}

struct CbTime {  // --gpu-timeline: one GPU command buffer
  size_t morsel;
  double encode_s, commit_s, retire_s, gpu_start_s, gpu_end_s;
  int queue;
};

template <class Q>
int run(const mcm::Args& a) {
  using Columns = typename Q::Columns;
  using Result = typename Q::Result;
  using Gpu = typename Q::Gpu;
  const std::string policy = a.str("policy", "");
  const std::vector<std::string> policies = {"gpu", "cpu", "split", "dynamic", "router", "cpu-router"};
  if (std::find(policies.begin(), policies.end(), policy) == policies.end())
    throw std::runtime_error("--policy must be one of gpu, cpu, split, dynamic, router, cpu-router");
  const double spin_ms = a.f64("gpu-spin-ms", 20);
  const int log2n = static_cast<int>(a.i64("morsel-log2", 20));
  if (log2n < 10 || log2n > 26) throw std::runtime_error("--morsel-log2 must be in [10, 26]");
  const uint32_t n = 1u << log2n;
  const std::string data_dir = a.str("data", "data/tpch-sf10");
  const int inflight = static_cast<int>(a.i64("inflight", 16));
  if (inflight < 1 || inflight > 64) throw std::runtime_error("--inflight must be in [1, 64]");
  const int queues = static_cast<int>(a.i64("queues", 1));
  if (queues < 1 || queues > 8) throw std::runtime_error("--queues must be in [1, 8]");
  const int reps = static_cast<int>(a.i64("reps", 3));
  const std::string out_path = a.str("out", "results/run_query.csv");
  const std::string expect_path = a.str("expect", data_dir + "/" + std::string(Q::kOp) + "_expected.txt");
  const double pred_ms = a.f64("pred-ms", 0);
  const std::string build_device = a.str("build-device", "gpu");
  if (build_device != "gpu" && build_device != "cpu") throw std::runtime_error("--build-device must be gpu or cpu");
  mcm::acquire_bench_lock();

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  std::vector<std::string> kernels{std::string(MCM_KERNEL_DIR) + "/" + Q::kKernelFile, std::string(MCM_KERNEL_DIR) + "/spin.metal"};
  if (Q::kHasBuild) kernels.push_back(std::string(MCM_KERNEL_DIR) + "/join.metal");
  mcm::MetalCtx ctx(kernels);
  mcm::MachineInfo mi = mcm::collect_machine_info(ctx.device());
  if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
  const int n_p = static_cast<int>(a.i64("p-workers", mi.p_cores));
  const int n_e = static_cast<int>(a.i64("e-workers", mi.e_cores));
  const bool gpu_on = policy != "cpu" && policy != "cpu-router";
  const bool cpu_on = policy != "gpu";
  if (cpu_on && n_p + n_e == 0) throw std::runtime_error("no CPU workers");
  const bool cpu_build = Q::kHasBuild && build_device == "cpu";
  if (cpu_build && (!cpu_on || n_p == 0)) throw std::runtime_error("--build-device cpu needs P workers (not --policy gpu)");
  std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores << " GPU cores=" << mi.gpu_cores
            << " | power=" << mi.power_source << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

  Columns cols = Q::load(data_dir);
  std::vector<Morsel> morsels;
  for (size_t b = 0; b < cols.rows; b += n) morsels.push_back({b, static_cast<uint32_t>(std::min<size_t>(n, cols.rows - b))});
  const size_t M = morsels.size();
  std::cerr << Q::kOp << ": " << cols.rows << " rows, " << M << " morsels of 2^" << log2n << " (" << (cols.rows * Q::kBytesPerRow >> 20)
            << " MiB) | policy " << policy << " | GPU "
            << (gpu_on ? "on, " + std::to_string(inflight) + " in flight" + (queues > 1 ? " on " + std::to_string(queues) + " queues" : "")
                       : "off")
            << " | CPU " << (cpu_on ? std::to_string(n_p) + " P + " + std::to_string(n_e) + " E workers" : "off")
            << (Q::kHasBuild ? " | build on " + build_device : std::string()) << "\n";

  Dispatcher disp;
  disp.policy = policy;
  disp.M = M;
  disp.t_gpu = a.f64("t-gpu", 0);
  disp.t_p = a.f64("t-p", 0);
  disp.t_e = a.f64("t-e", 0);
  disp.n_p = cpu_on ? n_p : 0;
  disp.n_e = cpu_on ? n_e : 0;
  disp.inflight = inflight;
  disp.gpu_on = gpu_on;
  double gpu_share = a.f64("gpu-share", -1);
  if (policy == "split" || policy == "router" || policy == "cpu-router") {
    if ((gpu_on && disp.t_gpu <= 0) || disp.t_p <= 0 || (disp.n_e > 0 && disp.t_e <= 0))
      throw std::runtime_error("--policy " + policy + " needs --t-gpu, --t-p and --t-e (predicted us per morsel; analysis/plan_e2e.py)");
    if (policy == "split") {
      if (gpu_share < 0) gpu_share = disp.rate(Dev::gpu) / disp.total_rate();
      disp.gpu_end = static_cast<size_t>(std::lround(gpu_share * static_cast<double>(M)));
      disp.gpu_end = std::min(disp.gpu_end, M);
      std::cerr << "split: GPU share " << gpu_share << " -> morsels [0, " << disp.gpu_end << "), CPU [" << disp.gpu_end << ", " << M << ")\n";
    }
  }

  // Reference (scalar) and DuckDB's answer, once.
  const Result ref_all = Q::reference(cols, 0, cols.rows);
  Result exp_all;
  const bool have_expect = Q::load_expected(expect_path, exp_all);
  if (have_expect && !(ref_all == exp_all)) std::cerr << "WARNING: the scalar reference disagrees with DuckDB\n";

  std::unique_ptr<Gpu> gpu;
  if (gpu_on) gpu = std::make_unique<Gpu>(ctx, cols, Gpu::default_max_groups(mi.gpu_cores));
  std::vector<MTL::CommandQueue*> gpu_queues{ctx.queue()};  // --queues: the probe's command buffers take them in turn
  for (int i = 1; gpu_on && i < queues; ++i) {
    gpu_queues.push_back(ctx.device()->newCommandQueue());
    if (!gpu_queues.back()) throw std::runtime_error("cannot create command queue " + std::to_string(i));
  }
  std::unique_ptr<mcm::join::GpuBuilder> builder;  // the GPU build phase (Q14, --build-device gpu)
  if constexpr (Q::kHasBuild) {
    if (build_device == "gpu")
      builder = std::make_unique<mcm::join::GpuBuilder>(ctx, cols.table, *cols.part[0], *cols.part[1], cols.parts);
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // this thread runs the build phase
  }
  std::unique_ptr<mcm::GpuSpin> spin;
  if (spin_ms > 0) spin = std::make_unique<mcm::GpuSpin>(ctx, spin_ms);
  if (cpu_on) {  // the core probes, once (bench.hpp): which cores the two QoS classes land on here
    std::thread([&] { mcm::place_cpu_thread("p"); }).join();
    if (n_e > 0) std::thread([&] { mcm::place_cpu_thread("e"); }).join();
  }

  std::ifstream in(out_path);
  const bool empty = !in.good() || in.peek() == std::ifstream::traits_type::eof();
  in.close();
  std::ofstream csv(out_path, std::ios::app);
  if (!csv) throw std::runtime_error("cannot open --out " + out_path);
  if (empty)
    csv << "machine,op,data,policy,n,morsels,p_workers,e_workers,inflight,spin_ms,t_gpu_us,t_p_us,t_e_us,gpu_share,pred_ms,rep,t_ms,"
           "gpu_morsels,p_morsels,e_morsels,gpu_busy_ms,p_busy_ms,e_busy_ms,gpu_finished_ms,p_finished_ms,e_finished_ms,ok,"
           "build_device,build_ms,gpu_queues\n";
  const std::string data_tag = data_dir.substr(data_dir.find_last_of('/') + 1);
  const std::string timeline_path = a.str("gpu-timeline", "");
  std::ofstream timeline;
  if (!timeline_path.empty()) {
    std::ifstream tin(timeline_path);
    const bool tempty = !tin.good() || tin.peek() == std::ifstream::traits_type::eof();
    tin.close();
    timeline.open(timeline_path, std::ios::app);
    if (!timeline) throw std::runtime_error("cannot open --gpu-timeline " + timeline_path);
    if (tempty)
      timeline << "op,policy,n,inflight,build_device,rep,seq,morsel,encode_us,commit_us,retire_us,gpu_start_us,gpu_end_us,queues,queue\n";
  }

  for (int rep = -1; rep < reps; ++rep) {  // -1: warm-up, untimed and unrecorded
    disp.next = 0;
    disp.gpu_next = 0;
    disp.gpu_in_flight = 0;
    disp.active_p = disp.n_p;
    disp.active_e = disp.n_e;
    if (policy == "split") disp.next = disp.gpu_end;
    if (spin) (*spin)();  // busy GPU clock at the start of the rep, whatever was set up in between
    std::atomic<int> ready{0}, go{0};
    // The probe morsels wait for the build phase. The GPU's driver thread waits on a flag of its own,
    // set and woken before the CPU workers' (publish_built): woken by the same notify_all as the 32
    // CPU workers, it resumed 0.08-0.15 ms after the build ended, against 0.003-0.009 ms waiting alone
    // (results/Apple-M3-Ultra_20261117-101927_e2e), and the CPU workers took the morsels first; woken
    // first, 0.007-0.062 ms (-103551_e2e).
    std::atomic<int> built{Q::kHasBuild ? 0 : 1}, gpu_built{Q::kHasBuild ? 0 : 1};
    auto publish_built = [&] {
      gpu_built.store(1);
      gpu_built.notify_one();
      built.store(1);
      built.notify_all();
    };
    std::atomic<int> cleared{0}, build_done{0};  // --build-device cpu: the P workers' barrier and finish line
    std::atomic<uint64_t> build_inserted{0};
    double build_us = 0;
    std::string build_error;
    const int parties = (gpu_on ? 1 : 0) + (cpu_on ? n_p + n_e : 0);
    DevStats st_gpu;
    std::vector<DevStats> st_cpu(static_cast<size_t>(cpu_on ? n_p + n_e : 0));
    Result got_gpu;
    std::vector<Result> got_cpu(st_cpu.size());
    Clock::time_point t0;
    double t0_s = 0;          // t0 on the host clock of the GPU timestamps
    std::vector<CbTime> tl;   // --gpu-timeline
    if (timeline.is_open()) tl.reserve(M);
    auto wait_go = [&] {
      ready.fetch_add(1);
      while (go.load() == 0) std::this_thread::yield();
    };

    std::thread gpu_thread;
    if (gpu_on) {
      gpu_thread = std::thread([&] {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        std::vector<typename Gpu::Slot> slots(static_cast<size_t>(inflight));
        for (auto& s : slots) s = gpu->new_slot();
        struct InFlight {
          MTL::CommandBuffer* cb;
          int slot;
          size_t morsel;
          double encode_s, commit_s;  // --gpu-timeline
          int queue;
        };
        size_t submitted = 0;  // command buffers committed this rep: the next one's queue
        const bool record = timeline.is_open();
        std::deque<InFlight> q;
        int free_slot = 0;
        wait_go();
        gpu_built.wait(0);
        auto t_busy0 = Clock::now();
        auto retire = [&] {  // the oldest command buffer: wait, merge, free its slot
          InFlight f = q.front();
          q.pop_front();
          f.cb->waitUntilCompleted();
          if (f.cb->status() != MTL::CommandBufferStatusCompleted) throw std::runtime_error("GPU command buffer failed");
          if (record) tl.push_back({f.morsel, f.encode_s, f.commit_s, host_s(), f.cb->GPUStartTime(), f.cb->GPUEndTime(), f.queue});
          got_gpu.merge(Gpu::read(slots[static_cast<size_t>(f.slot)]));
          f.cb->release();
          free_slot = f.slot;
          st_gpu.morsels++;
        };
        std::vector<int> free_slots;
        for (int i = 0; i < inflight; ++i) free_slots.push_back(i);
        while (true) {
          if (free_slots.empty()) {
            retire();
            free_slots.push_back(free_slot);
          }
          const size_t i = disp.take(Dev::gpu, static_cast<int>(q.size()));
          if (i == kNone) {
            if (q.empty()) break;
            retire();
            free_slots.push_back(free_slot);
            continue;
          }
          const int slot = free_slots.back();
          free_slots.pop_back();
          const double encode_s = record ? host_s() : 0;
          NS::AutoreleasePool* p = NS::AutoreleasePool::alloc()->init();
          const int qi = static_cast<int>(submitted++ % gpu_queues.size());
          MTL::CommandBuffer* cb = gpu_queues[static_cast<size_t>(qi)]->commandBuffer();
          cb->retain();
          MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
          gpu->encode(enc, morsels[i].begin, morsels[i].n, slots[static_cast<size_t>(slot)]);
          enc->endEncoding();
          cb->commit();
          p->release();
          q.push_back({cb, slot, i, encode_s, record ? host_s() : 0, qi});
        }
        st_gpu.busy_us = us_since(t_busy0, Clock::now());
        st_gpu.finished_us = us_since(t0, Clock::now());
        for (auto& s : slots) Gpu::free_slot(s);
      });
    }
    std::vector<std::thread> workers;
    for (int w = 0; cpu_on && w < n_p + n_e; ++w) {
      workers.emplace_back([&, w] {
        const Dev d = w < n_p ? Dev::p : Dev::e;
        pthread_set_qos_class_self_np(d == Dev::p ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, 0);
        DevStats& st = st_cpu[static_cast<size_t>(w)];
        Result& got = got_cpu[static_cast<size_t>(w)];
        wait_go();
        if constexpr (Q::kHasBuild) {
          if (cpu_build && d == Dev::p) {  // build phase: clear my share, wait for all, insert my share
            Q::clear_share(cols, w, n_p);
            cleared.fetch_add(1, std::memory_order_acq_rel);
            while (cleared.load(std::memory_order_acquire) < n_p) std::this_thread::yield();
            build_inserted.fetch_add(Q::build_share(cols, w, n_p), std::memory_order_relaxed);
            if (build_done.fetch_add(1, std::memory_order_acq_rel) == n_p - 1) {  // the last one: the table is complete
              build_us = us_since(t0, Clock::now());
              publish_built();
            }
            while (built.load(std::memory_order_acquire) == 0) std::this_thread::yield();  // P-cores: spin
          }
        }
        built.wait(0);
        while (true) {
          const size_t i = disp.take(d, 0);
          if (i == kNone) break;
          auto t1 = Clock::now();
          Q::cpu(cols, morsels[i].begin, morsels[i].n, got);
          st.busy_us += us_since(t1, Clock::now());
          st.morsels++;
          st.finished_us = us_since(t0, Clock::now());
        }
      });
    }
    while (ready.load() < parties) std::this_thread::yield();
    t0 = Clock::now();
    t0_s = host_s();
    go.store(1);
    if constexpr (Q::kHasBuild) {
      if (builder) {  // the GPU build phase, from this thread: one command buffer, the workers blocked
        try {
          build_inserted = builder->build_sync();
        } catch (const std::exception& e) {
          build_error = e.what();
        }
        build_us = us_since(t0, Clock::now());
        publish_built();
      }
    }
    if (gpu_thread.joinable()) gpu_thread.join();
    for (auto& t : workers) t.join();
    const double t_ms = us_since(t0, Clock::now()) / 1e3;
    if (Q::kHasBuild && build_error.empty() && build_inserted.load() != cols_rows_built(cols))
      build_error = "the build inserted " + std::to_string(build_inserted.load()) + " rows, not " + std::to_string(cols_rows_built(cols));
    if (!build_error.empty()) throw std::runtime_error("build phase: " + build_error);

    Result got = got_gpu;
    for (const Result& r : got_cpu) got.merge(r);
    const bool ok = got == ref_all && (!have_expect || got == exp_all);
    DevStats sp, se;
    for (int w = 0; w < static_cast<int>(st_cpu.size()); ++w) {
      DevStats& acc = w < n_p ? sp : se;
      acc.morsels += st_cpu[static_cast<size_t>(w)].morsels;
      acc.busy_us += st_cpu[static_cast<size_t>(w)].busy_us;
      acc.finished_us = std::max(acc.finished_us, st_cpu[static_cast<size_t>(w)].finished_us);
    }
    std::cerr << (rep < 0 ? "warm-up" : "rep " + std::to_string(rep)) << ": " << t_ms << " ms"
              << (Q::kHasBuild ? " (build " + std::to_string(build_us / 1e3) + " ms on " + build_device + ")" : std::string())
              << " | GPU " << st_gpu.morsels << " morsels"
              << " | P " << sp.morsels << " | E " << se.morsels << " | finished at GPU " << st_gpu.finished_us / 1e3 << " P "
              << sp.finished_us / 1e3 << " E " << se.finished_us / 1e3 << " ms | " << (ok ? "exact" : "MISMATCH") << "\n";
    if (rep < 0) continue;
    csv << mi.tag() << "," << Q::kOp << "," << data_tag << "," << policy << "," << n << "," << M << "," << (cpu_on ? n_p : 0) << ","
        << (cpu_on ? n_e : 0) << "," << (gpu_on ? inflight : 0) << "," << spin_ms << "," << disp.t_gpu << "," << disp.t_p << "," << disp.t_e << ","
        << (policy == "split" ? gpu_share : -1.0) << "," << pred_ms << "," << rep << "," << t_ms << "," << st_gpu.morsels << ","
        << sp.morsels << "," << se.morsels << "," << st_gpu.busy_us / 1e3 << "," << sp.busy_us / 1e3 << "," << se.busy_us / 1e3 << ","
        << st_gpu.finished_us / 1e3 << "," << sp.finished_us / 1e3 << "," << se.finished_us / 1e3 << "," << (ok ? 1 : 0) << ","
        << (Q::kHasBuild ? build_device : std::string("-")) << "," << build_us / 1e3 << "," << (gpu_on ? queues : 0) << "\n";
    for (size_t s = 0; s < tl.size(); ++s) {
      const CbTime& c = tl[s];
      auto us = [&](double x) { return (x - t0_s) * 1e6; };
      timeline << Q::kOp << "," << policy << "," << n << "," << inflight << "," << (Q::kHasBuild ? build_device : std::string("-")) << ","
               << rep << "," << s << "," << c.morsel << "," << us(c.encode_s) << "," << us(c.commit_s) << "," << us(c.retire_s) << ","
               << us(c.gpu_start_s) << "," << us(c.gpu_end_s) << "," << queues << "," << c.queue << "\n";
    }
    if (!ok) {
      std::cerr << "result:\n" << Q::describe(got) << "reference:\n" << Q::describe(ref_all);
      return 2;
    }
  }
  csv.close();
  std::cerr << "-> " << out_path << "\n";
  spin.reset();
  builder.reset();
  gpu.reset();
  for (size_t i = 1; i < gpu_queues.size(); ++i) gpu_queues[i]->release();
  pool->release();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const mcm::Args a(argc, argv);
    const std::string q = a.str("query", "");
    if (a.has("help") || (q != "q1" && q != "q6" && q != "q14")) {
      std::cerr << "usage: run_query --query q1|q6|q14 --policy gpu|cpu|split|dynamic|router|cpu-router [--morsel-log2 20]\n"
                   "                 [--data data/tpch-sf10] [--p-workers <P cores>] [--e-workers <E cores>] [--inflight 16]\n"
                   "                 [--t-gpu us] [--t-p us] [--t-e us] [--gpu-share f] [--pred-ms t] [--gpu-spin-ms 20] [--reps 3]\n"
                   "                 [--out results/run_query.csv] [--machine-json path] [--expect <data>/<query>_expected.txt]\n"
                   "                 [--build-device gpu|cpu]   (q14: the part hash table's build phase on the GPU or on the P workers)\n"
                   "                 [--gpu-timeline path] [--queues 1]   (diagnostics: per-command-buffer times; the GPU's queues)\n";
      return a.has("help") ? 0 : 1;
    }
    return q == "q1" ? run<Q1>(a) : q == "q6" ? run<Q6>(a) : run<Q14>(a);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
