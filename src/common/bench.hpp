#pragma once
// Helpers shared by the calibration benchmarks: cache flush, CPU P/E placement, core probe,
// one-benchmark-at-a-time lock, pipelined GPU dispatch.
#include <fcntl.h>
#include <pthread/qos.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/metal_ctx.hpp"
#include "common/timer.hpp"

namespace mcm {

// Cache flush: stream through a scratch buffer much larger than the CPU L2s plus the
// system-level cache (M1: 12 MB P-L2 + 8 MB SLC; M-series Max/Ultra: up to 96 MB SLC),
// so the next timed rep reads its input from DRAM. The SLC is shared by CPU and GPU, so
// this also evicts GPU input from the SLC. GPU-private L1/L2 are not touched (small).
struct CacheFlusher {
  explicit CacheFlusher(size_t bytes) : buf(bytes) { std::memset(buf.data(), 1, buf.size()); }
  void operator()() {
    const auto* p = static_cast<const uint64_t*>(buf.data());
    const size_t words = buf.size() / sizeof(uint64_t);
    uint64_t s = 0;
    for (size_t i = 0; i < words; i += 8) s += p[i];  // one load per 64 B line
    sink = s;
  }
  PageAlloc buf;
  volatile uint64_t sink = 0;
};

// Default --flush-mb: MCM_FLUSH_MB if set, else 64 MB (M1: 12 MB P-L2 + 8 MB SLC). Chips with a large
// SLC need more, e.g. 512 MB on an M3 Ultra (16 MB L2 per P-cluster, ~96 MB SLC over two dies).
inline long long default_flush_mb() {
  const char* e = std::getenv("MCM_FLUSH_MB");
  return e && std::atoll(e) > 0 ? std::atoll(e) : 64;
}

// Process-level part of CPU placement. Call once, before starting the worker thread.
// Thread-level QoS (BACKGROUND) and PRIO_DARWIN_THREAD do NOT keep a thread off the
// P-cores for a process started from a shell (measured on M1, macOS 15.7). Only the
// process-wide darwin background policy (what `taskpolicy -b` sets) moves it to E-cores.
inline void place_cpu_process(const std::string& qos) {
  if (qos != "p" && qos != "e") throw std::runtime_error("--qos must be p or e");
  if (qos == "e" && setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG) != 0)
    throw std::runtime_error("setpriority(PRIO_DARWIN_PROCESS, PRIO_DARWIN_BG) failed");
}

// Thread-level part: call first thing inside the worker thread. Also prints the core probe.
inline void place_cpu_thread(const std::string& qos) {
  pthread_set_qos_class_self_np(qos == "p" ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, 0);
  // Core-type probe: dependent FP multiply chain, ns per iteration tracks the core clock.
  volatile double x = 1;
  auto t0 = Clock::now();
  for (int i = 0; i < 20000000; ++i) x *= 1.0000000001;
  std::cerr << "core probe (qos=" << qos << "): " << us_since(t0, Clock::now()) * 1e3 / 2e7
            << " ns/iter (M1: ~3.5 on P-cores, ~6-9 on E-cores)\n";
}

// ---------------------------------------------------------------- one benchmark at a time
// Benchmark processes running at the same time share DRAM bandwidth and the GPU and silently
// contaminate each other's numbers (two agg sweeps once overlapped for six minutes). Each
// benchmark holds an exclusive lock for its whole lifetime and refuses to start if another
// process holds it. It also refuses while a sweep script (scripts/lib_sweep.sh) is running,
// unless that sweep started it: the sweep exports its pid as MCM_SWEEP_PID.
inline constexpr const char* kBenchLockFile = "/tmp/metal-cost-model-bench.lock";
inline constexpr const char* kSweepLockDir = "/tmp/metal-cost-model-sweep.lock";

// The protocol is AC power with Low Power Mode off (README). A benchmark refuses to start otherwise,
// and checks again when it exits: if the power source changed during the run, it exits with status 3
// so a sweep stops instead of mixing battery data into an AC run. MCM_ALLOW_BATTERY=1 skips both.
inline std::string power_state() {
  std::string out;
  if (FILE* f = popen("pmset -g batt 2>/dev/null | head -1; pmset -g 2>/dev/null | grep -i lowpowermode", "r")) {
    char buf[256];
    while (fgets(buf, sizeof(buf), f)) out += buf;
    pclose(f);
  }
  const bool ac = out.find("AC Power") != std::string::npos;
  const size_t lp = out.find("lowpowermode");
  const bool low = lp != std::string::npos && out.find('1', lp) != std::string::npos;
  return std::string(ac ? "AC" : "not AC") + (low ? ", Low Power Mode on" : "");
}

inline void require_ac_power() {
  const char* allow = std::getenv("MCM_ALLOW_BATTERY");
  if (allow && std::string(allow) == "1") return;
  const std::string s = power_state();
  if (s != "AC") throw std::runtime_error("power: " + s + "; plug in AC and turn Low Power Mode off (or set MCM_ALLOW_BATTERY=1)");
  std::atexit([] {
    const std::string now = power_state();
    if (now != "AC") {
      std::cerr << "error: the power state changed during the run (" << now << "); discard this run\n";
      std::_Exit(3);
    }
  });
}

inline void acquire_bench_lock() {
  require_ac_power();
  long sweep = 0;
  std::ifstream pf(std::string(kSweepLockDir) + "/pid");
  if (pf >> sweep && sweep > 0 && kill(static_cast<pid_t>(sweep), 0) == 0) {
    const char* mine = std::getenv("MCM_SWEEP_PID");
    if (!mine || std::atol(mine) != sweep)
      throw std::runtime_error("a sweep (pid " + std::to_string(sweep) +
                               ") is running; refusing to start a second benchmark");
  }
  static int fd = -1;  // stays open, and locked, until the process exits (a forked child inherits it)
  fd = open(kBenchLockFile, O_RDWR | O_CREAT, 0666);
  if (fd < 0) throw std::runtime_error(std::string("cannot open lock file ") + kBenchLockFile);
  if (flock(fd, LOCK_EX | LOCK_NB) != 0)
    throw std::runtime_error("another benchmark process is running; refusing to start");
}

// ---------------------------------------------------------------- pipelined GPU dispatch
// A morsel router keeps the GPU queue full: while the GPU runs morsel k, morsels k+1.. are
// already committed. Timing one synchronous round trip per morsel (encode, commit,
// waitUntilCompleted) instead charges every morsel the full launch + completion latency, plus a
// wake-up whenever the GPU idled in between (it does during the cache flush). run_pipelined
// encodes K command buffers first, commits them back to back, waits for all of them, and
// reports the GPU time per morsel over the batch: (last GPU end - first GPU start) / K.
//
// Consecutive command buffers overlap on the GPU (measured on M1: a buffer's own start-to-end
// time often exceeds the spacing of completions), so the spacing of completion times is NOT a
// per-morsel cost: with few morsels per batch it implied more than the DRAM peak bandwidth.
// The span / K includes one pipeline fill and drain, so it slightly overstates the cost of a
// queue that never drains; it is kept as the measurement, the spacing only as a diagnostic.
struct BatchTiming {
  double per_morsel_us = 0;  // (max GPUEndTime - min GPUStartTime) / K
  double spacing_us = 0;     // (max GPUEndTime - min GPUEndTime) / (K - 1); diagnostic only
  double encode_us = 0;      // mean host time to build one command buffer (CPU cost on the dispatcher)
  double exec_us = 0;        // mean GPUEndTime - GPUStartTime per command buffer (overlaps others)
  bool ok = true;
};

template <class EncodeFn>  // void(MTL::ComputeCommandEncoder*, int k): bind buffers and dispatch morsel k
BatchTiming run_pipelined(MTL::CommandQueue* q, MTL::ComputePipelineState* pso, int K, EncodeFn&& encode) {
  if (K < 2) throw std::runtime_error("a pipelined batch needs at least 2 morsels");
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  std::vector<MTL::CommandBuffer*> cbs(static_cast<size_t>(K));
  BatchTiming b;
  auto t0 = Clock::now();
  for (int k = 0; k < K; ++k) {
    cbs[k] = q->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cbs[k]->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    encode(enc, k);
    enc->endEncoding();
  }
  b.encode_us = us_since(t0, Clock::now()) / K;
  for (MTL::CommandBuffer* cb : cbs) cb->commit();
  for (MTL::CommandBuffer* cb : cbs) cb->waitUntilCompleted();
  double first_start = 1e300, first_end = 1e300, last_end = 0, exec = 0;
  for (MTL::CommandBuffer* cb : cbs) {
    if (cb->status() != MTL::CommandBufferStatusCompleted) b.ok = false;
    first_start = std::min(first_start, cb->GPUStartTime());
    first_end = std::min(first_end, cb->GPUEndTime());
    last_end = std::max(last_end, cb->GPUEndTime());
    exec += cb->GPUEndTime() - cb->GPUStartTime();
  }
  b.per_morsel_us = (last_end - first_start) / K * 1e6;
  b.spacing_us = (last_end - first_end) / (K - 1) * 1e6;
  b.exec_us = exec / K * 1e6;
  pool->release();
  return b;
}

// ---------------------------------------------------------------- private storage + blit
// The alternative to letting the kernel read host memory wrapped no-copy: copy the morsel into a
// MTLStorageModePrivate buffer with a blit, then run the kernel on the private copy. On UMA both
// live in the same DRAM, so the question is whether the GPU reads private memory faster (different
// page layout / compression) by enough to pay for the copy. The blit is its own term: it runs in its
// own command buffer, committed right before the kernel's, so each has its own GPU start/end time.
// The kernel's buffer is hazard-tracked (the default), so it waits for the blit.
struct BlitCopy {
  MTL::Buffer* src;
  MTL::Buffer* dst;
  size_t bytes;
};

struct BlitComputeTiming {
  double encode_us = 0;     // host: build + commit both command buffers
  double total_us = 0;      // host: start of encoding .. kernel completion (blit + kernel + overheads)
  double blit_exec_us = 0;  // blit command buffer GPUEndTime - GPUStartTime (0 without copies)
  double exec_us = 0;       // kernel command buffer GPUEndTime - GPUStartTime
  bool ok = true;
};

// One synchronous round trip: optional blit command buffer (copies; empty = data already resident in
// the private buffer), then the kernel's command buffer; waits for the kernel.
template <class EncodeFn>  // void(MTL::ComputeCommandEncoder*): bind buffers and dispatch
BlitComputeTiming run_blit_then_compute(MTL::CommandQueue* q, MTL::ComputePipelineState* pso,
                                        const std::vector<BlitCopy>& copies, EncodeFn&& encode) {
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
  BlitComputeTiming t;
  auto t0 = Clock::now();
  MTL::CommandBuffer* cbb = nullptr;
  if (!copies.empty()) {
    cbb = q->commandBuffer();
    MTL::BlitCommandEncoder* be = cbb->blitCommandEncoder();
    for (const BlitCopy& c : copies) be->copyFromBuffer(c.src, 0, c.dst, 0, c.bytes);
    be->endEncoding();
    cbb->commit();
  }
  MTL::CommandBuffer* cb = q->commandBuffer();
  MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
  enc->setComputePipelineState(pso);
  encode(enc);
  enc->endEncoding();
  cb->commit();
  auto t1 = Clock::now();
  cb->waitUntilCompleted();
  auto t2 = Clock::now();
  if (cbb) cbb->waitUntilCompleted();  // already done: the kernel waited for it
  t.encode_us = us_since(t0, t1);
  t.total_us = us_since(t0, t2);
  t.exec_us = (cb->GPUEndTime() - cb->GPUStartTime()) * 1e6;
  if (cb->status() != MTL::CommandBufferStatusCompleted) t.ok = false;
  if (cbb) {
    t.blit_exec_us = (cbb->GPUEndTime() - cbb->GPUStartTime()) * 1e6;
    if (cbb->status() != MTL::CommandBufferStatusCompleted) t.ok = false;
  }
  pool->release();
  return t;
}

inline MTL::Buffer* new_private_buffer(MTL::Device* dev, size_t bytes) {
  MTL::Buffer* b = dev->newBuffer(bytes, MTL::ResourceStorageModePrivate);
  if (!b) throw std::runtime_error("cannot allocate a private buffer of " + std::to_string(bytes) + " bytes");
  return b;
}

// ---------------------------------------------------------------- busy GPU (clock state)
// ALU-only GPU busy loop of about `ms` milliseconds (kernels/spin.metal gpu_spin; the benchmark's MetalCtx
// must compile spin.metal too), sync, untimed.
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

// ---------------------------------------------------------------- cache state (the router's cache policy)
// Where a morsel's input is at dispatch depends on what last touched it; a router decides that when it
// places the producer and the consumer. --cache of calib_filter names the state, set after the flush:
//   cold      : nothing (the flush: DRAM; every model so far);  hot: no flush at all (the old behaviour)
//   cpu-write : one CPU thread rewrites the morsel's rows in place, x += 0 with a runtime zero, so the
//               data is unchanged and every line is written: a CPU operator has just produced it. For
//               a CPU consumer it is the consumer thread itself (L2); for the GPU, the dispatching thread
//   cpu-read  : the same thread reads (sums) the rows: another scan has just used the column
//   gpu-write, gpu-read : a GPU kernel does the same (kernels/touch.metal, compiled with the benchmark's
//               kernels), one sync command buffer over the morsel's buffers, untimed
inline const std::vector<std::string> kCachePolicies = {"cold", "hot", "cpu-write", "cpu-read", "gpu-write", "gpu-read"};
inline bool cache_policy_touches(const std::string& p) { return p != "cold" && p != "hot"; }
inline bool cache_policy_on_gpu(const std::string& p) { return p.rfind("gpu-", 0) == 0; }

struct TouchParams {
  uint32_t n, zero;  // must match TouchParams in kernels/touch.metal
};

class Toucher {
 public:
  struct Range {  // rows [0, n) of one morsel: host pointer (CPU policies) or buffer + byte offset (GPU)
    uint32_t* host = nullptr;
    MTL::Buffer* buf = nullptr;
    size_t offset = 0;
    uint32_t n = 0;
  };

  // ctx may be null for a policy without a GPU step.
  Toucher(MetalCtx* ctx, const std::string& policy) : policy_(policy) {
    if (std::find(kCachePolicies.begin(), kCachePolicies.end(), policy) == kCachePolicies.end())
      throw std::runtime_error("--cache must be one of cold, hot, cpu-write, cpu-read, gpu-write, gpu-read");
    if (cache_policy_on_gpu(policy)) {
      if (!ctx) throw std::runtime_error("--cache " + policy + " needs the GPU");
      ctx_ = ctx;
      pso_ = ctx->pipeline(policy == "gpu-write" ? "touch_write" : "touch_read");
      out_ = ctx->device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
      tg_ = std::min<NS::UInteger>(256, pso_->maxTotalThreadsPerThreadgroup());
    }
  }
  ~Toucher() {
    if (out_) out_->release();
    if (pso_) pso_->release();
  }
  Toucher(const Toucher&) = delete;
  Toucher& operator=(const Toucher&) = delete;

  bool active() const { return cache_policy_touches(policy_); }
  void operator()(const std::vector<Range>& ranges) {
    if (!active()) return;
    if (cache_policy_on_gpu(policy_)) {
      gpu(ranges);
    } else {
      for (const Range& r : ranges) cpu(r);
    }
  }

 private:
  void cpu(const Range& r) {
    const uint32_t z = zero_;  // read from a volatile: the compiler cannot drop the stores
    if (policy_ == "cpu-write") {
      for (uint32_t i = 0; i < r.n; ++i) r.host[i] += z;
    } else {
      uint32_t s = 0;
      for (uint32_t i = 0; i < r.n; ++i) s += r.host[i];
      sink_ = s;
    }
  }
  void gpu(const std::vector<Range>& ranges) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MTL::CommandBuffer* cb = ctx_->queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso_);
    for (const Range& r : ranges) {
      const TouchParams p{r.n, 0};
      const NS::UInteger groups = std::clamp<NS::UInteger>((r.n + tg_ * 16 - 1) / (tg_ * 16), 1, 4096);
      enc->setBuffer(r.buf, r.offset, 0);
      enc->setBuffer(out_, 0, 1);
      enc->setBytes(&p, sizeof(p), 2);
      enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(tg_, 1, 1));
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) throw std::runtime_error("cache-state touch kernel failed");
    pool->release();
  }

  std::string policy_;
  MetalCtx* ctx_ = nullptr;
  MTL::ComputePipelineState* pso_ = nullptr;
  MTL::Buffer* out_ = nullptr;
  NS::UInteger tg_ = 256;
  volatile uint32_t zero_ = 0, sink_ = 0;
};

// Morsels per pipelined batch: up to K, each on its own buffer (so no morsel reads cache lines
// the previous one left behind), capped so the whole batch fits in cap_bytes; at least 2.
inline int batch_size(int K, size_t morsel_bytes, size_t cap_bytes) {
  const size_t fit = cap_bytes / std::max<size_t>(morsel_bytes, 1);
  return static_cast<int>(std::clamp<size_t>(fit, 2, static_cast<size_t>(std::max(K, 2))));
}

}  // namespace mcm
