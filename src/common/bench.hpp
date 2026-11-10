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

inline void acquire_bench_lock() {
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

// Morsels per pipelined batch: up to K, each on its own buffer (so no morsel reads cache lines
// the previous one left behind), capped so the whole batch fits in cap_bytes; at least 2.
inline int batch_size(int K, size_t morsel_bytes, size_t cap_bytes) {
  const size_t fit = cap_bytes / std::max<size_t>(morsel_bytes, 1);
  return static_cast<int>(std::clamp<size_t>(fit, 2, static_cast<size_t>(std::max(K, 2))));
}

}  // namespace mcm
