#pragma once
// Helpers shared by the calibration benchmarks: cache flush, CPU P/E placement, core probe.
#include <pthread/qos.h>
#include <sys/resource.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

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

}  // namespace mcm
