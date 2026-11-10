// calib_contention — shared-bandwidth contention between CPU and GPU (the κ term).
//
// Both sides run the filter_count streaming operator (v < thr) over their OWN 256 MB buffer
// (far larger than any cache), repeatedly, for a fixed duration per rep:
//   CPU side : T worker threads (--cpu-threads, 0 = no CPU load), each scanning its own slice
//   GPU side : back-to-back command buffers over the whole GPU buffer (--gpu on|off)
// and report each side's achieved bandwidth. Running a config with and without the other side
// gives the contention slowdown; analysis/contention.py turns that into κ.
//
// The CPU side runs in a forked child process, created before any Metal object exists.
// Reason: E-core placement needs the process-wide darwin background policy (see
// common/bench.hpp), and that policy would also demote the GPU submissions of the
// process that sets it. The parent only drives the GPU.
//
// Per rep: parent writes 'g' to the child, both sides start, both stop after --duration s.
// The child reports {bytes, seconds} back over a pipe.
//
// Output: one CSV row per (rep, side).

#include <sys/wait.h>
#include <unistd.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <atomic>
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

using namespace mcm;

namespace {

constexpr uint32_t THR = 0x80000000u;  // ~50% selectivity on uniform random data

struct FilterParams {
  uint32_t n, threshold;
};

std::atomic<uint32_t> g_sink{0};

struct SideResult {
  double bytes = 0, seconds = 0;
};

void fill_random(PageAlloc& a) {
  auto* v = static_cast<uint32_t*>(a.data());
  const size_t n = a.size() / sizeof(uint32_t);
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < n; ++i) {  // xorshift64, cheap and good enough for a predicate
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    v[i] = static_cast<uint32_t>(x >> 32);
  }
}

bool read_full(int fd, void* p, size_t n) {
  auto* c = static_cast<char*>(p);
  while (n) {
    ssize_t r = read(fd, c, n);
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

// ---------------------------------------------------------------- CPU side (child process)
[[noreturn]] void cpu_child(int cmd_fd, int res_fd, const std::string& qos, int threads, size_t bytes,
                            double duration_s) {
  try {
    place_cpu_process(qos);
    PageAlloc buf(bytes);
    fill_random(buf);
    const auto* v = static_cast<const uint32_t*>(buf.data());
    const size_t n = buf.size() / sizeof(uint32_t);
    const size_t slice = n / threads;
    constexpr size_t CHUNK = 64 * 1024;  // elements (256 KB) between clock checks

    // Probe once per process, on a thread with the right QoS (prints to stderr).
    std::thread([&] { place_cpu_thread(qos); }).join();

    // Ready: the parent waits for this before the first rep, so rep 0 does not start while this
    // process is still filling its buffer or running the probe.
    const char ready = 'r';
    if (write(res_fd, &ready, 1) != 1) _exit(1);

    char cmd;
    while (read_full(cmd_fd, &cmd, 1) && cmd == 'g') {
      std::vector<double> tb(threads, 0), ts(threads, 0);
      std::vector<std::thread> ws;
      for (int t = 0; t < threads; ++t) {
        ws.emplace_back([&, t] {
          pthread_set_qos_class_self_np(qos == "p" ? QOS_CLASS_USER_INTERACTIVE : QOS_CLASS_BACKGROUND, 0);
          const uint32_t* base = v + t * slice;
          uint64_t done = 0;
          uint32_t c = 0;
          size_t pos = 0;
          auto t0 = Clock::now();
          double el = 0;
          while (el < duration_s * 1e6) {
            const size_t len = std::min(CHUNK, slice - pos);
            for (size_t i = 0; i < len; ++i) c += base[pos + i] < THR;
            done += len;
            pos += len;
            if (pos == slice) pos = 0;
            el = us_since(t0, Clock::now());
          }
          g_sink.fetch_add(c, std::memory_order_relaxed);  // keep the loop alive
          tb[t] = static_cast<double>(done) * sizeof(uint32_t);
          ts[t] = el * 1e-6;
        });
      }
      for (auto& w : ws) w.join();
      SideResult r;
      for (int t = 0; t < threads; ++t) r.bytes += tb[t];
      r.seconds = *std::max_element(ts.begin(), ts.end());
      if (write(res_fd, &r, sizeof(r)) != sizeof(r)) _exit(1);
    }
    _exit(0);
  } catch (const std::exception& e) {
    std::cerr << "cpu child error: " << e.what() << "\n";
    _exit(1);
  }
}

// ---------------------------------------------------------------- GPU side (parent)
struct GpuStreamer {
  MetalCtx& ctx;
  MTL::ComputePipelineState* pso;
  MTL::Buffer* in;
  MTL::Buffer* out;
  uint32_t n;

  GpuStreamer(MetalCtx& c, const PageAlloc& mem) : ctx(c) {
    pso = ctx.pipeline("filter_count");
    in = wrap_no_copy(ctx.device(), mem);
    out = ctx.device()->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    n = static_cast<uint32_t>(mem.size() / sizeof(uint32_t));
  }
  ~GpuStreamer() {
    in->release();
    out->release();
    pso->release();
  }

  MTL::CommandBuffer* submit() {
    const NS::UInteger tg = std::min<NS::UInteger>(256, pso->maxTotalThreadsPerThreadgroup());
    FilterParams p{n, THR};
    MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
    cb->retain();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    enc->setBuffer(in, 0, 0);
    enc->setBuffer(out, 0, 1);
    enc->setBytes(&p, sizeof(p), 2);
    enc->dispatchThreadgroups(MTL::Size(4096, 1, 1), MTL::Size(tg, 1, 1));
    enc->endEncoding();
    cb->commit();
    return cb;
  }

  // Stream for duration_s with two command buffers in flight.
  // Returns host-side bytes/seconds; gpu_busy_s = sum of GPU-side execution times.
  SideResult stream(double duration_s, double& gpu_busy_s) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    SideResult r;
    gpu_busy_s = 0;
    auto t0 = Clock::now();
    MTL::CommandBuffer* prev = submit();
    uint64_t done = 0;
    while (true) {
      MTL::CommandBuffer* cur = us_since(t0, Clock::now()) < duration_s * 1e6 ? submit() : nullptr;
      prev->waitUntilCompleted();
      gpu_busy_s += prev->GPUEndTime() - prev->GPUStartTime();
      ++done;
      prev->release();
      if (!cur) break;
      prev = cur;
    }
    r.seconds = us_since(t0, Clock::now()) * 1e-6;
    r.bytes = static_cast<double>(done) * n * sizeof(uint32_t);
    pool->release();
    return r;
  }
};

void usage() {
  std::cerr << "usage: calib_contention [--cpu-threads 0..N] [--qos p|e] [--gpu on|off]\n"
               "                        [--duration 1.5] [--reps 5] [--mb 256]\n"
               "                        [--out results/calib_contention.csv] [--machine-json path]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help")) {
      usage();
      return 0;
    }
    const int threads = static_cast<int>(a.i64("cpu-threads", 0));
    const std::string qos = a.str("qos", "p");
    const bool gpu = a.str("gpu", "on") == "on";
    const double dur = a.f64("duration", 1.5);
    const int reps = static_cast<int>(a.i64("reps", 5));
    const size_t bytes = static_cast<size_t>(a.i64("mb", 256)) << 20;
    const std::string out_path = a.str("out", "results/calib_contention.csv");
    if (threads < 0 || (threads == 0 && !gpu)) throw std::runtime_error("nothing to run");
    if (qos != "p" && qos != "e") throw std::runtime_error("--qos must be p or e");
    acquire_bench_lock();  // before fork: the CPU child inherits the lock

    // Fork the CPU side BEFORE any Metal/Foundation object exists in this process.
    int cmd[2] = {-1, -1}, res[2] = {-1, -1};
    pid_t child = -1;
    if (threads > 0) {
      if (pipe(cmd) || pipe(res)) throw std::runtime_error("pipe failed");
      child = fork();
      if (child < 0) throw std::runtime_error("fork failed");
      if (child == 0) {
        close(cmd[1]);
        close(res[0]);
        cpu_child(cmd[0], res[1], qos, threads, bytes, dur);
      }
      close(cmd[0]);
      close(res[1]);
    }

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MetalCtx ctx(std::string(MCM_KERNEL_DIR) + "/filter.metal");
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n"
              << "config: cpu=" << threads << "x" << qos << " gpu=" << (gpu ? "on" : "off")
              << " | " << (bytes >> 20) << " MB per side | " << dur << " s x " << reps << " reps\n";

    PageAlloc gmem(gpu ? bytes : 4096);
    std::unique_ptr<GpuStreamer> g;
    if (gpu) {
      fill_random(gmem);
      g = std::make_unique<GpuStreamer>(ctx, gmem);
      double busy;
      g->stream(0.2, busy);  // make resident + wake the GPU, untimed
    }

    bool exists = std::ifstream(out_path).good();
    std::ofstream f(out_path, std::ios::app);
    if (!exists) f << "machine,op,cpu_qos,cpu_threads,gpu,rep,side,bytes,seconds,gbps,gpu_busy_s\n";
    const std::string cfg = mi.tag() + ",filter_stream," + qos + "," + std::to_string(threads) + "," +
                            (gpu ? "on" : "off");

    if (threads > 0) {  // wait until the CPU side has its buffer filled and its probe done
      char ready = 0;
      if (!read_full(res[0], &ready, 1) || ready != 'r') throw std::runtime_error("cpu child failed to start");
    }
    for (int r = 0; r < reps; ++r) {
      if (threads > 0 && write(cmd[1], "g", 1) != 1) throw std::runtime_error("child gone");
      if (gpu) {
        double busy = 0;
        SideResult s = g->stream(dur, busy);
        f << cfg << "," << r << ",gpu," << s.bytes << "," << s.seconds << "," << s.bytes / s.seconds / 1e9
          << "," << busy << "\n";
        std::cerr << "rep " << r << " gpu " << s.bytes / s.seconds / 1e9 << " GB/s";
      }
      if (threads > 0) {
        SideResult s;
        if (!read_full(res[0], &s, sizeof(s))) throw std::runtime_error("child died");
        f << cfg << "," << r << ",cpu," << s.bytes << "," << s.seconds << "," << s.bytes / s.seconds / 1e9
          << ",0\n";
        std::cerr << (gpu ? " | " : "rep " + std::to_string(r) + " ") << "cpu "
                  << s.bytes / s.seconds / 1e9 << " GB/s";
      }
      std::cerr << "\n";
    }

    if (threads > 0) {
      write(cmd[1], "q", 1);
      close(cmd[1]);
      int st = 0;
      waitpid(child, &st, 0);
      if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) throw std::runtime_error("cpu child failed");
    }
    g.reset();
    pool->release();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
