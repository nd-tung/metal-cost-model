// calib_agg — aggregation calibration microbenchmark.
//
// Operator: SUM(val)                      (no --keys: G = 1)
//           SUM(val) GROUP BY key         (--keys: dense keys, G = max - min + 1 over the column)
// over n rows of TPC-H columns exported by scripts/gen_tpch.py. A morsel of size n is the
// first n rows. Aggregation is into a dense array indexed by key - min (no hash table).
// Sums are uint32 and wrap mod 2^32 on both devices (M1 GPUs have no 64-bit atomic add),
// so the GPU result is checked exactly against the CPU reference, every bin.
//
// Same devices, states and cache convention as calib_filter:
//   device=gpu, state=warm : buffers already used by the GPU
//   device=gpu, state=cold : fresh host allocations wrapped no-copy right before the run
//   device=cpu, qos=p|e    : one worker thread (see common/bench.hpp for E placement)
//   --cache cold (default) : 64 MB flush before every timed rep
//   --storage private      : blit keys + values into private buffers each rep, then the kernel;
//                            state warm|cold describes the blit source, resident = no blit (sync only)
//
// GPU strategy (--gpu-path auto picks by G):
//   sum    : G = 1, SIMD reduction + one atomic per threadgroup
//   tg     : G small enough for threadgroup memory, bins per threadgroup, merged at the end
//   global : one device atomic per row
//
// GPU dispatch (--dispatch, CSV columns dispatch + batch), as in calib_filter:
//   sync (default) : one command buffer per timed rep, encode .. waitUntilCompleted.
//   pipelined      : one timed rep = K morsels on K distinct key/value buffers, encoded first and
//                    committed back to back, all adding into one table; t_total is the GPU time per
//                    morsel over the batch, (last GPU end - first GPU start) / K; gpu_spacing_us is
//                    a diagnostic (see common/bench.hpp). The check compares the table with K x the
//                    reference.
//
// Not timed on either device: zeroing the aggregation table before each rep.
// Timed: the aggregation pass (CPU) / encode .. waitUntilCompleted (GPU), as in calib_filter.

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

using namespace mcm;

namespace {

constexpr uint32_t TG_BINS = 4096;  // must match kernels/agg.metal

struct AggParams {
  uint32_t n, base, groups, copies;
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

std::vector<uint32_t> load_u32(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open column " + path);
  std::vector<uint32_t> v(static_cast<size_t>(f.tellg()) / sizeof(uint32_t));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(uint32_t)));
  return v;
}

std::string stem(const std::string& path) {
  std::string s = path.substr(path.find_last_of('/') + 1);
  return s.substr(0, s.find('.'));
}

// Reference: the plain loop, written independently of the timed CPU kernel below.
void reference_agg(const uint32_t* keys, const uint32_t* vals, uint64_t n, uint32_t base, uint32_t* acc) {
  for (uint64_t i = 0; i < n; ++i) acc[keys ? keys[i] - base : 0] += vals[i];
}

// Timed CPU kernel into acc (size G, zeroed by the caller).
// Small G gets CPU_COPIES interleaved private tables, merged at the end (merge is timed):
// with one table, consecutive rows that hit the same bin serialize on store-to-load
// forwarding (~1 ns/row at G = 7 on M1). This is the CPU counterpart of the per-simdgroup
// bin copies in agg_group_tg, so neither device gets the naive version.
constexpr uint32_t CPU_COPIES = 8;
constexpr uint32_t CPU_COPY_MAX_G = 4096;  // 8 x 4096 x 4 B = 128 KB, M1 P-core L1D size

void cpu_agg(const uint32_t* keys, const uint32_t* vals, uint64_t n, uint32_t base, uint32_t G,
             uint32_t* acc, uint32_t* scratch /* CPU_COPIES * G, zeroed */) {
  if (!keys) {
    uint32_t s = 0;
    for (uint64_t i = 0; i < n; ++i) s += vals[i];  // vectorized by the compiler
    acc[0] = s;
    return;
  }
  if (G > CPU_COPY_MAX_G) {
    for (uint64_t i = 0; i < n; ++i) acc[keys[i] - base] += vals[i];
    return;
  }
  uint64_t i = 0;
  for (; i + CPU_COPIES <= n; i += CPU_COPIES)
    for (uint32_t c = 0; c < CPU_COPIES; ++c) scratch[c * G + keys[i + c] - base] += vals[i + c];
  for (; i < n; ++i) scratch[keys[i] - base] += vals[i];
  for (uint32_t c = 0; c < CPU_COPIES; ++c)
    for (uint32_t b = 0; b < G; ++b) acc[b] += scratch[c * G + b];
}

// ---------------------------------------------------------------- GPU path
struct GpuRunner {
  MetalCtx& ctx;
  MTL::ComputePipelineState* pso;
  MTL::Buffer* out;       // aggregation table (sync dispatch)
  MTL::Buffer* out_pipe;  // the same, hazard-untracked, for pipelined batches: the morsels only
                          // add to it atomically, so back-to-back command buffers may overlap
  std::string path;
  uint32_t groups, copies = 1;
  NS::UInteger tg;  // threads per threadgroup

  GpuRunner(MetalCtx& c, const std::string& p, uint32_t G) : ctx(c), path(p), groups(G) {
    pso = ctx.pipeline(path == "sum" ? "agg_sum" : path == "tg" ? "agg_group_tg" : "agg_group_global");
    out = ctx.device()->newBuffer(std::max<size_t>(G, 1) * sizeof(uint32_t), MTL::ResourceStorageModeShared);
    out_pipe = ctx.device()->newBuffer(std::max<size_t>(G, 1) * sizeof(uint32_t),
                                       MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    tg = std::min<NS::UInteger>(256, pso->maxTotalThreadsPerThreadgroup());
    if (path == "tg") {
      const uint32_t nsg = static_cast<uint32_t>(tg / 32);
      copies = (G * nsg <= TG_BINS) ? nsg : 1;
      if (G * copies > TG_BINS) throw std::runtime_error("G too large for --gpu-path tg");
    }
  }
  ~GpuRunner() {
    out_pipe->release();
    out->release();
    pso->release();
  }

  void zero_out() { std::memset(out->contents(), 0, groups * sizeof(uint32_t)); }
  void zero_out_pipe() { std::memset(out_pipe->contents(), 0, groups * sizeof(uint32_t)); }
  const uint32_t* result() const { return static_cast<const uint32_t*>(out->contents()); }
  const uint32_t* result_pipe() const { return static_cast<const uint32_t*>(out_pipe->contents()); }

  // Binds buffers and dispatches one morsel (the pipeline state is already set).
  void encode(MTL::ComputeCommandEncoder* enc, MTL::Buffer* keys, MTL::Buffer* vals, MTL::Buffer* o,
              uint32_t n, uint32_t base) const {
    NS::UInteger tgs = (n + tg * 16 - 1) / (tg * 16);  // ~16 rows per thread
    // tg path merges G bins per threadgroup at the end, so keep the threadgroup count low.
    tgs = std::clamp<NS::UInteger>(tgs, 1, path == "tg" ? 128 : 4096);
    AggParams p{n, base, groups, copies};
    if (keys) enc->setBuffer(keys, 0, 0);
    enc->setBuffer(vals, 0, 1);
    enc->setBuffer(o, 0, 2);
    enc->setBytes(&p, sizeof(p), 3);
    enc->dispatchThreadgroups(MTL::Size(tgs, 1, 1), MTL::Size(tg, 1, 1));
  }

  // Pipelined: one batch, morsel k on (keys[k], vals[k]), all adding into out_pipe.
  void run_batch(const std::vector<MTL::Buffer*>& keys, const std::vector<MTL::Buffer*>& vals, uint32_t n,
                 uint32_t base, Row& r) {
    BatchTiming b = run_pipelined(ctx.queue(), pso, static_cast<int>(vals.size()),
                                  [&](MTL::ComputeCommandEncoder* enc, int k) {
                                    encode(enc, keys[k], vals[k], out_pipe, n, base);
                                  });
    r.t_encode = b.encode_us;
    r.t_total = b.per_morsel_us;
    r.spacing = b.spacing_us;
    r.gpu_exec = b.exec_us;
    r.batch = static_cast<int>(vals.size());
    if (!b.ok) r.ok = false;
  }

  // Private storage, sync: blit (keys, vals) into (pk, pv) (skipped when vals is null), then the
  // kernel on the private copies.
  void run_private(MTL::Buffer* keys, MTL::Buffer* vals, MTL::Buffer* pk, MTL::Buffer* pv, uint32_t n,
                   uint32_t base, Row& r) {
    std::vector<BlitCopy> copies;
    if (vals) {
      if (keys) copies.push_back({keys, pk, n * sizeof(uint32_t)});
      copies.push_back({vals, pv, n * sizeof(uint32_t)});
    }
    BlitComputeTiming t = run_blit_then_compute(ctx.queue(), pso, copies, [&](MTL::ComputeCommandEncoder* enc) {
      encode(enc, pk, pv, out, n, base);
    });
    r.t_encode = t.encode_us;
    r.t_total = t.total_us;
    r.gpu_exec = t.exec_us;
    r.t_blit = t.blit_exec_us;
    if (!t.ok) r.ok = false;
  }

  // Sync: one morsel, timed as one round trip.
  void run(MTL::Buffer* keys, MTL::Buffer* vals, uint32_t n, uint32_t base, Row& r) {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    auto t0 = Clock::now();
    MTL::CommandBuffer* cb = ctx.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* enc = cb->computeCommandEncoder();
    enc->setComputePipelineState(pso);
    encode(enc, keys, vals, out, n, base);
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
};

// ---------------------------------------------------------------- CSV
void write_rows(const std::string& path, const std::string& machine, const std::string& op,
                const std::string& data, const std::string& cache, uint32_t groups,
                const std::string& gpath, uint64_t bytes_per_row, const std::string& dispatch,
                const std::string& storage, const std::vector<Row>& rows) {
  bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (!exists)
    f << "machine,op,data,thr,device,qos,state,n,bytes,rep,t_wrap_us,t_encode_us,t_total_us,gpu_exec_us,ok,"
         "cache,groups,path,dispatch,batch,gpu_spacing_us,storage,t_blit_us\n";
  for (const Row& r : rows) {
    f << machine << "," << op << "," << data << ",0," << r.device << "," << r.qos << "," << r.state << ","
      << r.n << "," << r.n * bytes_per_row << "," << r.rep << "," << r.t_wrap << "," << r.t_encode << ","
      << r.t_total << "," << r.gpu_exec << "," << (r.ok ? 1 : 0) << "," << cache << "," << groups << ","
      << (r.device == "gpu" ? gpath : (groups > 1 && groups <= CPU_COPY_MAX_G) ? "copies8" : "scalar") << ","
      << (r.device == "gpu" ? dispatch : "-") << "," << r.batch << "," << r.spacing << ","
      << (r.device == "gpu" ? storage : "-") << "," << r.t_blit << "\n";
  }
}

void usage() {
  std::cerr << "usage: calib_agg --device gpu|cpu [--state warm|cold|resident] [--qos p|e]\n"
               "                 --vals data/tpch-sf10/l_extendedprice.u32\n"
               "                 [--keys data/tpch-sf10/l_shipmode.u32]      (omit for plain SUM)\n"
               "                 [--gpu-path auto|sum|tg|global]\n"
               "                 [--min-log2 10] [--max-log2 25] [--reps 15]\n"
               "                 [--cache cold|hot] [--flush-mb 64]\n"
               "                 [--dispatch sync|pipelined] [--batch 16] [--batch-mb 1024]   (GPU only)\n"
               "                 [--storage shared|private]   (GPU only; private: blit, then kernel)\n"
               "                 [--out results/calib_agg.csv] [--machine-json path]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    if (a.has("help") || !a.has("device") || !a.has("vals")) {
      usage();
      return a.has("help") ? 0 : 1;
    }
    const std::string device = a.str("device", "gpu");
    const std::string state = a.str("state", "warm");
    const std::string qos = a.str("qos", "p");
    const int min_l2 = static_cast<int>(a.i64("min-log2", 10));
    int max_l2 = static_cast<int>(a.i64("max-log2", 25));
    const int reps = static_cast<int>(a.i64("reps", 15));
    const std::string out_path = a.str("out", "results/calib_agg.csv");
    const std::string cache = a.str("cache", "cold");
    if (cache != "cold" && cache != "hot") throw std::runtime_error("--cache must be cold or hot");
    if (max_l2 > 30) throw std::runtime_error("max-log2 > 30 not supported (uint32 n)");
    const std::string dispatch = a.str("dispatch", "sync");
    if (dispatch != "sync" && dispatch != "pipelined") throw std::runtime_error("--dispatch must be sync or pipelined");
    const bool pipelined = dispatch == "pipelined";
    if (pipelined && device != "gpu") throw std::runtime_error("--dispatch pipelined applies to --device gpu only");
    const std::string storage = a.str("storage", "shared");
    if (storage != "shared" && storage != "private") throw std::runtime_error("--storage must be shared or private");
    if (storage == "private" && device != "gpu") throw std::runtime_error("--storage applies to --device gpu only");
    if (storage == "private" && pipelined) throw std::runtime_error("--storage private supports --dispatch sync only");
    if (state == "resident" && storage != "private") throw std::runtime_error("--state resident needs --storage private");
    const int batch = static_cast<int>(a.i64("batch", 16));
    const size_t batch_cap = static_cast<size_t>(a.i64("batch-mb", 1024)) << 20;
    acquire_bench_lock();  // one benchmark process at a time (common/bench.hpp)

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    MetalCtx ctx(std::string(MCM_KERNEL_DIR) + "/agg.metal");
    MachineInfo mi = collect_machine_info(ctx.device());
    if (a.has("machine-json")) std::ofstream(a.str("machine-json", "")) << mi.to_json();
    std::cerr << "machine: " << mi.cpu_brand << " | P=" << mi.p_cores << " E=" << mi.e_cores
              << " GPU cores=" << mi.gpu_cores << " | mem=" << (mi.mem_bytes >> 30) << " GiB"
              << " | macOS " << mi.os_version << " | power=" << mi.power_source
              << (mi.low_power_mode ? " (LOW POWER MODE ON)" : "") << "\n";

    // ---- data
    const std::string vals_path = a.str("vals", "");
    const std::string keys_path = a.str("keys", "");
    const bool grouped = !keys_path.empty();
    std::vector<uint32_t> vcol = load_u32(vals_path), kcol;
    uint32_t base = 0, G = 1;
    if (grouped) {
      kcol = load_u32(keys_path);
      if (kcol.size() != vcol.size()) throw std::runtime_error("--keys and --vals have different row counts");
      auto [mn, mx] = std::minmax_element(kcol.begin(), kcol.end());
      base = *mn;
      G = *mx - *mn + 1;
    }
    int fit = 0;
    while ((2ull << fit) <= vcol.size()) ++fit;
    if (fit < max_l2) {
      std::cerr << "column has " << vcol.size() << " rows; capping max-log2 at " << fit << "\n";
      max_l2 = fit;
    }
    const uint64_t max_n = 1ull << max_l2;
    const uint64_t bpr = grouped ? 8 : 4;  // bytes read per row
    PageAlloc mvals(max_n * sizeof(uint32_t)), mkeys(grouped ? max_n * sizeof(uint32_t) : 4096);
    std::memcpy(mvals.data(), vcol.data(), max_n * sizeof(uint32_t));
    if (grouped) std::memcpy(mkeys.data(), kcol.data(), max_n * sizeof(uint32_t));
    std::vector<uint32_t>().swap(vcol);
    std::vector<uint32_t>().swap(kcol);
    const auto* mv = static_cast<const uint32_t*>(mvals.data());
    const auto* mk = grouped ? static_cast<const uint32_t*>(mkeys.data()) : nullptr;

    std::string gpath = a.str("gpu-path", "auto");
    if (gpath == "auto") gpath = !grouped ? "sum" : G <= TG_BINS ? "tg" : "global";
    if (gpath == "sum" && grouped) throw std::runtime_error("--gpu-path sum needs no --keys");
    if (gpath != "sum" && !grouped) throw std::runtime_error("--gpu-path tg/global need --keys");
    const std::string op = grouped ? "agg_group" : "agg_sum";
    const std::string data = grouped ? stem(vals_path) + "_by_" + stem(keys_path) : "sum_" + stem(vals_path);
    std::cerr << "data: " << data << " | G=" << G << " | base=" << base
              << (device == "gpu" ? " | gpu path=" + gpath : "") << " | cache=" << cache << "\n";

    CacheFlusher flusher(static_cast<size_t>(a.i64("flush-mb", 64)) << 20);
    auto flush = [&] { if (cache == "cold") flusher(); };

    std::vector<Row> rows;
    std::vector<uint32_t> ref(G);
    auto reference = [&](uint64_t n) {
      std::fill(ref.begin(), ref.end(), 0u);
      reference_agg(mk, mv, n, base, ref.data());
    };
    const size_t ref_bytes = G * sizeof(uint32_t);

    if (device == "gpu") {
      GpuRunner g(ctx, gpath, G);
      if (gpath == "tg") std::cerr << "tg path: " << g.copies << " bin copies per threadgroup\n";
      auto copy_in = [&](PageAlloc& k, PageAlloc& v, uint64_t n) {
        std::memcpy(v.data(), mv, n * sizeof(uint32_t));
        if (grouped) std::memcpy(k.data(), mk, n * sizeof(uint32_t));
      };
      for (int l2 = min_l2; l2 <= max_l2; ++l2) {
        const uint32_t n = 1u << l2;
        reference(n);
        const size_t kb = grouped ? n * sizeof(uint32_t) : 4096;
        const int K = pipelined ? batch_size(batch, n * bpr, batch_cap) : 1;
        // Expected table after one timed rep: the reference, or K x the reference for a pipelined
        // batch (all K morsels add into one table; uint32 wraps the same way on both sides).
        std::vector<uint32_t> expect(ref);
        for (uint32_t& e : expect) e *= static_cast<uint32_t>(K);
        // Zeroing stays outside timed() so it happens before the cache flush, as it always has.
        auto zero = [&] { pipelined ? g.zero_out_pipe() : g.zero_out(); };
        auto timed = [&](const std::vector<MTL::Buffer*>& kbufs, const std::vector<MTL::Buffer*>& vbufs, Row& row) {
          if (pipelined) {
            g.run_batch(kbufs, vbufs, n, base, row);
            return std::memcmp(g.result_pipe(), expect.data(), ref_bytes) == 0;
          }
          g.run(kbufs[0], vbufs[0], n, base, row);
          return std::memcmp(g.result(), expect.data(), ref_bytes) == 0;
        };
        auto alloc_morsels = [&](std::vector<std::unique_ptr<PageAlloc>>& km, std::vector<std::unique_ptr<PageAlloc>>& vm) {
          for (int k = 0; k < K; ++k) {
            km.push_back(std::make_unique<PageAlloc>(kb));
            vm.push_back(std::make_unique<PageAlloc>(n * sizeof(uint32_t)));
            copy_in(*km.back(), *vm.back(), n);
          }
        };
        auto release = [](std::vector<MTL::Buffer*>& kbufs, std::vector<MTL::Buffer*>& vbufs) {
          for (MTL::Buffer* b : kbufs) if (b) b->release();
          for (MTL::Buffer* b : vbufs) b->release();
        };
        if (storage == "private") {
          MTL::Buffer* pk = grouped ? new_private_buffer(ctx.device(), n * sizeof(uint32_t)) : nullptr;
          MTL::Buffer* pv = new_private_buffer(ctx.device(), n * sizeof(uint32_t));
          auto run_priv = [&](MTL::Buffer* kb_, MTL::Buffer* vb_, Row& row) {
            g.run_private(kb_, vb_, pk, pv, n, base, row);
            return std::memcmp(g.result(), ref.data(), ref_bytes) == 0;
          };
          auto wrap = [&](PageAlloc& km, PageAlloc& vm, MTL::Buffer*& bk, MTL::Buffer*& bv) {
            bk = grouped ? wrap_no_copy(ctx.device(), km) : nullptr;
            bv = wrap_no_copy(ctx.device(), vm);
          };
          auto unwrap = [](MTL::Buffer* bk, MTL::Buffer* bv) {
            if (bk) bk->release();
            bv->release();
          };
          Row warm{};
          if (state == "warm" || state == "resident") {
            PageAlloc km(kb), vm(n * sizeof(uint32_t));
            copy_in(km, vm, n);
            MTL::Buffer *bk, *bv;
            wrap(km, vm, bk, bv);
            for (int w = 0; w < 3; ++w) {  // fills the private copies, makes everything resident; untimed
              g.zero_out();
              run_priv(bk, bv, warm);
            }
            for (int r = 0; r < reps; ++r) {
              Row row{"gpu", "-", state, n, r};
              g.zero_out();
              flush();
              row.ok = (state == "warm" ? run_priv(bk, bv, row) : run_priv(nullptr, nullptr, row)) && row.ok;
              rows.push_back(row);
            }
            unwrap(bk, bv);
          } else if (state == "cold") {
            {  // warm the pipeline, the blit path and the private buffers themselves
              PageAlloc km(kb), vm(n * sizeof(uint32_t));
              copy_in(km, vm, n);
              MTL::Buffer *bk, *bv;
              wrap(km, vm, bk, bv);
              g.zero_out();
              run_priv(bk, bv, warm);
              unwrap(bk, bv);
            }
            for (int r = 0; r < reps; ++r) {
              PageAlloc km(kb), vm(n * sizeof(uint32_t));
              copy_in(km, vm, n);  // CPU-touched, GPU-unseen
              Row row{"gpu", "-", "cold", n, r};
              g.zero_out();
              flush();
              auto t0 = Clock::now();
              MTL::Buffer *bk, *bv;
              wrap(km, vm, bk, bv);
              row.t_wrap = us_since(t0, Clock::now());
              row.ok = run_priv(bk, bv, row) && row.ok;
              unwrap(bk, bv);
              rows.push_back(row);
            }
          } else {
            throw std::runtime_error("--state must be warm, cold or resident");
          }
          if (pk) pk->release();
          pv->release();
        } else if (state == "warm") {
          std::vector<std::unique_ptr<PageAlloc>> kmem, vmem;
          alloc_morsels(kmem, vmem);
          std::vector<MTL::Buffer*> kbufs, vbufs;
          for (int k = 0; k < K; ++k) {
            kbufs.push_back(grouped ? wrap_no_copy(ctx.device(), *kmem[k]) : nullptr);
            vbufs.push_back(wrap_no_copy(ctx.device(), *vmem[k]));
          }
          Row warm{};
          for (int w = 0; w < 3; ++w) {  // make resident, untimed
            zero();
            timed(kbufs, vbufs, warm);
          }
          for (int r = 0; r < reps; ++r) {
            Row row{"gpu", "-", "warm", n, r};
            zero();
            flush();
            row.ok = timed(kbufs, vbufs, row) && row.ok;
            rows.push_back(row);
          }
          release(kbufs, vbufs);
        } else if (state == "cold") {
          {  // warm the pipeline itself, so we measure buffer residency, not first-dispatch setup
            PageAlloc tk(4096), tv(4096);
            std::memset(tk.data(), 0, 4096);
            std::memset(tv.data(), 0, 4096);
            static_cast<uint32_t*>(tk.data())[0] = base;  // the one row must be a valid key
            MTL::Buffer* bk = grouped ? wrap_no_copy(ctx.device(), tk) : nullptr;
            MTL::Buffer* bv = wrap_no_copy(ctx.device(), tv);
            Row warm{};
            g.zero_out();
            g.run(bk, bv, 1, base, warm);
            if (bk) bk->release();
            bv->release();
          }
          for (int r = 0; r < reps; ++r) {
            std::vector<std::unique_ptr<PageAlloc>> kmem, vmem;
            alloc_morsels(kmem, vmem);  // CPU-touched, GPU-unseen
            Row row{"gpu", "-", "cold", n, r};
            std::vector<MTL::Buffer*> kbufs, vbufs;
            kbufs.reserve(K);
            vbufs.reserve(K);
            zero();
            flush();
            auto t0 = Clock::now();
            for (int k = 0; k < K; ++k) {
              kbufs.push_back(grouped ? wrap_no_copy(ctx.device(), *kmem[k]) : nullptr);
              vbufs.push_back(wrap_no_copy(ctx.device(), *vmem[k]));
            }
            row.t_wrap = us_since(t0, Clock::now()) / K;
            row.ok = timed(kbufs, vbufs, row) && row.ok;
            release(kbufs, vbufs);
            rows.push_back(row);
          }
        } else {
          throw std::runtime_error("--state must be warm or cold");
        }
        std::cerr << "gpu " << state << " " << storage << " " << dispatch << (pipelined ? " x" + std::to_string(K) : "")
                  << " n=2^" << l2 << " done\n";
      }
    } else if (device == "cpu") {
      place_cpu_process(qos);
      std::thread worker([&] {
        place_cpu_thread(qos);
        std::vector<uint32_t> acc(G), scratch(G <= CPU_COPY_MAX_G ? CPU_COPIES * G : 0);
        for (int l2 = min_l2; l2 <= max_l2; ++l2) {
          const uint64_t n = 1ull << l2;
          reference(n);
          for (int r = 0; r < reps; ++r) {
            Row row{"cpu", qos, r == 0 ? "first" : "warm", n, r};
            std::fill(acc.begin(), acc.end(), 0u);
            std::fill(scratch.begin(), scratch.end(), 0u);
            flush();
            asm volatile("" ::: "memory");
            auto t0 = Clock::now();
            cpu_agg(mk, mv, n, base, G, acc.data(), scratch.data());
            asm volatile("" ::: "memory");
            row.t_total = us_since(t0, Clock::now());
            row.ok = std::memcmp(acc.data(), ref.data(), ref_bytes) == 0;
            rows.push_back(row);
          }
          std::cerr << "cpu " << qos << " n=2^" << l2 << " done\n";
        }
      });
      worker.join();
    } else {
      throw std::runtime_error("--device must be gpu or cpu");
    }

    size_t bad = std::count_if(rows.begin(), rows.end(), [](const Row& r) { return !r.ok; });
    write_rows(out_path, mi.tag(), op, data, cache, G, gpath, bpr, dispatch, storage, rows);
    std::cerr << rows.size() << " rows -> " << out_path << (bad ? " | WARNING: " : "")
              << (bad ? std::to_string(bad) + " rows failed the correctness check" : "") << "\n";
    pool->release();
    return bad ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
