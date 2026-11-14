// val_stream — a diagnostic, not a query: why the fused Q1 kernel streams at 81% of the filter's
// pipelined rate on the M3 Ultra (README, Q1). It runs kernels/stream.metal on Q1's columns through
// the validation harness (bench/val_query.hpp: morsel sizes, dispatch, cache flush, checks and CSV
// as in val_q1), with Q1's grid (256 threads per threadgroup, 16 rows per thread, Q1's cap) and
// Q1's two dispatches per morsel:
//   --kernel stream --cols C (1..7, default 7): the first C of Q1's columns, summed per thread.
//       Q1's loads without its aggregation. op = stream<C>, bytes = 4 C n.
//   --kernel agg: l_shipdate alone, with Q1's aggregation (36 uint64 accumulators per thread) on
//       values derived from it. Q1's arithmetic and registers on one stream. op = agg1, bytes = 4 n.
//   --rows-per-thread R (default 16, Q1's): R rows per thread. The default cap scales by 16 / R, so
//       in rows it stays Q1's. op gets the suffix _r<R>.
// GPU only. CSV thr = 0. There is no DuckDB answer; every morsel is checked against a scalar loop.

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/args.hpp"
#include "common/metal_ctx.hpp"
#include "engine/q1.hpp"
#include "engine/q1_gpu.hpp"
#include "val_query.hpp"

namespace {

using mcm::q1::Columns;
constexpr int kAggVals = mcm::q1::kVals;  // agg: Q1's 36 values; stream: 1

struct Variant {
  bool agg = false;
  int cols = 7;  // columns loaded per row
  int rows_per_thread = 16;
  int vals() const { return agg ? kAggVals : 1; }
};
Variant g_v;  // set by main() before the harness runs

struct Result {
  uint64_t v[kAggVals] = {};  // the first g_v.vals() are used
  void merge(const Result& o) {
    for (int i = 0; i < kAggVals; ++i) v[i] += o.v[i];
  }
  bool operator==(const Result& o) const { return std::memcmp(v, o.v, sizeof(v)) == 0; }
};

// agg_partial's values derived from l_shipdate (kernels/stream.metal does the same).
struct Derived {
  uint32_t rf, ls, q, e, d, t;
};
inline Derived derive(uint32_t x) { return {'A' + (x & 31u), 'F' + ((x >> 5) & 15u), x & 63u, x, x & 7u, (x >> 3) & 7u}; }

// Plain scalar loops, the correctness reference.
Result reference(const Columns& c, size_t begin, size_t n) {
  Result r;
  if (!g_v.agg) {
    for (int k = 0; k < g_v.cols; ++k)
      for (size_t i = begin; i < begin + n; ++i) r.v[0] += c.col(k)[i];
    return r;
  }
  for (size_t i = begin; i < begin + n; ++i) {
    const uint32_t x = c.shipdate()[i];
    if (x > mcm::q1::kCutoff) continue;
    const Derived w = derive(x);
    uint64_t* s = r.v + mcm::q1::slot_of(w.rf, w.ls) * mcm::q1::kFields;
    const uint64_t dp = uint64_t(w.e) * (100 - w.d);
    s[mcm::q1::COUNT] += 1;
    s[mcm::q1::SUM_QTY] += w.q;
    s[mcm::q1::SUM_BASE_PRICE] += w.e;
    s[mcm::q1::SUM_DISC_PRICE] += dp;
    s[mcm::q1::SUM_CHARGE] += dp * (100 + w.t);
    s[mcm::q1::SUM_DISC] += w.d;
  }
  return r;
}

struct GpuParams {
  uint32_t n, cutoff;  // must match StreamParams in kernels/stream.metal
};
struct GpuFinalParams {
  uint32_t rows, vals;  // must match StreamFinalParams
};

// The harness's Gpu interface (bench/val_query.hpp), as engine/q1_gpu.hpp.
class Gpu {
 public:
  static constexpr NS::UInteger kThreads = 256;
  static NS::UInteger default_max_groups(int gpu_cores) {
    return std::max<NS::UInteger>(1, mcm::q1::Gpu::default_max_groups(gpu_cores) * 16 / g_v.rows_per_thread);
  }
  // stream_final threads per value: agg as q1_final (64 for each of the 36 values); stream sums
  // its one value with 1024, so even 2^20 rows of partials take few steps.
  static NS::UInteger final_threads() { return g_v.agg ? 64 : 1024; }

  struct Slot {
    MTL::Buffer* partials = nullptr;  // max_groups() x vals uint64
    MTL::Buffer* out = nullptr;       // vals uint64
  };

  Gpu(mcm::MetalCtx& ctx, const Columns& c, NS::UInteger max_groups) : ctx_(ctx), max_groups_(max_groups) {
    if (max_groups_ == 0) throw std::runtime_error("val_stream: max_groups must be at least 1");
    partial_ = g_v.agg ? ctx.pipeline("agg_partial") : ctx.pipeline("stream_partial", {static_cast<uint32_t>(g_v.cols)});
    final_ = ctx.pipeline("stream_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads || final_->maxTotalThreadsPerThreadgroup() < final_threads())
      throw std::runtime_error("val_stream kernels: too few threads per threadgroup available");
    for (int i = 0; i < 7; ++i) cols_[i] = mcm::wrap_no_copy(ctx.device(), *c.mem[i]);
  }
  ~Gpu() {
    for (MTL::Buffer* b : cols_) b->release();
    final_->release();
    partial_->release();
  }
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;

  Slot new_slot() const {
    Slot s;
    s.partials = ctx_.device()->newBuffer(max_groups_ * g_v.vals() * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    s.out = ctx_.device()->newBuffer(g_v.vals() * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    return s;
  }
  static void free_slot(Slot& s) {
    s.partials->release();
    s.out->release();
    s = Slot{};
  }
  static Result read(const Slot& s) {
    Result r;
    std::memcpy(r.v, s.out->contents(), g_v.vals() * sizeof(uint64_t));
    return r;
  }

  MTL::ComputePipelineState* partial_pso() const { return partial_; }
  NS::UInteger max_groups() const { return max_groups_; }

  NS::UInteger groups_for(uint32_t n) const {
    const NS::UInteger per_group = kThreads * static_cast<NS::UInteger>(g_v.rows_per_thread);
    return std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, max_groups_);
  }

  void set_final_pass(bool on) { final_pass_ = on; }
  bool final_pass() const { return final_pass_; }
  void final_on_cpu(const Slot& s, uint32_t n) const {
    const auto* p = static_cast<const uint64_t*>(s.partials->contents());
    auto* out = static_cast<uint64_t*>(s.out->contents());
    const NS::UInteger rows = groups_for(n);
    const int vals = g_v.vals();
    for (int v = 0; v < vals; ++v) {
      uint64_t sum = 0;
      for (NS::UInteger g = 0; g < rows; ++g) sum += p[g * vals + v];
      out[v] = sum;
    }
  }

  // The partial kernel on morsel [begin, begin + n), then stream_final (same serial encoder).
  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger groups = groups_for(n);
    const GpuParams p{n, mcm::q1::kCutoff};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 7; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(s.partials, 0, 7);
    enc->setBytes(&p, sizeof(p), 8);
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    if (!final_pass_) return;
    const GpuFinalParams f{static_cast<uint32_t>(groups), static_cast<uint32_t>(g_v.vals())};
    enc->setComputePipelineState(final_);
    enc->setBuffer(s.partials, 0, 0);
    enc->setBuffer(s.out, 0, 1);
    enc->setBytes(&f, sizeof(f), 2);
    enc->dispatchThreadgroups(MTL::Size(static_cast<NS::UInteger>(g_v.vals()), 1, 1), MTL::Size(final_threads(), 1, 1));
  }

 private:
  mcm::MetalCtx& ctx_;
  NS::UInteger max_groups_;
  bool final_pass_ = true;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* final_ = nullptr;
  MTL::Buffer* cols_[7] = {};
};

// The harness's query interface; the strings depend on the variant and are set by main().
struct Stream {
  static inline std::string kOp = "stream7";
  static constexpr const char* kKernelFile = "stream.metal";
  static inline std::string kPartialKernel = "stream_partial";
  static inline size_t kBytesPerRow = 7 * sizeof(uint32_t);
  static constexpr uint32_t kThr = 0;
  using Columns = ::Columns;
  using Result = ::Result;
  using Gpu = ::Gpu;
  static std::string what() {
    std::ostringstream o;
    if (g_v.agg)
      o << "agg: Q1's aggregation on l_shipdate alone";
    else
      o << "stream: the first " << g_v.cols << " of Q1's columns, summed";
    o << ", " << g_v.rows_per_thread << " rows per thread (diagnostic, not a query)";
    return o.str();
  }
  static Columns load(const std::string& dir) { return mcm::q1::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return ::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { out.merge(::reference(c, begin, n)); }
  static bool load_expected(const std::string&, Result&) { return false; }
  static std::string describe(const Result& r) {
    std::ostringstream o;
    o << "  " << kOp << ": v[0]=" << r.v[0];
    if (g_v.agg) o << " (count of group AF) v[24]=" << r.v[24] << " (count of group RF)";
    o << "\n";
    return o.str();
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const mcm::Args a(argc, argv);
    const std::string kernel = a.str("kernel", "stream");
    if (kernel != "stream" && kernel != "agg") throw std::runtime_error("--kernel must be stream or agg");
    g_v.agg = kernel == "agg";
    g_v.cols = static_cast<int>(a.i64("cols", g_v.agg ? 1 : 7));
    if (g_v.agg ? g_v.cols != 1 : (g_v.cols < 1 || g_v.cols > 7))
      throw std::runtime_error("--cols must be 1..7 (--kernel agg: 1)");
    g_v.rows_per_thread = static_cast<int>(a.i64("rows-per-thread", 16));
    if (g_v.rows_per_thread < 1 || g_v.rows_per_thread > 4096) throw std::runtime_error("--rows-per-thread must be in [1, 4096]");
    if (a.has("device") && a.str("device", "") != "gpu") throw std::runtime_error("val_stream is a GPU diagnostic: --device gpu");
    Stream::kOp = kernel + std::to_string(g_v.cols) + (g_v.rows_per_thread != 16 ? "_r" + std::to_string(g_v.rows_per_thread) : "");
    Stream::kPartialKernel = g_v.agg ? "agg_partial" : "stream_partial";
    Stream::kBytesPerRow = static_cast<size_t>(g_v.cols) * sizeof(uint32_t);
    if (a.has("help") || !a.has("device"))
      std::cerr << "val_stream (diagnostic): --kernel stream|agg [--cols 1..7] [--rows-per-thread 16], then:\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return mcm::valq::run<Stream>(argc, argv);
}
