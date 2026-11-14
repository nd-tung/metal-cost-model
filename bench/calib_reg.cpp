// calib_reg — calibration microbenchmark for register aggregation on the GPU (kernels/agg_reg.metal):
// SUM GROUP BY over G dense keys with F sums per group, all U = G x F uint64 accumulators in
// registers and all updated for every row (branchless). analysis/model.py fits the GPU's cost per
// update from it (register_agg) for kernels that aggregate this way, such as Q1's (6 groups x 6
// fields). GPU only: a CPU group-by updates one private table per row (calib_agg).
//
// The protocol, CSV and checks are the query harness's (bench/val_query.hpp: sync or pipelined
// dispatch, cache flush, busy GPU, partials + final pass, every morsel checked against a scalar
// loop), on the engine's grid for register aggregation (engine/q1_gpu.hpp: 256 threads per
// threadgroup, 16 rows per thread, at most max(128, 16 x GPU cores) threadgroups).
//   --keys l_shipmode    dense codes; G = max - min + 1 (at most 8)
//   --vals l_extendedprice
//   --fields F           sums per group (1..8); the f-th adds val >> f
//   --cols C             columns loaded per row (2..7): the key, the value, and the first C - 2 of
//                        l_quantity, l_discount, l_tax, l_shipdate, l_returnflag (Q1's other
//                        columns but l_linestatus), added to the value. Q1 loads 7.
// CSV: op = agg_reg, thr = U = G x F (accumulators updated per row), bytes = 4 C n, data = the
// data directory's name.

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "common/args.hpp"
#include "common/metal_ctx.hpp"
#include "engine/columns.hpp"
#include "engine/q1_gpu.hpp"
#include "val_query.hpp"

namespace {

constexpr int kMaxG = 8, kMaxF = 8, kMaxCols = 7;
constexpr const char* kExtra[kMaxCols - 2] = {"l_quantity", "l_discount", "l_tax", "l_shipdate", "l_returnflag"};

struct Cfg {
  std::string keys = "l_shipmode", vals = "l_extendedprice";
  int fields = 6;
  int cols = 2;       // columns loaded per row: keys, vals, cols - 2 of kExtra
  int groups = 0;     // from the key column (load)
  uint32_t base = 0;  // smallest key
  int accs() const { return groups * fields; }
};
Cfg g_cfg;  // set by main() and load() before the harness needs it

// The partial kernel for (G, F); kernels/agg_reg.metal instantiates G = 7 with F = 1..8.
std::string partial_kernel_name() { return "agg_reg_g" + std::to_string(g_cfg.groups) + "_f" + std::to_string(g_cfg.fields); }

struct Columns {
  size_t rows = 0;
  std::unique_ptr<mcm::PageAlloc> mem[kMaxCols];  // keys, vals, then the extra columns (g_cfg.cols in all)
  const uint32_t* col(int i) const { return static_cast<const uint32_t*>(mem[i]->data()); }
  const uint32_t* keys() const { return col(0); }
  const uint32_t* vals() const { return col(1); }
};

struct Result {
  uint64_t v[kMaxG * kMaxF] = {};  // the first g_cfg.accs() are used
  void merge(const Result& o) {
    for (int i = 0; i < kMaxG * kMaxF; ++i) v[i] += o.v[i];
  }
  bool operator==(const Result& o) const { return std::memcmp(v, o.v, sizeof(v)) == 0; }
};

// Plain scalar loop, the correctness reference.
Result reference(const Columns& c, size_t begin, size_t n) {
  Result r;
  for (size_t i = begin; i < begin + n; ++i) {
    uint64_t* s = r.v + (c.keys()[i] - g_cfg.base) * g_cfg.fields;
    uint32_t v = c.vals()[i];  // uint32 sum, as the kernel's
    for (int k = 2; k < g_cfg.cols; ++k) v += c.col(k)[i];
    for (int f = 0; f < g_cfg.fields; ++f) s[f] += v >> f;
  }
  return r;
}

struct GpuParams {
  uint32_t n, base;  // must match RegParams in kernels/agg_reg.metal
};
struct GpuFinalParams {
  uint32_t rows, vals;  // must match RegFinalParams
};

// The harness's Gpu interface (bench/val_query.hpp), as engine/q1_gpu.hpp.
class Gpu {
 public:
  static constexpr NS::UInteger kThreads = 256;
  static constexpr NS::UInteger kRowsPerThread = 16;
  static constexpr NS::UInteger kFinalThreads = 64;  // per value, as q1_final
  static NS::UInteger default_max_groups(int gpu_cores) { return mcm::q1::Gpu::default_max_groups(gpu_cores); }

  struct Slot {
    MTL::Buffer* partials = nullptr;  // max_groups() x U uint64
    MTL::Buffer* out = nullptr;       // U uint64
  };

  Gpu(mcm::MetalCtx& ctx, const Columns& c, NS::UInteger max_groups) : ctx_(ctx), max_groups_(max_groups) {
    if (max_groups_ == 0) throw std::runtime_error("calib_reg: max_groups must be at least 1");
    partial_ = ctx.pipeline(partial_kernel_name().c_str(), {static_cast<uint32_t>(g_cfg.cols)});
    final_ = ctx.pipeline("agg_reg_final");
    if (partial_->maxTotalThreadsPerThreadgroup() < kThreads || final_->maxTotalThreadsPerThreadgroup() < kFinalThreads)
      throw std::runtime_error("agg_reg kernels: too few threads per threadgroup available");
    for (int i = 0; i < g_cfg.cols; ++i) cols_[i] = mcm::wrap_no_copy(ctx.device(), *c.mem[i]);
  }
  ~Gpu() {
    for (MTL::Buffer* b : cols_)
      if (b) b->release();
    final_->release();
    partial_->release();
  }
  Gpu(const Gpu&) = delete;
  Gpu& operator=(const Gpu&) = delete;

  Slot new_slot() const {
    Slot s;
    s.partials = ctx_.device()->newBuffer(max_groups_ * g_cfg.accs() * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    s.out = ctx_.device()->newBuffer(g_cfg.accs() * sizeof(uint64_t), MTL::ResourceStorageModeShared);
    return s;
  }
  static void free_slot(Slot& s) {
    s.partials->release();
    s.out->release();
    s = Slot{};
  }
  static Result read(const Slot& s) {
    Result r;
    std::memcpy(r.v, s.out->contents(), g_cfg.accs() * sizeof(uint64_t));
    return r;
  }

  MTL::ComputePipelineState* partial_pso() const { return partial_; }
  NS::UInteger max_groups() const { return max_groups_; }

  NS::UInteger groups_for(uint32_t n) const {
    const NS::UInteger per_group = kThreads * kRowsPerThread;
    return std::clamp<NS::UInteger>((n + per_group - 1) / per_group, 1, max_groups_);
  }

  void set_final_pass(bool on) { final_pass_ = on; }
  bool final_pass() const { return final_pass_; }
  void final_on_cpu(const Slot& s, uint32_t n) const {
    const auto* p = static_cast<const uint64_t*>(s.partials->contents());
    auto* out = static_cast<uint64_t*>(s.out->contents());
    const NS::UInteger rows = groups_for(n);
    const int vals = g_cfg.accs();
    for (int v = 0; v < vals; ++v) {
      uint64_t sum = 0;
      for (NS::UInteger g = 0; g < rows; ++g) sum += p[g * vals + v];
      out[v] = sum;
    }
  }

  // agg_reg_partial on morsel [begin, begin + n), then agg_reg_final (same serial encoder).
  void encode(MTL::ComputeCommandEncoder* enc, size_t begin, uint32_t n, const Slot& s) const {
    const NS::UInteger groups = groups_for(n);
    const GpuParams p{n, g_cfg.base};
    enc->setComputePipelineState(partial_);
    for (int i = 0; i < 2; ++i) enc->setBuffer(cols_[i], begin * sizeof(uint32_t), static_cast<NS::UInteger>(i));
    enc->setBuffer(s.partials, 0, 2);
    enc->setBytes(&p, sizeof(p), 3);
    for (int k = 2; k < kMaxCols; ++k)  // x0..x4 at 4..8; slots the variant does not read get the value column
      enc->setBuffer(cols_[k < g_cfg.cols ? k : 1], begin * sizeof(uint32_t), static_cast<NS::UInteger>(k + 2));
    enc->dispatchThreadgroups(MTL::Size(groups, 1, 1), MTL::Size(kThreads, 1, 1));
    if (!final_pass_) return;
    const GpuFinalParams f{static_cast<uint32_t>(groups), static_cast<uint32_t>(g_cfg.accs())};
    enc->setComputePipelineState(final_);
    enc->setBuffer(s.partials, 0, 0);
    enc->setBuffer(s.out, 0, 1);
    enc->setBytes(&f, sizeof(f), 2);
    enc->dispatchThreadgroups(MTL::Size(static_cast<NS::UInteger>(g_cfg.accs()), 1, 1), MTL::Size(kFinalThreads, 1, 1));
  }

 private:
  mcm::MetalCtx& ctx_;
  NS::UInteger max_groups_;
  bool final_pass_ = true;
  MTL::ComputePipelineState* partial_ = nullptr;
  MTL::ComputePipelineState* final_ = nullptr;
  MTL::Buffer* cols_[kMaxCols] = {};
};

// The harness's operator interface. kThr (U) is known once load() has read the key column.
struct Reg {
  static constexpr const char* kOp = "agg_reg";
  static constexpr const char* kKernelFile = "agg_reg.metal";
  static inline std::string kPartialKernel = "agg_reg_g7_f6";  // set by load()
  static inline size_t kBytesPerRow = 2 * sizeof(uint32_t);  // set by main: 4 B x --cols
  static inline uint32_t kThr = 0;
  using Columns = ::Columns;
  using Result = ::Result;
  using Gpu = ::Gpu;
  static std::string what() {
    std::ostringstream o;
    std::string v = g_cfg.vals;
    for (int k = 2; k < g_cfg.cols; ++k) v += std::string(" + ") + kExtra[k - 2];
    o << "register aggregation: SUM((" << v << ") >> f), f < " << g_cfg.fields << ", GROUP BY " << g_cfg.keys
      << " (G = " << g_cfg.groups << "): U = " << g_cfg.accs() << " accumulators per thread, " << g_cfg.cols
      << " columns per row";
    return o.str();
  }
  static Columns load(const std::string& dir) {
    Columns c;
    for (int k = 0; k < g_cfg.cols; ++k) {  // one column at a time: only the ones this variant reads
      const char* const name[1] = {k == 0 ? g_cfg.keys.c_str() : k == 1 ? g_cfg.vals.c_str() : kExtra[k - 2]};
      std::unique_ptr<mcm::PageAlloc> m[1];
      const size_t rows = mcm::load_u32_columns(dir, name, m);
      if (k > 0 && rows != c.rows) throw std::runtime_error(std::string(name[0]) + ": row count differs from " + g_cfg.keys);
      c.rows = rows;
      c.mem[k] = std::move(m[0]);
    }
    const auto [lo, hi] = std::minmax_element(c.keys(), c.keys() + c.rows);
    g_cfg.base = *lo;
    g_cfg.groups = static_cast<int>(*hi - *lo + 1);
    if (g_cfg.groups > kMaxG)
      throw std::runtime_error(g_cfg.keys + " has " + std::to_string(g_cfg.groups) + " key values; at most 8");
    kThr = static_cast<uint32_t>(g_cfg.accs());
    kPartialKernel = partial_kernel_name();
    return c;
  }
  static Result reference(const Columns& c, size_t begin, size_t n) { return ::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { out.merge(::reference(c, begin, n)); }
  static bool load_expected(const std::string&, Result&) { return false; }
  static std::string describe(const Result& r) {
    std::ostringstream o;
    o << "  agg_reg U=" << g_cfg.accs() << ": group 0 sum=" << r.v[0] << ", group " << g_cfg.groups - 1
      << " sum=" << r.v[(g_cfg.groups - 1) * g_cfg.fields] << "\n";
    return o.str();
  }
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const mcm::Args a(argc, argv);
    g_cfg.keys = a.str("keys", g_cfg.keys);
    g_cfg.vals = a.str("vals", g_cfg.vals);
    g_cfg.fields = static_cast<int>(a.i64("fields", g_cfg.fields));
    if (g_cfg.fields < 1 || g_cfg.fields > kMaxF) throw std::runtime_error("--fields must be in [1, 8]");
    g_cfg.cols = static_cast<int>(a.i64("cols", g_cfg.cols));
    if (g_cfg.cols < 2 || g_cfg.cols > kMaxCols) throw std::runtime_error("--cols must be in [2, 7]");
    Reg::kBytesPerRow = static_cast<size_t>(g_cfg.cols) * sizeof(uint32_t);
    if (a.has("device") && a.str("device", "") != "gpu") throw std::runtime_error("calib_reg is GPU only: --device gpu");
    if (a.has("help") || !a.has("device"))
      std::cerr << "calib_reg: [--keys l_shipmode] [--vals l_extendedprice] [--fields 6] [--cols 2], then:\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return mcm::valq::run<Reg>(argc, argv);
}
