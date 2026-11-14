// val_q1 — TPC-H Q1 per morsel on CPU-P, CPU-E and the GPU: held-out validation data for the cost
// model (never used for fitting; analysis/predict_q1.py compares it with the model's prediction).
// The fused Q1 kernels are src/engine/q1.hpp and kernels/q1.metal; the protocol (morsel sizes,
// dispatch, cache flush, checks, CSV) is bench/val_query.hpp. CSV: op = q1, thr = the shipdate
// cutoff, bytes = 28 n (7 uint32 columns).
//
// --gpu-loads dependent (diagnostic; default eager): the GPU morsels run q1_partial_dependent, the
// kernel before 13 Nov 2026, instead of q1_partial. It computes the same result but reads
// l_returnflag and l_linestatus only for rows that pass the shipdate test; q1_partial issues all 7
// loads of a row at once (kernels/q1.metal). CSV column gpu_kernel; analysis/fit.py names these
// configurations -dependent, so the predictors never take them for the query.

#include <iostream>
#include <stdexcept>
#include <string>

#include "common/args.hpp"
#include "engine/q1.hpp"
#include "engine/q1_gpu.hpp"
#include "val_query.hpp"

namespace {

struct Q1 {
  static constexpr const char* kOp = "q1";
  static constexpr const char* kKernelFile = "q1.metal";
  static inline const char* kPartialKernel = "q1_partial";  // main: q1_partial_dependent for --gpu-loads dependent
  static constexpr size_t kBytesPerRow = mcm::q1::kBytesPerRow;
  static constexpr uint32_t kThr = mcm::q1::kCutoff;
  using Columns = mcm::q1::Columns;
  using Result = mcm::q1::Result;
  // The harness builds Gpu(ctx, cols, max_groups); this one runs kPartialKernel.
  struct Gpu : mcm::q1::Gpu {
    Gpu(mcm::MetalCtx& ctx, const Columns& c, NS::UInteger max_groups)
        : mcm::q1::Gpu(ctx, c, max_groups, kPartialKernel) {}
  };
  static std::string what() { return "Q1, shipdate <= " + std::to_string(mcm::q1::kCutoff); }
  static Columns load(const std::string& dir) { return mcm::q1::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q1::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q1::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q1::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q1::describe(r); }
};

}  // namespace

int main(int argc, char** argv) {
  bool usage = false;
  try {
    const mcm::Args a(argc, argv);
    usage = a.has("help") || !a.has("device");
    const std::string loads = a.str("gpu-loads", "eager");
    if (loads != "eager" && loads != "dependent") throw std::runtime_error("--gpu-loads must be eager or dependent");
    if (loads == "dependent" && a.str("device", "gpu") != "gpu")
      throw std::runtime_error("--gpu-loads applies to --device gpu only");
    if (loads == "dependent") Q1::kPartialKernel = "q1_partial_dependent";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  const int rc = mcm::valq::run<Q1>(argc, argv);
  if (usage) std::cerr << "              [--gpu-loads eager|dependent (dependent: diagnostic, q1_partial_dependent)]\n";
  return rc;
}
