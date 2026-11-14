// val_q6 — TPC-H Q6 per morsel on CPU-P, CPU-E and the GPU: held-out validation data for the cost
// model (never used for fitting; analysis/predict_q6.py compares it with the model's prediction).
// The fused Q6 kernels are src/engine/q6.hpp and kernels/q6.metal; the protocol (morsel sizes,
// dispatch, cache flush, checks, CSV) is bench/val_query.hpp. CSV: op = q6, thr = the l_quantity
// bound (2400, x100), bytes = 16 n (4 uint32 columns, all read for every row).

#include <string>

#include "engine/q6.hpp"
#include "engine/q6_gpu.hpp"
#include "val_query.hpp"

namespace {

struct Q6 {
  static constexpr const char* kOp = "q6";
  static constexpr const char* kKernelFile = "q6.metal";
  static constexpr const char* kPartialKernel = "q6_partial";
  static constexpr size_t kBytesPerRow = mcm::q6::kBytesPerRow;
  static constexpr uint32_t kThr = mcm::q6::kQtyHi;
  using Columns = mcm::q6::Columns;
  using Result = mcm::q6::Result;
  using Gpu = mcm::q6::Gpu;
  static std::string what() {
    using namespace mcm::q6;
    return "Q6, shipdate in [" + std::to_string(kDateLo) + ", " + std::to_string(kDateHi) + "), discount in [" +
           std::to_string(kDiscLo) + ", " + std::to_string(kDiscHi) + "], quantity < " + std::to_string(kQtyHi);
  }
  static Columns load(const std::string& dir) { return mcm::q6::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q6::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q6::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q6::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q6::describe(r); }
};

}  // namespace

int main(int argc, char** argv) { return mcm::valq::run<Q6>(argc, argv); }
