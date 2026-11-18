// val_q14 — TPC-H Q14 per morsel on CPU-P, CPU-E and the GPU: held-out validation data for the cost
// model (never used for fitting; analysis/predict_q14.py compares it with the model's prediction,
// composed from the filter, join-probe and aggregation terms). The fused Q14 kernels are
// src/engine/q14.hpp and kernels/q14.metal; the protocol is bench/val_query.hpp. CSV: op = q14,
// thr = the month's first day, bytes = 16 n (the 4 lineitem columns; the kernels read 4 B of the
// rows outside the month and all 16 B plus a probe for the rows inside it).

#include <string>

#include "engine/q14.hpp"
#include "engine/q14_gpu.hpp"
#include "val_query.hpp"

namespace {

struct Q14 {
  static constexpr const char* kOp = "q14";
  static constexpr const char* kKernelFile = "q14.metal";
  static constexpr const char* kPartialKernel = "q14_partial";
  static constexpr size_t kBytesPerRow = mcm::q14::kBytesPerRow;
  static constexpr uint32_t kThr = mcm::q14::kDateLo;
  using Columns = mcm::q14::Columns;
  using Result = mcm::q14::Result;
  using Gpu = mcm::q14::Gpu;
  static std::string what() {
    return "Q14, shipdate in [" + std::to_string(mcm::q14::kDateLo) + ", " + std::to_string(mcm::q14::kDateHi) +
           "), lineitem x part on partkey, promo share of revenue";
  }
  static Columns load(const std::string& dir) { return mcm::q14::load_columns(dir); }
  static Result reference(const Columns& c, size_t begin, size_t n) { return mcm::q14::reference(c, begin, n); }
  static void cpu(const Columns& c, size_t begin, size_t n, Result& out) { mcm::q14::cpu(c, begin, n, out); }
  static bool load_expected(const std::string& path, Result& r) { return mcm::q14::load_expected(path, r); }
  static std::string describe(const Result& r) { return mcm::q14::describe(r); }
};

}  // namespace

int main(int argc, char** argv) { return mcm::valq::run<Q14>(argc, argv); }
