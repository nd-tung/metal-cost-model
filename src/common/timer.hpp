#pragma once
#include <chrono>

namespace mcm {

using Clock = std::chrono::steady_clock;

inline double us_since(Clock::time_point t0, Clock::time_point t1) {
  return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

}  // namespace mcm
