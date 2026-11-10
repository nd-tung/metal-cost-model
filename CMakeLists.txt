cmake_minimum_required(VERSION 3.20)
project(metal_cost_model LANGUAGES CXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal-cost-model builds only on macOS (Apple Silicon).")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/third_party/metal-cpp" CACHE PATH "Path to metal-cpp headers")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(FATAL_ERROR "metal-cpp not found at ${METAL_CPP_DIR}. Run scripts/fetch_metal_cpp.sh first.")
endif()

add_library(mcm_common STATIC
  src/metal_impl.cpp
  src/common/machine_info.cpp
  src/common/metal_ctx.cpp
)
target_include_directories(mcm_common PUBLIC src)
target_include_directories(mcm_common SYSTEM PUBLIC "${METAL_CPP_DIR}")
target_link_libraries(mcm_common PUBLIC "-framework Metal" "-framework Foundation")
target_compile_definitions(mcm_common PUBLIC MCM_KERNEL_DIR="${CMAKE_SOURCE_DIR}/kernels")
target_compile_options(mcm_common PUBLIC -O3 -Wall -Wextra -Wno-unused-parameter)

add_executable(calib_filter bench/calib_filter.cpp)
target_link_libraries(calib_filter PRIVATE mcm_common)

add_executable(calib_agg bench/calib_agg.cpp)
target_link_libraries(calib_agg PRIVATE mcm_common)

add_executable(calib_contention bench/calib_contention.cpp)
target_link_libraries(calib_contention PRIVATE mcm_common)

add_executable(val_q1 bench/val_q1.cpp)
target_link_libraries(val_q1 PRIVATE mcm_common)
