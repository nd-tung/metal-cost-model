#pragma once
#include <cstdint>
#include <string>

namespace MTL { class Device; }

namespace mcm {

struct MachineInfo {
  std::string hw_model;        // e.g. MacBookAir10,1
  std::string cpu_brand;       // e.g. Apple M1
  std::string os_version;      // e.g. 26.1
  int64_t mem_bytes = 0;
  int p_cores = 0;             // hw.perflevel0.physicalcpu
  int e_cores = 0;             // hw.perflevel1.physicalcpu
  int gpu_cores = 0;           // from ioreg gpu-core-count (0 if unknown)
  int64_t page_size = 0;
  std::string gpu_name;
  uint64_t gpu_recommended_max_working_set = 0;
  bool low_power_mode = false;
  std::string power_source;    // "AC" / "Battery"

  // Short machine tag used in CSV rows, e.g. "Apple-M1".
  std::string tag() const;
  std::string to_json() const;
};

MachineInfo collect_machine_info(MTL::Device* dev);

}  // namespace mcm
