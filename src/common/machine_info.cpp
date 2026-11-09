#include "common/machine_info.hpp"

#include <sys/sysctl.h>
#include <unistd.h>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cctype>
#include <cstdio>
#include <sstream>

namespace mcm {
namespace {

std::string sysctl_str(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return "";
  std::string s(len, '\0');
  if (sysctlbyname(name, s.data(), &len, nullptr, 0) != 0) return "";
  while (!s.empty() && s.back() == '\0') s.pop_back();
  return s;
}

int64_t sysctl_i64(const char* name) {
  int64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return 0;
  if (len == sizeof(int32_t)) return static_cast<int64_t>(*reinterpret_cast<int32_t*>(&v));
  return v;
}

std::string shell_out(const char* cmd) {
  std::string out;
  FILE* p = popen(cmd, "r");
  if (!p) return out;
  char buf[256];
  while (fgets(buf, sizeof(buf), p)) out += buf;
  pclose(p);
  return out;
}

int first_int(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && !std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
  if (i == s.size()) return 0;
  return std::atoi(s.c_str() + i);
}

std::string json_escape(const std::string& s) {
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') { o += "\\n"; continue; }
    o += c;
  }
  return o;
}

}  // namespace

std::string MachineInfo::tag() const {
  std::string t = cpu_brand;
  for (char& c : t)
    if (c == ' ') c = '-';
  return t;
}

std::string MachineInfo::to_json() const {
  std::ostringstream o;
  o << "{\n"
    << "  \"hw_model\": \"" << json_escape(hw_model) << "\",\n"
    << "  \"cpu_brand\": \"" << json_escape(cpu_brand) << "\",\n"
    << "  \"os_version\": \"" << json_escape(os_version) << "\",\n"
    << "  \"mem_bytes\": " << mem_bytes << ",\n"
    << "  \"p_cores\": " << p_cores << ",\n"
    << "  \"e_cores\": " << e_cores << ",\n"
    << "  \"gpu_cores\": " << gpu_cores << ",\n"
    << "  \"page_size\": " << page_size << ",\n"
    << "  \"gpu_name\": \"" << json_escape(gpu_name) << "\",\n"
    << "  \"gpu_recommended_max_working_set\": " << gpu_recommended_max_working_set << ",\n"
    << "  \"low_power_mode\": " << (low_power_mode ? "true" : "false") << ",\n"
    << "  \"power_source\": \"" << json_escape(power_source) << "\"\n"
    << "}\n";
  return o.str();
}

MachineInfo collect_machine_info(MTL::Device* dev) {
  MachineInfo m;
  m.hw_model = sysctl_str("hw.model");
  m.cpu_brand = sysctl_str("machdep.cpu.brand_string");
  m.os_version = sysctl_str("kern.osproductversion");
  m.mem_bytes = sysctl_i64("hw.memsize");
  m.p_cores = static_cast<int>(sysctl_i64("hw.perflevel0.physicalcpu"));
  m.e_cores = static_cast<int>(sysctl_i64("hw.perflevel1.physicalcpu"));
  m.page_size = getpagesize();
  m.gpu_cores = first_int(shell_out("ioreg -rc AGXAccelerator -d1 2>/dev/null | grep gpu-core-count"));
  m.low_power_mode = first_int(shell_out("pmset -g 2>/dev/null | grep -i lowpowermode")) == 1;
  std::string batt = shell_out("pmset -g batt 2>/dev/null | head -1");
  m.power_source = batt.find("AC Power") != std::string::npos ? "AC"
                   : batt.find("Battery") != std::string::npos ? "Battery" : "unknown";
  if (dev) {
    m.gpu_name = dev->name()->utf8String();
    m.gpu_recommended_max_working_set = dev->recommendedMaxWorkingSetSize();
  }
  return m;
}

}  // namespace mcm
