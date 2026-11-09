#pragma once
// Minimal --key value argument parser.
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace mcm {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + k);
      k = k.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[k] = argv[++i];
      } else {
        kv_[k] = "1";  // flag
      }
    }
  }
  std::string str(const std::string& k, const std::string& def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  long long i64(const std::string& k, long long def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atoll(it->second.c_str());
  }
  double f64(const std::string& k, double def) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : std::atof(it->second.c_str());
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }

 private:
  std::map<std::string, std::string> kv_;
};

}  // namespace mcm
