#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  int src = -1;
  int dst = -1;
  bool list_only = false;
  bool all_pairs = false;
  bool json = false;
  std::size_t size_bytes = 256ull * 1024ull * 1024ull;
  int warmup = 20;
  int iters = 200;
};

struct DeviceInfo {
  int index = -1;
  std::string name;
};

struct BenchResult {
  int src = -1;
  int dst = -1;
  std::size_t size_bytes = 0;
  int warmup = 0;
  int iters = 0;
  double elapsed_ms = 0.0;
  double decimal_gb_s = 0.0;
  double binary_gib_s = 0.0;
  double total_bytes = 0.0;
};

std::string cuda_error_string(cudaError_t rc) {
  const char* text = cudaGetErrorString(rc);
  return text == nullptr ? "unknown CUDA error" : std::string(text);
}

void cuda_check(cudaError_t rc, const char* where) {
  if (rc != cudaSuccess) {
    throw std::runtime_error(std::string(where) + ": " + cuda_error_string(rc));
  }
}

void enable_peer_access(int current_device, int peer_device) {
  cuda_check(cudaSetDevice(current_device), "cudaSetDevice");
  cudaError_t rc = cudaDeviceEnablePeerAccess(peer_device, 0);
  if (rc == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError();
    return;
  }
  cuda_check(rc, "cudaDeviceEnablePeerAccess");
}

int parse_int(const std::string& text, const char* flag) {
  try {
    std::size_t consumed = 0;
    int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      throw std::runtime_error("");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid value for ") + flag + ": " + text);
  }
}

std::size_t parse_size_bytes(const std::string& text, const char* flag) {
  try {
    std::size_t consumed = 0;
    unsigned long long value = std::stoull(text, &consumed);
    if (consumed != text.size()) {
      throw std::runtime_error("");
    }
    return static_cast<std::size_t>(value);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid value for ") + flag + ": " + text);
  }
}

std::string format_size_mib(std::size_t size_bytes) {
  std::ostringstream out;
  out << (size_bytes / (1024ull * 1024ull)) << " MiB";
  return out.str();
}

std::string json_escape(const std::string& text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    std::string arg = argv[index];
    auto require_value = [&](const char* flag) -> std::string {
      if (index + 1 >= argc) {
        throw std::runtime_error(std::string(flag) + " requires a value");
      }
      ++index;
      return argv[index];
    };

    if (arg == "--help" || arg == "-h") {
      std::cout
          << "1CatLinkStat NVLink Microbench\n\n"
          << "Options:\n"
          << "  --list               List GPUs and peer-accessible pairs.\n"
          << "  --all-pairs          Benchmark every bidirectional peer pair.\n"
          << "  --src <gpu>          Source GPU index for a one-way test.\n"
          << "  --dst <gpu>          Destination GPU index for a one-way test.\n"
          << "  --size-mib <mib>     Transfer size per iteration in MiB. Default: 256.\n"
          << "  --bytes <bytes>      Transfer size per iteration in bytes.\n"
          << "  --warmup <iters>     Warmup iterations. Default: 20.\n"
          << "  --iters <iters>      Measured iterations. Default: 200.\n"
          << "  --json               Emit JSON instead of plain text.\n";
      std::exit(0);
    } else if (arg == "--list") {
      options.list_only = true;
    } else if (arg == "--all-pairs") {
      options.all_pairs = true;
    } else if (arg == "--json") {
      options.json = true;
    } else if (arg == "--src") {
      options.src = parse_int(require_value("--src"), "--src");
    } else if (arg == "--dst") {
      options.dst = parse_int(require_value("--dst"), "--dst");
    } else if (arg == "--size-mib") {
      options.size_bytes = static_cast<std::size_t>(parse_size_bytes(require_value("--size-mib"), "--size-mib")) *
                           1024ull * 1024ull;
    } else if (arg == "--bytes") {
      options.size_bytes = parse_size_bytes(require_value("--bytes"), "--bytes");
    } else if (arg == "--warmup") {
      options.warmup = parse_int(require_value("--warmup"), "--warmup");
    } else if (arg == "--iters") {
      options.iters = parse_int(require_value("--iters"), "--iters");
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if ((options.src >= 0) != (options.dst >= 0)) {
    throw std::runtime_error("--src and --dst must be used together");
  }
  if (options.size_bytes == 0) {
    throw std::runtime_error("transfer size must be greater than zero");
  }
  if (options.warmup < 0 || options.iters <= 0) {
    throw std::runtime_error("warmup must be >= 0 and iters must be > 0");
  }
  return options;
}

std::vector<DeviceInfo> collect_devices() {
  int device_count = 0;
  cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  std::vector<DeviceInfo> devices;
  devices.reserve(static_cast<std::size_t>(device_count));
  for (int index = 0; index < device_count; ++index) {
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, index), "cudaGetDeviceProperties");
    devices.push_back(DeviceInfo{index, prop.name});
  }
  return devices;
}

bool can_access_peer(int src, int dst) {
  int access = 0;
  cuda_check(cudaDeviceCanAccessPeer(&access, src, dst), "cudaDeviceCanAccessPeer");
  return access == 1;
}

std::vector<std::pair<int, int>> collect_bidirectional_pairs(const std::vector<DeviceInfo>& devices) {
  std::vector<std::pair<int, int>> pairs;
  for (std::size_t left = 0; left < devices.size(); ++left) {
    for (std::size_t right = left + 1; right < devices.size(); ++right) {
      const int src = devices[left].index;
      const int dst = devices[right].index;
      if (can_access_peer(src, dst) && can_access_peer(dst, src)) {
        pairs.emplace_back(src, dst);
      }
    }
  }
  return pairs;
}

BenchResult run_one_way(int src, int dst, const Options& options) {
  void* src_buffer = nullptr;
  void* dst_buffer = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() noexcept {
    if (stop != nullptr) {
      cudaSetDevice(dst);
      cudaEventDestroy(stop);
    }
    if (start != nullptr) {
      cudaSetDevice(dst);
      cudaEventDestroy(start);
    }
    if (stream != nullptr) {
      cudaSetDevice(dst);
      cudaStreamDestroy(stream);
    }
    if (dst_buffer != nullptr) {
      cudaSetDevice(dst);
      cudaFree(dst_buffer);
    }
    if (src_buffer != nullptr) {
      cudaSetDevice(src);
      cudaFree(src_buffer);
    }
  };

  try {
    enable_peer_access(src, dst);
    enable_peer_access(dst, src);

    cuda_check(cudaSetDevice(src), "cudaSetDevice(src)");
    cuda_check(cudaMalloc(&src_buffer, options.size_bytes), "cudaMalloc(src)");
    cuda_check(cudaMemset(src_buffer, 0x5A, options.size_bytes), "cudaMemset(src)");

    cuda_check(cudaSetDevice(dst), "cudaSetDevice(dst)");
    cuda_check(cudaMalloc(&dst_buffer, options.size_bytes), "cudaMalloc(dst)");
    cuda_check(cudaMemset(dst_buffer, 0, options.size_bytes), "cudaMemset(dst)");
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cuda_check(cudaEventCreate(&start), "cudaEventCreate(start)");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      cuda_check(
          cudaMemcpyPeerAsync(dst_buffer, dst, src_buffer, src, options.size_bytes, stream),
          "cudaMemcpyPeerAsync(warmup)");
    }
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(warmup)");

    cuda_check(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < options.iters; ++iteration) {
      cuda_check(
          cudaMemcpyPeerAsync(dst_buffer, dst, src_buffer, src, options.size_bytes, stream),
          "cudaMemcpyPeerAsync(benchmark)");
    }
    cuda_check(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");

    BenchResult result;
    result.src = src;
    result.dst = dst;
    result.size_bytes = options.size_bytes;
    result.warmup = options.warmup;
    result.iters = options.iters;
    result.elapsed_ms = static_cast<double>(elapsed_ms);
    result.total_bytes = static_cast<double>(options.size_bytes) * static_cast<double>(options.iters);
    const double seconds = result.elapsed_ms / 1000.0;
    result.decimal_gb_s = result.total_bytes / seconds / 1'000'000'000.0;
    result.binary_gib_s = result.total_bytes / seconds / static_cast<double>(1ull << 30);

    cleanup();
    return result;
  } catch (...) {
    cleanup();
    throw;
  }
}

void print_list(const std::vector<DeviceInfo>& devices, const std::vector<std::pair<int, int>>& pairs) {
  std::cout << "1CatLinkStat NVLink Microbench\n\n";
  std::cout << "Detected NVIDIA GPUs:\n";
  for (const auto& device : devices) {
    std::cout << "  GPU" << device.index << "  " << device.name << '\n';
  }
  std::cout << "\nBidirectional peer-accessible pairs:\n";
  if (pairs.empty()) {
    std::cout << "  none\n";
    return;
  }
  for (const auto& pair : pairs) {
    std::cout << "  GPU" << pair.first << " <-> GPU" << pair.second << '\n';
  }
}

void print_results_text(const std::vector<DeviceInfo>& devices, const std::vector<BenchResult>& results) {
  std::cout << "1CatLinkStat NVLink Microbench\n\n";
  for (const auto& result : results) {
    std::cout << "GPU" << result.src << " (" << devices[static_cast<std::size_t>(result.src)].name << ")"
              << " -> GPU" << result.dst << " (" << devices[static_cast<std::size_t>(result.dst)].name << ")\n";
    std::cout << "  chunk:   " << format_size_mib(result.size_bytes) << '\n';
    std::cout << "  warmup:  " << result.warmup << '\n';
    std::cout << "  iters:   " << result.iters << '\n';
    std::cout << "  time:    " << std::fixed << std::setprecision(3) << result.elapsed_ms << " ms\n";
    std::cout << "  rate:    " << std::fixed << std::setprecision(3) << result.decimal_gb_s << " GB/s"
              << " (" << result.binary_gib_s << " GiB/s)\n";
    std::cout << "  bytes:   " << std::fixed << std::setprecision(0) << result.total_bytes << "\n\n";
  }

  if (results.size() == 2 && results[0].src == results[1].dst && results[0].dst == results[1].src) {
    const double aggregate = results[0].decimal_gb_s + results[1].decimal_gb_s;
    std::cout << "Aggregate sequential throughput: " << std::fixed << std::setprecision(3) << aggregate << " GB/s\n";
  }
}

void print_results_json(const std::vector<DeviceInfo>& devices, const std::vector<BenchResult>& results) {
  std::cout << "{\n";
  std::cout << "  \"tool\": \"1CatLinkStat NVLink Microbench\",\n";
  std::cout << "  \"results\": [\n";
  for (std::size_t index = 0; index < results.size(); ++index) {
    const auto& result = results[index];
    std::cout << "    {\n";
    std::cout << "      \"src\": " << result.src << ",\n";
    std::cout << "      \"dst\": " << result.dst << ",\n";
    std::cout << "      \"src_name\": \"" << json_escape(devices[static_cast<std::size_t>(result.src)].name) << "\",\n";
    std::cout << "      \"dst_name\": \"" << json_escape(devices[static_cast<std::size_t>(result.dst)].name) << "\",\n";
    std::cout << "      \"size_bytes\": " << result.size_bytes << ",\n";
    std::cout << "      \"warmup\": " << result.warmup << ",\n";
    std::cout << "      \"iters\": " << result.iters << ",\n";
    std::cout << "      \"elapsed_ms\": " << std::fixed << std::setprecision(6) << result.elapsed_ms << ",\n";
    std::cout << "      \"total_bytes\": " << std::fixed << std::setprecision(0) << result.total_bytes << ",\n";
    std::cout << "      \"decimal_gb_s\": " << std::fixed << std::setprecision(6) << result.decimal_gb_s << ",\n";
    std::cout << "      \"binary_gib_s\": " << std::fixed << std::setprecision(6) << result.binary_gib_s << '\n';
    std::cout << "    }" << (index + 1 == results.size() ? '\n' : ',') ;
  }
  std::cout << "  ]\n";
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    cuda_check(cudaFree(0), "cudaFree(0)");
    const Options options = parse_options(argc, argv);
    const std::vector<DeviceInfo> devices = collect_devices();
    if (devices.size() < 2) {
      throw std::runtime_error("at least two NVIDIA GPUs are required for the microbenchmark");
    }

    const std::vector<std::pair<int, int>> pairs = collect_bidirectional_pairs(devices);
    if (options.list_only) {
      print_list(devices, pairs);
      return 0;
    }

    std::vector<BenchResult> results;
    if (options.src >= 0 && options.dst >= 0) {
      if (options.src >= static_cast<int>(devices.size()) || options.dst >= static_cast<int>(devices.size())) {
        throw std::runtime_error("GPU index out of range");
      }
      if (!can_access_peer(options.src, options.dst)) {
        throw std::runtime_error("selected GPUs do not have peer access in the requested direction");
      }
      results.push_back(run_one_way(options.src, options.dst, options));
    } else if (options.all_pairs) {
      if (pairs.empty()) {
        throw std::runtime_error("no bidirectional peer-accessible GPU pairs were found");
      }
      for (const auto& pair : pairs) {
        results.push_back(run_one_way(pair.first, pair.second, options));
        results.push_back(run_one_way(pair.second, pair.first, options));
      }
    } else {
      if (pairs.empty()) {
        throw std::runtime_error("no bidirectional peer-accessible GPU pairs were found");
      }
      const auto pair = pairs.front();
      results.push_back(run_one_way(pair.first, pair.second, options));
      results.push_back(run_one_way(pair.second, pair.first, options));
    }

    if (options.json) {
      print_results_json(devices, results);
    } else {
      print_results_text(devices, results);
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "1CatLinkStat-bench: " << exc.what() << '\n';
    return 1;
  }
}
