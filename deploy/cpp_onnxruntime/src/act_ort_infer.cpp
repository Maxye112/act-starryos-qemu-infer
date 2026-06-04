#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace {

constexpr int kImageW = 224;
constexpr int kImageH = 224;
constexpr int kImageC = 3;

struct ActParams {
  std::vector<float> state_q01;
  std::vector<float> state_q99;
  std::vector<float> action_q01;
  std::vector<float> action_q99;
  std::vector<float> latent;
  std::array<float, 3> image_mean{0.485f, 0.456f, 0.406f};
  std::array<float, 3> image_std{0.229f, 0.224f, 0.225f};
  int state_dim = 2;
  int latent_dim = 32;
  int action_chunk_size = 8;
  int action_dim = 3;
};

struct Options {
  std::string model_path =
      "artifacts/onnx_quant/balancedcalib_static_qdq_conv_matmul_keep_action_head_fp16.onnx";
  std::string image_path = "output/dataset/videos/observation.images.fpv/chunk-000/frame_000000.jpg";
  std::string params_path = "deploy/cpp_onnxruntime/config/act_params.json";
  std::string eval_manifest_path;
  std::string dataset_root = "output/dataset";
  std::array<float, 2> state{0.0f, 0.0f};
  int threads = 0;
  int runs = 1;
  int warmup = 1;
  int eval_limit = 0;
  int eval_progress = 100;
  float deadband = 0.01f;
  float eval_turn_eps = 1e-6f;
  bool use_arena = true;
  bool use_mem_pattern = true;
  bool use_spinning = false;
  bool print_chunk = false;
  bool print_mem = true;
  bool track_allocator = false;
  bool eval_feedback_state = true;
};

struct MemStats {
  long vm_rss_kb = -1;
  long vm_hwm_kb = -1;
  long vm_size_kb = -1;
  long statm_rss_kb = -1;
  long statm_size_kb = -1;
  long ru_maxrss_kb = -1;
};

struct StageStats {
  std::string name;
  double ms = 0.0;
  MemStats mem;
};

struct AllocStats {
  std::atomic<size_t> current_bytes{0};
  std::atomic<size_t> peak_bytes{0};
  std::atomic<size_t> total_allocated_bytes{0};
  std::atomic<size_t> alloc_count{0};
  std::atomic<size_t> free_count{0};
  std::atomic<size_t> max_alloc_bytes{0};
};

struct TrackingAllocator {
  OrtAllocator ort{};
  OrtMemoryInfo* memory_info = nullptr;
  AllocStats stats;
};

struct AllocHeader {
  void* raw;
  size_t size;
};

struct DatasetSample {
  int index = 0;
  int episode_index = -1;
  std::string image_path;
  std::array<float, 2> state{0.0f, 0.0f};
  std::array<float, 3> action{0.0f, 0.0f, 0.0f};
};

enum class TurnLabel {
  Straight,
  Left,
  Right,
};

struct TurnThresholdStats {
  float eps = 0.0f;
  int gt_left = 0;
  int gt_right = 0;
  int gt_straight_ignored = 0;
  int turn_total = 0;
  int turn_correct = 0;
  int left_correct = 0;
  int right_correct = 0;
  int turn_pred_straight = 0;
  int turn_pred_opposite = 0;
  int turn_pred_nonstraight = 0;
  int turn_correct_ignore_pred_straight = 0;
};

[[noreturn]] void Die(const std::string& message) {
  std::cerr << "error: " << message << "\n";
  std::exit(1);
}

std::string ReadFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) Die("failed to open " + path);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<float> ParseNumberArray(const std::string& text, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = text.find(needle);
  if (pos == std::string::npos) Die("missing JSON key: " + key);
  pos = text.find('[', pos);
  if (pos == std::string::npos) Die("missing JSON array for key: " + key);
  const size_t end = text.find(']', pos);
  if (end == std::string::npos) Die("unterminated JSON array for key: " + key);

  std::vector<float> values;
  const char* p = text.c_str() + pos + 1;
  const char* last = text.c_str() + end;
  while (p < last) {
    while (p < last && (std::isspace(static_cast<unsigned char>(*p)) || *p == ',')) ++p;
    if (p >= last) break;
    char* next = nullptr;
    const float value = std::strtof(p, &next);
    if (next == p) Die("invalid numeric value in key: " + key);
    values.push_back(value);
    p = next;
  }
  return values;
}

int ParseIntValue(const std::string& text, const std::string& key, int fallback) {
  const std::string needle = "\"" + key + "\"";
  size_t pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  char* next = nullptr;
  const long value = std::strtol(text.c_str() + pos + 1, &next, 10);
  return next == text.c_str() + pos + 1 ? fallback : static_cast<int>(value);
}

ActParams LoadParams(const std::string& path) {
  const std::string text = ReadFile(path);
  ActParams p;
  p.state_q01 = ParseNumberArray(text, "state_q01");
  p.state_q99 = ParseNumberArray(text, "state_q99");
  p.action_q01 = ParseNumberArray(text, "action_q01");
  p.action_q99 = ParseNumberArray(text, "action_q99");
  p.latent = ParseNumberArray(text, "latent");
  auto mean = ParseNumberArray(text, "image_mean");
  auto std = ParseNumberArray(text, "image_std");
  for (int i = 0; i < 3 && i < static_cast<int>(mean.size()); ++i) p.image_mean[i] = mean[i];
  for (int i = 0; i < 3 && i < static_cast<int>(std.size()); ++i) p.image_std[i] = std[i];
  p.state_dim = ParseIntValue(text, "state_dim", p.state_dim);
  p.latent_dim = ParseIntValue(text, "latent_dim", p.latent_dim);
  p.action_chunk_size = ParseIntValue(text, "action_chunk_size", p.action_chunk_size);
  p.action_dim = ParseIntValue(text, "action_dim", p.action_dim);

  if (static_cast<int>(p.state_q01.size()) != p.state_dim ||
      static_cast<int>(p.state_q99.size()) != p.state_dim ||
      static_cast<int>(p.latent.size()) != p.latent_dim ||
      static_cast<int>(p.action_q01.size()) != p.action_dim ||
      static_cast<int>(p.action_q99.size()) != p.action_dim) {
    Die("invalid parameter dimensions in " + path);
  }
  return p;
}

void PrintUsage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0 << " [options]\n"
      << "  --model PATH       ONNX model path\n"
      << "  --image PATH       Input RGB image path\n"
      << "  --params PATH      JSON params path\n"
      << "  --eval-manifest PATH  CSV manifest for whole-dataset evaluation\n"
      << "  --dataset-root PATH   Root used to resolve relative image paths in manifest\n"
      << "  --state L R        Raw state [left_vel right_vel]\n"
      << "  --threads N        Intra-op/inter-op threads, 0 = hardware_concurrency\n"
      << "  --runs N           Timed inference runs\n"
      << "  --warmup N         Warmup runs\n"
      << "  --deadband X       Straight threshold for left-right diff\n"
      << "  --eval-turn-eps X  GT/pred turn threshold for dataset evaluation\n"
      << "  --eval-limit N     Limit dataset evaluation samples, 0 = all\n"
      << "  --eval-progress N  Print progress every N samples, 0 = silent\n"
      << "  --eval-open-loop-state  Use GT state for every eval frame instead of previous prediction\n"
      << "  --no-arena         Disable ORT CPU arena\n"
      << "  --no-mem-pattern   Disable ORT memory pattern\n"
      << "  --spin             Enable ORT thread spinning\n"
      << "  --track-allocator  Register a CPU allocator that tracks ORT heap allocations\n"
      << "  --no-mem           Do not print /proc/self/status memory stats\n"
      << "  --print-chunk      Print all action chunk steps\n";
}

Options ParseArgs(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> const char* {
      if (i + 1 >= argc) Die("missing value for " + name);
      return argv[++i];
    };
    if (arg == "--model") opt.model_path = require_value(arg);
    else if (arg == "--image") opt.image_path = require_value(arg);
    else if (arg == "--params") opt.params_path = require_value(arg);
    else if (arg == "--eval-manifest") opt.eval_manifest_path = require_value(arg);
    else if (arg == "--dataset-root") opt.dataset_root = require_value(arg);
    else if (arg == "--threads") opt.threads = std::max(0, std::atoi(require_value(arg)));
    else if (arg == "--runs") opt.runs = std::max(1, std::atoi(require_value(arg)));
    else if (arg == "--warmup") opt.warmup = std::max(0, std::atoi(require_value(arg)));
    else if (arg == "--eval-limit") opt.eval_limit = std::max(0, std::atoi(require_value(arg)));
    else if (arg == "--eval-progress") opt.eval_progress = std::max(0, std::atoi(require_value(arg)));
    else if (arg == "--deadband") opt.deadband = std::strtof(require_value(arg), nullptr);
    else if (arg == "--eval-turn-eps") opt.eval_turn_eps = std::strtof(require_value(arg), nullptr);
    else if (arg == "--state") {
      opt.state[0] = std::strtof(require_value(arg), nullptr);
      opt.state[1] = std::strtof(require_value(arg), nullptr);
    } else if (arg == "--no-arena") opt.use_arena = false;
    else if (arg == "--no-mem-pattern") opt.use_mem_pattern = false;
    else if (arg == "--spin") opt.use_spinning = true;
    else if (arg == "--track-allocator") opt.track_allocator = true;
    else if (arg == "--eval-open-loop-state") opt.eval_feedback_state = false;
    else if (arg == "--no-mem") opt.print_mem = false;
    else if (arg == "--print-chunk") opt.print_chunk = true;
    else if (arg == "-h" || arg == "--help") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      Die("unknown argument: " + arg);
    }
  }
  return opt;
}

long ParseStatusKb(const std::string& text, const std::string& key) {
  const size_t pos = text.find(key);
  if (pos == std::string::npos) return -1;
  const char* p = text.c_str() + pos + key.size();
  while (*p && !std::isdigit(static_cast<unsigned char>(*p))) ++p;
  char* next = nullptr;
  const long value = std::strtol(p, &next, 10);
  return next == p ? -1 : value;
}

MemStats ReadMemStats() {
  MemStats stats;
  {
    std::ifstream file("/proc/self/status");
    if (file) {
      const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
      stats.vm_rss_kb = ParseStatusKb(text, "VmRSS:");
      stats.vm_hwm_kb = ParseStatusKb(text, "VmHWM:");
      stats.vm_size_kb = ParseStatusKb(text, "VmSize:");
    }
  }
  {
    std::ifstream file("/proc/self/statm");
    long pages_size = -1;
    long pages_rss = -1;
    if (file >> pages_size >> pages_rss) {
      const long page_kb = static_cast<long>(sysconf(_SC_PAGESIZE)) / 1024;
      if (page_kb > 0) {
        stats.statm_size_kb = pages_size * page_kb;
        stats.statm_rss_kb = pages_rss * page_kb;
      }
    }
  }
  {
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      stats.ru_maxrss_kb = usage.ru_maxrss;
    }
  }
  return stats;
}

void PrintMemStats(const std::string& label) {
  const MemStats stats = ReadMemStats();
  if (stats.vm_rss_kb < 0 && stats.vm_hwm_kb < 0 && stats.vm_size_kb < 0 &&
      stats.statm_rss_kb < 0 && stats.statm_size_kb < 0 && stats.ru_maxrss_kb < 0) {
    return;
  }
  std::cout << "mem_" << label << "_kb:"
            << " VmRSS=" << stats.vm_rss_kb
            << " VmHWM=" << stats.vm_hwm_kb
            << " VmSize=" << stats.vm_size_kb
            << " statm_RSS=" << stats.statm_rss_kb
            << " statm_Size=" << stats.statm_size_kb
            << " ru_maxrss=" << stats.ru_maxrss_kb << "\n";
}

void PrintStageStats(const std::vector<StageStats>& stages) {
  std::cout << "stage_profile:\n";
  for (const auto& stage : stages) {
    std::cout << "  " << stage.name << "_ms: " << stage.ms
              << "  VmRSS_kb=" << stage.mem.vm_rss_kb
              << "  VmHWM_kb=" << stage.mem.vm_hwm_kb
              << "  statm_RSS_kb=" << stage.mem.statm_rss_kb
              << "  ru_maxrss_kb=" << stage.mem.ru_maxrss_kb << "\n";
  }
}

void UpdatePeak(std::atomic<size_t>& peak, size_t value) {
  size_t old = peak.load(std::memory_order_relaxed);
  while (value > old && !peak.compare_exchange_weak(old, value, std::memory_order_relaxed)) {
  }
}

void* ORT_API_CALL TrackingAlloc(OrtAllocator* this_, size_t size) {
  if (size == 0) return nullptr;
  auto* allocator = reinterpret_cast<TrackingAllocator*>(this_);
  constexpr size_t kAlignment = 64;
  const size_t total = sizeof(AllocHeader) + size + kAlignment - 1;
  void* raw = std::malloc(total);
  if (!raw) return nullptr;
  const auto start = reinterpret_cast<uintptr_t>(raw) + sizeof(AllocHeader);
  const auto aligned = (start + kAlignment - 1) & ~(static_cast<uintptr_t>(kAlignment) - 1);
  auto* header = reinterpret_cast<AllocHeader*>(aligned - sizeof(AllocHeader));
  header->raw = raw;
  header->size = size;

  const size_t current = allocator->stats.current_bytes.fetch_add(size, std::memory_order_relaxed) + size;
  allocator->stats.total_allocated_bytes.fetch_add(size, std::memory_order_relaxed);
  allocator->stats.alloc_count.fetch_add(1, std::memory_order_relaxed);
  UpdatePeak(allocator->stats.peak_bytes, current);
  UpdatePeak(allocator->stats.max_alloc_bytes, size);
  return reinterpret_cast<void*>(aligned);
}

void ORT_API_CALL TrackingFree(OrtAllocator* this_, void* p) {
  if (!p) return;
  auto* allocator = reinterpret_cast<TrackingAllocator*>(this_);
  auto* raw = static_cast<AllocHeader*>(p) - 1;
  allocator->stats.current_bytes.fetch_sub(raw->size, std::memory_order_relaxed);
  allocator->stats.free_count.fetch_add(1, std::memory_order_relaxed);
  std::free(raw->raw);
}

const OrtMemoryInfo* ORT_API_CALL TrackingInfo(const OrtAllocator* this_) {
  auto* allocator = reinterpret_cast<const TrackingAllocator*>(this_);
  return allocator->memory_info;
}

void InitTrackingAllocator(TrackingAllocator& allocator) {
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault,
                                                      &allocator.memory_info));
  allocator.ort.version = ORT_API_VERSION;
  allocator.ort.Alloc = TrackingAlloc;
  allocator.ort.Free = TrackingFree;
  allocator.ort.Info = TrackingInfo;
  allocator.ort.Reserve = TrackingAlloc;
  allocator.ort.GetStats = nullptr;
  allocator.ort.AllocOnStream = nullptr;
  allocator.ort.Shrink = nullptr;
}

void ReleaseTrackingAllocator(TrackingAllocator& allocator) {
  if (allocator.memory_info) {
    Ort::GetApi().ReleaseMemoryInfo(allocator.memory_info);
    allocator.memory_info = nullptr;
  }
}

void PrintAllocatorStats(const std::string& label, const TrackingAllocator* allocator) {
  if (!allocator) return;
  const auto& s = allocator->stats;
  std::cout << "alloc_" << label << "_bytes:"
            << " current=" << s.current_bytes.load(std::memory_order_relaxed)
            << " peak=" << s.peak_bytes.load(std::memory_order_relaxed)
            << " total_allocated=" << s.total_allocated_bytes.load(std::memory_order_relaxed)
            << " alloc_count=" << s.alloc_count.load(std::memory_order_relaxed)
            << " free_count=" << s.free_count.load(std::memory_order_relaxed)
            << " max_alloc=" << s.max_alloc_bytes.load(std::memory_order_relaxed) << "\n";
}

std::vector<float> LoadAndPreprocessImage(const std::string& path, const ActParams& params) {
  int src_w = 0, src_h = 0, src_c = 0;
  stbi_uc* raw = stbi_load(path.c_str(), &src_w, &src_h, &src_c, 3);
  if (!raw) Die("failed to decode image: " + path);

  std::vector<float> nchw(kImageC * kImageH * kImageW);
  const float scale_x = static_cast<float>(src_w) / kImageW;
  const float scale_y = static_cast<float>(src_h) / kImageH;

  for (int y = 0; y < kImageH; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
    const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, src_h - 1);
    const int y1 = std::min(y0 + 1, src_h - 1);
    const float wy = src_y - y0;
    for (int x = 0; x < kImageW; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
      const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, src_w - 1);
      const int x1 = std::min(x0 + 1, src_w - 1);
      const float wx = src_x - x0;

      const stbi_uc* p00 = raw + (y0 * src_w + x0) * 3;
      const stbi_uc* p01 = raw + (y0 * src_w + x1) * 3;
      const stbi_uc* p10 = raw + (y1 * src_w + x0) * 3;
      const stbi_uc* p11 = raw + (y1 * src_w + x1) * 3;

      for (int c = 0; c < 3; ++c) {
        const float top = p00[c] * (1.0f - wx) + p01[c] * wx;
        const float bot = p10[c] * (1.0f - wx) + p11[c] * wx;
        const float pixel = (top * (1.0f - wy) + bot * wy) / 255.0f;
        nchw[c * kImageH * kImageW + y * kImageW + x] =
            (pixel - params.image_mean[c]) / params.image_std[c];
      }
    }
  }

  stbi_image_free(raw);
  return nchw;
}

std::array<float, 2> NormalizeState(const std::array<float, 2>& state, const ActParams& params) {
  std::array<float, 2> out{};
  for (int i = 0; i < 2; ++i) {
    float denom = params.state_q99[i] - params.state_q01[i];
    if (denom == 0.0f) denom = 1e-8f;
    out[i] = 2.0f * (state[i] - params.state_q01[i]) / denom - 1.0f;
  }
  return out;
}

std::vector<float> DenormalizeAction(const float* action, int count, const ActParams& params) {
  std::vector<float> out(count);
  for (int i = 0; i < count; ++i) {
    const int d = i % params.action_dim;
    float denom = params.action_q99[d] - params.action_q01[d];
    if (denom == 0.0f) denom = 1e-8f;
    out[i] = (action[i] + 1.0f) * 0.5f * denom + params.action_q01[d];
  }
  return out;
}

const char* TurnDecision(float left, float right, float deadband) {
  const float diff = left - right;
  if (std::abs(diff) < deadband) return "straight";
  return diff > 0.0f ? "right" : "left";
}

TurnLabel LabelFromDiff(float diff, float eps) {
  if (diff > eps) return TurnLabel::Right;
  if (diff < -eps) return TurnLabel::Left;
  return TurnLabel::Straight;
}

const char* LabelName(TurnLabel label) {
  switch (label) {
    case TurnLabel::Left:
      return "left";
    case TurnLabel::Right:
      return "right";
    case TurnLabel::Straight:
    default:
      return "straight";
  }
}

void UpdateTurnThresholdStats(TurnThresholdStats& stats, float gt_diff, float pred_diff) {
  const TurnLabel gt_label = LabelFromDiff(gt_diff, stats.eps);
  const TurnLabel pred_label = LabelFromDiff(pred_diff, stats.eps);
  if (gt_label == TurnLabel::Straight) {
    ++stats.gt_straight_ignored;
    return;
  }

  ++stats.turn_total;
  if (gt_label == TurnLabel::Left) ++stats.gt_left;
  if (gt_label == TurnLabel::Right) ++stats.gt_right;

  if (pred_label != TurnLabel::Straight) {
    ++stats.turn_pred_nonstraight;
    if (gt_label == pred_label) ++stats.turn_correct_ignore_pred_straight;
  }

  if (gt_label == pred_label) {
    ++stats.turn_correct;
    if (gt_label == TurnLabel::Left) ++stats.left_correct;
    if (gt_label == TurnLabel::Right) ++stats.right_correct;
  } else if (pred_label == TurnLabel::Straight) {
    ++stats.turn_pred_straight;
  } else {
    ++stats.turn_pred_opposite;
  }
}

std::vector<std::string> SplitCsvLine(const std::string& line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (char ch : line) {
    if (ch == '"') {
      quoted = !quoted;
    } else if (ch == ',' && !quoted) {
      fields.push_back(field);
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(field);
  return fields;
}

float ParseFloatField(const std::string& value, const std::string& name, int line_no) {
  char* end = nullptr;
  const float parsed = std::strtof(value.c_str(), &end);
  if (end == value.c_str()) {
    Die("invalid float field " + name + " at manifest line " + std::to_string(line_no));
  }
  return parsed;
}

std::string JoinPath(const std::string& root, const std::string& path) {
  if (path.empty() || path[0] == '/') return path;
  if (root.empty() || root == ".") return path;
  if (root.back() == '/') return root + path;
  return root + "/" + path;
}

std::vector<DatasetSample> LoadEvalManifest(const std::string& path, const std::string& dataset_root) {
  std::ifstream file(path);
  if (!file) Die("failed to open eval manifest: " + path);

  std::vector<DatasetSample> samples;
  std::string line;
  int line_no = 0;
  bool has_episode_index = false;
  while (std::getline(file, line)) {
    ++line_no;
    if (line.empty()) continue;
    if (line_no == 1 && line.find("image_path") != std::string::npos) {
      has_episode_index = line.find("episode_index") != std::string::npos;
      continue;
    }

    const std::vector<std::string> fields = SplitCsvLine(line);
    const size_t expected = has_episode_index ? 8 : 7;
    if (fields.size() < expected) {
      Die("expected at least " + std::to_string(expected) +
          " CSV fields at manifest line " + std::to_string(line_no));
    }

    DatasetSample sample;
    sample.index = static_cast<int>(ParseFloatField(fields[0], "index", line_no));
    size_t offset = 1;
    if (has_episode_index) {
      sample.episode_index = static_cast<int>(ParseFloatField(fields[1], "episode_index", line_no));
      offset = 2;
    }
    sample.image_path = JoinPath(dataset_root, fields[offset + 0]);
    sample.state[0] = ParseFloatField(fields[offset + 1], "state_left", line_no);
    sample.state[1] = ParseFloatField(fields[offset + 2], "state_right", line_no);
    sample.action[0] = ParseFloatField(fields[offset + 3], "gt_left", line_no);
    sample.action[1] = ParseFloatField(fields[offset + 4], "gt_right", line_no);
    sample.action[2] = ParseFloatField(fields[offset + 5], "gt_gripper", line_no);
    samples.push_back(std::move(sample));
  }

  if (samples.empty()) Die("eval manifest has no samples: " + path);
  return samples;
}

std::vector<float> RunOneFrame(
    Ort::Session& session,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const ActParams& params,
    const DatasetSample& sample,
    Ort::MemoryInfo& memory_info,
    Ort::RunOptions& run_options) {
  std::vector<float> image = LoadAndPreprocessImage(sample.image_path, params);
  const std::array<float, 2> state_norm = NormalizeState(sample.state, params);
  std::vector<float> state_vec = {state_norm[0], state_norm[1]};
  std::vector<float> latent = params.latent;

  std::array<int64_t, 4> image_shape{1, 3, kImageH, kImageW};
  std::array<int64_t, 2> state_shape{1, params.state_dim};
  std::array<int64_t, 2> latent_shape{1, params.latent_dim};

  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, image.data(), image.size(), image_shape.data(), image_shape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, state_vec.data(), state_vec.size(), state_shape.data(), state_shape.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, latent.data(), latent.size(), latent_shape.data(), latent_shape.size()));

  auto outputs = session.Run(run_options, input_names.data(), inputs.data(), inputs.size(),
                             output_names.data(), output_names.size());
  const float* action_norm = outputs[0].GetTensorData<float>();
  auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
  return DenormalizeAction(action_norm, static_cast<int>(out_info.GetElementCount()), params);
}

int EvaluateManifest(
    Ort::Session& session,
    const std::vector<const char*>& input_names,
    const std::vector<const char*>& output_names,
    const Options& opt,
    const ActParams& params,
    TrackingAllocator* tracking_allocator_ptr) {
  std::vector<DatasetSample> samples = LoadEvalManifest(opt.eval_manifest_path, opt.dataset_root);
  if (opt.eval_limit > 0 && opt.eval_limit < static_cast<int>(samples.size())) {
    samples.resize(static_cast<size_t>(opt.eval_limit));
  }

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::RunOptions run_options;

  double diff_abs_sum = 0.0;
  double diff_sq_sum = 0.0;
  double diff_max_abs = 0.0;
  double left_abs_sum = 0.0;
  double right_abs_sum = 0.0;
  int diff_acc_005 = 0;
  int diff_acc_010 = 0;
  int diff_acc_020 = 0;
  int gt_left_count = 0;
  int gt_right_count = 0;
  int gt_straight_ignored = 0;
  int turn_total = 0;
  int turn_correct = 0;
  int left_correct = 0;
  int right_correct = 0;
  int turn_pred_straight = 0;
  int turn_pred_opposite = 0;
  int pred_left_count = 0;
  int pred_right_count = 0;
  int pred_straight_count = 0;
  int episode_resets = 0;
  int last_episode = std::numeric_limits<int>::min();
  std::array<float, 2> feedback_state{0.0f, 0.0f};
  std::vector<TurnThresholdStats> threshold_stats = {
      {1e-6f},
      {0.005f},
      {0.01f},
      {0.02f},
  };
  bool has_requested_eps = false;
  for (const auto& stats : threshold_stats) {
    if (std::abs(stats.eps - opt.eval_turn_eps) < 1e-9f) has_requested_eps = true;
  }
  if (!has_requested_eps) threshold_stats.push_back({opt.eval_turn_eps});

  const auto start = std::chrono::steady_clock::now();
  for (size_t i = 0; i < samples.size(); ++i) {
    const DatasetSample& sample = samples[i];
    DatasetSample infer_sample = sample;
    const bool reset_state =
        i == 0 || (sample.episode_index >= 0 && sample.episode_index != last_episode);
    if (reset_state) {
      feedback_state = sample.state;
      last_episode = sample.episode_index;
      ++episode_resets;
    }
    if (opt.eval_feedback_state) {
      infer_sample.state = feedback_state;
    }

    std::vector<float> pred = RunOneFrame(session, input_names, output_names, params, infer_sample,
                                          memory_info, run_options);
    if (pred.size() < 2) Die("model output has fewer than 2 action values");

    const float pred_left = pred[0];
    const float pred_right = pred[1];
    const float gt_left = sample.action[0];
    const float gt_right = sample.action[1];
    const float pred_diff = pred_left - pred_right;
    const float gt_diff = gt_left - gt_right;
    const float diff_err = pred_diff - gt_diff;
    const double abs_diff_err = std::abs(diff_err);
    for (auto& stats : threshold_stats) {
      UpdateTurnThresholdStats(stats, gt_diff, pred_diff);
    }

    diff_abs_sum += abs_diff_err;
    diff_sq_sum += static_cast<double>(diff_err) * diff_err;
    diff_max_abs = std::max(diff_max_abs, abs_diff_err);
    left_abs_sum += std::abs(pred_left - gt_left);
    right_abs_sum += std::abs(pred_right - gt_right);
    if (abs_diff_err <= 0.005) ++diff_acc_005;
    if (abs_diff_err <= 0.010) ++diff_acc_010;
    if (abs_diff_err <= 0.020) ++diff_acc_020;

    const TurnLabel gt_label = LabelFromDiff(gt_diff, opt.eval_turn_eps);
    const TurnLabel pred_label = LabelFromDiff(pred_diff, opt.eval_turn_eps);
    if (pred_label == TurnLabel::Left) ++pred_left_count;
    else if (pred_label == TurnLabel::Right) ++pred_right_count;
    else ++pred_straight_count;

    if (gt_label == TurnLabel::Straight) {
      ++gt_straight_ignored;
    } else {
      ++turn_total;
      if (gt_label == TurnLabel::Left) ++gt_left_count;
      if (gt_label == TurnLabel::Right) ++gt_right_count;
      if (gt_label == pred_label) {
        ++turn_correct;
        if (gt_label == TurnLabel::Left) ++left_correct;
        if (gt_label == TurnLabel::Right) ++right_correct;
      } else if (pred_label == TurnLabel::Straight) {
        ++turn_pred_straight;
      } else {
        ++turn_pred_opposite;
      }
    }

    feedback_state = {pred_left, pred_right};

    if (opt.eval_progress > 0 && (i + 1) % static_cast<size_t>(opt.eval_progress) == 0) {
      std::cout << "eval_progress: " << (i + 1) << "/" << samples.size() << "\n";
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const double elapsed_ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
  const double n = static_cast<double>(samples.size());

  if (opt.print_mem) PrintMemStats("after_eval");
  PrintAllocatorStats("after_eval", tracking_allocator_ptr);
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "dataset_eval:\n";
  std::cout << "  samples: " << samples.size() << "\n";
  std::cout << "  eval_state_mode: " << (opt.eval_feedback_state ? "feedback_predicted_state" : "open_loop_gt_state") << "\n";
  std::cout << "  episode_resets: " << episode_resets << "\n";
  std::cout << "  elapsed_ms: " << elapsed_ms << "\n";
  std::cout << "  avg_frame_ms: " << (elapsed_ms / n) << "\n";
  std::cout << "  diff_mae: " << (diff_abs_sum / n) << "\n";
  std::cout << "  diff_rmse: " << std::sqrt(diff_sq_sum / n) << "\n";
  std::cout << "  diff_max_abs: " << diff_max_abs << "\n";
  std::cout << "  left_mae: " << (left_abs_sum / n) << "\n";
  std::cout << "  right_mae: " << (right_abs_sum / n) << "\n";
  std::cout << "  diff_acc_abs_le_0.005: " << (static_cast<double>(diff_acc_005) / n) << "\n";
  std::cout << "  diff_acc_abs_le_0.010: " << (static_cast<double>(diff_acc_010) / n) << "\n";
  std::cout << "  diff_acc_abs_le_0.020: " << (static_cast<double>(diff_acc_020) / n) << "\n";
  std::cout << "  eval_turn_eps: " << opt.eval_turn_eps << "\n";
  std::cout << "  gt_left: " << gt_left_count << "\n";
  std::cout << "  gt_right: " << gt_right_count << "\n";
  std::cout << "  gt_straight_ignored: " << gt_straight_ignored << "\n";
  std::cout << "  pred_left_all: " << pred_left_count << "\n";
  std::cout << "  pred_right_all: " << pred_right_count << "\n";
  std::cout << "  pred_straight_all: " << pred_straight_count << "\n";
  std::cout << "  turn_total: " << turn_total << "\n";
  std::cout << "  turn_correct: " << turn_correct << "\n";
  std::cout << "  turn_accuracy_ignore_gt_straight: "
            << (turn_total > 0 ? static_cast<double>(turn_correct) / turn_total : 0.0) << "\n";
  std::cout << "  left_turn_accuracy: "
            << (gt_left_count > 0 ? static_cast<double>(left_correct) / gt_left_count : 0.0) << "\n";
  std::cout << "  right_turn_accuracy: "
            << (gt_right_count > 0 ? static_cast<double>(right_correct) / gt_right_count : 0.0) << "\n";
  std::cout << "  turn_pred_straight: " << turn_pred_straight << "\n";
  std::cout << "  turn_pred_opposite: " << turn_pred_opposite << "\n";
  std::cout << "turn_thresholds:\n";
  for (const auto& stats : threshold_stats) {
    const double turn_acc =
        stats.turn_total > 0 ? static_cast<double>(stats.turn_correct) / stats.turn_total : 0.0;
    const double left_acc =
        stats.gt_left > 0 ? static_cast<double>(stats.left_correct) / stats.gt_left : 0.0;
    const double right_acc =
        stats.gt_right > 0 ? static_cast<double>(stats.right_correct) / stats.gt_right : 0.0;
    const double pred_turn_coverage =
        stats.turn_total > 0 ? static_cast<double>(stats.turn_pred_nonstraight) / stats.turn_total : 0.0;
    const double ignore_pred_straight_acc =
        stats.turn_pred_nonstraight > 0
            ? static_cast<double>(stats.turn_correct_ignore_pred_straight) / stats.turn_pred_nonstraight
            : 0.0;
    std::cout << "  eps=" << stats.eps
              << " gt_left=" << stats.gt_left
              << " gt_right=" << stats.gt_right
              << " gt_straight_ignored=" << stats.gt_straight_ignored
              << " turn_total=" << stats.turn_total
              << " turn_correct=" << stats.turn_correct
              << " turn_accuracy=" << turn_acc
              << " left_accuracy=" << left_acc
              << " right_accuracy=" << right_acc
              << " pred_turn_coverage_on_gt_turn=" << pred_turn_coverage
              << " ignore_pred_straight_accuracy=" << ignore_pred_straight_acc
              << " turn_pred_straight=" << stats.turn_pred_straight
              << " turn_pred_opposite=" << stats.turn_pred_opposite << "\n";
  }
  return 0;
}

int ThreadCount(const Options& opt) {
  if (opt.threads > 0) return opt.threads;
  const unsigned hw = std::thread::hardware_concurrency();
  return hw == 0 ? 1 : static_cast<int>(hw);
}

template <typename T>
std::vector<int64_t> ShapeOf(const Ort::TensorTypeAndShapeInfo& info) {
  (void)sizeof(T);
  return info.GetShape();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<StageStats> stages;
    auto measure = [&](const std::string& name, auto&& fn) -> decltype(fn()) {
      const auto start = std::chrono::steady_clock::now();
      if constexpr (std::is_void_v<decltype(fn())>) {
        fn();
        const auto end = std::chrono::steady_clock::now();
        stages.push_back({name, std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(),
                          ReadMemStats()});
      } else {
        auto result = fn();
        const auto end = std::chrono::steady_clock::now();
        stages.push_back({name, std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(),
                          ReadMemStats()});
        return result;
      }
    };

    const Options opt = ParseArgs(argc, argv);
    const ActParams params = measure("load_params", [&] { return LoadParams(opt.params_path); });
    if (opt.print_mem) PrintMemStats("after_params");

    const int threads = std::max(1, ThreadCount(opt));
    Ort::Env env = measure("create_env", [&] { return Ort::Env(ORT_LOGGING_LEVEL_WARNING, "act_ort_infer"); });
    TrackingAllocator tracking_allocator;
    TrackingAllocator* tracking_allocator_ptr = nullptr;
    if (opt.track_allocator) {
      InitTrackingAllocator(tracking_allocator);
      env.RegisterAllocator(&tracking_allocator.ort);
      tracking_allocator_ptr = &tracking_allocator;
      PrintAllocatorStats("after_register", tracking_allocator_ptr);
    }
    Ort::SessionOptions session_options;
    measure("configure_session_options", [&] {
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
      session_options.SetIntraOpNumThreads(threads);
      session_options.SetInterOpNumThreads(1);
      session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      if (!opt.use_arena) session_options.DisableCpuMemArena();
      if (!opt.use_mem_pattern) session_options.DisableMemPattern();
      if (opt.track_allocator) session_options.AddConfigEntry("session.use_env_allocators", "1");
      session_options.AddConfigEntry("session.intra_op.allow_spinning", opt.use_spinning ? "1" : "0");
      session_options.AddConfigEntry("session.inter_op.allow_spinning", opt.use_spinning ? "1" : "0");
    });

    Ort::Session session = measure("create_session", [&] { return Ort::Session(env, opt.model_path.c_str(), session_options); });
    if (opt.print_mem) PrintMemStats("after_session");
    PrintAllocatorStats("after_session", tracking_allocator_ptr);
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names_owned;
    std::vector<const char*> input_names;
    std::vector<std::string> output_names_owned;
    std::vector<const char*> output_names;
    measure("read_io_names", [&] {
      for (size_t i = 0; i < session.GetInputCount(); ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names_owned.emplace_back(name.get());
      }
      for (const auto& name : input_names_owned) input_names.push_back(name.c_str());

      for (size_t i = 0; i < session.GetOutputCount(); ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names_owned.emplace_back(name.get());
      }
      for (const auto& name : output_names_owned) output_names.push_back(name.c_str());
    });

    if (input_names_owned.size() != 3 || output_names_owned.empty()) {
      Die("expected 3 inputs and at least 1 output");
    }

    if (!opt.eval_manifest_path.empty()) {
      return EvaluateManifest(session, input_names, output_names, opt, params, tracking_allocator_ptr);
    }

    std::vector<float> image = measure("image_decode_resize_normalize", [&] {
      return LoadAndPreprocessImage(opt.image_path, params);
    });
    const std::array<float, 2> state_norm = measure("normalize_state", [&] {
      return NormalizeState(opt.state, params);
    });
    std::vector<float> state_vec = measure("prepare_state_tensor_data", [&] {
      return std::vector<float>{state_norm[0], state_norm[1]};
    });
    std::vector<float> latent = measure("prepare_latent_tensor_data", [&] {
      return params.latent;
    });
    if (opt.print_mem) PrintMemStats("after_preprocess");
    PrintAllocatorStats("after_preprocess", tracking_allocator_ptr);

    std::array<int64_t, 4> image_shape{1, 3, kImageH, kImageW};
    std::array<int64_t, 2> state_shape{1, params.state_dim};
    std::array<int64_t, 2> latent_shape{1, params.latent_dim};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> inputs;
    measure("create_input_tensors", [&] {
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, image.data(), image.size(), image_shape.data(), image_shape.size()));
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, state_vec.data(), state_vec.size(), state_shape.data(), state_shape.size()));
      inputs.emplace_back(Ort::Value::CreateTensor<float>(
          memory_info, latent.data(), latent.size(), latent_shape.data(), latent_shape.size()));
    });

    Ort::RunOptions run_options;
    measure("warmup_total", [&] {
      for (int i = 0; i < opt.warmup; ++i) {
        auto outputs = session.Run(run_options, input_names.data(), inputs.data(), inputs.size(),
                                   output_names.data(), output_names.size());
        (void)outputs;
      }
    });
    if (opt.print_mem) PrintMemStats("after_warmup");
    PrintAllocatorStats("after_warmup", tracking_allocator_ptr);

    std::vector<Ort::Value> outputs;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < opt.runs; ++i) {
      outputs = session.Run(run_options, input_names.data(), inputs.data(), inputs.size(),
                            output_names.data(), output_names.size());
    }
    const auto end = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count() /
        static_cast<double>(opt.runs);
    if (opt.print_mem) PrintMemStats("after_runs");
    PrintAllocatorStats("after_runs", tracking_allocator_ptr);
    stages.push_back({"inference_avg", ms, ReadMemStats()});

    const float* action_norm = outputs[0].GetTensorData<float>();
    auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto out_shape = out_info.GetShape();
    size_t action_count = out_info.GetElementCount();
    std::vector<float> action = DenormalizeAction(action_norm, static_cast<int>(action_count), params);

    std::cout << "model: " << opt.model_path << "\n";
    std::cout << "image: " << opt.image_path << "\n";
    std::cout << "threads: " << threads << "  avg_latency_ms: " << ms << "\n";
    std::cout << "pid: " << getpid() << "\n";
    PrintStageStats(stages);
    std::cout << "state_raw: [" << opt.state[0] << ", " << opt.state[1] << "]\n";
    std::cout << "state_norm: [" << state_norm[0] << ", " << state_norm[1] << "]\n";
    std::cout << "output_shape:";
    for (int64_t d : out_shape) std::cout << " " << d;
    std::cout << "\n";

    const float left = action[0];
    const float right = action[1];
    const float gripper = action.size() > 2 ? action[2] : 0.0f;
    std::cout << "first_step: left_vel=" << left << " right_vel=" << right
              << " gripper_target=" << gripper
              << " diff=" << (left - right)
              << " decision=" << TurnDecision(left, right, opt.deadband) << "\n";

    if (opt.print_chunk) {
      const int steps = static_cast<int>(action_count) / params.action_dim;
      for (int t = 0; t < steps; ++t) {
        const int base = t * params.action_dim;
        std::cout << "step[" << t << "]:";
        for (int d = 0; d < params.action_dim; ++d) {
          std::cout << " " << action[base + d];
        }
        std::cout << "\n";
      }
    }

    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "onnxruntime error: " << e.what() << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
