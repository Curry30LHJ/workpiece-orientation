#include <windows.h>
#include <psapi.h>

#include "feature_extractor.h"
#include "preprocess.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace {

using workpiece::ppshitu::BatchTensor;
using workpiece::ppshitu::FeatureExtractor;
using workpiece::ppshitu::ImageTensor;
using workpiece::ppshitu::PredictionBatch;
using workpiece::ppshitu::PreprocessOptions;
using workpiece::ppshitu::PredictorOptions;

enum class ExitCode : int {
  kSuccess = 0,
  kInvalidArgument = 2,
  kInputUnreadable = 3,
  kModelLoadFailed = 4,
  kInferenceFailed = 5,
  kReportWriteFailed = 6,
  kCpuUnsupported = 7,
};

struct Options {
  std::filesystem::path model_dir;
  std::filesystem::path image_list;
  std::filesystem::path report;
  std::filesystem::path dump_inputs;
  std::filesystem::path dump_embeddings;
  PreprocessOptions preprocess;
  int threads = 1;
  int workers = 1;
  int batch_size = 1;
  int warmup = 0;
  int iterations = 1;
  bool preprocess_only = false;
};

std::string WideToUtf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                       static_cast<int>(value.size()), nullptr,
                                       0, nullptr, nullptr);
  if (size <= 0) {
    throw std::runtime_error("unable to encode a Windows path as UTF-8");
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), size, nullptr, nullptr);
  return result;
}

std::string PathToUtf8(const std::filesystem::path& path) {
  return WideToUtf8(path.wstring());
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '\"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          const char hex[] = "0123456789abcdef";
          output << "\\u00" << hex[(character >> 4) & 0x0f]
                 << hex[character & 0x0f];
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

bool ReplaceFile(const std::filesystem::path& temporary,
                 const std::filesystem::path& destination) {
  return MoveFileExW(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}

bool WriteJsonAtomically(const std::filesystem::path& path,
                         const std::string& json) {
  if (path.empty()) {
    return false;
  }
  std::filesystem::path temporary = path;
  temporary += L".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return false;
    }
    stream.write(json.data(), static_cast<std::streamsize>(json.size()));
    stream.flush();
    if (!stream) {
      return false;
    }
  }
  return ReplaceFile(temporary, path);
}

bool WriteFloat32Atomically(const std::filesystem::path& path,
                            const std::vector<float>& values) {
  if (path.empty()) {
    return false;
  }
  std::filesystem::path temporary = path;
  temporary += L".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return false;
    }
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    stream.flush();
    if (!stream) {
      return false;
    }
  }
  return ReplaceFile(temporary, path);
}

bool WriteErrorReport(const std::filesystem::path& report_path,
                      const std::string& code, const std::string& message) {
  std::ostringstream report;
  report << "{\"schema_version\":1,\"ok\":false,\"error\":{"
         << "\"code\":\"" << JsonEscape(code) << "\","
         << "\"message\":\"" << JsonEscape(message) << "\"}}";
  return WriteJsonAtomically(report_path, report.str());
}

void PrintHelp() {
  std::cout
      << "PP-ShiTuV2 native recognition benchmark\n\n"
      << "Required:\n"
      << "  --model-dir <path>    Paddle inference model directory\n"
      << "  --image-list <path>   UTF-8 JSON image-list file\n"
      << "  --report <path>       JSON report output\n\n"
      << "Options:\n"
      << "  --threads <n>         Paddle CPU threads (default: 1)\n"
      << "  --workers <n>         Independent predictors (default: 1)\n"
      << "  --batch-size <n>      Images per predictor Run() (default: 1)\n"
      << "  --warmup <n>          Unmeasured iterations (default: 0)\n"
      << "  --iterations <n>      Measured iterations (default: 1)\n"
      << "  --dump-embeddings <path>  Write normalized float32 rows\n"
      << "  --preprocess-only     Stop after NCHW preprocessing\n"
      << "  --dump-inputs <path>  Write row-major NCHW float32 values\n"
      << "  --input-width <n>     Resize width (default: 224)\n"
      << "  --input-height <n>    Resize height (default: 224)\n"
      << "  --scale <value>       Pixel scale (default: 1/255)\n"
      << "  --mean-rgb <r,g,b>    RGB mean values\n"
      << "  --std-rgb <r,g,b>     RGB standard deviations\n"
      << "  --help                Show this help\n";
}

std::optional<int> ParseNonNegativeInteger(const std::wstring& value) {
  try {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed, 10);
    if (consumed != value.size() || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return std::nullopt;
    }
    return static_cast<int>(parsed);
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<float> ParseFiniteFloat(const std::wstring& value) {
  try {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
      return std::nullopt;
    }
    return parsed;
  } catch (...) {
    return std::nullopt;
  }
}

std::optional<std::array<float, 3>> ParseFloatTriplet(
    const std::wstring& value) {
  std::array<float, 3> result{};
  std::size_t begin = 0;
  for (std::size_t index = 0; index < result.size(); ++index) {
    const std::size_t end = value.find(L',', begin);
    if ((index < result.size() - 1 && end == std::wstring::npos) ||
        (index == result.size() - 1 && end != std::wstring::npos)) {
      return std::nullopt;
    }
    const std::wstring part = value.substr(begin, end - begin);
    const auto parsed = ParseFiniteFloat(part);
    if (!parsed) {
      return std::nullopt;
    }
    result[index] = *parsed;
    begin = end == std::wstring::npos ? value.size() : end + 1;
  }
  return result;
}

bool IsValueArgument(const std::wstring& key) {
  return key == L"--model-dir" || key == L"--image-list" ||
         key == L"--report" || key == L"--threads" ||
         key == L"--workers" || key == L"--batch-size" ||
         key == L"--warmup" || key == L"--iterations" ||
         key == L"--dump-embeddings" ||
         key == L"--dump-inputs" || key == L"--input-width" ||
         key == L"--input-height" || key == L"--scale" ||
         key == L"--mean-rgb" || key == L"--std-rgb";
}

std::optional<Options> ParseArguments(int argc, wchar_t** argv,
                                      std::string* error) {
  std::unordered_map<std::wstring, std::wstring> values;
  bool preprocess_only = false;
  for (int index = 1; index < argc; ++index) {
    const std::wstring key = argv[index];
    if (key == L"--help") {
      continue;
    }
    if (key == L"--preprocess-only") {
      if (preprocess_only) {
        *error = "duplicate argument: --preprocess-only";
        return std::nullopt;
      }
      preprocess_only = true;
      continue;
    }
    if (!IsValueArgument(key)) {
      *error = "unknown argument: " + WideToUtf8(key);
      return std::nullopt;
    }
    if (index + 1 >= argc) {
      *error = "missing value for " + WideToUtf8(key);
      return std::nullopt;
    }
    if (values.find(key) != values.end()) {
      *error = "duplicate argument: " + WideToUtf8(key);
      return std::nullopt;
    }
    values.emplace(key, argv[++index]);
  }

  for (const wchar_t* required : {L"--model-dir", L"--image-list",
                                  L"--report"}) {
    if (values.find(required) == values.end()) {
      *error = "missing required argument: " + WideToUtf8(required);
      return std::nullopt;
    }
  }

  Options options;
  options.model_dir = values.at(L"--model-dir");
  options.image_list = values.at(L"--image-list");
  options.report = values.at(L"--report");
  options.preprocess_only = preprocess_only;
  if (const auto found = values.find(L"--threads"); found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "threads must be a positive integer";
      return std::nullopt;
    }
    options.threads = *parsed;
  }
  if (const auto found = values.find(L"--workers"); found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "workers must be a positive integer";
      return std::nullopt;
    }
    options.workers = *parsed;
  }
  if (const auto found = values.find(L"--batch-size");
      found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "batch size must be a positive integer";
      return std::nullopt;
    }
    options.batch_size = *parsed;
  }
  if (const auto found = values.find(L"--warmup"); found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed) {
      *error = "warmup must be a non-negative integer";
      return std::nullopt;
    }
    options.warmup = *parsed;
  }
  if (const auto found = values.find(L"--iterations");
      found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "iterations must be a positive integer";
      return std::nullopt;
    }
    options.iterations = *parsed;
  }
  if (const auto found = values.find(L"--input-width");
      found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "input width must be a positive integer";
      return std::nullopt;
    }
    options.preprocess.width = *parsed;
  }
  if (const auto found = values.find(L"--input-height");
      found != values.end()) {
    const auto parsed = ParseNonNegativeInteger(found->second);
    if (!parsed || *parsed == 0) {
      *error = "input height must be a positive integer";
      return std::nullopt;
    }
    options.preprocess.height = *parsed;
  }
  if (const auto found = values.find(L"--scale"); found != values.end()) {
    const auto parsed = ParseFiniteFloat(found->second);
    if (!parsed || *parsed <= 0.0f) {
      *error = "scale must be finite and positive";
      return std::nullopt;
    }
    options.preprocess.scale = *parsed;
  }
  if (const auto found = values.find(L"--mean-rgb"); found != values.end()) {
    const auto parsed = ParseFloatTriplet(found->second);
    if (!parsed) {
      *error = "mean-rgb must contain three finite comma-separated values";
      return std::nullopt;
    }
    options.preprocess.mean = *parsed;
  }
  if (const auto found = values.find(L"--std-rgb"); found != values.end()) {
    const auto parsed = ParseFloatTriplet(found->second);
    if (!parsed || (*parsed)[0] <= 0.0f || (*parsed)[1] <= 0.0f ||
        (*parsed)[2] <= 0.0f) {
      *error = "std-rgb must contain three finite positive values";
      return std::nullopt;
    }
    options.preprocess.std = *parsed;
  }
  if (const auto found = values.find(L"--dump-inputs");
      found != values.end()) {
    options.dump_inputs = found->second;
  }
  if (const auto found = values.find(L"--dump-embeddings");
      found != values.end()) {
    options.dump_embeddings = found->second;
  }
  if (options.preprocess_only && options.dump_inputs.empty()) {
    *error = "--dump-inputs is required with --preprocess-only";
    return std::nullopt;
  }
  if (!options.preprocess_only && !options.dump_inputs.empty()) {
    *error = "--dump-inputs requires --preprocess-only";
    return std::nullopt;
  }
  if (options.preprocess_only && !options.dump_embeddings.empty()) {
    *error = "--dump-embeddings cannot be used with --preprocess-only";
    return std::nullopt;
  }
  if (!options.preprocess_only && options.dump_embeddings.empty()) {
    *error = "--dump-embeddings is required for feature extraction";
    return std::nullopt;
  }
  return options;
}

class JsonStringArrayParser {
 public:
  explicit JsonStringArrayParser(std::string_view input) : input_(input) {}

  std::vector<std::string> Parse() {
    SkipWhitespace();
    Expect('[');
    SkipWhitespace();
    std::vector<std::string> result;
    if (Consume(']')) {
      throw std::runtime_error("image list must not be empty");
    }
    while (true) {
      result.push_back(ParseString());
      SkipWhitespace();
      if (Consume(']')) {
        break;
      }
      Expect(',');
      SkipWhitespace();
    }
    SkipWhitespace();
    if (position_ != input_.size()) {
      throw std::runtime_error("unexpected data after image list");
    }
    return result;
  }

 private:
  void SkipWhitespace() {
    while (position_ < input_.size() &&
           (input_[position_] == ' ' || input_[position_] == '\t' ||
            input_[position_] == '\r' || input_[position_] == '\n')) {
      ++position_;
    }
  }

  bool Consume(char expected) {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void Expect(char expected) {
    if (!Consume(expected)) {
      throw std::runtime_error(std::string("expected '") + expected + "'");
    }
  }

  static int HexValue(char character) {
    if (character >= '0' && character <= '9') {
      return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
      return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
      return character - 'A' + 10;
    }
    return -1;
  }

  std::uint32_t ParseHex4() {
    if (position_ + 4 > input_.size()) {
      throw std::runtime_error("truncated unicode escape");
    }
    std::uint32_t value = 0;
    for (int count = 0; count < 4; ++count) {
      const int digit = HexValue(input_[position_++]);
      if (digit < 0) {
        throw std::runtime_error("invalid unicode escape");
      }
      value = value * 16 + static_cast<std::uint32_t>(digit);
    }
    return value;
  }

  static void AppendUtf8(std::uint32_t code_point, std::string* output) {
    if (code_point == 0 || code_point > 0x10ffff) {
      throw std::runtime_error("invalid unicode code point in path");
    }
    if (code_point <= 0x7f) {
      output->push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7ff) {
      output->push_back(static_cast<char>(0xc0 | (code_point >> 6)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else if (code_point <= 0xffff) {
      output->push_back(static_cast<char>(0xe0 | (code_point >> 12)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else {
      output->push_back(static_cast<char>(0xf0 | (code_point >> 18)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3f)));
      output->push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
      output->push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    }
  }

  std::string ParseString() {
    Expect('\"');
    std::string output;
    while (position_ < input_.size()) {
      const unsigned char character =
          static_cast<unsigned char>(input_[position_++]);
      if (character == '\"') {
        return output;
      }
      if (character < 0x20) {
        throw std::runtime_error("unescaped control character in path");
      }
      if (character != '\\') {
        output.push_back(static_cast<char>(character));
        continue;
      }
      if (position_ >= input_.size()) {
        throw std::runtime_error("truncated escape in path");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '\"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u': {
          std::uint32_t code_point = ParseHex4();
          if (code_point >= 0xd800 && code_point <= 0xdbff) {
            if (position_ + 2 > input_.size() || input_[position_] != '\\' ||
                input_[position_ + 1] != 'u') {
              throw std::runtime_error("missing low unicode surrogate");
            }
            position_ += 2;
            const std::uint32_t low = ParseHex4();
            if (low < 0xdc00 || low > 0xdfff) {
              throw std::runtime_error("invalid low unicode surrogate");
            }
            code_point =
                0x10000 + ((code_point - 0xd800) << 10) + (low - 0xdc00);
          } else if (code_point >= 0xdc00 && code_point <= 0xdfff) {
            throw std::runtime_error("unexpected low unicode surrogate");
          }
          AppendUtf8(code_point, &output);
          break;
        }
        default:
          throw std::runtime_error("invalid escape in path");
      }
    }
    throw std::runtime_error("unterminated path string");
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

std::vector<std::filesystem::path> ReadImageList(
    const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("unable to open image-list file");
  }
  std::string content((std::istreambuf_iterator<char>(stream)),
                      std::istreambuf_iterator<char>());
  if (content.size() >= 3 &&
      static_cast<unsigned char>(content[0]) == 0xef &&
      static_cast<unsigned char>(content[1]) == 0xbb &&
      static_cast<unsigned char>(content[2]) == 0xbf) {
    content.erase(0, 3);
  }
  const std::vector<std::string> encoded_paths =
      JsonStringArrayParser(content).Parse();
  std::vector<std::filesystem::path> paths;
  paths.reserve(encoded_paths.size());
  for (const std::string& encoded_path : encoded_paths) {
    if (encoded_path.empty()) {
      throw std::runtime_error("image paths must not be empty");
    }
    paths.push_back(std::filesystem::u8path(encoded_path));
  }
  return paths;
}

std::string BuildPreprocessReport(
    const Options& options,
    const std::vector<std::filesystem::path>& image_paths,
    const BatchTensor& batch) {
  std::ostringstream report;
  report << "{\"schema_version\":1,\"ok\":true,"
         << "\"mode\":\"preprocess_only\","
         << "\"input_count\":" << image_paths.size() << ','
         << "\"ordered_images\":[";
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    if (index != 0) {
      report << ',';
    }
    report << '\"' << JsonEscape(PathToUtf8(image_paths[index])) << '\"';
  }
  report << "],\"tensor_shape\":[" << batch.batch << ',' << batch.channels
         << ',' << batch.height << ',' << batch.width << "],"
         << "\"preprocess\":{"
         << "\"width\":" << options.preprocess.width << ','
         << "\"height\":" << options.preprocess.height << ','
         << "\"scale\":" << options.preprocess.scale << "}}";
  return report.str();
}

int Fail(const std::filesystem::path& report_path, ExitCode exit_code,
         const std::string& code, const std::string& message) {
  if (!WriteErrorReport(report_path, code, message)) {
    std::cerr << message << '\n';
    return static_cast<int>(ExitCode::kReportWriteFailed);
  }
  return static_cast<int>(exit_code);
}

int RunPreprocessOnly(const Options& options) {
  std::vector<std::filesystem::path> image_paths;
  try {
    image_paths = ReadImageList(options.image_list);
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kInputUnreadable,
                "IMAGE_LIST_INVALID", exception.what());
  }

  std::vector<ImageTensor> images;
  images.reserve(image_paths.size());
  for (const std::filesystem::path& image_path : image_paths) {
    try {
      images.push_back(
          workpiece::ppshitu::LoadAndPreprocess(image_path, options.preprocess));
    } catch (const std::exception& exception) {
      return Fail(options.report, ExitCode::kInputUnreadable,
                  "IMAGE_UNREADABLE",
                  PathToUtf8(image_path) + ": " + exception.what());
    }
  }

  BatchTensor batch;
  try {
    batch = workpiece::ppshitu::StackBatch(images);
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kInvalidArgument,
                "INVALID_ARGUMENT", exception.what());
  }
  if (!WriteFloat32Atomically(options.dump_inputs, batch.nchw)) {
    return Fail(options.report, ExitCode::kReportWriteFailed,
                "DUMP_WRITE_FAILED", "unable to write preprocessing dump");
  }
  if (!WriteJsonAtomically(
          options.report,
          BuildPreprocessReport(options, image_paths, batch))) {
    std::cerr << "unable to write preprocessing report\n";
    return static_cast<int>(ExitCode::kReportWriteFailed);
  }
  return static_cast<int>(ExitCode::kSuccess);
}

struct ChunkResult {
  std::size_t begin = 0;
  std::size_t rows = 0;
  std::size_t columns = 0;
  std::vector<float> embeddings;
  double decode_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double normalize_ms = 0.0;
};

struct IterationResult {
  std::vector<float> embeddings;
  std::size_t rows = 0;
  std::size_t columns = 0;
  double decode_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double normalize_ms = 0.0;
  double total_ms = 0.0;
};

ChunkResult ProcessChunk(
    FeatureExtractor* extractor,
    const std::vector<std::filesystem::path>& image_paths,
    std::size_t begin, std::size_t end,
    const PreprocessOptions& preprocess_options) {
  ChunkResult chunk;
  chunk.begin = begin;
  std::vector<ImageTensor> images;
  images.reserve(end - begin);
  for (std::size_t index = begin; index < end; ++index) {
    ImageTensor image = workpiece::ppshitu::LoadAndPreprocess(
        image_paths[index], preprocess_options);
    chunk.decode_ms += image.decode_ms;
    chunk.preprocess_ms += image.preprocess_ms;
    images.push_back(std::move(image));
  }
  const BatchTensor batch = workpiece::ppshitu::StackBatch(images);
  PredictionBatch prediction = extractor->Predict(batch);
  chunk.rows = prediction.rows;
  chunk.columns = prediction.columns;
  chunk.embeddings = std::move(prediction.embeddings);
  chunk.inference_ms = prediction.inference_ms;
  chunk.normalize_ms = prediction.normalize_ms;
  if (chunk.rows != end - begin) {
    throw std::runtime_error("predictor returned an unexpected chunk row count");
  }
  return chunk;
}

IterationResult RunOneIteration(
    const Options& options,
    const std::vector<std::filesystem::path>& image_paths,
    const std::vector<std::unique_ptr<FeatureExtractor>>& extractors) {
  const auto total_start = std::chrono::steady_clock::now();
  const std::size_t batch_size = static_cast<std::size_t>(options.batch_size);
  const std::size_t chunk_count =
      (image_paths.size() + batch_size - 1) / batch_size;
  std::vector<std::optional<ChunkResult>> chunks(chunk_count);

  auto run_chunk = [&](std::size_t chunk_index, std::size_t worker_index) {
    const std::size_t begin = chunk_index * batch_size;
    const std::size_t end = std::min(begin + batch_size, image_paths.size());
    chunks[chunk_index] = ProcessChunk(extractors[worker_index].get(),
                                       image_paths, begin, end,
                                       options.preprocess);
  };

  if (extractors.size() == 1) {
    for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
      run_chunk(chunk, 0);
    }
  } else {
    std::atomic<std::size_t> next_chunk{0};
    std::mutex failure_mutex;
    std::exception_ptr failure;
    std::vector<std::thread> workers;
    workers.reserve(extractors.size());
    for (std::size_t worker = 0; worker < extractors.size(); ++worker) {
      workers.emplace_back([&, worker]() {
        while (true) {
          const std::size_t chunk =
              next_chunk.fetch_add(1, std::memory_order_relaxed);
          if (chunk >= chunk_count) {
            return;
          }
          try {
            run_chunk(chunk, worker);
          } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) {
              failure = std::current_exception();
            }
            return;
          }
        }
      });
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    if (failure) {
      std::rethrow_exception(failure);
    }
  }

  IterationResult result;
  result.rows = image_paths.size();
  for (const std::optional<ChunkResult>& candidate : chunks) {
    if (!candidate) {
      throw std::runtime_error("a scheduled prediction chunk did not finish");
    }
    const ChunkResult& chunk = *candidate;
    if (result.columns == 0) {
      result.columns = chunk.columns;
      result.embeddings.resize(result.rows * result.columns);
    }
    if (chunk.columns != result.columns ||
        chunk.embeddings.size() != chunk.rows * chunk.columns) {
      throw std::runtime_error("recognition feature dimension changed");
    }
    std::copy(chunk.embeddings.begin(), chunk.embeddings.end(),
              result.embeddings.begin() + chunk.begin * result.columns);
    result.decode_ms += chunk.decode_ms;
    result.preprocess_ms += chunk.preprocess_ms;
    result.inference_ms += chunk.inference_ms;
    result.normalize_ms += chunk.normalize_ms;
  }
  const auto total_end = std::chrono::steady_clock::now();
  result.total_ms =
      std::chrono::duration<double, std::milli>(total_end - total_start)
          .count();
  return result;
}

double Percentile(const std::vector<double>& samples, double quantile) {
  if (samples.empty()) {
    throw std::invalid_argument("timing samples must not be empty");
  }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const double position = (sorted.size() - 1) * quantile;
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double fraction = position - lower;
  return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

void AppendTimingSeries(std::ostringstream* report, const std::string& name,
                        const std::vector<double>& samples, bool comma) {
  *report << '\"' << name << "\":{\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      *report << ',';
    }
    *report << samples[index];
  }
  *report << "],\"p50\":" << Percentile(samples, 0.50)
          << ",\"p95\":" << Percentile(samples, 0.95)
          << ",\"p99\":" << Percentile(samples, 0.99)
          << ",\"max\":" << *std::max_element(samples.begin(), samples.end())
          << '}';
  if (comma) {
    *report << ',';
  }
}

std::uint64_t PeakWorkingSetBytes() {
  PROCESS_MEMORY_COUNTERS counters{};
  counters.cb = static_cast<DWORD>(sizeof(counters));
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters,
                            static_cast<DWORD>(sizeof(counters)))) {
    throw std::runtime_error("unable to query process memory counters");
  }
  return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
}

std::string BuildInferenceReport(
    const Options& options,
    const std::vector<std::filesystem::path>& image_paths,
    const std::vector<std::unique_ptr<FeatureExtractor>>& extractors,
    const IterationResult& last, const std::vector<double>& decode_samples,
    const std::vector<double>& preprocess_samples,
    const std::vector<double>& inference_samples,
    const std::vector<double>& normalize_samples,
    const std::vector<double>& total_samples, std::uint64_t peak_working_set) {
  std::ostringstream report;
  report << std::setprecision(10)
         << "{\"schema_version\":1,\"ok\":true,"
         << "\"mode\":\"feature_extraction\","
         << "\"input_count\":" << image_paths.size() << ','
         << "\"feature_dimension\":" << last.columns << ','
         << "\"worker_count\":" << options.workers << ','
         << "\"batch_size\":" << options.batch_size << ','
         << "\"threads\":" << options.threads << ','
         << "\"warmup\":" << options.warmup << ','
         << "\"iterations\":" << options.iterations << ','
         << "\"peak_working_set_bytes\":" << peak_working_set << ','
         << "\"stage_aggregation\":\"sum_worker_time\","
         << "\"ordered_images\":[";
  for (std::size_t index = 0; index < image_paths.size(); ++index) {
    if (index != 0) {
      report << ',';
    }
    report << '\"' << JsonEscape(PathToUtf8(image_paths[index])) << '\"';
  }
  report << "],\"predictor_instance_ids\":[";
  for (std::size_t index = 0; index < extractors.size(); ++index) {
    if (index != 0) {
      report << ',';
    }
    report << extractors[index]->instance_id();
  }
  report << "],\"timings_ms\":{";
  AppendTimingSeries(&report, "decode", decode_samples, true);
  AppendTimingSeries(&report, "preprocess", preprocess_samples, true);
  AppendTimingSeries(&report, "inference", inference_samples, true);
  AppendTimingSeries(&report, "normalize", normalize_samples, true);
  AppendTimingSeries(&report, "total", total_samples, false);
  report << "}}";
  return report.str();
}

int RunInference(const Options& options) {
  std::vector<std::filesystem::path> image_paths;
  try {
    image_paths = ReadImageList(options.image_list);
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kInputUnreadable,
                "IMAGE_LIST_INVALID", exception.what());
  }

  std::vector<std::unique_ptr<FeatureExtractor>> extractors;
  extractors.reserve(static_cast<std::size_t>(options.workers));
  try {
    const PredictorOptions predictor_options{options.model_dir, options.threads,
                                             true};
    for (int worker = 0; worker < options.workers; ++worker) {
      extractors.push_back(
          std::make_unique<FeatureExtractor>(predictor_options));
    }
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kModelLoadFailed,
                "MODEL_LOAD_FAILED", exception.what());
  }

  try {
    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      RunOneIteration(options, image_paths, extractors);
    }
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kInferenceFailed,
                "WARMUP_FAILED", exception.what());
  }

  std::vector<double> decode_samples;
  std::vector<double> preprocess_samples;
  std::vector<double> inference_samples;
  std::vector<double> normalize_samples;
  std::vector<double> total_samples;
  IterationResult last;
  try {
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      IterationResult current =
          RunOneIteration(options, image_paths, extractors);
      decode_samples.push_back(current.decode_ms);
      preprocess_samples.push_back(current.preprocess_ms);
      inference_samples.push_back(current.inference_ms);
      normalize_samples.push_back(current.normalize_ms);
      total_samples.push_back(current.total_ms);
      last = std::move(current);
    }
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kInferenceFailed,
                "INFERENCE_FAILED", exception.what());
  }

  if (!WriteFloat32Atomically(options.dump_embeddings, last.embeddings)) {
    return Fail(options.report, ExitCode::kReportWriteFailed,
                "DUMP_WRITE_FAILED", "unable to write embedding dump");
  }
  std::uint64_t peak_working_set = 0;
  try {
    peak_working_set = PeakWorkingSetBytes();
  } catch (const std::exception& exception) {
    return Fail(options.report, ExitCode::kReportWriteFailed,
                "MEMORY_QUERY_FAILED", exception.what());
  }
  if (!WriteJsonAtomically(
          options.report,
          BuildInferenceReport(options, image_paths, extractors, last,
                               decode_samples, preprocess_samples,
                               inference_samples, normalize_samples,
                               total_samples, peak_working_set))) {
    std::cerr << "unable to write inference report\n";
    return static_cast<int>(ExitCode::kReportWriteFailed);
  }
  return static_cast<int>(ExitCode::kSuccess);
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  for (int index = 1; index < argc; ++index) {
    if (std::wstring(argv[index]) == L"--help") {
      PrintHelp();
      return static_cast<int>(ExitCode::kSuccess);
    }
  }

  std::filesystem::path report_hint;
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::wstring(argv[index]) == L"--report") {
      report_hint = argv[index + 1];
      break;
    }
  }
  std::string error;
  const auto options = ParseArguments(argc, argv, &error);
  if (!options) {
    return Fail(report_hint, ExitCode::kInvalidArgument, "INVALID_ARGUMENT",
                error);
  }
  if (!cv::checkHardwareSupport(CV_CPU_AVX)) {
    return Fail(options->report, ExitCode::kCpuUnsupported,
                "CPU_FEATURE_UNSUPPORTED", "AVX CPU support is required");
  }
  if (options->preprocess_only) {
    return RunPreprocessOnly(*options);
  }
  return RunInference(*options);
}
