#include <windows.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

#include <opencv2/core.hpp>

namespace {

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
  int threads = 1;
};

std::string WideToUtf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                       static_cast<int>(value.size()), nullptr,
                                       0, nullptr, nullptr);
  if (size <= 0) {
    return {};
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
                      result.data(), size, nullptr, nullptr);
  return result;
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

bool WriteErrorReport(const std::filesystem::path& report_path,
                      const std::string& code, const std::string& message) {
  if (report_path.empty()) {
    return false;
  }
  std::filesystem::path temporary = report_path;
  temporary += L".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      return false;
    }
    stream << "{\"schema_version\":1,\"ok\":false,\"error\":{"
           << "\"code\":\"" << JsonEscape(code) << "\","
           << "\"message\":\"" << JsonEscape(message) << "\"}}";
    stream.flush();
    if (!stream) {
      return false;
    }
  }
  return MoveFileExW(temporary.c_str(), report_path.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
}

void PrintHelp() {
  std::cout
      << "PP-ShiTuV2 native recognition benchmark\n\n"
      << "Required:\n"
      << "  --model-dir <path>   Paddle inference model directory\n"
      << "  --image-list <path>  UTF-8 JSON image-list file\n"
      << "  --report <path>      JSON report output\n\n"
      << "Options:\n"
      << "  --threads <n>        Paddle CPU threads (default: 1)\n"
      << "  --help               Show this help\n";
}

std::optional<int> ParsePositiveOrZeroInteger(const std::wstring& value) {
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

std::optional<Options> ParseArguments(int argc, wchar_t** argv,
                                      std::filesystem::path* report_hint,
                                      std::string* error) {
  std::unordered_map<std::wstring, std::wstring> values;
  for (int index = 1; index < argc; ++index) {
    const std::wstring key = argv[index];
    if (key == L"--help") {
      continue;
    }
    if (key != L"--model-dir" && key != L"--image-list" &&
        key != L"--report" && key != L"--threads") {
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

  const auto report = values.find(L"--report");
  if (report != values.end()) {
    *report_hint = std::filesystem::path(report->second);
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
  if (const auto threads = values.find(L"--threads"); threads != values.end()) {
    const auto parsed = ParsePositiveOrZeroInteger(threads->second);
    if (!parsed || *parsed == 0) {
      *error = "threads must be a positive integer";
      return std::nullopt;
    }
    options.threads = *parsed;
  }
  return options;
}

int Fail(const std::filesystem::path& report_path, ExitCode exit_code,
         const std::string& code, const std::string& message) {
  if (!WriteErrorReport(report_path, code, message)) {
    std::cerr << message << '\n';
    return static_cast<int>(ExitCode::kReportWriteFailed);
  }
  return static_cast<int>(exit_code);
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
  std::string error;
  const auto options = ParseArguments(argc, argv, &report_hint, &error);
  if (!options) {
    return Fail(report_hint, ExitCode::kInvalidArgument, "INVALID_ARGUMENT",
                error);
  }
  if (!cv::checkHardwareSupport(CV_CPU_AVX)) {
    return Fail(options->report, ExitCode::kCpuUnsupported,
                "CPU_FEATURE_UNSUPPORTED", "AVX CPU support is required");
  }
  return Fail(options->report, ExitCode::kInferenceFailed, "NOT_IMPLEMENTED",
              "feature extraction is not implemented yet");
}
