#include "native_service.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

void PrintHelp() {
  std::cout
      << "PP-ShiTuV2 native recognition service\n\n"
      << "Required:\n"
      << "  --serve                 Run the persistent stdin/stdout service\n"
      << "  --model-dir <path>      ASCII Paddle inference model directory\n\n"
      << "Options:\n"
      << "  --threads <n>           Paddle CPU threads (default: 1)\n"
      << "  --max-frame-bytes <n>   Maximum request payload (default: 268435456)\n"
      << "  --max-batch <n>         Maximum images per request (default: 256)\n"
      << "  --input-width <n>       Resize width (default: 224)\n"
      << "  --input-height <n>      Resize height (default: 224)\n"
      << "  --scale <value>         Pixel scale (default: 1/255)\n"
      << "  --mean-rgb <r,g,b>      RGB mean values\n"
      << "  --std-rgb <r,g,b>       RGB standard deviations\n"
      << "  --help                  Show this help\n";
}

bool ParsePositiveInt(const std::wstring& value, const char* name, int* out,
                      std::string* error) {
  try {
    std::size_t consumed = 0;
    const long long parsed = std::stoll(value, &consumed, 10);
    if (consumed != value.size() || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      *error = std::string(name) + " must be a positive integer";
      return false;
    }
    *out = static_cast<int>(parsed);
    return true;
  } catch (...) {
    *error = std::string(name) + " must be a positive integer";
    return false;
  }
}

bool ParsePositiveUint32(const std::wstring& value, const char* name,
                         std::uint32_t* out, std::string* error) {
  try {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 10);
    if (consumed != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
      *error = std::string(name) + " must be a positive uint32";
      return false;
    }
    *out = static_cast<std::uint32_t>(parsed);
    return true;
  } catch (...) {
    *error = std::string(name) + " must be a positive uint32";
    return false;
  }
}

bool ParseFiniteFloat(const std::wstring& value, const char* name, float* out,
                      std::string* error) {
  try {
    std::size_t consumed = 0;
    const float parsed = std::stof(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
      *error = std::string(name) + " must be finite";
      return false;
    }
    *out = parsed;
    return true;
  } catch (...) {
    *error = std::string(name) + " must be finite";
    return false;
  }
}

bool ParseFloatTriplet(const std::wstring& value, const char* name,
                       std::array<float, 3>* out, std::string* error) {
  std::array<float, 3> parsed{};
  std::size_t begin = 0;
  for (std::size_t index = 0; index < parsed.size(); ++index) {
    const std::size_t end = value.find(L',', begin);
    if ((index < parsed.size() - 1 && end == std::wstring::npos) ||
        (index == parsed.size() - 1 && end != std::wstring::npos)) {
      *error = std::string(name) + " must contain three comma-separated values";
      return false;
    }
    const std::wstring part = value.substr(begin, end - begin);
    if (!ParseFiniteFloat(part, name, &parsed[index], error)) {
      return false;
    }
    begin = end == std::wstring::npos ? value.size() : end + 1;
  }
  *out = parsed;
  return true;
}

bool ParseArguments(int argc, wchar_t** argv,
                    workpiece::ppshitu::ServiceOptions* options,
                    std::string* error) {
  bool serve = false;
  bool model_seen = false;
  bool threads_seen = false;
  bool frame_seen = false;
  bool batch_seen = false;
  bool width_seen = false;
  bool height_seen = false;
  bool scale_seen = false;
  bool mean_seen = false;
  bool std_seen = false;

  for (int index = 1; index < argc; ++index) {
    const std::wstring key = argv[index];
    if (key == L"--serve") {
      if (serve) {
        *error = "duplicate argument: --serve";
        return false;
      }
      serve = true;
      continue;
    }
    if (key == L"--help") {
      continue;
    }
    if (key == L"--model-dir" || key == L"--threads" ||
        key == L"--max-frame-bytes" || key == L"--max-batch" ||
        key == L"--input-width" || key == L"--input-height" ||
        key == L"--scale" || key == L"--mean-rgb" || key == L"--std-rgb") {
      if (index + 1 >= argc) {
        *error = "missing value for service option";
        return false;
      }
      const std::wstring value = argv[++index];
      if (key == L"--model-dir") {
        if (model_seen) {
          *error = "duplicate argument: --model-dir";
          return false;
        }
        model_seen = true;
        options->model_dir = value;
      } else if (key == L"--threads") {
        if (threads_seen ||
            !ParsePositiveInt(value, "threads", &options->threads, error)) {
          if (threads_seen) {
            *error = "duplicate argument: --threads";
          }
          return false;
        }
        threads_seen = true;
      } else if (key == L"--max-frame-bytes") {
        if (frame_seen ||
            !ParsePositiveUint32(value, "max-frame-bytes",
                                 &options->max_frame_bytes, error)) {
          if (frame_seen) {
            *error = "duplicate argument: --max-frame-bytes";
          }
          return false;
        }
        frame_seen = true;
      } else if (key == L"--max-batch") {
        if (batch_seen ||
            !ParsePositiveUint32(value, "max-batch", &options->max_batch,
                                 error)) {
          if (batch_seen) {
            *error = "duplicate argument: --max-batch";
          }
          return false;
        }
        batch_seen = true;
      } else if (key == L"--input-width") {
        if (width_seen ||
            !ParsePositiveInt(value, "input-width", &options->preprocess.width,
                              error)) {
          if (width_seen) {
            *error = "duplicate argument: --input-width";
          }
          return false;
        }
        width_seen = true;
      } else if (key == L"--input-height") {
        if (height_seen ||
            !ParsePositiveInt(value, "input-height",
                              &options->preprocess.height, error)) {
          if (height_seen) {
            *error = "duplicate argument: --input-height";
          }
          return false;
        }
        height_seen = true;
      } else if (key == L"--scale") {
        if (scale_seen ||
            !ParseFiniteFloat(value, "scale", &options->preprocess.scale,
                              error) ||
            options->preprocess.scale <= 0.0F) {
          if (scale_seen) {
            *error = "duplicate argument: --scale";
          } else if (error->empty()) {
            *error = "scale must be finite and positive";
          }
          return false;
        }
        scale_seen = true;
      } else if (key == L"--mean-rgb") {
        if (mean_seen ||
            !ParseFloatTriplet(value, "mean-rgb", &options->preprocess.mean,
                               error)) {
          if (mean_seen) {
            *error = "duplicate argument: --mean-rgb";
          }
          return false;
        }
        mean_seen = true;
      } else if (key == L"--std-rgb") {
        if (std_seen ||
            !ParseFloatTriplet(value, "std-rgb", &options->preprocess.std,
                               error) ||
            options->preprocess.std[0] <= 0.0F ||
            options->preprocess.std[1] <= 0.0F ||
            options->preprocess.std[2] <= 0.0F) {
          if (std_seen) {
            *error = "duplicate argument: --std-rgb";
          } else {
            *error = "std-rgb must contain three finite positive values";
          }
          return false;
        }
        std_seen = true;
      }
      continue;
    }
    *error = "unknown argument";
    return false;
  }

  if (!serve) {
    *error = "--serve is required";
    return false;
  }
  if (!model_seen || options->model_dir.empty()) {
    *error = "--model-dir is required";
    return false;
  }
  return true;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  for (int index = 1; index < argc; ++index) {
    if (std::wstring(argv[index]) == L"--help") {
      PrintHelp();
      return 0;
    }
  }

  workpiece::ppshitu::ServiceOptions options;
  std::string error;
  if (!ParseArguments(argc, argv, &options, &error)) {
    std::cerr << "INVALID_ARGUMENT: " << error << '\n';
    return 2;
  }

#ifdef _WIN32
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  return workpiece::ppshitu::RunNativeService(options);
}
