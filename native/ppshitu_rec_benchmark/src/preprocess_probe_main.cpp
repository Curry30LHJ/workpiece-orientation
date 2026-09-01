#include "preprocess.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

int ParsePositiveInt(const char* text, const char* name) {
  if (text == nullptr || *text == '\0') {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  try {
    std::size_t consumed = 0;
    const long long value = std::stoll(text, &consumed, 10);
    if (consumed != std::string(text).size() || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<int>(value);
  } catch (const std::invalid_argument&) {
    throw;
  } catch (...) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
}

unsigned char ParseHexNibble(const char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<unsigned char>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<unsigned char>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<unsigned char>(value - 'A' + 10);
  }
  throw std::invalid_argument("raw RGB bytes contain invalid hex");
}

std::vector<std::uint8_t> DecodeHex(const std::string& value) {
  if (value.size() % 2 != 0) {
    throw std::invalid_argument("raw RGB byte count must be even");
  }
  std::vector<std::uint8_t> bytes;
  bytes.reserve(value.size() / 2);
  for (std::size_t index = 0; index < value.size(); index += 2) {
    const unsigned char high = ParseHexNibble(value[index]);
    const unsigned char low = ParseHexNibble(value[index + 1]);
    bytes.push_back(static_cast<std::uint8_t>((high << 4) | low));
  }
  return bytes;
}

void PrintUsage() {
  std::cerr << "usage: ppshitu_preprocess_probe --raw-rgb <width> <height> "
               "<channels> <hex-bytes> [--output-width <n>] "
               "[--output-height <n>]\n";
}

int RunRawRgb(int argc, char** argv) {
  if (argc < 6) {
    throw std::invalid_argument("raw RGB arguments are incomplete");
  }
  const int width = ParsePositiveInt(argv[2], "width");
  const int height = ParsePositiveInt(argv[3], "height");
  const int channels = ParsePositiveInt(argv[4], "channels");
  const std::vector<std::uint8_t> bytes = DecodeHex(argv[5]);

  workpiece::ppshitu::PreprocessOptions options;
  for (int index = 6; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--output-width" || argument == "--output-height") {
      if (index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      const int value = ParsePositiveInt(argv[++index], argument.c_str());
      if (argument == "--output-width") {
        options.width = value;
      } else {
        options.height = value;
      }
      continue;
    }
    throw std::invalid_argument("unknown argument: " + argument);
  }

  const workpiece::ppshitu::ImageTensor output =
      workpiece::ppshitu::PreprocessRgb(bytes.data(), bytes.size(), width,
                                        height, channels, options);
  if (!output.nchw.empty()) {
    std::cout.write(
        reinterpret_cast<const char*>(output.nchw.data()),
        static_cast<std::streamsize>(output.nchw.size() * sizeof(float)));
  }
  std::cout.flush();
  if (!std::cout) {
    throw std::runtime_error("unable to write preprocessing output");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  try {
    if (argc >= 2 && std::string(argv[1]) == "--raw-rgb") {
      return RunRawRgb(argc, argv);
    }
    PrintUsage();
    return 2;
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 3;
  }
}
