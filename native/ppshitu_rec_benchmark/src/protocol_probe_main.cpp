#include "ppshitu_protocol.h"

#include <cstdlib>
#include <iostream>
#include <string>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

namespace {

bool ParseMaxPayload(int argc, char** argv, std::uint32_t* value) {
  if (value == nullptr) {
    return false;
  }
  *value = workpiece::ppshitu::protocol::kDefaultMaxFrameBytes;
  for (int index = 1; index < argc; ++index) {
    if (std::string(argv[index]) != "--max-payload" || index + 1 >= argc) {
      return false;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(argv[++index], &end, 10);
    if (end == nullptr || *end != '\0' || parsed == 0 ||
        parsed > workpiece::ppshitu::protocol::kDefaultMaxFrameBytes) {
      return false;
    }
    *value = static_cast<std::uint32_t>(parsed);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  _setmode(_fileno(stdin), _O_BINARY);
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  std::uint32_t max_payload = 0;
  if (!ParseMaxPayload(argc, argv, &max_payload)) {
    std::cerr << "usage: protocol_probe --max-payload <positive-bytes>\n";
    return 3;
  }

  workpiece::ppshitu::protocol::Frame frame;
  std::string error;
  const auto status = workpiece::ppshitu::protocol::ReadFrame(
      std::cin, &frame, max_payload, &error);
  if (status != workpiece::ppshitu::protocol::ReadStatus::kFrame) {
    std::cerr << "REJECTED: " << error << '\n';
    return 2;
  }
  if (frame.kind == workpiece::ppshitu::protocol::Kind::kClose &&
      !frame.payload.empty()) {
    std::cerr << "REJECTED: CLOSE payload must be empty\n";
    return 2;
  }
  std::cerr << "OK\n";
  return 0;
}
