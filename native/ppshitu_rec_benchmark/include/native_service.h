#pragma once

#include "preprocess.h"

#include <cstdint>
#include <filesystem>

namespace workpiece::ppshitu {

struct ServiceOptions {
  std::filesystem::path model_dir;
  int threads = 1;
  std::uint32_t max_frame_bytes = 256U * 1024U * 1024U;
  std::uint32_t max_batch = 256;
  PreprocessOptions preprocess;
};

// Run one persistent native predictor on stdin/stdout.  The function returns
// zero for a clean CLOSE/EOF and a non-zero value for startup, protocol, or
// inference failures.  stdout is reserved exclusively for protocol frames;
// diagnostics are written to stderr.
int RunNativeService(const ServiceOptions& options);

}  // namespace workpiece::ppshitu
