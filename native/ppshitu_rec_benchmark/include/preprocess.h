#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace workpiece::ppshitu {

struct PreprocessOptions {
  int width = 224;
  int height = 224;
  std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
  std::array<float, 3> std{0.229f, 0.224f, 0.225f};
  float scale = 1.0f / 255.0f;
};

struct ImageTensor {
  std::vector<float> nchw;
  int channels = 3;
  int height = 0;
  int width = 0;
};

struct BatchTensor {
  std::vector<float> nchw;
  std::size_t batch = 0;
  int channels = 3;
  int height = 0;
  int width = 0;
};

ImageTensor LoadAndPreprocess(const std::filesystem::path& path,
                              const PreprocessOptions& options);

BatchTensor StackBatch(const std::vector<ImageTensor>& images);

}  // namespace workpiece::ppshitu

