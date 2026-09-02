#include "preprocess.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace workpiece::ppshitu {
namespace {

void ValidateOptions(const PreprocessOptions& options) {
  if (options.width <= 0 || options.height <= 0) {
    throw std::invalid_argument("preprocess dimensions must be positive");
  }
  if (!std::isfinite(options.scale) || options.scale <= 0.0F) {
    throw std::invalid_argument("preprocess scale must be finite and positive");
  }
  for (std::size_t channel = 0; channel < 3; ++channel) {
    if (!std::isfinite(options.mean[channel]) ||
        !std::isfinite(options.std[channel]) || options.std[channel] <= 0.0F) {
      throw std::invalid_argument(
          "preprocess mean/std must be finite and std must be positive");
    }
  }
}

std::vector<unsigned char> ReadBytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("unable to open image");
  }
  const std::streamsize length = stream.tellg();
  if (length <= 0 ||
      static_cast<unsigned long long>(length) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("image is empty or too large");
  }
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(reinterpret_cast<char*>(bytes.data()), length)) {
    throw std::runtime_error("unable to read image bytes");
  }
  return bytes;
}

std::size_t CheckedRgbBytes(const int width, const int height,
                            const int channels) {
  if (width <= 0 || height <= 0 || channels != 3) {
    throw std::invalid_argument(
        "RGB image dimensions must be positive and channels must be 3");
  }
  const auto width_size = static_cast<std::size_t>(width);
  const auto height_size = static_cast<std::size_t>(height);
  const auto channels_size = static_cast<std::size_t>(channels);
  if (width_size > std::numeric_limits<std::size_t>::max() / height_size ||
      width_size * height_size >
          std::numeric_limits<std::size_t>::max() / channels_size) {
    throw std::invalid_argument("RGB image dimensions overflow");
  }
  return width_size * height_size * channels_size;
}

ImageTensor PreprocessRgbMat(const cv::Mat& rgb,
                             const PreprocessOptions& options,
                             const double decode_ms) {
  ValidateOptions(options);
  if (rgb.empty() || rgb.type() != CV_8UC3 || !rgb.isContinuous()) {
    throw std::invalid_argument(
        "RGB image must be a non-empty contiguous 8-bit three-channel matrix");
  }

  const auto preprocess_start = std::chrono::steady_clock::now();
  cv::Mat resized;
  cv::resize(rgb, resized, cv::Size(options.width, options.height), 0.0, 0.0,
             cv::INTER_LINEAR);

  const std::size_t plane =
      static_cast<std::size_t>(options.width) *
      static_cast<std::size_t>(options.height);
  if (plane > std::numeric_limits<std::size_t>::max() / 3U) {
    throw std::invalid_argument("preprocess output dimensions overflow");
  }
  ImageTensor output;
  output.height = options.height;
  output.width = options.width;
  output.decode_ms = decode_ms;
  output.nchw.resize(plane * 3U);
  for (int row = 0; row < options.height; ++row) {
    const auto* pixels = resized.ptr<cv::Vec3b>(row);
    for (int column = 0; column < options.width; ++column) {
      const std::size_t position =
          static_cast<std::size_t>(row) * options.width + column;
      for (std::size_t channel = 0; channel < 3; ++channel) {
        const float normalized =
            (static_cast<float>(pixels[column][static_cast<int>(channel)]) *
                 options.scale -
             options.mean[channel]) /
            options.std[channel];
        if (!std::isfinite(normalized)) {
          throw std::runtime_error("preprocessing produced a non-finite value");
        }
        output.nchw[channel * plane + position] = normalized;
      }
    }
  }
  const auto preprocess_end = std::chrono::steady_clock::now();
  output.preprocess_ms = std::chrono::duration<double, std::milli>(
                             preprocess_end - preprocess_start)
                             .count();
  return output;
}

}  // namespace

ImageTensor PreprocessRgb(const std::uint8_t* data,
                          const std::size_t data_size,
                          const int width,
                          const int height,
                          const int channels,
                          const PreprocessOptions& options) {
  const std::size_t expected = CheckedRgbBytes(width, height, channels);
  if (data == nullptr || data_size == 0 || data_size != expected) {
    throw std::invalid_argument("RGB image byte count does not match dimensions");
  }
  const std::size_t row_stride = static_cast<std::size_t>(width) * 3U;
  cv::Mat view(height, width, CV_8UC3,
               const_cast<std::uint8_t*>(data), row_stride);
  // Clone so the output never aliases a caller-owned pipe buffer.
  return PreprocessRgbMat(view.clone(), options, 0.0);
}

ImageTensor LoadAndPreprocess(const std::filesystem::path& path,
                              const PreprocessOptions& options) {
  ValidateOptions(options);
  const auto decode_start = std::chrono::steady_clock::now();
  const std::vector<unsigned char> bytes = ReadBytes(path);
  if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("encoded image is too large to decode");
  }
  const cv::Mat encoded(1, static_cast<int>(bytes.size()), CV_8UC1,
                        const_cast<unsigned char*>(bytes.data()));
  const cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (bgr.empty()) {
    throw std::runtime_error("unable to decode image");
  }
  if (bgr.depth() != CV_8U || bgr.channels() != 3) {
    throw std::runtime_error("image must be an 8-bit three-channel image");
  }
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  const auto decode_end = std::chrono::steady_clock::now();
  const double decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();
  return PreprocessRgbMat(rgb.clone(), options, decode_ms);
}

BatchTensor StackBatch(const std::vector<ImageTensor>& images) {
  if (images.empty()) {
    throw std::invalid_argument("at least one image is required");
  }
  const ImageTensor& first = images.front();
  const std::size_t values_per_image = first.nchw.size();
  if (values_per_image == 0 || first.channels != 3 || first.height <= 0 ||
      first.width <= 0) {
    throw std::invalid_argument("invalid image tensor");
  }
  if (images.size() > std::numeric_limits<std::size_t>::max() /
                           values_per_image) {
    throw std::invalid_argument("batch tensor dimensions overflow");
  }

  BatchTensor batch;
  batch.batch = images.size();
  batch.channels = first.channels;
  batch.height = first.height;
  batch.width = first.width;
  batch.nchw.reserve(values_per_image * images.size());
  for (const ImageTensor& image : images) {
    if (image.channels != batch.channels || image.height != batch.height ||
        image.width != batch.width || image.nchw.size() != values_per_image) {
      throw std::invalid_argument("image tensors must have identical shapes");
    }
    batch.nchw.insert(batch.nchw.end(), image.nchw.begin(), image.nchw.end());
  }
  return batch;
}

}  // namespace workpiece::ppshitu
