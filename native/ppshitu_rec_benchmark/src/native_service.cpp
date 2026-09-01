#include "native_service.h"

#include "feature_extractor.h"
#include "ppshitu_protocol.h"

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace workpiece::ppshitu {
namespace {

constexpr char kServiceVersion[] = "ppshitu-native-cpp/1";

std::string WideToUtf8(const std::wstring& value) {
  if (value.empty()) {
    return {};
  }
  const int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                       static_cast<int>(value.size()), nullptr,
                                       0, nullptr, nullptr);
  if (size <= 0) {
    throw std::runtime_error("unable to encode path as UTF-8");
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  if (WideCharToMultiByte(CP_UTF8, 0, value.data(),
                          static_cast<int>(value.size()), result.data(), size,
                          nullptr, nullptr) <= 0) {
    throw std::runtime_error("unable to encode path as UTF-8");
  }
  return result;
}

std::string PathToUtf8(const std::filesystem::path& path) {
  return WideToUtf8(path.wstring());
}

bool IsAsciiPath(const std::filesystem::path& path) {
  for (const wchar_t character : path.wstring()) {
    if (character > 0x7f) {
      return false;
    }
  }
  return true;
}

class Sha256Hasher {
 public:
  Sha256Hasher() {
    if (BCryptOpenAlgorithmProvider(&algorithm_, BCRYPT_SHA256_ALGORITHM,
                                    nullptr, 0) < 0) {
      throw std::runtime_error("unable to open SHA-256 provider");
    }
    DWORD object_length = 0;
    DWORD returned = 0;
    if (BCryptGetProperty(algorithm_, BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&object_length),
                          sizeof(object_length), &returned, 0) < 0 ||
        object_length == 0) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
      algorithm_ = nullptr;
      throw std::runtime_error("unable to query SHA-256 provider");
    }
    object_.resize(object_length);
    if (BCryptCreateHash(algorithm_, &hash_, object_.data(), object_length,
                         nullptr, 0, 0) < 0) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
      algorithm_ = nullptr;
      throw std::runtime_error("unable to create SHA-256 hash");
    }
  }

  Sha256Hasher(const Sha256Hasher&) = delete;
  Sha256Hasher& operator=(const Sha256Hasher&) = delete;

  ~Sha256Hasher() {
    if (hash_ != nullptr) {
      BCryptDestroyHash(hash_);
    }
    if (algorithm_ != nullptr) {
      BCryptCloseAlgorithmProvider(algorithm_, 0);
    }
  }

  void Update(const unsigned char* data, std::size_t length) {
    while (length != 0) {
      const ULONG chunk = static_cast<ULONG>(std::min<std::size_t>(
          length, std::numeric_limits<ULONG>::max()));
      if (BCryptHashData(hash_, const_cast<PUCHAR>(data), chunk, 0) < 0) {
        throw std::runtime_error("unable to update SHA-256 hash");
      }
      data += chunk;
      length -= chunk;
    }
  }

  std::string Finish() {
    std::array<unsigned char, 32> digest{};
    if (BCryptFinishHash(hash_, digest.data(),
                         static_cast<ULONG>(digest.size()), 0) < 0) {
      throw std::runtime_error("unable to finish SHA-256 hash");
    }
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(digest.size() * 2);
    for (const unsigned char value : digest) {
      result.push_back(hex[value >> 4]);
      result.push_back(hex[value & 0x0f]);
    }
    return result;
  }

 private:
  BCRYPT_ALG_HANDLE algorithm_ = nullptr;
  BCRYPT_HASH_HANDLE hash_ = nullptr;
  std::vector<unsigned char> object_;
};

void HashFile(Sha256Hasher* hasher, const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("unable to open model file for hashing: " +
                             PathToUtf8(path));
  }
  std::vector<unsigned char> buffer(64 * 1024);
  while (stream) {
    stream.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = stream.gcount();
    if (count > 0) {
      hasher->Update(buffer.data(), static_cast<std::size_t>(count));
    }
  }
  if (!stream.eof()) {
    throw std::runtime_error("unable to read model file for hashing: " +
                             PathToUtf8(path));
  }
}

std::string ModelFingerprint(const std::filesystem::path& root) {
  if (!std::filesystem::is_directory(root)) {
    throw std::runtime_error("model directory does not exist");
  }
  std::vector<std::filesystem::path> files;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end(), [&](const auto& left, const auto& right) {
    return left.lexically_relative(root).generic_u8string() <
           right.lexically_relative(root).generic_u8string();
  });
  Sha256Hasher hasher;
  for (const auto& file : files) {
    const std::string relative =
        file.lexically_relative(root).generic_u8string();
    hasher.Update(reinterpret_cast<const unsigned char*>(relative.data()),
                  relative.size());
    HashFile(&hasher, file);
  }
  return hasher.Finish();
}

void ValidateOptions(const ServiceOptions& options) {
  if (options.model_dir.empty()) {
    throw std::invalid_argument("model directory is required");
  }
  if (options.threads <= 0) {
    throw std::invalid_argument("threads must be positive");
  }
  if (options.max_frame_bytes == 0 ||
      options.max_frame_bytes > protocol::kDefaultMaxFrameBytes) {
    throw std::invalid_argument("max frame bytes are out of range");
  }
  if (options.max_batch == 0 || options.max_batch > protocol::kMaxBatch) {
    throw std::invalid_argument("max batch is out of range");
  }
  if (options.preprocess.width <= 0 || options.preprocess.height <= 0 ||
      options.preprocess.width > static_cast<int>(protocol::kMaxImageWidth) ||
      options.preprocess.height >
          static_cast<int>(protocol::kMaxImageHeight)) {
    throw std::invalid_argument("preprocess dimensions are out of range");
  }
  if (!std::isfinite(options.preprocess.scale) ||
      options.preprocess.scale <= 0.0F) {
    throw std::invalid_argument("preprocess scale must be finite and positive");
  }
  for (std::size_t channel = 0; channel < 3; ++channel) {
    if (!std::isfinite(options.preprocess.mean[channel]) ||
        !std::isfinite(options.preprocess.std[channel]) ||
        options.preprocess.std[channel] <= 0.0F) {
      throw std::invalid_argument(
          "preprocess mean/std must be finite and std must be positive");
    }
  }
  const auto model_file = options.model_dir / L"inference.pdmodel";
  const auto params_file = options.model_dir / L"inference.pdiparams";
  if (!std::filesystem::is_regular_file(model_file) ||
      !std::filesystem::is_regular_file(params_file)) {
    throw std::runtime_error(
        "model directory must contain inference.pdmodel and inference.pdiparams");
  }
  if (!IsAsciiPath(options.model_dir)) {
    throw std::runtime_error(
        "native Paddle model path must be ASCII; stage the model before loading");
  }
}

std::string BoundedDiagnostic(const std::string& value) {
  constexpr std::size_t kDiagnosticLimit = 4096;
  if (value.size() <= kDiagnosticLimit) {
    return value;
  }
  return value.substr(0, kDiagnosticLimit);
}

bool SendError(const std::uint64_t request_id,
               const char* code,
               const std::string& message,
               const std::string& diagnostic,
               const std::uint32_t max_frame_bytes) {
  protocol::Frame frame;
  protocol::ErrorPayload payload;
  payload.code = code;
  payload.message = BoundedDiagnostic(message);
  payload.diagnostic = BoundedDiagnostic(diagnostic);
  std::string error;
  if (!protocol::EncodeError(payload, request_id, &frame, &error) ||
      !protocol::WriteFrame(std::cout, frame, max_frame_bytes, &error)) {
    std::cerr << "unable to write ERROR frame: " << error << '\n';
    return false;
  }
  return true;
}

double ElapsedMilliseconds(const std::chrono::steady_clock::time_point start,
                           const std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int RunNativeService(const ServiceOptions& options) {
  try {
    ValidateOptions(options);
    if (!cv::checkHardwareSupport(CV_CPU_AVX)) {
      throw std::runtime_error("AVX CPU support is required");
    }

    // Compute the source identity before constructing the predictor.  The
    // staged directory is expected to contain exactly the files consumed by
    // FeatureExtractor, and this digest is diagnostic only.
    const std::string model_sha256 = ModelFingerprint(options.model_dir);
    const PredictorOptions predictor_options{options.model_dir, options.threads,
                                             true};
    auto extractor = std::make_unique<FeatureExtractor>(predictor_options);

    // Run one tiny, deterministic inference so HELLO can advertise the
    // negotiated output dimension before the first client request.
    const std::vector<std::uint8_t> black_pixel{0, 0, 0};
    const ImageTensor warmup_image = PreprocessRgb(
        black_pixel.data(), black_pixel.size(), 1, 1, 3, options.preprocess);
    const BatchTensor warmup_batch = StackBatch({warmup_image});
    const PredictionBatch warmup_prediction = extractor->Predict(warmup_batch);
    if (extractor->feature_dimension() == 0 ||
        extractor->feature_dimension() > protocol::kMaxFeatureDimension ||
        warmup_prediction.columns != extractor->feature_dimension()) {
      throw std::runtime_error("unable to determine recognition feature dimension");
    }

    protocol::Hello hello;
    hello.service_version = kServiceVersion;
    hello.model_sha256 = model_sha256;
    hello.feature_dim = static_cast<std::uint32_t>(extractor->feature_dimension());
    hello.max_batch = options.max_batch;
    hello.threads = static_cast<std::uint32_t>(options.threads);
    protocol::Frame hello_frame;
    std::string error;
    if (!protocol::EncodeHello(hello, 0, &hello_frame, &error) ||
        !protocol::WriteFrame(std::cout, hello_frame, options.max_frame_bytes,
                              &error)) {
      std::cerr << "native service HELLO failed: " << error << '\n';
      return 4;
    }

    while (true) {
      protocol::Frame request;
      const protocol::ReadStatus status = protocol::ReadFrame(
          std::cin, &request, options.max_frame_bytes, &error);
      if (status == protocol::ReadStatus::kCleanEof) {
        return 0;
      }
      if (status == protocol::ReadStatus::kError) {
        std::cerr << "native service protocol read failed: " << error << '\n';
        return 5;
      }
      if (request.kind == protocol::Kind::kClose) {
        if (!request.payload.empty()) {
          if (!SendError(request.request_id, "INVALID_CLOSE",
                         "CLOSE payload must be empty", "request has trailing bytes",
                         options.max_frame_bytes)) {
            return 5;
          }
          continue;
        }
        return 0;
      }
      if (request.kind != protocol::Kind::kPredict) {
        if (!SendError(request.request_id, "UNSUPPORTED_REQUEST",
                       "only PREDICT and CLOSE requests are supported",
                       "unexpected frame kind", options.max_frame_bytes)) {
          return 5;
        }
        continue;
      }

      protocol::PredictPayload payload;
      if (!protocol::DecodePredict(request, options.max_batch, &payload,
                                   &error)) {
        if (!SendError(request.request_id, "INVALID_PREDICT", error,
                       "request payload validation failed",
                       options.max_frame_bytes)) {
          return 5;
        }
        continue;
      }

      try {
        const auto preprocess_start = std::chrono::steady_clock::now();
        std::vector<ImageTensor> images;
        images.reserve(payload.images.size());
        for (const protocol::ImageRecord& record : payload.images) {
          images.push_back(PreprocessRgb(
              record.rgb.data(), record.rgb.size(),
              static_cast<int>(record.width), static_cast<int>(record.height),
              static_cast<int>(record.channels), options.preprocess));
        }
        const BatchTensor batch = StackBatch(images);
        const auto preprocess_end = std::chrono::steady_clock::now();
        const PredictionBatch prediction = extractor->Predict(batch);

        protocol::ResultPayload result;
        result.rows = static_cast<std::uint32_t>(prediction.rows);
        result.columns = static_cast<std::uint32_t>(prediction.columns);
        result.preprocess_ms = static_cast<float>(
            ElapsedMilliseconds(preprocess_start, preprocess_end));
        result.inference_ms = static_cast<float>(prediction.timings.inference_ms);
        result.postprocess_ms = static_cast<float>(prediction.timings.postprocess_ms);
        result.embeddings = prediction.embeddings;
        protocol::Frame response;
        if (!protocol::EncodeResult(result, request.request_id, &response,
                                    &error) ||
            !protocol::WriteFrame(std::cout, response, options.max_frame_bytes,
                                  &error)) {
          if (!SendError(request.request_id, "RESULT_WRITE_FAILED",
                         "unable to encode or write RESULT", error,
                         options.max_frame_bytes)) {
            return 5;
          }
        }
      } catch (const std::exception& exception) {
        if (!SendError(request.request_id, "INFERENCE_FAILED", exception.what(),
                       "native predictor request failed",
                       options.max_frame_bytes)) {
          return 5;
        }
      }
    }
  } catch (const std::exception& exception) {
    std::cerr << "NATIVE_SERVICE_STARTUP_FAILED: " << exception.what() << '\n';
    return 4;
  } catch (...) {
    std::cerr << "NATIVE_SERVICE_STARTUP_FAILED: unknown exception\n";
    return 4;
  }
}

}  // namespace workpiece::ppshitu
