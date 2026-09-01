#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace workpiece::ppshitu::protocol {

constexpr std::uint16_t kProtocolVersion = 1;
constexpr std::size_t kFrameHeaderBytes = 20;
constexpr std::uint32_t kDefaultMaxFrameBytes = 256U * 1024U * 1024U;
constexpr std::uint32_t kMaxBatch = 4096;
constexpr std::uint32_t kMaxImageWidth = 8192;
constexpr std::uint32_t kMaxImageHeight = 8192;
constexpr std::uint32_t kMaxFeatureDimension = 65536;
constexpr std::uint16_t kMaxStringBytes = 65535;

static_assert(kFrameHeaderBytes == 4 + 2 + 2 + 4 + 8,
              "native protocol header size must remain 20 bytes");

enum class Kind : std::uint16_t {
  kHello = 1,
  kPredict = 2,
  kResult = 3,
  kError = 4,
  kClose = 5,
};

struct Frame {
  Kind kind = Kind::kClose;
  std::uint64_t request_id = 0;
  std::vector<std::uint8_t> payload;
};

enum class ReadStatus {
  kFrame,
  kCleanEof,
  kError,
};

struct Hello {
  std::string service_version;
  std::string model_sha256;
  std::uint32_t feature_dim = 0;
  std::uint32_t max_batch = 0;
  std::uint32_t threads = 0;
};

struct ImageRecord {
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint16_t channels = 0;
  std::vector<std::uint8_t> rgb;
};

struct PredictPayload {
  std::vector<ImageRecord> images;
};

struct ResultPayload {
  std::uint32_t rows = 0;
  std::uint32_t columns = 0;
  float preprocess_ms = 0.0F;
  float inference_ms = 0.0F;
  float postprocess_ms = 0.0F;
  std::vector<float> embeddings;
};

struct ErrorPayload {
  std::string code;
  std::string message;
  std::string diagnostic;
};

ReadStatus ReadFrame(std::istream& input,
                     Frame* frame,
                     std::uint32_t max_payload_bytes,
                     std::string* error);

bool WriteFrame(std::ostream& output,
                const Frame& frame,
                std::uint32_t max_payload_bytes,
                std::string* error);

bool EncodeHello(const Hello& hello,
                 std::uint64_t request_id,
                 Frame* frame,
                 std::string* error);

bool DecodeHello(const Frame& frame, Hello* hello, std::string* error);

bool DecodePredict(const Frame& frame,
                   std::uint32_t max_batch,
                   PredictPayload* payload,
                   std::string* error);

bool EncodeResult(const ResultPayload& result,
                  std::uint64_t request_id,
                  Frame* frame,
                  std::string* error);

bool DecodeResult(const Frame& frame,
                  std::uint32_t max_batch,
                  ResultPayload* result,
                  std::string* error);

bool EncodeError(const ErrorPayload& payload,
                 std::uint64_t request_id,
                 Frame* frame,
                 std::string* error);

bool DecodeError(const Frame& frame, ErrorPayload* payload, std::string* error);

}  // namespace workpiece::ppshitu::protocol
