#include "ppshitu_protocol.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace workpiece::ppshitu::protocol {
namespace {

constexpr std::array<std::uint8_t, 4> kMagic = {'P', 'P', 'S', 'H'};

bool IsKnownKind(const std::uint16_t value) {
  return value >= static_cast<std::uint16_t>(Kind::kHello) &&
         value <= static_cast<std::uint16_t>(Kind::kClose);
}

void SetError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

template <typename T>
bool IsPositive(T value) {
  return value > static_cast<T>(0);
}

void AppendU16(std::vector<std::uint8_t>* output, const std::uint16_t value) {
  output->push_back(static_cast<std::uint8_t>(value & 0xffU));
  output->push_back(static_cast<std::uint8_t>((value >> 8U) & 0xffU));
}

void AppendU32(std::vector<std::uint8_t>* output, const std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output->push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

void AppendF32(std::vector<std::uint8_t>* output, const float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit");
  std::memcpy(&bits, &value, sizeof(bits));
  AppendU32(output, bits);
}

bool AppendString(std::vector<std::uint8_t>* output,
                  const std::string& value,
                  const bool allow_empty,
                  const char* name,
                  std::string* error) {
  if (!allow_empty && value.empty()) {
    SetError(error, name);
    return false;
  }
  if (value.size() > kMaxStringBytes) {
    SetError(error, "string exceeds UTF-8 length limit");
    return false;
  }
  AppendU16(output, static_cast<std::uint16_t>(value.size()));
  output->insert(output->end(), value.begin(), value.end());
  return true;
}

class Cursor {
 public:
  explicit Cursor(const std::vector<std::uint8_t>& bytes) : bytes_(bytes) {}

  bool ReadU16(std::uint16_t* value) {
    if (value == nullptr || offset_ + 2 > bytes_.size()) {
      return false;
    }
    *value = static_cast<std::uint16_t>(bytes_[offset_]) |
             static_cast<std::uint16_t>(bytes_[offset_ + 1]) << 8U;
    offset_ += 2;
    return true;
  }

  bool ReadU32(std::uint32_t* value) {
    if (value == nullptr || offset_ + 4 > bytes_.size()) {
      return false;
    }
    *value = static_cast<std::uint32_t>(bytes_[offset_]) |
             static_cast<std::uint32_t>(bytes_[offset_ + 1]) << 8U |
             static_cast<std::uint32_t>(bytes_[offset_ + 2]) << 16U |
             static_cast<std::uint32_t>(bytes_[offset_ + 3]) << 24U;
    offset_ += 4;
    return true;
  }

  bool ReadU64(std::uint64_t* value) {
    if (value == nullptr || offset_ + 8 > bytes_.size()) {
      return false;
    }
    *value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
      *value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
    }
    return true;
  }

  bool ReadF32(float* value) {
    std::uint32_t bits = 0;
    if (!ReadU32(&bits) || value == nullptr) {
      return false;
    }
    static_assert(sizeof(bits) == sizeof(*value), "float must be 32-bit");
    std::memcpy(value, &bits, sizeof(bits));
    return true;
  }

  bool ReadBytes(const std::size_t count, std::vector<std::uint8_t>* value) {
    if (value == nullptr || count > bytes_.size() - offset_) {
      return false;
    }
    value->assign(bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
                  bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
    offset_ += count;
    return true;
  }

  bool ReadString(const bool allow_empty, std::string* value) {
    std::uint16_t length = 0;
    if (value == nullptr || !ReadU16(&length) ||
        static_cast<std::size_t>(length) > bytes_.size() - offset_) {
      return false;
    }
    if (!allow_empty && length == 0) {
      return false;
    }
    value->assign(reinterpret_cast<const char*>(bytes_.data() + offset_), length);
    offset_ += length;
    return true;
  }

  std::size_t remaining() const noexcept { return bytes_.size() - offset_; }

 private:
  const std::vector<std::uint8_t>& bytes_;
  std::size_t offset_ = 0;
};

bool ReadExact(std::istream& input, std::uint8_t* destination, std::size_t count) {
  while (count != 0) {
    input.read(reinterpret_cast<char*>(destination),
               static_cast<std::streamsize>(count));
    const std::streamsize read_count = input.gcount();
    if (read_count <= 0) {
      return false;
    }
    destination += static_cast<std::size_t>(read_count);
    count -= static_cast<std::size_t>(read_count);
  }
  return true;
}

bool ValidateMaxPayload(const std::uint32_t max_payload_bytes,
                        std::string* error) {
  if (max_payload_bytes == 0) {
    SetError(error, "max payload must be positive");
    return false;
  }
  return true;
}

bool ValidateFrame(const Frame& frame,
                   const std::uint32_t max_payload_bytes,
                   std::string* error) {
  if (!ValidateMaxPayload(max_payload_bytes, error)) {
    return false;
  }
  const auto kind = static_cast<std::uint16_t>(frame.kind);
  if (!IsKnownKind(kind)) {
    SetError(error, "unknown frame kind");
    return false;
  }
  if (frame.payload.size() > max_payload_bytes ||
      frame.payload.size() > std::numeric_limits<std::uint32_t>::max()) {
    SetError(error, "payload exceeds frame limit");
    return false;
  }
  return true;
}

bool ValidateTimings(const float preprocess_ms,
                     const float inference_ms,
                     const float postprocess_ms,
                     std::string* error) {
  if (!std::isfinite(preprocess_ms) || preprocess_ms < 0.0F ||
      !std::isfinite(inference_ms) || inference_ms < 0.0F ||
      !std::isfinite(postprocess_ms) || postprocess_ms < 0.0F) {
    SetError(error, "timings must be finite and non-negative");
    return false;
  }
  return true;
}

bool ValidateEmbeddingNorms(const std::vector<float>& embeddings,
                            const std::uint32_t rows,
                            const std::uint32_t columns,
                            std::string* error) {
  const std::size_t expected = static_cast<std::size_t>(rows) * columns;
  if (embeddings.size() != expected) {
    SetError(error, "embedding count does not match dimensions");
    return false;
  }
  for (std::uint32_t row = 0; row < rows; ++row) {
    double squared_norm = 0.0;
    for (std::uint32_t column = 0; column < columns; ++column) {
      const float value = embeddings[static_cast<std::size_t>(row) * columns + column];
      if (!std::isfinite(value)) {
        SetError(error, "embedding contains non-finite value");
        return false;
      }
      squared_norm += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm <= 0.0 || std::abs(norm - 1.0) > 1e-3) {
      SetError(error, "embedding must have unit norm");
      return false;
    }
  }
  return true;
}

}  // namespace

ReadStatus ReadFrame(std::istream& input,
                     Frame* frame,
                     const std::uint32_t max_payload_bytes,
                     std::string* error) {
  if (frame == nullptr || !ValidateMaxPayload(max_payload_bytes, error)) {
    if (frame == nullptr) {
      SetError(error, "frame output is null");
    }
    return ReadStatus::kError;
  }

  std::array<std::uint8_t, kFrameHeaderBytes> header{};
  input.read(reinterpret_cast<char*>(header.data()), 1);
  if (input.gcount() == 0) {
    if (input.eof()) {
      return ReadStatus::kCleanEof;
    }
    SetError(error, "unable to read frame header");
    return ReadStatus::kError;
  }
  if (!ReadExact(input, header.data() + 1, kFrameHeaderBytes - 1)) {
    SetError(error, "truncated frame header");
    return ReadStatus::kError;
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
    SetError(error, "frame magic is invalid");
    return ReadStatus::kError;
  }

  const std::uint16_t version = static_cast<std::uint16_t>(header[4]) |
                                static_cast<std::uint16_t>(header[5]) << 8U;
  if (version != kProtocolVersion) {
    SetError(error, "frame version is unsupported");
    return ReadStatus::kError;
  }
  const std::uint16_t kind_value = static_cast<std::uint16_t>(header[6]) |
                                   static_cast<std::uint16_t>(header[7]) << 8U;
  if (!IsKnownKind(kind_value)) {
    SetError(error, "unknown frame kind");
    return ReadStatus::kError;
  }
  std::uint32_t payload_length = 0;
  for (unsigned shift = 0; shift < 32; shift += 8) {
    payload_length |= static_cast<std::uint32_t>(header[8 + shift / 8]) << shift;
  }
  if (payload_length > max_payload_bytes) {
    SetError(error, "payload exceeds frame limit");
    return ReadStatus::kError;
  }
  std::uint64_t request_id = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    request_id |= static_cast<std::uint64_t>(header[12 + shift / 8]) << shift;
  }

  frame->kind = static_cast<Kind>(kind_value);
  frame->request_id = request_id;
  frame->payload.assign(payload_length, 0);
  if (payload_length != 0 &&
      !ReadExact(input, frame->payload.data(), payload_length)) {
    frame->payload.clear();
    SetError(error, "truncated frame payload");
    return ReadStatus::kError;
  }
  return ReadStatus::kFrame;
}

bool WriteFrame(std::ostream& output,
                const Frame& frame,
                const std::uint32_t max_payload_bytes,
                std::string* error) {
  if (!ValidateFrame(frame, max_payload_bytes, error)) {
    return false;
  }
  std::array<std::uint8_t, kFrameHeaderBytes> header{};
  std::copy(kMagic.begin(), kMagic.end(), header.begin());
  header[4] = static_cast<std::uint8_t>(kProtocolVersion & 0xffU);
  header[5] = static_cast<std::uint8_t>((kProtocolVersion >> 8U) & 0xffU);
  const auto kind = static_cast<std::uint16_t>(frame.kind);
  header[6] = static_cast<std::uint8_t>(kind & 0xffU);
  header[7] = static_cast<std::uint8_t>((kind >> 8U) & 0xffU);
  const auto payload_length = static_cast<std::uint32_t>(frame.payload.size());
  for (unsigned shift = 0; shift < 32; shift += 8) {
    header[8 + shift / 8] =
        static_cast<std::uint8_t>((payload_length >> shift) & 0xffU);
  }
  for (unsigned shift = 0; shift < 64; shift += 8) {
    header[12 + shift / 8] =
        static_cast<std::uint8_t>((frame.request_id >> shift) & 0xffULL);
  }
  output.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
  if (!frame.payload.empty()) {
    output.write(reinterpret_cast<const char*>(frame.payload.data()),
                 static_cast<std::streamsize>(frame.payload.size()));
  }
  output.flush();
  if (!output) {
    SetError(error, "unable to write frame");
    return false;
  }
  return true;
}

bool EncodeHello(const Hello& hello,
                 const std::uint64_t request_id,
                 Frame* frame,
                 std::string* error) {
  if (frame == nullptr || !IsPositive(hello.feature_dim) ||
      hello.feature_dim > kMaxFeatureDimension || !IsPositive(hello.max_batch) ||
      hello.max_batch > kMaxBatch || !IsPositive(hello.threads)) {
    SetError(error, "HELLO numeric field is invalid");
    return false;
  }
  std::vector<std::uint8_t> payload;
  payload.reserve(32 + hello.service_version.size() + hello.model_sha256.size());
  AppendU16(&payload, kProtocolVersion);
  if (!AppendString(&payload, hello.service_version, false, "service_version is empty", error) ||
      !AppendString(&payload, hello.model_sha256, false, "model_sha256 is empty", error)) {
    return false;
  }
  AppendU32(&payload, hello.feature_dim);
  AppendU32(&payload, hello.max_batch);
  AppendU32(&payload, hello.threads);
  frame->kind = Kind::kHello;
  frame->request_id = request_id;
  frame->payload = std::move(payload);
  return true;
}

bool DecodeHello(const Frame& frame, Hello* hello, std::string* error) {
  if (hello == nullptr || frame.kind != Kind::kHello) {
    SetError(error, "expected HELLO frame");
    return false;
  }
  Cursor cursor(frame.payload);
  std::uint16_t version = 0;
  if (!cursor.ReadU16(&version) || version != kProtocolVersion ||
      !cursor.ReadString(false, &hello->service_version) ||
      !cursor.ReadString(false, &hello->model_sha256) ||
      !cursor.ReadU32(&hello->feature_dim) ||
      !cursor.ReadU32(&hello->max_batch) ||
      !cursor.ReadU32(&hello->threads) ||
      cursor.remaining() != 0 || hello->feature_dim == 0 ||
      hello->feature_dim > kMaxFeatureDimension || hello->max_batch == 0 ||
      hello->max_batch > kMaxBatch || hello->threads == 0) {
    SetError(error, "HELLO payload is invalid");
    return false;
  }
  return true;
}

bool DecodePredict(const Frame& frame,
                   const std::uint32_t max_batch,
                   PredictPayload* payload,
                   std::string* error) {
  if (payload == nullptr || frame.kind != Kind::kPredict || max_batch == 0 ||
      max_batch > kMaxBatch) {
    SetError(error, "expected bounded PREDICT frame");
    return false;
  }
  Cursor cursor(frame.payload);
  std::uint32_t count = 0;
  if (!cursor.ReadU32(&count) || count == 0 || count > max_batch) {
    SetError(error, "PREDICT batch count is invalid");
    return false;
  }
  payload->images.clear();
  payload->images.reserve(count);
  for (std::uint32_t index = 0; index < count; ++index) {
    ImageRecord image;
    std::uint32_t data_length = 0;
    if (!cursor.ReadU32(&image.width) || !cursor.ReadU32(&image.height) ||
        !cursor.ReadU16(&image.channels) || !cursor.ReadU32(&data_length) ||
        image.channels != 3 || image.width == 0 || image.height == 0 ||
        image.width > kMaxImageWidth || image.height > kMaxImageHeight) {
      SetError(error, "PREDICT image header is invalid");
      return false;
    }
    const std::uint64_t expected = static_cast<std::uint64_t>(image.width) *
                                   image.height * image.channels;
    if (expected > std::numeric_limits<std::uint32_t>::max() ||
        data_length != expected || !cursor.ReadBytes(data_length, &image.rgb)) {
      SetError(error, "PREDICT image byte count is invalid");
      return false;
    }
    payload->images.push_back(std::move(image));
  }
  if (cursor.remaining() != 0) {
    SetError(error, "PREDICT payload has trailing bytes");
    return false;
  }
  return true;
}

bool EncodeResult(const ResultPayload& result,
                  const std::uint64_t request_id,
                  Frame* frame,
                  std::string* error) {
  if (frame == nullptr || result.rows == 0 || result.rows > kMaxBatch ||
      result.columns == 0 || result.columns > kMaxFeatureDimension ||
      !ValidateTimings(result.preprocess_ms, result.inference_ms,
                       result.postprocess_ms, error) ||
      !ValidateEmbeddingNorms(result.embeddings, result.rows, result.columns,
                              error)) {
    if (frame == nullptr) {
      SetError(error, "result output is null");
    }
    return false;
  }
  std::vector<std::uint8_t> payload;
  payload.reserve(20 + result.embeddings.size() * sizeof(float));
  AppendU32(&payload, result.rows);
  AppendU32(&payload, result.columns);
  AppendF32(&payload, result.preprocess_ms);
  AppendF32(&payload, result.inference_ms);
  AppendF32(&payload, result.postprocess_ms);
  for (const float value : result.embeddings) {
    AppendF32(&payload, value);
  }
  frame->kind = Kind::kResult;
  frame->request_id = request_id;
  frame->payload = std::move(payload);
  return true;
}

bool DecodeResult(const Frame& frame,
                  const std::uint32_t max_batch,
                  ResultPayload* result,
                  std::string* error) {
  if (result == nullptr || frame.kind != Kind::kResult || max_batch == 0 ||
      max_batch > kMaxBatch) {
    SetError(error, "expected bounded RESULT frame");
    return false;
  }
  Cursor cursor(frame.payload);
  if (!cursor.ReadU32(&result->rows) || !cursor.ReadU32(&result->columns) ||
      !cursor.ReadF32(&result->preprocess_ms) ||
      !cursor.ReadF32(&result->inference_ms) ||
      !cursor.ReadF32(&result->postprocess_ms) || result->rows == 0 ||
      result->rows > max_batch || result->columns == 0 ||
      result->columns > kMaxFeatureDimension ||
      !ValidateTimings(result->preprocess_ms, result->inference_ms,
                       result->postprocess_ms, error)) {
    SetError(error, "RESULT header is invalid");
    return false;
  }
  const std::uint64_t expected = static_cast<std::uint64_t>(result->rows) *
                                 result->columns;
  if (expected > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      cursor.remaining() != expected * sizeof(float)) {
    SetError(error, "RESULT embedding byte count is invalid");
    return false;
  }
  result->embeddings.resize(static_cast<std::size_t>(expected));
  for (float& value : result->embeddings) {
    if (!cursor.ReadF32(&value)) {
      SetError(error, "RESULT embedding data is truncated");
      return false;
    }
  }
  return ValidateEmbeddingNorms(result->embeddings, result->rows,
                                result->columns, error);
}

bool EncodeError(const ErrorPayload& payload,
                 const std::uint64_t request_id,
                 Frame* frame,
                 std::string* error) {
  if (frame == nullptr) {
    SetError(error, "error output is null");
    return false;
  }
  std::vector<std::uint8_t> encoded;
  if (!AppendString(&encoded, payload.code, false, "error code is empty", error) ||
      !AppendString(&encoded, payload.message, false, "error message is empty", error) ||
      !AppendString(&encoded, payload.diagnostic, true, "", error)) {
    return false;
  }
  frame->kind = Kind::kError;
  frame->request_id = request_id;
  frame->payload = std::move(encoded);
  return true;
}

bool DecodeError(const Frame& frame, ErrorPayload* payload, std::string* error) {
  if (payload == nullptr || frame.kind != Kind::kError) {
    SetError(error, "expected ERROR frame");
    return false;
  }
  Cursor cursor(frame.payload);
  if (!cursor.ReadString(false, &payload->code) ||
      !cursor.ReadString(false, &payload->message) ||
      !cursor.ReadString(true, &payload->diagnostic) || cursor.remaining() != 0) {
    SetError(error, "ERROR payload is invalid");
    return false;
  }
  return true;
}

}  // namespace workpiece::ppshitu::protocol
