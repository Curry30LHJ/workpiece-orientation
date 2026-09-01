#include "feature_extractor.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace workpiece::ppshitu {
namespace {

std::atomic<std::uint64_t> next_instance_id{1};

std::string NarrowAsciiPath(const std::filesystem::path& path) {
  const std::wstring wide = path.wstring();
  std::string narrow;
  narrow.reserve(wide.size());
  for (const wchar_t character : wide) {
    if (character > 0x7f) {
      throw std::invalid_argument(
          "Paddle model path must be ASCII; stage the model before loading");
    }
    narrow.push_back(static_cast<char>(character));
  }
  return narrow;
}

std::size_t CheckedElementCount(const std::vector<int>& shape) {
  if (shape.empty()) {
    throw std::runtime_error("predictor returned an empty output shape");
  }
  std::size_t count = 1;
  for (const int dimension : shape) {
    if (dimension <= 0 ||
        count > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(dimension)) {
      throw std::runtime_error("predictor returned an invalid output shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

}  // namespace

FeatureExtractor::FeatureExtractor(const PredictorOptions& options) {
  if (options.cpu_threads <= 0) {
    throw std::invalid_argument("CPU thread count must be positive");
  }
  const std::filesystem::path model_file =
      options.model_dir / L"inference.pdmodel";
  const std::filesystem::path params_file =
      options.model_dir / L"inference.pdiparams";
  if (!std::filesystem::is_regular_file(model_file) ||
      !std::filesystem::is_regular_file(params_file)) {
    throw std::runtime_error(
        "model directory must contain inference.pdmodel and inference.pdiparams");
  }

  paddle_infer::Config config;
  config.SetModel(NarrowAsciiPath(model_file), NarrowAsciiPath(params_file));
  config.DisableGpu();
  if (options.enable_mkldnn) {
    config.EnableMKLDNN();
    config.SetMkldnnCacheCapacity(10);
  }
  config.SetCpuMathLibraryNumThreads(options.cpu_threads);
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  config.SwitchIrOptim(true);
  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  predictor_ = paddle_infer::CreatePredictor(config);
  if (!predictor_) {
    throw std::runtime_error("Paddle failed to create a predictor");
  }
  const std::vector<std::string> input_names = predictor_->GetInputNames();
  const std::vector<std::string> output_names = predictor_->GetOutputNames();
  if (input_names.size() != 1 || output_names.size() != 1) {
    throw std::runtime_error(
        "recognition model must expose exactly one input and one output");
  }
  input_name_ = input_names.front();
  output_name_ = output_names.front();
  instance_id_ = next_instance_id.fetch_add(1, std::memory_order_relaxed);
}

PredictionBatch FeatureExtractor::Predict(const BatchTensor& batch) {
  if (batch.batch == 0 || batch.channels != 3 || batch.height <= 0 ||
      batch.width <= 0) {
    throw std::invalid_argument("invalid prediction batch shape");
  }
  const std::size_t expected_values =
      batch.batch * static_cast<std::size_t>(batch.channels) *
      static_cast<std::size_t>(batch.height) *
      static_cast<std::size_t>(batch.width);
  if (batch.nchw.size() != expected_values ||
      batch.batch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("prediction batch data does not match its shape");
  }

  const auto inference_start = std::chrono::steady_clock::now();
  std::unique_ptr<paddle_infer::Tensor> input =
      predictor_->GetInputHandle(input_name_);
  input->Reshape({static_cast<int>(batch.batch), batch.channels, batch.height,
                  batch.width});
  input->CopyFromCpu(batch.nchw.data());
  if (!predictor_->Run()) {
    throw std::runtime_error("Paddle predictor Run() returned false");
  }
  std::unique_ptr<paddle_infer::Tensor> output =
      predictor_->GetOutputHandle(output_name_);
  if (output->type() != paddle_infer::FLOAT32) {
    throw std::runtime_error("recognition output must be float32");
  }
  const std::vector<int> output_shape = output->shape();
  if (output_shape.empty() || output_shape.front() <= 0 ||
      static_cast<std::size_t>(output_shape.front()) != batch.batch) {
    throw std::runtime_error("recognition output batch dimension changed");
  }
  const std::size_t output_count = CheckedElementCount(output_shape);
  if (output_count % batch.batch != 0) {
    throw std::runtime_error("recognition output cannot be split by batch row");
  }

  PredictionBatch result;
  result.rows = batch.batch;
  result.columns = output_count / batch.batch;
  result.embeddings.resize(output_count);
  output->CopyToCpu(result.embeddings.data());
  const auto inference_end = std::chrono::steady_clock::now();

  const auto normalize_start = inference_end;
  for (std::size_t row = 0; row < result.rows; ++row) {
    const std::size_t begin = row * result.columns;
    double squared_norm = 0.0;
    for (std::size_t column = 0; column < result.columns; ++column) {
      const float value = result.embeddings[begin + column];
      if (!std::isfinite(value)) {
        throw std::runtime_error("recognition output contains a non-finite value");
      }
      squared_norm += static_cast<double>(value) * value;
    }
    const double norm = std::sqrt(squared_norm);
    if (!std::isfinite(norm) || norm == 0.0) {
      throw std::runtime_error("recognition output has zero or invalid norm");
    }
    for (std::size_t column = 0; column < result.columns; ++column) {
      result.embeddings[begin + column] =
          static_cast<float>(result.embeddings[begin + column] / norm);
    }
  }
  const auto normalize_end = std::chrono::steady_clock::now();
  result.inference_ms = std::chrono::duration<double, std::milli>(
                            inference_end - inference_start)
                            .count();
  result.normalize_ms = std::chrono::duration<double, std::milli>(
                            normalize_end - normalize_start)
                            .count();
  return result;
}

}  // namespace workpiece::ppshitu
