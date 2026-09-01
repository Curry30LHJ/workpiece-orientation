#pragma once

#include "preprocess.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <paddle_inference_api.h>

namespace workpiece::ppshitu {

struct PredictorOptions {
  std::filesystem::path model_dir;
  int cpu_threads = 1;
  bool enable_mkldnn = true;
};

struct StageTimings {
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double postprocess_ms = 0.0;
};

struct PredictionBatch {
  std::vector<float> embeddings;
  std::size_t rows = 0;
  std::size_t columns = 0;
  StageTimings timings;
  // Retained for compatibility with the benchmark report schema.  The
  // inference value mirrors timings.inference_ms and normalize_ms mirrors
  // the postprocess stage.
  double inference_ms = 0.0;
  double normalize_ms = 0.0;
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const PredictorOptions& options);

  PredictionBatch Predict(const BatchTensor& batch);
  std::size_t feature_dimension() const noexcept { return feature_dimension_; }
  std::uint64_t instance_id() const noexcept { return instance_id_; }

 private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::string input_name_;
  std::string output_name_;
  std::size_t feature_dimension_ = 0;
  std::uint64_t instance_id_ = 0;
};

}  // namespace workpiece::ppshitu
