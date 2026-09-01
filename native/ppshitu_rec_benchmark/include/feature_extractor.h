#pragma once

#include "preprocess.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <paddle_inference_api.h>

namespace workpiece::ppshitu {

struct PredictorOptions {
  std::filesystem::path model_dir;
  int cpu_threads = 1;
  bool enable_mkldnn = true;
};

struct PredictionBatch {
  std::vector<float> embeddings;
  std::size_t rows = 0;
  std::size_t columns = 0;
  double inference_ms = 0.0;
  double normalize_ms = 0.0;
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(const PredictorOptions& options);

  PredictionBatch Predict(const BatchTensor& batch);
  std::uint64_t instance_id() const noexcept { return instance_id_; }

 private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;
  std::string input_name_;
  std::string output_name_;
  std::uint64_t instance_id_ = 0;
};

}  // namespace workpiece::ppshitu
