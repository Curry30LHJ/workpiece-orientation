# PP-ShiTuV2 Native C++ Inference Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a Windows x64 native Paddle Inference benchmark for the existing PP-ShiTuV2 recognition model, then decide from measured embedding parity and latency evidence whether native product integration is justified.

**Architecture:** Keep the current Qt and Python backend unchanged. Add an isolated C++ executable that performs Unicode-safe image loading, the exact PaddleClas recognition preprocessing, batch feature extraction, L2 normalization, and structured timing; add a Python comparison driver that runs the current `RecPredictor` against the same ordered inputs, checks final fast-path decisions, and evaluates the 15% integration gate.

**Tech Stack:** C++17, Visual Studio 2019 x64, CMake, Paddle Inference Windows CPU AVX/MKL, OpenCV 4.6 Windows development package, Python 3.10, NumPy, pytest, PowerShell 5.1.

## Global Constraints

- Work only on `feature/20260901/ppshitu-cpp-inference`, based on `5c600532689e8eb7cfc32217beb7181affe9f0e1`.
- Do not change PP-ShiTu weights, ALIKED, LightGlue, geometry rules, Ridge parameters, fusion thresholds, Qt behavior, workpiece libraries, or template-cache formats.
- The native program covers the recognition model only; do not add main-body detection or FAISS.
- Use MSVC 2019 x64 and an official Paddle Inference 3.x Windows CPU AVX/MKL package. Record the exact archive SHA-256 and runtime version.
- Do not download dependencies from CMake or during a normal build. Local dependency roots are explicit PowerShell parameters and remain outside Git.
- Preserve input order. Invalid images, invalid arguments, missing model files, non-finite embeddings, and dimension changes are explicit failures; no silent skipping.
- Direct speed comparisons use the common decode/preprocess/feature/L2 boundary. Full-backend improvement is a clearly labelled projection until a later product integration exists.
- Native/Python cosine similarity must be at least `0.9999`, maximum absolute embedding error at most `0.001`, and labels/review decisions must match exactly.
- Native feature-extraction five-image P95 and projected full-pipeline P95 must each improve by at least 15% before recommending integration.
- The i5-8250U five-image `P95 <= 100 ms` claim requires a later real full-backend measurement on that machine; a projection is not acceptance evidence.
- Follow red-green-refactor. No production behavior is written before its corresponding failing test has been observed.

---

## File Map

- `native/ppshitu_rec_benchmark/CMakeLists.txt`: offline MSVC build graph and Paddle/OpenCV linkage.
- `native/ppshitu_rec_benchmark/include/preprocess.h`: Unicode image loading and NCHW preprocessing API.
- `native/ppshitu_rec_benchmark/src/preprocess.cpp`: BGR-to-RGB, resize, normalize, HWC-to-NCHW implementation.
- `native/ppshitu_rec_benchmark/include/feature_extractor.h`: predictor options, timing, and ordered embedding API.
- `native/ppshitu_rec_benchmark/src/feature_extractor.cpp`: Paddle predictor creation, real batch execution, L2 normalization.
- `native/ppshitu_rec_benchmark/src/main.cpp`: CLI parsing, JSON-array input, worker isolation, measurement, embedding dump, JSON report.
- `native/ppshitu_rec_benchmark/NOTICE.md`: PaddleClas-derived implementation attribution.
- `native/ppshitu_rec_benchmark/dependencies.json`: actual downloaded archive URLs, versions, and SHA-256 values.
- `scripts/build_cpp_inference_benchmark.ps1`: deterministic offline configure/build/runtime-DLL staging.
- `scripts/compare_cpp_python_inference.py`: Python baseline, native invocation, embedding/decision comparison, and gate calculation.
- `tests/test_cpp_inference_benchmark_contract.py`: build/CLI/report/order/error behavior.
- `tests/test_compare_cpp_python_inference.py`: comparison math, projection, and gate behavior.
- `docs/verification/ppshitu-cpp-inference-evaluation-results.md`: commands, artifacts, parity, performance, and final integration decision.

---

### Task 1: Comparison Metrics and Integration-Gate Contract

**Files:**
- Create: `scripts/compare_cpp_python_inference.py`
- Create: `tests/test_compare_cpp_python_inference.py`

**Interfaces:**
- Produces: `read_f32_matrix(path: Path, rows: int, columns: int) -> np.ndarray`
- Produces: `compare_embeddings(reference: np.ndarray, candidate: np.ndarray, *, min_cosine: float = 0.9999, max_abs_error: float = 0.001) -> dict[str, object]`
- Produces: `project_pipeline_p95(*, python_pipeline_p95_ms: float, python_feature_p95_ms: float, native_feature_p95_ms: float) -> float`
- Produces: `evaluate_integration_gate(*, embedding_metrics: Mapping[str, object], labels_match: bool, review_decisions_match: bool, python_feature_p95_ms: float, native_feature_p95_ms: float, python_pipeline_p95_ms: float) -> dict[str, object]`
- Consumed by: Task 5 comparison orchestration and Task 6 verification report.

- [ ] **Step 1: Write failing tests for hand-derived embedding and gate outcomes**

```python
def test_compare_embeddings_rejects_a_rotated_vector_even_when_shapes_match():
    module = load_comparison_module()
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    result = module.compare_embeddings(reference, candidate)

    assert result["passed"] is False
    assert result["minimum_cosine"] == pytest.approx(0.0)
    assert result["maximum_absolute_error"] == pytest.approx(1.0)


def test_integration_gate_requires_both_measured_and_projected_fifteen_percent_gain():
    module = load_comparison_module()
    result = module.evaluate_integration_gate(
        embedding_metrics={"passed": True},
        labels_match=True,
        review_decisions_match=True,
        python_feature_p95_ms=100.0,
        native_feature_p95_ms=80.0,
        python_pipeline_p95_ms=132.0,
    )

    assert result["native_feature_improvement_ratio"] == pytest.approx(0.20)
    assert result["projected_pipeline_p95_ms"] == pytest.approx(112.0)
    assert result["projected_pipeline_improvement_ratio"] == pytest.approx(20.0 / 132.0)
    assert result["passed"] is True
```

Name the mutations these catch before saving: a dot product without normalization, an absolute-error maximum computed over the wrong axis, a projection that double-counts feature time, or a gate that accepts only one of the two 15% requirements.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_compare_cpp_python_inference.py -q -p no:cacheprovider
```

Expected: collection/import failure because `scripts/compare_cpp_python_inference.py` does not exist.

- [ ] **Step 3: Implement the minimal comparison functions**

Use float64 for metric accumulation and reject zero-norm/non-finite rows explicitly:

```python
def compare_embeddings(reference, candidate, *, min_cosine=0.9999, max_abs_error=0.001):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] == 0:
        raise ValueError("embedding matrices must be non-empty and have identical 2-D shapes")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("embedding matrices must contain only finite values")
    ref_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    if np.any(ref_norm == 0.0) or np.any(candidate_norm == 0.0):
        raise ValueError("embedding rows must have non-zero norm")
    cosines = np.sum(reference * candidate, axis=1) / (ref_norm * candidate_norm)
    minimum_cosine = float(np.min(cosines))
    maximum_absolute_error = float(np.max(np.abs(reference - candidate)))
    return {
        "rows": int(reference.shape[0]),
        "columns": int(reference.shape[1]),
        "minimum_cosine": minimum_cosine,
        "maximum_absolute_error": maximum_absolute_error,
        "passed": minimum_cosine >= min_cosine and maximum_absolute_error <= max_abs_error,
    }
```

`project_pipeline_p95` must reject negative values and `python_feature_p95_ms > python_pipeline_p95_ms`, then return `pipeline - python_feature + native_feature`. The gate returns every input, both improvement ratios, the projection formula operands, individual booleans, and one overall `passed` boolean.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the Step 2 command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add scripts/compare_cpp_python_inference.py tests/test_compare_cpp_python_inference.py
git commit -m "test: define native inference comparison gate"
```

---

### Task 2: Offline Build and Executable Contract

**Files:**
- Create: `native/ppshitu_rec_benchmark/CMakeLists.txt`
- Create: `native/ppshitu_rec_benchmark/src/main.cpp`
- Create: `native/ppshitu_rec_benchmark/NOTICE.md`
- Create: `scripts/build_cpp_inference_benchmark.ps1`
- Create: `tests/test_cpp_inference_benchmark_contract.py`

**Interfaces:**
- Produces: `ppshitu_rec_benchmark.exe` with stable exit codes and `--help`.
- Produces: PowerShell parameters `-PaddleInferenceRoot`, `-OpenCvRoot`, `-BuildDir`, `-Configuration Release`, and `-Clean`.
- Produces: error JSON with `schema_version=1`, `ok=false`, and `{code,message}`.
- Consumed by: Tasks 3-6.

- [ ] **Step 1: Write failing behavioral tests for missing dependency roots and CLI validation**

```python
def test_build_script_rejects_missing_paddle_root_before_configuring(tmp_path):
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(REPO_ROOT / "scripts" / "build_cpp_inference_benchmark.ps1"),
         "-PaddleInferenceRoot", str(tmp_path / "missing-paddle"),
         "-OpenCvRoot", str(tmp_path / "missing-opencv"),
         "-BuildDir", str(tmp_path / "build")],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "PaddleInferenceRoot" in result.stderr


@pytest.mark.integration
def test_native_cli_rejects_zero_threads_with_stable_json(native_exe, tmp_path):
    report = tmp_path / "error.json"
    result = subprocess.run(
        [str(native_exe), "--model-dir", "missing", "--image-list", "missing.json",
         "--report", str(report), "--threads", "0"],
        text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert json.loads(report.read_text(encoding="utf-8"))["error"]["code"] == "INVALID_ARGUMENT"
```

The native fixture reads `WORKPIECE_CPP_BENCHMARK_EXE` and fails with a clear integration prerequisite message when the variable is explicitly set to a missing file. Normal non-integration runs skip only when the variable is absent.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_benchmark_contract.py -q -p no:cacheprovider
```

Expected: failure because the build script is absent.

- [ ] **Step 3: Add an offline-only CMake graph and PowerShell build validator**

The CMake target uses no `FetchContent`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(ppshitu_rec_benchmark LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED PATHS "${OPENCV_ROOT}" NO_DEFAULT_PATH)
add_executable(ppshitu_rec_benchmark
    src/main.cpp)
target_include_directories(ppshitu_rec_benchmark PRIVATE
    include "${PADDLE_INFERENCE_ROOT}/paddle/include" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(ppshitu_rec_benchmark PRIVATE
    "${PADDLE_INFERENCE_ROOT}/paddle/lib/paddle_inference.lib" ${OpenCV_LIBS})
```

The PowerShell script resolves each path with `Resolve-Path -LiteralPath`, verifies `paddle_inference_api.h`, `paddle_inference.lib`, `paddle_inference.dll`, and `OpenCVConfig.cmake`, invokes the VS 2019 generator with `-A x64`, builds `Release`, and copies the required Paddle/OpenCV runtime DLLs beside the executable. It must not remove a build directory unless its resolved path is under the supplied `BuildDir` itself.

- [ ] **Step 4: Add the minimal argument parser and error report writer**

`main.cpp` first implements only argument validation and JSON escaping. Use `wmain`
on Windows, convert option names to UTF-8 only for parsing, and retain path values as
`std::filesystem::path`; this prevents Chinese model/report paths from being damaged by
the active console code page. Use these stable exit codes:

```cpp
enum class ExitCode : int {
  kSuccess = 0,
  kInvalidArgument = 2,
  kInputUnreadable = 3,
  kModelLoadFailed = 4,
  kInferenceFailed = 5,
  kReportWriteFailed = 6,
  kCpuUnsupported = 7,
};
```

Write the report through a temporary sibling file and rename it only after the stream closes successfully, so a crash cannot leave a valid-looking partial report.
Before model loading, check AVX support with OpenCV's runtime CPU feature query. Return `kCpuUnsupported` and `CPU_FEATURE_UNSUPPORTED` when AVX is absent; do not allow Paddle to fail later with an opaque illegal-instruction crash.

- [ ] **Step 5: Build with the downloaded local dependency roots and verify GREEN**

Use the fixed local roots:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_cpp_inference_benchmark.ps1 `
  -PaddleInferenceRoot "E:\Project\wang\ai_区分正反\.native-deps\paddle_inference" `
  -OpenCvRoot "E:\Project\wang\ai_区分正反\.native-deps\opencv" `
  -BuildDir "E:\Project\wang\ai_区分正反\.native-build\ppshitu-cpp"
$env:WORKPIECE_CPP_BENCHMARK_EXE = "E:\Project\wang\ai_区分正反\.native-build\ppshitu-cpp\Release\ppshitu_rec_benchmark.exe"
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_benchmark_contract.py -q -p no:cacheprovider
```

Expected: missing-root test and compiled CLI validation tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add native/ppshitu_rec_benchmark scripts/build_cpp_inference_benchmark.ps1 tests/test_cpp_inference_benchmark_contract.py
git commit -m "build: add offline native inference harness"
```

---

### Task 3: Unicode-Safe Preprocessing Parity

**Files:**
- Modify: `native/ppshitu_rec_benchmark/CMakeLists.txt`
- Create: `native/ppshitu_rec_benchmark/include/preprocess.h`
- Create: `native/ppshitu_rec_benchmark/src/preprocess.cpp`
- Modify: `native/ppshitu_rec_benchmark/src/main.cpp`
- Modify: `tests/test_cpp_inference_benchmark_contract.py`

**Interfaces:**
- Produces: `ImageTensor LoadAndPreprocess(const std::filesystem::path&, const PreprocessOptions&)`.
- Produces: `BatchTensor StackBatch(const std::vector<ImageTensor>&)` with contiguous `NCHW float32` data.
- Produces: diagnostic CLI mode `--preprocess-only --dump-inputs <file.f32>` for parity tests; it still validates ordered image input and writes the normal report.
- Produces: preprocessing arguments `--input-width`, `--input-height`, `--scale`, `--mean-rgb`, and `--std-rgb`; defaults match the current YAML, while the Python driver passes the values it actually read from that YAML.
- Consumed by: Task 4 feature extractor.

- [ ] **Step 1: Write a failing test using a hand-checked 2×2 RGB fixture in a Chinese path**

Create a lossless PNG whose RGB pixels are `[(0,0,0), (255,0,0), (0,255,0), (0,0,255)]`. Invoke the executable with a test-only preprocessing configuration of width/height 2 and identity resize. Read the dumped float32 NCHW tensor and compare against these literal channel values after `(value / 255 - mean) / std`:

```python
expected_r = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
expected_g = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
expected_b = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
expected = np.concatenate([
    (expected_r - 0.485) / 0.229,
    (expected_g - 0.456) / 0.224,
    (expected_b - 0.406) / 0.225,
]).astype(np.float32)
np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
```

Also assert that two image paths are returned in the same order and that one unreadable image fails the entire invocation with `IMAGE_UNREADABLE`.

- [ ] **Step 2: Run the compiled integration test and verify RED**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_benchmark_contract.py -q -p no:cacheprovider -m integration
```

Expected: failure because `--preprocess-only` is not implemented.

- [ ] **Step 3: Implement Unicode decoding and exact preprocessing**

Use the following public data types:

```cpp
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
```

Read bytes with `std::ifstream(path, std::ios::binary)` using `std::filesystem::path`, decode with `cv::imdecode`, convert `cv::COLOR_BGR2RGB`, resize with `cv::INTER_LINEAR`, normalize each RGB channel, and write channel-major floats. Reject empty, non-three-channel, and non-finite output.
Parse the UTF-8 JSON image-list file as a strict top-level array of strings; reject
non-string elements and malformed escapes. Convert decoded UTF-8 strings with
`std::filesystem::u8path` before opening them. Add `src/preprocess.cpp` to the CMake
target only in this task, after the source exists.

- [ ] **Step 4: Rebuild and verify GREEN**

Run the Task 2 build command, then the Task 3 integration command. Expected: all preprocessing/order/error tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add native/ppshitu_rec_benchmark tests/test_cpp_inference_benchmark_contract.py
git commit -m "feat: match PaddleClas native preprocessing"
```

---

### Task 4: Paddle Predictor, Real Batch, and Isolated Worker Pool

**Files:**
- Modify: `native/ppshitu_rec_benchmark/CMakeLists.txt`
- Create: `native/ppshitu_rec_benchmark/include/feature_extractor.h`
- Create: `native/ppshitu_rec_benchmark/src/feature_extractor.cpp`
- Modify: `native/ppshitu_rec_benchmark/src/main.cpp`
- Modify: `tests/test_cpp_inference_benchmark_contract.py`

**Interfaces:**
- Produces: `FeatureExtractor(PredictorOptions)` with one owned `std::shared_ptr<paddle_infer::Predictor>`.
- Produces: `PredictionBatch FeatureExtractor::Predict(const BatchTensor&)`.
- Produces: `RunConfiguration{threads, workers, batch_size, warmup, iterations}`.
- Produces: `.f32` embedding dump ordered as row-major `[image_count, feature_dimension]`.
- Consumed by: Task 5 Python comparison and Task 6 performance sweep.

- [ ] **Step 1: Write failing real-model tests for embedding shape, normalization, order, and worker isolation**

```python
@pytest.mark.integration
def test_native_real_batch_preserves_order_and_returns_unit_embeddings(native_run, five_images):
    report, embeddings = native_run(five_images, workers=1, batch_size=5, warmup=1, iterations=1)
    assert report["ok"] is True
    assert report["input_count"] == 5
    assert embeddings.shape == (5, report["feature_dimension"])
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), np.ones(5), rtol=0, atol=1e-5)
    assert report["ordered_images"] == [str(path) for path in five_images]


@pytest.mark.integration
def test_four_workers_report_four_distinct_predictor_instances(native_run, five_images):
    report, _ = native_run(five_images, workers=4, batch_size=1, warmup=1, iterations=2)
    assert report["worker_count"] == 4
    assert len(set(report["predictor_instance_ids"])) == 4
```

The worker test catches the unsafe mutation where four threads share one predictor.

- [ ] **Step 2: Run the integration tests and verify RED**

Run the Task 3 integration command. Expected: failure because model prediction/report fields are absent.

- [ ] **Step 3: Implement predictor creation and exact L2 normalization**

```cpp
struct PredictorOptions {
  std::filesystem::path model_dir;
  int cpu_threads = 1;
  bool enable_mkldnn = true;
};

struct StageTimings {
  double decode_ms = 0.0;
  double preprocess_ms = 0.0;
  double inference_ms = 0.0;
  double normalize_ms = 0.0;
  double total_ms = 0.0;
};

struct PredictionBatch {
  std::vector<float> embeddings;
  std::size_t rows = 0;
  std::size_t columns = 0;
  StageTimings timings;
};
```

Configure `SetModel(inference.pdmodel, inference.pdiparams)`, `DisableGpu()`, `EnableMKLDNN()`, `SetMkldnnCacheCapacity(10)`, `SetCpuMathLibraryNumThreads(cpu_threads)`, `SwitchIrOptim(true)`, `EnableMemoryOptim()`, and `DisableGlogInfo()`. Reshape the input to `[N,3,H,W]`; confirm output first dimension equals `N`; L2-normalize every row and reject zero/non-finite norms.
Add `src/feature_extractor.cpp` to the CMake target only in this task, after the source
exists.

- [ ] **Step 4: Implement real batch and worker scheduling**

For `workers=1`, call one predictor with batches of `batch_size`. For `workers>1`, construct exactly `workers` independent `FeatureExtractor` instances before warmup and bind each worker thread permanently to one instance. Use an atomic next-index counter and write results into pre-sized slots by original index. Never call one predictor from two threads.

Warmups are excluded from samples. Each measured iteration reports decode, preprocess, inference, normalize, and total arrays plus linear P50/P95/P99/max summaries. Hash input files and model files outside the timed section. On Windows, sample the process peak working set with `GetProcessMemoryInfo` after the measured loop and store it as `peak_working_set_bytes`; do not estimate memory from model-file size.

- [ ] **Step 5: Rebuild and verify GREEN**

Run the Task 2 build command and all `test_cpp_inference_benchmark_contract.py` tests with the native executable environment variable set. Expected: pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add native/ppshitu_rec_benchmark tests/test_cpp_inference_benchmark_contract.py
git commit -m "feat: benchmark native PP-ShiTu batch inference"
```

---

### Task 5: Python Baseline, Embedding Parity, and Fast-Path Decision Replay

**Files:**
- Modify: `scripts/compare_cpp_python_inference.py`
- Modify: `tests/test_compare_cpp_python_inference.py`

**Interfaces:**
- Produces: `EmbeddingRun`, a frozen dataclass with `embeddings: np.ndarray`, `ordered_images: tuple[str, ...]`, `timings_ms: dict[str, object]`, and `model_fingerprint: str`.
- Produces: `run_python_embeddings(project_root: Path, model_dir: Path, config_path: Path, image_paths: Sequence[Path], *, warmup: int, iterations: int, threads: int) -> EmbeddingRun` using the repository's current `RecPredictor` configuration.
- Produces: `run_native_embeddings(executable: Path, model_dir: Path, config_path: Path, image_paths: Sequence[Path], output_dir: Path, *, warmup: int, iterations: int, workers: int, batch_size: int, threads: int) -> EmbeddingRun` using one native executable invocation per configuration.
- Produces: `replay_fast_decisions(classifier: OrientationClassifier, cache: TemplateCache, image_paths: Sequence[Path], native_slot_embeddings: Mapping[str, np.ndarray], *, library_revision: int) -> dict[str, object]` using existing geometry and Ridge data without changing product code.
- Produces: comparison JSON containing input hashes, embedding metrics, label/review parity, timings, projection operands, and gate result.
- Consumed by: Task 6 real experiment and verification document.

- [ ] **Step 1: Write failing tests for native report validation and decision mismatches**

Use a fake executable that writes a complete native report and deterministic `.f32` rows. Assert the runner rejects wrong image order, wrong feature dimensions, missing timing stages, a nonzero native exit, and a single changed final label. Do not assert fake call counts; assert the comparison result and stable error raised by the real runner.

```python
def test_comparison_rejects_native_report_with_reordered_images(tmp_path):
    report = valid_native_report(tmp_path)
    report["ordered_images"] = list(reversed(report["ordered_images"]))
    path = tmp_path / "native.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ordered_images"):
        validate_native_report(path, expected_images=list(reversed(report["ordered_images"])))
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the Task 1 focused command. Expected: new tests fail because orchestration functions are absent.

- [ ] **Step 3: Implement Python baseline with the current production configuration**

Load images through the repository's Unicode-safe reader, convert BGR to RGB exactly as `OrientationClassifier._global_embeddings` does, and construct `RecPredictor` through `create_rec_predictor` and `prepare_paddle_model_path`. Apply warmup/iterations outside load time, L2-normalize through the current predictor, and write ordered float32 rows.
Read width, height, scale, mean, and standard deviation from `deploy/configs/inference_general.yaml`; pass those resolved values to the native CLI and record them in both reports. Stage the native model through the existing ASCII-path compatibility helper so the C++ model path contains only ASCII characters without modifying the source model.

- [ ] **Step 4: Implement native invocation and strict report validation**

Write the ordered image list as UTF-8 JSON, invoke the native executable once for each configuration, require a zero exit and `ok=true`, verify model/image hashes and order, then read the embedding matrix using the exact report dimensions. Capture stdout/stderr in the comparison report without treating log text as a pass condition.

- [ ] **Step 5: Implement final-decision replay without product-path edits**

For each query, use the loaded classifier's existing `FastGeometryProcessor` to produce the raw/front/back slots. Save those slots as lossless PNG files, extract native embeddings in one ordered invocation, then create a temporary `FastOrientationEngine` whose embed callback verifies the actual slot hashes before returning the recorded rows. Run the same immutable `FastRuntimeCache` and compare these observable fields with the normal Python prediction:

```python
DECISION_FIELDS = (
    "label", "needs_review", "global_label", "geometry_applied",
    "front_score", "back_score", "margin",
)
```

Exact-match categorical/boolean fields. Compare score fields with an absolute tolerance of `1e-5`; any tolerance failure marks decision parity false and records the image and field.

- [ ] **Step 6: Run focused and real-model comparison tests and verify GREEN**

Run:

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_compare_cpp_python_inference.py tests\test_cpp_inference_benchmark_contract.py -q -p no:cacheprovider
```

Then run the integration subset with `WORKPIECE_CPP_BENCHMARK_EXE` set. Expected: all pass.

- [ ] **Step 7: Commit Task 5**

```powershell
git add scripts/compare_cpp_python_inference.py tests/test_compare_cpp_python_inference.py
git commit -m "feat: compare native and Python orientation decisions"
```

---

### Task 6: Dependency Lock, Full Dataset Evidence, and Performance Sweep

**Files:**
- Create: `native/ppshitu_rec_benchmark/dependencies.json`
- Create: `docs/verification/ppshitu-cpp-inference-evaluation-results.md`
- Modify: `scripts/compare_cpp_python_inference.py` only if a defect is first reproduced by a failing test.
- Modify: `tests/test_compare_cpp_python_inference.py` only for defects discovered by the real experiment.

**Interfaces:**
- Consumes: exact native executable, existing model, existing three datasets, and current packaged Python benchmark evidence.
- Produces: immutable raw JSON reports under ignored `release_staging/cpp-inference-evaluation/`.
- Produces: committed dependency hashes and summarized verification evidence.

- [ ] **Step 1: Download and hash dependencies outside Git**

Use these official archives:

```text
Paddle Inference 3.0.0 Windows CPU AVX/MKL MSVC2019:
https://paddle-inference-lib.bj.bcebos.com/3.0.0/cxx_c/Windows/CPU/x86-64_avx-mkl-vs2019/paddle_inference.zip

OpenCV 4.6.0 Windows VC14/VC15:
https://github.com/opencv/opencv/releases/download/4.6.0/opencv-4.6.0-vc14_vc15.exe
```

Store/extract only under `E:\Project\wang\ai_区分正反\.native-deps`. Calculate each archive SHA-256 with `Get-FileHash -Algorithm SHA256` and write the actual values, byte sizes, resolved versions, and URLs into `dependencies.json`. Do not invent hashes from documentation.

- [ ] **Step 2: Build and capture the native artifact identity**

Run the Task 2 build command. Record executable SHA-256, all staged runtime DLL SHA-256 values, compiler version, Paddle version, OpenCV version, and Git commit in the ignored raw report directory and the verification Markdown.

- [ ] **Step 3: Run embedding and decision parity across all repository datasets**

Discover the dataset roots explicitly and record their paths and image counts. Use all readable images for embedding parity and the existing evaluation split/protocol for label/review parity. Empty or unavailable dataset roots make this step fail; they are not silently skipped.

Run the comparison script once per dataset and once for the fixed ordered five-image batch. Require every correctness gate from the design to pass before running performance conclusions.

- [ ] **Step 4: Run the same-input performance sweep**

For each of `1×1`, `1×2`, `1×4`, real batch-5, `2×2`, and `4×1`, use warmup 50 and 200 measured five-image iterations. Also run 1000 measured single-image iterations for 1, 2, and 4 threads. Save raw native and Python timing arrays, not only summaries.

Obtain the current full `predict_batch` P95 from a fresh run of `scripts/benchmark_batch_inference.py` against the verified CPU package. Calculate the projection only from the three documented operands:

```text
projected_pipeline_p95 = python_pipeline_p95
                       - python_feature_p95
                       + native_feature_p95
```

- [ ] **Step 5: Write the evidence-backed integration decision**

The verification document must state one of exactly two conclusions:

```text
INTEGRATE: all correctness gates pass, native feature P95 improves >=15%,
and projected pipeline P95 improves >=15%.

DO NOT INTEGRATE: at least one named gate failed; preserve the current Python
backend and prioritize the measured next bottleneck.
```

Do not claim i5 acceptance unless the real target machine produced a complete full-backend report.

- [ ] **Step 6: Run focused regression and commit Task 6**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest `
  tests\test_compare_cpp_python_inference.py `
  tests\test_cpp_inference_benchmark_contract.py `
  tests\test_orientation_classifier.py `
  tests\test_fast_orientation.py -q -p no:cacheprovider
git add native/ppshitu_rec_benchmark/dependencies.json docs/verification/ppshitu-cpp-inference-evaluation-results.md
git commit -m "docs: record native PP-ShiTu evaluation"
```

---

### Task 7: Full Regression, Branch Audit, and Push

**Files:**
- Modify only files proven necessary by a failing regression test.
- Verify all files committed in Tasks 1-6.

**Interfaces:**
- Produces: a clean branch except for the user's pre-existing ignored/untracked `runtime_reports/`.
- Produces: remote `origin/feature/20260901/ppshitu-cpp-inference` at the verified final commit.

- [ ] **Step 1: Run the full Python suite**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=E:\Project\wang\pp_813\.pytest-cpp-final-20260901
```

Use a fresh explicit directory and record the exact summary. If the managed sandbox produces Windows ACL errors, rerun the identical command outside the sandbox; do not classify ACL failures as code failures or passes.

- [ ] **Step 2: Run the full Qt 5.14.2 suite**

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1
```

Record all target totals and exit code. No Qt source should change in this branch.

- [ ] **Step 3: Run the forced native integration suite**

```powershell
$env:WORKPIECE_CPP_BENCHMARK_EXE = "E:\Project\wang\ai_区分正反\.native-build\ppshitu-cpp\Release\ppshitu_rec_benchmark.exe"
E:\python\anaconda3\envs\shitu\python.exe -m pytest `
  tests\test_cpp_inference_benchmark_contract.py `
  tests\test_compare_cpp_python_inference.py -q -p no:cacheprovider -m integration
```

Confirm the tests actually ran rather than all being skipped.

- [ ] **Step 4: Audit scope and artifacts**

```powershell
git diff 5c600532689e8eb7cfc32217beb7181affe9f0e1...HEAD --stat
git status --short --branch
git log --oneline --decorate 5c600532689e8eb7cfc32217beb7181affe9f0e1..HEAD
```

Confirm no dependency archive, model, generated embedding, build directory, report JSON, workpiece library, or geometry-rule data is tracked. Confirm every committed file traces to the approved evaluation design.

- [ ] **Step 5: Push the verified branch**

```powershell
git push origin feature/20260901/ppshitu-cpp-inference
```

- [ ] **Step 6: Completion audit**

Map every design deliverable and gate to an authoritative file, test result, raw report, or Git remote ref. Do not mark the goal complete if the native build, embedding parity, dataset decision parity, performance sweep, verification decision, full regressions, or remote push lacks evidence.
