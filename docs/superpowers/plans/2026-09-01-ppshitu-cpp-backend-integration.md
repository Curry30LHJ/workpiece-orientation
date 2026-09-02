# PP-ShiTuV2 Native C++ Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, long-lived native C++ PP-ShiTu feature backend to the existing Python orientation service, preserving geometry, cache, catalog, Qt, and TCP behavior while making single-process and batch throughput measurable on the i5-8250U target.

**Architecture:** Keep the Python TCP service as the business and decision layer. OrientationClassifier selects either the current Python RecPredictor or a NativePPClient; the client owns a persistent ppshitu_rec_service.exe child process that loads one Paddle Inference predictor and exchanges RGB image batches over a versioned binary stdin/stdout protocol. Batch sessions use one independent native client/process per worker slot, while Qt only forwards configuration and continues using the existing JSON TCP protocol.

**Tech Stack:** C++17, Visual Studio 2019 x64, CMake, Paddle Inference Windows CPU AVX/MKL, OpenCV 4.6, Python 3.10, NumPy, pytest, PowerShell 5.1, Qt 5.14.2, QtTest.

## Global Constraints

- pp_backend=native_cpp is supported only with compute_device=cpu and inference_mode=fast_geometry; python remains the default.
- Do not modify PP-ShiTu weights, ALIKED, LightGlue, geometry rules, Ridge parameters, fusion thresholds, template-cache formats, or the existing Qt TCP request/response schema.
- Native and Python embeddings must preserve row order and dimension; cosine similarity is at least 0.9999 and maximum absolute error is at most 0.001 for every comparison row.
- Existing 5+5 libraries and unequal template counts (1+1, 5+10, 10+15, and more than 30 per side) recover and predict without rebuilding.
- Native startup failures, service exits, protocol errors, timeouts, and dimension changes are explicit errors; native mode never silently falls back to Python.
- The source model fingerprint remains cache identity. A staged ASCII native model fingerprint is diagnostic/integrity-only (it may be compared at process startup, but must never replace or alter the cache key).
- Windows pipes are binary; stdout carries only frames and stderr carries logs. Frame size, image dimensions, batch count, and request wait time are bounded.
- Native startup does not create a Python RecPredictor or import Paddle unless a failing test proves a non-Paddle component requires it.
- Follow red-green-refactor and observe each focused failing test before changing production behavior.
- Work relative to E:\Project\wang\pp_813 and preserve the pre-existing untracked runtime_reports directory.

## Dependencies and checkpoints

- Tasks 1 and 2 are independent of Paddle and must be green before any service code is built.
- Task 3 depends on the C++ protocol target only for the final CMake graph; its preprocessing probe must remain runnable without a model or Paddle DLL.
- Task 4 depends on Tasks 2 and 3. Task 5 depends on Task 4's wire contract but can use the fake fixture before a real model is available.
- Task 6 depends on Tasks 3 and 5. Task 7 depends on Task 6. Task 8 depends on Task 7 and updates Qt only after the Python hello schema is fixed.
- Task 9 is the correctness gate; Task 10 is the performance/rollout gate. Do not change the default backend before both gates are recorded.
- After each task, run its focused command, inspect the diff, and commit only the files listed for that task. If a test exposes a cross-task defect, add the smallest targeted change to the owning task and document the reason in the commit body.

---

## File map

- native/ppshitu_rec_benchmark/include/ppshitu_protocol.h and src/ppshitu_protocol.cpp: canonical C++ header, payload codecs, and bounded stream I/O.
- native/ppshitu_rec_benchmark/src/protocol_probe_main.cpp: dependency-free protocol compatibility probe used by the Python contract tests.
- native/ppshitu_rec_benchmark/include/preprocess.h and src/preprocess.cpp: shared file/RGB preprocessing.
- native/ppshitu_rec_benchmark/src/preprocess_probe_main.cpp: native preprocessing contract probe that does not load Paddle.
- native/ppshitu_rec_benchmark/include/feature_extractor.h and src/feature_extractor.cpp: one Paddle predictor, batch execution, L2 normalization, and metadata.
- native/ppshitu_rec_benchmark/include/native_service.h and src/native_service.cpp: service loop translating frames into predictor calls.
- native/ppshitu_rec_benchmark/src/service_main.cpp: Windows service executable entry point.
- native/ppshitu_rec_benchmark/CMakeLists.txt: shared static libraries plus benchmark and service targets.
- src/native_pp_protocol.py: Python codec matching C++ bytes.
- src/native_pp_client.py: persistent subprocess client, handshake, timeout, stderr capture, and cleanup.
- src/orientation_classifier.py: backend selection, ASCII model staging, scalar extraction, batch sessions, and lifecycle.
- src/orientation_tcp_service.py: CLI validation, runtime loading, hello metadata, startup errors, and shutdown.
- qt_app/appconfig.h, appconfig.cpp, app_config.json.example: native backend configuration.
- qt_app/backendprocessmanager.h, backendprocessmanager.cpp, appheader.h, appheader.cpp, mainwindow.cpp: argument forwarding and diagnostics.
- tests/test_native_pp_protocol.py, tests/test_native_pp_client.py, tests/fixtures/fake_native_pp_service.py: protocol/client tests without Paddle.
- tests/test_cpp_inference_service_contract.py: compiled service handshake, order, malformed-frame, and real-model tests.
- tests/test_orientation_classifier.py, tests/test_orientation_tcp_service.py: backend selection, lifecycle, batch isolation, and metadata regression.
- qt_app/tests/test_appconfig.cpp, test_backendprocessmanager.cpp, test_appheader.cpp: Qt configuration and diagnostic tests.
- docs/verification/ppshitu-cpp-backend-integration-results.md: parity, performance, and rollout decision.

---

### Task 1: Define the Python wire-contract codec

**Files:**
- Create: src/native_pp_protocol.py
- Create: tests/test_native_pp_protocol.py

**Interfaces:**
- NativeProtocolError(ValueError) is the single validation/codec exception shared by the client and tests.
- Frame(kind: int, request_id: int, payload: bytes).
- encode_frame(kind: int, request_id: int, payload: bytes, *, max_payload_bytes: int) -> bytes.
- decode_frame_header(header: bytes, *, max_payload_bytes: int) -> tuple[int, int, int].
- encode_predict(images: Sequence[np.ndarray], request_id: int, *, max_payload_bytes: int) -> bytes.
- decode_hello(frame: Frame) -> NativeHello and decode_result(frame: Frame, expected_request_id: int) -> NativeResult.
- NativeHello(service_version: str, model_sha256: str, feature_dim: int, max_batch: int, threads: int).
- NativeResult(embeddings: np.ndarray, timings_ms: dict[str, float]); the client adds a measured `transport_ms` (write-to-read wall time minus the service-reported stages) without changing embedding values.

- [ ] **Step 1: Write the failing byte-level tests**

Use a 20-byte little-endian header and a 16-byte image record:

~~~python
HEADER = struct.Struct("<4sHHIQ")
IMAGE_HEADER = struct.Struct("<IIH2xI")

def test_frame_header_is_fixed_and_little_endian():
    encoded = encode_frame(KIND_PREDICT, 7, b"abc", max_payload_bytes=1024)
    assert encoded[:20] == HEADER.pack(b"PPSH", 1, KIND_PREDICT, 3, 7)
    assert len(encoded) == 23

def test_predict_rejects_empty_bad_channels_and_oversized_payload():
    with pytest.raises(NativeProtocolError, match="empty batch"):
        encode_predict([], 1, max_payload_bytes=1024)
    with pytest.raises(NativeProtocolError, match="3 channels"):
        encode_predict([np.zeros((2, 2, 1), np.uint8)], 1, max_payload_bytes=1024)
    with pytest.raises(NativeProtocolError, match="frame"):
        encode_predict([np.zeros((64, 64, 3), np.uint8)], 1, max_payload_bytes=32)

def test_result_decode_preserves_rows_and_checks_request_id():
    frame = make_result_frame(9, [[1.0, 0.0], [0.0, 1.0]])
    result = decode_result(frame, expected_request_id=9)
    np.testing.assert_allclose(result.embeddings, np.eye(2, dtype=np.float32))
    with pytest.raises(NativeProtocolError, match="request_id"):
        decode_result(frame, expected_request_id=10)
~~~

Also cover bad magic/version, truncated frames, zero dimensions, non-finite timing values, and a byte count different from count times dimension times four. Define test-only helpers such as `make_result_frame` in the test module so they use the public codec, not duplicated private parsing. Define ERROR payload strings as `u16 code_len + code + u16 message_len + message + u16 diagnostic_len + diagnostic` (all UTF-8, diagnostic may be empty); a decoder must cap each field and never echo image bytes.

- [ ] **Step 2: Run the focused tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_native_pp_protocol.py -q -p no:cacheprovider
~~~

Expected: collection/import failure because src/native_pp_protocol.py does not exist.

- [ ] **Step 3: Implement the codec**

Define MAGIC=b"PPSH", PROTOCOL_VERSION=1, kinds HELLO=1, PREDICT=2, RESULT=3, ERROR=4, CLOSE=5, and DEFAULT_MAX_FRAME_BYTES=256*1024*1024. Use struct formats <4sHHIQ, <IIH2xI, and <IIfff. Encode HELLO payload as `u16 payload_version + u16 service_len + service + u16 digest_len + digest + u32 feature_dim + u32 max_batch + u32 threads`; the payload version must equal PROTOCOL_VERSION. The model digest is the lowercase SHA-256 over sorted relative paths and file bytes using the existing `model_directory_sha256` algorithm. Encode PREDICT as count followed by width, height, channels, data_len, and tightly packed RGB bytes for each row in order. Encode RESULT as count, dimension, three float timings, and row-major little-endian float32 data. Validate C-contiguous HWC uint8 RGB arrays, checked multiplication, finite values, and unit norms; enforce independent positive count/dimension/width/height limits before allocating.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command. Expected: all codec and validation tests pass.

- [ ] **Step 5: Commit Task 1**

~~~powershell
git add src/native_pp_protocol.py tests/test_native_pp_protocol.py
git commit -m "test: define native PP-ShiTu wire protocol"
~~~

---

### Task 2: Add the C++ codec with the same wire contract

**Files:**
- Create: native/ppshitu_rec_benchmark/include/ppshitu_protocol.h
- Create: native/ppshitu_rec_benchmark/src/ppshitu_protocol.cpp
- Create: native/ppshitu_rec_benchmark/src/protocol_probe_main.cpp
- Modify: native/ppshitu_rec_benchmark/CMakeLists.txt
- Modify: tests/test_native_pp_protocol.py

**Interfaces:**
- workpiece::ppshitu::protocol::Frame.
- ReadStatus ReadFrame(std::istream&, Frame*, uint32_t max_payload_bytes, std::string*).
- bool WriteFrame(std::ostream&, const Frame&, uint32_t max_payload_bytes, std::string*).
- EncodeHello, DecodePredict, EncodeResult, and EncodeError with the exact Task 1 field order.

- [ ] **Step 1: Write the failing C++ byte-compatibility test**

Add `protocol_probe_main.cpp`, a dependency-free probe with `--max-payload <bytes>` that reads exactly one frame from stdin, writes `OK` or `REJECTED` to stderr, and exits 0/2 without writing stdout. Add an environment-gated pytest fixture named `WORKPIECE_CPP_PROTOCOL_PROBE_EXE`. Send the Python-generated CLOSE bytes with request_id 4 and assert the probe accepts them; send a payload length of 257 with a 256-byte limit and assert rejection. If the variable is explicitly set to a missing file, fail with a prerequisite message rather than skipping.

- [ ] **Step 2: Run the test and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_native_pp_protocol.py -q -p no:cacheprovider
~~~

Expected: Python tests pass and the compiled-probe test reports that the C++ probe has not been built.

- [ ] **Step 3: Implement bounded C++ serialization**

Serialize fields individually; never reinterpret-cast a padded C++ struct. Add static_assert for 20 header bytes. Return kCleanEof only when EOF occurs before the first header byte. Check magic, version, kind range, checked payload length, and all stream reads. Encode UTF-8 strings with a two-byte length and reject values over 65535 bytes. Keep stdout strictly binary. Define `ppshitu_protocol` with `target_include_directories(... PUBLIC include)` and no Paddle/OpenCV dependency so the probe and service share one codec implementation.

- [ ] **Step 4: Build and run the compatibility test**

Compile a `ppshitu_protocol` static library from `ppshitu_protocol.cpp` and a `ppshitu_protocol_probe` executable from `protocol_probe_main.cpp`, set `WORKPIECE_CPP_PROTOCOL_PROBE_EXE` to the resulting executable, and run the Step 2 command. Expected: Python and C++ accept/reject identical frames and produce identical bytes. The probe must not link Paddle or OpenCV.

- [ ] **Step 5: Commit Task 2**

~~~powershell
git add native/ppshitu_rec_benchmark/include/ppshitu_protocol.h native/ppshitu_rec_benchmark/src/ppshitu_protocol.cpp native/ppshitu_rec_benchmark/src/protocol_probe_main.cpp native/ppshitu_rec_benchmark/CMakeLists.txt tests/test_native_pp_protocol.py
git commit -m "feat: add native PP-ShiTu binary protocol codec"
~~~

---

### Task 3: Refactor native preprocessing and predictor into a shared core

**Files:**
- Modify: native/ppshitu_rec_benchmark/include/preprocess.h
- Modify: native/ppshitu_rec_benchmark/src/preprocess.cpp
- Modify: native/ppshitu_rec_benchmark/include/feature_extractor.h
- Modify: native/ppshitu_rec_benchmark/src/feature_extractor.cpp
- Create: native/ppshitu_rec_benchmark/src/preprocess_probe_main.cpp
- Modify: native/ppshitu_rec_benchmark/CMakeLists.txt
- Modify: native/ppshitu_rec_benchmark/src/main.cpp
- Modify: tests/test_cpp_inference_benchmark_contract.py

**Interfaces:**
- ImageTensor PreprocessRgb(const uint8_t*, size_t, int width, int height, int channels, const PreprocessOptions&).
- Existing LoadAndPreprocess and StackBatch remain source-compatible.
- size_t FeatureExtractor::feature_dimension() const noexcept.
- StageTimings{double preprocess_ms, double inference_ms, double postprocess_ms}; `PredictionBatch` exposes these timings while retaining its existing `inference_ms` and `normalize_ms` fields for benchmark JSON compatibility.
- ppshitu_core static library linked by the benchmark and service; the lightweight preprocessing library is linked by the preprocessing probe.

- [ ] **Step 1: Write the failing in-memory RGB test**

Create `preprocess_probe_main.cpp` with a `--raw-rgb <width> <height> <channels> <hex-bytes>` mode and a CTest/pytest entry point that sends raw tightly packed RGB bytes and compares its emitted channel-major float32 values with `LoadAndPreprocess` for the same 2x2 image. Assert that an incorrect byte count and a non-three-channel record return distinct nonzero errors. This catches row-stride confusion in the service path without loading Paddle.

- [ ] **Step 2: Run the focused native tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_benchmark_contract.py -q -p no:cacheprovider -m integration
~~~

Expected: the new in-memory probe target is absent.

- [ ] **Step 3: Implement one RGB preprocessing path**

Construct a CV_8UC3 view from the supplied bytes, treat the bytes as already-RGB (no channel swap), clone before returning, resize with INTER_LINEAR to the configured dimensions, apply scale 1/255 and the current mean/std, and write contiguous channel-major float32 values. Route file decoding through this helper only after the existing BGR `imdecode` result is converted to RGB once. Reject nonpositive dimensions, channel counts other than 3, overflow, empty data, and non-finite values.

- [ ] **Step 4: Split the CMake graph without changing benchmark semantics**

Use a shared preprocessing library plus the existing feature core; link the benchmark and service to the core, preserve the `ppshitu_protocol` target from Task 2 for the service, and link the probe only to preprocessing/OpenCV so it does not load Paddle:

~~~cmake
add_library(ppshitu_preprocess STATIC src/preprocess.cpp)
target_include_directories(ppshitu_preprocess PUBLIC include)
target_link_libraries(ppshitu_preprocess PUBLIC ${OpenCV_LIBS})
add_library(ppshitu_core STATIC src/feature_extractor.cpp)
target_include_directories(ppshitu_core PUBLIC include)
target_link_libraries(ppshitu_core PUBLIC ppshitu_preprocess "${PADDLE_INFERENCE_LIBRARY}" ${OpenCV_LIBS} Psapi Bcrypt)
add_executable(ppshitu_rec_benchmark src/main.cpp)
target_link_libraries(ppshitu_rec_benchmark PRIVATE ppshitu_core)
add_executable(ppshitu_preprocess_probe src/preprocess_probe_main.cpp)
target_link_libraries(ppshitu_preprocess_probe PRIVATE ppshitu_preprocess)
~~~

Keep existing benchmark JSON fields and exit codes; only add feature dimension and preprocessing values. Do not duplicate the `ppshitu_core` target when replacing the current monolithic target.

- [ ] **Step 5: Rebuild and run the existing native contract suite**

Run the current local dependency build, set `WORKPIECE_CPP_BENCHMARK_EXE` and `WORKPIECE_CPP_PREPROCESS_PROBE_EXE`, and run tests/test_cpp_inference_benchmark_contract.py. Expected: preprocessing, Unicode path, order, normalization, and worker tests remain green.

- [ ] **Step 6: Commit Task 3**

~~~powershell
git add native/ppshitu_rec_benchmark tests/test_cpp_inference_benchmark_contract.py
git commit -m "refactor: share native PP-ShiTu inference core"
~~~

---

### Task 4: Implement the long-lived C++ feature service

**Files:**
- Create: native/ppshitu_rec_benchmark/include/native_service.h
- Create: native/ppshitu_rec_benchmark/src/native_service.cpp
- Create: native/ppshitu_rec_benchmark/src/service_main.cpp
- Modify: native/ppshitu_rec_benchmark/CMakeLists.txt
- Modify: scripts/build_cpp_inference_benchmark.ps1
- Create: tests/test_cpp_inference_service_contract.py

**Interfaces:**
- ServiceOptions{model_dir, threads, max_frame_bytes, max_batch, preprocess}.
- int RunNativeService(const ServiceOptions&).
- `ppshitu_rec_service.exe --serve --model-dir <path> --threads <n> --max-frame-bytes <n> --max-batch <n> --input-width <n> --input-height <n> --scale <f> --mean-rgb <r,g,b> --std-rgb <r,g,b>`; preprocessing defaults must match `PreprocessOptions` exactly.
- The pytest `NativeRawProcess` fixture starts the executable with a temporary model directory, exposes `read_frame()`, `write_predict(images, request_id)`, `write_raw(bytes)`, and `stderr_tail()`, and always closes/terminates the child in teardown.

- [ ] **Step 1: Write failing process-level service tests**

Add `WORKPIECE_CPP_SERVICE_EXE` and test HELLO, an ordered two-image RESULT, a malformed complete frame ERROR, truncated-frame termination, and no text bytes on stdout:

~~~python
@pytest.mark.integration
def test_service_returns_ordered_batch(native_service, native_model_dir, images):
    with NativeRawProcess(native_service, native_model_dir, max_batch=8) as process:
        hello = process.read_frame()
        assert hello.kind == KIND_HELLO
        assert hello.payload.feature_dim == 512
        process.write_predict([images[1], images[0]], request_id=12)
        result = process.read_frame()
        assert result.request_id == 12
        assert result.payload.embeddings.shape[0] == 2
~~~

- [ ] **Step 2: Run the service tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_service_contract.py -q -p no:cacheprovider -m integration
~~~

Expected: a clear prerequisite failure because ppshitu_rec_service.exe is not built.

- [ ] **Step 3: Implement service startup and HELLO**

Use `wmain` and set stdin/stdout to binary mode on Windows. Check AVX, the exact `inference.pdmodel`/`inference.pdiparams` files currently consumed by `FeatureExtractor`, ASCII-staged path, thread count, frame limit, and max batch before constructing FeatureExtractor; if a future alias is added, keep the check and loader in one shared helper. Create exactly one predictor on the service thread. On startup failure, write a bounded `ERROR` diagnostic when a request ID is available or a stable code to stderr before exiting nonzero. Send HELLO containing service version, staged model SHA-256, feature dimension, max batch, and threads, then flush stdout.

- [ ] **Step 4: Implement the bounded request loop**

Decode each PREDICT image record, call PreprocessRgb, stack tensors, run FeatureExtractor, and emit one ordered RESULT with preprocess, inference, and postprocess timings. Reject empty batches, bad dimensions, oversized frames, non-finite embeddings, and predictor exceptions with stable ERROR frames. A malformed complete request gets an ERROR frame tied to its request ID; a truncated frame cannot be trusted and causes stderr diagnostics plus nonzero exit. CLOSE and clean EOF return success; an unreusable stream exits nonzero. All diagnostics and uncaught exception text go to stderr.

- [ ] **Step 5: Add service target and stage both executables**

Add the service target:

~~~cmake
add_executable(ppshitu_rec_service src/service_main.cpp src/native_service.cpp)
target_link_libraries(ppshitu_rec_service PRIVATE ppshitu_core ppshitu_protocol)
~~~

Update scripts/build_cpp_inference_benchmark.ps1 to assert and stage both ppshitu_rec_benchmark.exe and ppshitu_rec_service.exe while retaining existing DLL conflict checks and benchmark output location.

- [ ] **Step 6: Rebuild and run service/benchmark tests**

Build with the local .native-deps roots, set both executable variables, and run:

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_cpp_inference_benchmark_contract.py tests\test_cpp_inference_service_contract.py -q -p no:cacheprovider
~~~

Expected: service handshake, ordered result, malformed-frame error, model-load error, and existing benchmark tests pass.

- [ ] **Step 7: Commit Task 4**

~~~powershell
git add native/ppshitu_rec_benchmark scripts/build_cpp_inference_benchmark.ps1 tests/test_cpp_inference_service_contract.py
git commit -m "feat: add persistent native PP-ShiTu service"
~~~

---

### Task 5: Implement the Python persistent native client

**Files:**
- Create: src/native_pp_client.py
- Create: tests/test_native_pp_client.py
- Create: tests/fixtures/fake_native_pp_service.py

**Interfaces:**
- NativePPError(RuntimeError) with code and details.
- Client-side image validation raises the shared `NativeProtocolError` before any bytes are written to the child.
- NativePPClient(executable: Path, model_dir: Path, *, threads: int = 1, max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES, max_batch: int = 256, request_timeout_s: float = 30.0, preprocess: Mapping[str, object] | None = None, env: Mapping[str, str] | None = None).
- start() -> NativeHello, predict(images: Sequence[np.ndarray]) -> NativeResult, close() -> None.
- Read-only properties ready, feature_dim, hello, and last_stderr.

- [ ] **Step 1: Write failing fake-service tests**

The fixture supports normal, sleep, exit, and bad-dimension modes. Test order and stable poisoning:

~~~python
def test_client_waits_for_hello_and_preserves_order(fake_service, tmp_path):
    client = NativePPClient(fake_service, tmp_path, request_timeout_s=1.0)
    hello = client.start()
    assert hello.feature_dim == 2
    result = client.predict([
        np.full((2, 2, 3), 2, np.uint8),
        np.full((2, 2, 3), 1, np.uint8),
    ])
    assert [row[0] for row in result.embeddings] == [2.0, 1.0]
    client.close()

@pytest.mark.parametrize(
    "mode,code",
    [("sleep", "NATIVE_PP_TIMEOUT"),
     ("exit", "NATIVE_PP_SERVICE_EXITED"),
     ("bad-dimension", "NATIVE_PP_DIMENSION_MISMATCH")],
)
def test_client_poisoned_process_is_not_reused(fake_service, tmp_path, mode, code):
    client = NativePPClient(
        fake_service, tmp_path, request_timeout_s=0.05,
        env={"FAKE_NATIVE_PP_MODE": mode},
    )
    client.start()
    with pytest.raises(NativePPError) as raised:
        client.predict([np.zeros((2, 2, 3), np.uint8)])
    assert raised.value.code == code
    with pytest.raises(NativePPError) as reused:
        client.predict([np.zeros((2, 2, 3), np.uint8)])
    assert reused.value.code == code
    client.close()
    client.close()

def test_client_rejects_invalid_images_before_writing(fake_service, tmp_path):
    client = NativePPClient(fake_service, tmp_path)
    client.start()
    with pytest.raises(NativeProtocolError, match="uint8|3 channels|contiguous"):
        client.predict([np.zeros((2, 2, 3), np.float32)])
    client.close()
~~~

- [ ] **Step 2: Run client tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_native_pp_client.py -q -p no:cacheprovider
~~~

Expected: import failure because src/native_pp_client.py is absent.

- [ ] **Step 3: Implement process startup and reader threads**

Launch Popen with binary stdin/stdout, stderr=PIPE, bufsize=0, and text=False. Start a daemon stdout frame-reader queue and a daemon stderr ring-buffer before waiting for HELLO; anonymous Windows pipes cannot be reliably waited with selectors. Protect one in-flight request with an RLock, use nonzero monotonic request IDs, and reject mismatched response IDs/kinds. Keep an optional `env: Mapping[str, str] | None` constructor argument only for test fixture process variables; production preprocessing options remain a separate mapping and are converted to service CLI flags.

- [ ] **Step 4: Implement timeout, poisoning, and cleanup**

On timeout, EOF, malformed frame, nonzero child exit, or ERROR frame, store the stable code and stderr tail, close stdin, wait at most two seconds, then terminate/kill if required, and mark the client unusable before raising. close() sends CLOSE only when healthy, closes handles, joins readers boundedly, and never raises on repeated calls.

- [ ] **Step 5: Implement RGB validation and timing propagation**

Require a nonempty sequence of HWC uint8 three-channel C-contiguous arrays. Pass preprocessing values and the staged model path as arguments. Decode copied float32 rows, check finite values, negotiated dimension, and norm within 1e-4, and return NativeResult without reordering. During `start()`, compare the HELLO staged model SHA-256 with the digest of the exact staged directory passed to the child; a mismatch poisons the client with `NATIVE_PP_MODEL_MISMATCH`. The fake fixture reads `FAKE_NATIVE_PP_MODE` from its environment and never adds a test-only field to the production preprocessing mapping.

- [ ] **Step 6: Run client tests and commit**

Run the Step 2 command. Expected: normal, order, timeout, exit, dimension, stderr, invalid-image, and idempotent-close tests pass.

~~~powershell
git add src/native_pp_client.py tests/test_native_pp_client.py tests/fixtures/fake_native_pp_service.py
git commit -m "feat: add persistent native PP-ShiTu Python client"
~~~

---

### Task 6: Select the native backend in OrientationClassifier

**Files:**
- Modify: src/orientation_classifier.py
- Modify: tests/test_orientation_classifier.py
- Modify: src/native_pp_client.py only if a failing integration test demonstrates a client defect.

**Interfaces:**
- PP_BACKENDS = ("python", "native_cpp") and DEFAULT_PP_BACKEND = "python".
- OrientationClassifier.__init__(..., pp_backend: str = "python", native_pp_client: NativePPClient | None = None, native_request_timeout_s: float = 30.0).
- OrientationClassifier.load(..., pp_backend: str = "python", native_pp_executable: Path | None = None, native_request_timeout_s: float = 30.0).
- close() -> None and `native_backend_info() -> dict[str, object]` returning keys `backend`, `service_version`, `model_sha256`, `feature_dim`, and `last_error`.
- _global_embeddings still returns list[np.ndarray].

- [ ] **Step 1: Write failing selection tests**

Use a fake client and patch the import boundary so native loading fails if paddle is imported. Assert native mode starts one client, leaves global_predictor as None, and Python mode still constructs the current predictor. Assert native GPU or legacy mode raises ComputeDeviceError with code NATIVE_PP_UNSUPPORTED_MODE.

- [ ] **Step 2: Run focused classifier tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py -q -p no:cacheprovider
~~~

Expected: new tests fail because the load signature and native branch do not exist.

- [ ] **Step 3: Add explicit backend validation and native load**

Validate pp_backend before loading models. For native mode require a regular, non-symlink executable, CPU, and fast_geometry. Call `prepare_paddle_model_path(model_dir)` to retain an ASCII staging directory for the lifetime of the classifier, compute the source fingerprint before staging, resolve CPU threads from existing environment/default without importing Paddle, create/start NativePPClient, and construct the classifier with `global_predictor=None`. Store the staged path on `self.native_model_dir` for every batch worker; keep it until process exit via the existing compatibility cleanup, while `close()` guarantees all child processes have released it. Keep the current Paddle branch behavior and keyword compatibility.

- [ ] **Step 4: Route scalar extraction and diagnostics**

Make `_global_embeddings` preserve the existing BGR→RGB semantics, but wrap each reversed view with `np.ascontiguousarray` before calling `native_pp_client.predict(rgb_images).embeddings` in native mode under the existing inference lock. Validate negotiated dimension against fast-cache slot dimension. `native_backend_info` returns backend, service version, native model SHA-256, dimension, and last error; Python mode reports backend python and no native error.

- [ ] **Step 5: Create independent native batch sessions**

Branch `_prepare_batch_pool_locked`'s worker factory. Capture the validated classifier fields `self.native_pp_executable` and `self.native_model_dir`, and use the existing pool's `threads_per_worker` value:

~~~python
def create_native_worker(slot: int) -> _BatchSession:
    client = NativePPClient(
        self.native_pp_executable,
        self.native_model_dir,
        threads=threads_per_worker,
        request_timeout_s=self.native_request_timeout_s,
    )
    hello = client.start()
    engine = FastOrientationEngine(
        lambda images, _client=client: _client.predict(
            [np.ascontiguousarray(image[:, :, ::-1]) for image in images]
        ).embeddings,
        self.fast_engine.geometry,
        image_reader=self.fast_engine.image_reader,
        deduplicate_identical_slots=self.fast_engine.deduplicate_identical_slots,
    )
    return _BatchSession(
        predictor=client,
        engine=engine,
        lock=threading.Lock(),
        feature_dim=hello.feature_dim,
    )
~~~

Before using the factory, extend `_BatchSession` with `feature_dim: int | None = None`; verify it against the scalar client's negotiated dimension and the existing fast-cache slot dimension. Each slot owns one client/process. Preserve BatchInferencePool order. If pool creation fails before native readiness, a native scalar client may service the request; never create or retry through the Python Paddle predictor in native mode. A poisoned native request is never retried.

- [ ] **Step 6: Close resources exactly once**

Extend `_BatchSession.close` for NativePPClient, add classifier `close()` to close the batch pool then scalar client, and make service shutdown call it. Store `native_request_timeout_s` as a positive finite value and pass it to every client. Preserve concurrent prepare/close guarantees and idempotence; calling `close()` twice must not terminate a newly created replacement session.

- [ ] **Step 7: Run regression tests and commit**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_classifier.py tests\test_fast_orientation.py -q -p no:cacheprovider
git add src/orientation_classifier.py tests/test_orientation_classifier.py
git commit -m "feat: select native PP-ShiTu backend in classifier"
~~~

Expected: existing Python tests plus native fake-client selection, order, independent worker, no-fallback, and close tests pass.

---

### Task 7: Wire TCP service CLI, hello metadata, and startup errors

**Files:**
- Modify: src/orientation_tcp_service.py
- Modify: tests/test_orientation_tcp_service.py

**Interfaces:**
- Parser options --pp-backend python|native_cpp and --native-pp-executable path.
- _load_runtime(..., pp_backend: str = "python", native_pp_executable: Path | None = None).
- Hello adds pp_backend, native_service_version, native_model_sha256, and feature_dim.
- RuntimeSnapshot adds `pp_backend: str = "python"`, `native_service_version: str = ""`, `native_model_sha256: str = ""`, and `feature_dim: int | None = None`; `ServiceRuntime.set_ready` copies these values from `classifier.native_backend_info()` while retaining the existing snapshot fields.
- Startup action mappings include NATIVE_PP_CONFIG_INVALID, NATIVE_PP_UNSUPPORTED_MODE, NATIVE_PP_STARTUP_FAILED, NATIVE_PP_MODEL_MISMATCH, and NATIVE_PP_DIMENSION_MISMATCH.

- [ ] **Step 1: Write failing service tests**

Assert parser values, missing/non-file executable rejection before model loading, forwarding of native keywords to a fake OrientationClassifier.load, hello metadata, and classifier close exactly once during runtime shutdown.

- [ ] **Step 2: Run focused service tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py -q -p no:cacheprovider
~~~

Expected: new parser and metadata assertions fail.

- [ ] **Step 3: Validate and forward native arguments**

Add strict backend choices and pass pp_backend/native_pp_executable through main, loader thread, and OrientationClassifier.load. Native executable must be a regular file; native GPU/non-fast combinations raise coded startup errors. Do not require model_sha256 for development Python mode; preserve packaged checks.

- [ ] **Step 4: Publish metadata only after HELLO**

Have `ServiceRuntime.set_ready` call `classifier.native_backend_info()` exactly once after classifier construction and copy validated `backend`, `service_version`, `model_sha256`, and `feature_dim` into the new snapshot fields. For Python mode, store `pp_backend="python"` and empty native fields. `hello` remains `ready=false` while native child startup is incomplete and becomes `ready=true` only after protocol, dimension, and model checks. Keep existing JSON fields and batch capability fields, adding the four native metadata keys without renaming any old key.

- [ ] **Step 5: Map errors and close resources**

Catch NativePPError/ComputeDeviceError separately, preserve stable codes and stderr tail in logs/hello failure, and call classifier.close() on failed transfer and normal shutdown. Prediction errors use the existing error envelope with NATIVE_PP_* codes. Ensure `_shutdown_runtime_components` prefers `classifier.close()` and falls back to the legacy `close_batch_pool()` only for classifiers that do not expose `close()`.

- [ ] **Step 6: Run service tests and commit**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_orientation_tcp_service.py tests\test_orientation_classifier.py -q -p no:cacheprovider
git add src/orientation_tcp_service.py tests/test_orientation_tcp_service.py
git commit -m "feat: expose native PP-ShiTu backend in TCP service"
~~~

Expected: Python-default behavior is unchanged and native parser, metadata, startup, and cleanup tests pass.

---

### Task 8: Add Qt configuration, forwarding, identity, and diagnostics

**Files:**
- Modify: qt_app/appconfig.h
- Modify: qt_app/appconfig.cpp
- Modify: qt_app/app_config.json.example
- Modify: qt_app/backendprocessmanager.h
- Modify: qt_app/backendprocessmanager.cpp
- Modify: qt_app/appheader.h
- Modify: qt_app/appheader.cpp
- Modify: qt_app/mainwindow.cpp
- Modify: qt_app/tests/test_appconfig.cpp
- Modify: qt_app/tests/test_backendprocessmanager.cpp
- Modify: qt_app/tests/test_appheader.cpp

**Interfaces:**
- `AppConfig` adds `QString ppBackend = "python"` and optional `QString nativePpExecutable` (empty in Python mode).
- `BackendStatusDetails` adds `QString ppBackend`, `int nativeFeatureDim` (use `-1` when absent), `QString nativeServiceVersion`, and `QString nativeModelSha256`.
- backendArguments emits --pp-backend and emits --native-pp-executable only for native_cpp.
- identityMatches treats a missing backend field as python for old development services and requires native_cpp for native config.
- `BackendProcessManager` adds `backendMetadataUpdated(const QJsonObject &metadata)` and emits it immediately before `backendReady()` on a validated hello; `MainWindow` connects that signal and maps the four native fields into `BackendStatusDetails` without changing the existing `backendReady()` signal.

- [ ] **Step 1: Write failing Qt configuration tests**

Add rows for absent pp_backend (Python default), a valid native config with a Chinese executable path, unknown backend, native GPU/legacy combinations, and a missing executable. The native fixture uses compute_device=cpu and inference_mode=fast_geometry.

- [ ] **Step 2: Run the config target and verify RED**

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1 -Target test_appconfig
~~~

Expected: new tests fail because fields are not parsed.

- [ ] **Step 3: Implement strict parsing without breaking defaults**

Parse pp_backend as python or native_cpp; absent means python. Parse native_pp_executable as an optional non-empty string, resolve it relative to the config file, and require an existing file only for native mode. Enforce CPU/fast_geometry for native mode. Preserve packaged edition, fingerprint, and data-root checks. Add pp_backend: "python" to the example and omit the native executable field there.

- [ ] **Step 4: Forward arguments and check identity**

Update backendArguments and assert exact QStringList values for Python/native configurations. Include backend in identity comparison with a compatibility default for old Python services; report a coded mismatch for a native service with the wrong backend.

- [ ] **Step 5: Show native diagnostics**

Emit the validated hello object through `backendMetadataUpdated`, map `pp_backend`, `feature_dim`, `native_service_version`, and `native_model_sha256` into `BackendStatusDetails`, and append PP-ShiTu backend, feature dimension, service version, and native model digest to the existing model-detail panel. Clear stale native fields on every new loading/unavailable transition. Keep the top-right state green only after hello ready; loading/reconnect remains a single state.

- [ ] **Step 6: Run all Qt tests and commit**

~~~powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1
git add qt_app/appconfig.h qt_app/appconfig.cpp qt_app/app_config.json.example qt_app/backendprocessmanager.h qt_app/backendprocessmanager.cpp qt_app/appheader.h qt_app/appheader.cpp qt_app/mainwindow.cpp qt_app/tests
git commit -m "feat: configure native PP-ShiTu backend from Qt"
~~~

Expected: all Qt 5.14.2 targets plus config, forwarding, identity, and diagnostic tests pass.

---

### Task 9: End-to-end parity, cache compatibility, and failure regression

**Files:**
- Create: tests/test_native_backend_regression.py
- Modify: tests/test_orientation_classifier.py and tests/test_orientation_tcp_service.py only when a reproduced failing test demonstrates a regression.

**Interfaces:**
- run_backend_pair(...) returns Python and native decision dictionaries for identical RGB slots.
- Regression evidence covers old caches, unequal template counts, no fallback, and child cleanup.

- [ ] **Step 1: Write failing regression tests**

Use deterministic fake embeddings for 1+1, 5+10, 10+15, and 35+35 template counts. Load a serialized existing 5+5 FastRuntimeCache and predict with both backends. Force a native child exit during batch and assert each affected item reports NativePPError/NATIVE_PP_SERVICE_EXITED and that the Python predictor was not called.

- [ ] **Step 2: Run regression tests and verify RED**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_native_backend_regression.py -q -p no:cacheprovider
~~~

Expected: failures identify absent backend plumbing or a fixed-template assumption.

- [ ] **Step 3: Apply minimum compatibility fixes**

Keep FastRuntimeCache signatures/count validation and source model fingerprint unchanged. Feed native output through the same FastOrientationEngine callback as Python so template count and orientation never enter the native protocol. Attach native errors to item results and prevent a poisoned client from returning to the pool.

- [ ] **Step 4: Run parity and lifecycle tests**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests\test_native_backend_regression.py tests\test_orientation_classifier.py tests\test_orientation_tcp_service.py tests\test_fast_orientation.py -q -p no:cacheprovider
~~~

Expected: old cache recovery, unequal counts, decision parity, no-fallback, order, and cleanup pass.

- [ ] **Step 5: Commit Task 9**

~~~powershell
git add tests/test_native_backend_regression.py tests/test_orientation_classifier.py tests/test_orientation_tcp_service.py
git commit -m "test: verify native backend parity and rollback"
~~~

---

### Task 10: Measure performance and document rollout

**Files:**
- Create: docs/verification/ppshitu-cpp-backend-integration-results.md
- Create: native/ppshitu_rec_benchmark/dependencies.json
- Modify: scripts/compare_cpp_python_inference.py only behind a failing timing test.
- Modify: tests/test_compare_cpp_python_inference.py only for that reproduced defect.

**Interfaces:**
- Raw reports under ignored release_staging/cpp-backend-evaluation.
- Markdown table with model hashes, dimension, startup, transport, model, Python/native P50/P95/P99, CPU utilization, and peak memory.
- One explicit result: INTEGRATE AS OPT-IN or KEEP PYTHON DEFAULT.

- [ ] **Step 1: Record actual dependency identities**

Write the downloaded Paddle Inference/OpenCV archive URLs, versions, byte sizes, and SHA-256 values using Get-FileHash. Do not commit archives, models, DLLs, build directories, or generated embeddings.

- [ ] **Step 2: Run parity across all five datasets**

Use identical ordered lists and source model. Require cosine >= 0.9999, absolute error <= 0.001, exact labels/review/geometry flags, and all template counts. A missing or unreadable dataset fails the measurement instead of being skipped.

- [ ] **Step 3: Run warm sweeps**

~~~text
single image: 1 worker with 1, 2, and 4 predictor threads; 1000 iterations
five images: 1x1, 1x2, 1x4, 2x2, and 4x1; 50 warmups plus 200 iterations
95 images: 4x1 and production batch pool; 20 warmups plus 50 iterations
~~~

Record C++ preprocessing, model execution, Python-to-C++ transfer, Python postprocessing, and complete predict_batch wall time as P50/P95/P99. Report batch total and per-image throughput separately.

- [ ] **Step 4: Calculate projection and compare target hardware**

Use only:

~~~text
projected_pipeline_p95 = python_pipeline_p95
                       - python_feature_p95
                       + native_feature_p95
~~~

State all three operands and measured service overhead. Accept the i5-8250U claim only with a complete backend run on that machine; label a projection as a projection.

- [ ] **Step 5: Write the evidence-backed rollout decision**

Use INTEGRATE AS OPT-IN only when all correctness gates pass and both native-feature and projected-pipeline P95 gains are at least 15%. Otherwise use KEEP PYTHON DEFAULT, name the failed gate, and retain native only for diagnostics. Document rollback as setting pp_backend to python and restarting, with no library/cache rebuild.

- [ ] **Step 6: Run complete regression and commit evidence**

~~~powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=E:\Project\wang\pp_813\.pytest-native-final-20260901
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_qt5_tests.ps1
git add docs/verification/ppshitu-cpp-backend-integration-results.md native/ppshitu_rec_benchmark/dependencies.json
git commit -m "docs: record native PP-ShiTu backend integration evidence"
~~~

The branch is complete only when native build, parity, lifecycle, full Python and Qt suites, target-hardware evidence, and rollback instructions are present in the verification document.
