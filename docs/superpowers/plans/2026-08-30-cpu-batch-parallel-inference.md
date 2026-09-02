# CPU Batch Parallel Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use the checkbox syntax for tracking.

**Goal:** 修复冻结 CPU 后端的 src.orientation_classifier 导入失败，并在 i5-8250U 上为批量检测增加可观测、可回退的并行推理路径。

**Architecture:** 保留现有单张 predict 和识别逻辑；新增 predict_batch 协议和一个有界的进程内推理会话池。每个批量 worker 使用独立 Paddle predictor，读取同一不可变工件库快照，按输入索引返回结果；能力协商失败或会话池不可用时自动回到旧逐张路径。PyInstaller 使用显式 src 包和归档审计。

**Tech Stack:** Python 3.10、PaddlePaddle/PaddleClas 3.2.2/2.6.0、OpenCV、PyInstaller 6.22.2、pytest、Qt 5.14.2/C++17、Qt JSON/TCP。

## Global Constraints

- 不重新训练、微调或替换 PP-ShiTu 权重。
- 不改变 PP-ShiTu 预处理、几何规则、ALIKED/LightGlue、Ridge 和现有融合阈值。
- 单张 predict 的请求字段、返回字段和行为保持兼容。
- CPU 批量默认最多 4 个独立会话；先比较 4×1 与 2×2，不创建 8 个模型会话。
- 批量输入不得静默截断、丢弃或重排；空列表和无效请求必须显式报错。
- 旧 5+5 缓存、正反面不等模板数量和旧 Qt/后端必须继续可用。
- 目标硬件为 Windows 10/11 x64、i5-8250U；稳态 5 图批次 P95 ≤100 ms 是实测目标，不能用开发机数据代替。
- 每个任务先写失败测试、确认失败，再写最小实现；每个任务完成后运行对应测试并提交。

---

### Task 1: 固化 src 包和冻结入口契约

**Files:**
- Create: src/__init__.py
- Modify: release_tools/backend_bundle.py
- Modify: deploy/orientation_backend.spec
- Modify: scripts/build_portable_backend.ps1
- Test: tests/test_backend_bundle.py
- Test: tests/test_backend_spec.py
- Test: tests/test_orientation_service_environment.py

**Interfaces:**
- Produces production_src_modules() -> tuple[str, ...] in release_tools.backend_bundle.
- deploy/orientation_backend.spec consumes that tuple as explicit hidden imports.
- Build validation consumes assert_frozen_backend_modules(backend_dir: Path) -> None.

- [ ] **Step 1: Write the failing package/import tests**

Add tests that require an explicit package marker and a complete production module list:

~~~
def test_src_is_an_explicit_package():
    assert (Path(__file__).parents[1] / "src" / "__init__.py").is_file()


def test_production_src_module_list_contains_runtime_dependencies():
    from release_tools.backend_bundle import production_src_modules

    modules = set(production_src_modules())
    assert {
        "src.orientation_tcp_service",
        "src.orientation_classifier",
        "src.workpiece_catalog",
        "src.workpiece_library",
        "src.runtime_data",
        "src.fast_orientation",
        "src.fast_geometry",
        "src.fast_ridge",
        "src.geometry_calibration",
        "src.geometry_mask_profiles",
    } <= modules
    assert "src.aliked_lightglue_matcher" not in modules
~~~

Extend test_backend_spec.py to assert that the spec references production_src_modules() and does not depend on loose project source files at runtime.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

~~~
pytest tests/test_backend_bundle.py::test_src_is_an_explicit_package tests/test_backend_bundle.py::test_production_src_module_list_contains_runtime_dependencies tests/test_backend_spec.py -q
~~~

Expected: failure because src/__init__.py and production_src_modules do not yet exist.

- [ ] **Step 3: Implement the explicit package and module contract**

Create an empty src/__init__.py. Add one immutable tuple in release_tools/backend_bundle.py containing only modules needed by the fast service, and export production_src_modules. Update the spec:

~~~
from release_tools.backend_bundle import (
    edition_for,
    production_src_modules,
    pyinstaller_excludes,
    validate_installed_distributions,
)

hiddenimports = [
    *production_src_modules(),
    "paddle.base.core",
    "paddle.inference",
    "paddleclas.deploy.python.predict_rec",
    "paddleclas.deploy.utils.config",
    "paddleclas.deploy.utils.predictor",
]
~~~

Keep excluded local matcher modules excluded. Add a post-build archive check that opens the generated PYZ/PKG listing and fails with missing fully-qualified module names; never copy Python source files into the release as a workaround.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run:

~~~
pytest tests/test_backend_bundle.py tests/test_backend_spec.py tests/test_orientation_service_environment.py -q
~~~

Expected: all focused tests pass, including existing edition and exclusion checks.

- [ ] **Step 5: Commit**

~~~
git add src/__init__.py release_tools/backend_bundle.py deploy/orientation_backend.spec scripts/build_portable_backend.ps1 tests/test_backend_bundle.py tests/test_backend_spec.py tests/test_orientation_service_environment.py
git commit -m "fix: make frozen backend src imports explicit"
~~~

### Task 2: Add a generic bounded parallel execution pool

**Files:**
- Create: src/parallel_inference.py
- Test: tests/test_parallel_inference.py

**Interfaces:**
- BatchWorkItem(index: int, image_path: Path), frozen dataclass.
- BatchItemResult(index: int, image_path: Path, value: object | None, error: BaseException | None), frozen dataclass.
- BatchInferencePool(worker_count: int, create_worker: Callable[[int], object], run_item: Callable[[object, BatchWorkItem], object]).
- BatchInferencePool.ready -> bool, worker_count -> int, submit_many(items: Sequence[BatchWorkItem]) -> list[BatchItemResult], close() -> None.

- [ ] **Step 1: Write failing concurrency, order, and error tests**

Create deterministic tests using a threading.Barrier and an active counter:

~~~
def test_submit_many_overlaps_workers_and_restores_input_order():
    pool = make_blocking_pool(worker_count=4)
    items = [BatchWorkItem(i, Path(f"{i}.png")) for i in range(5)]
    results = pool.submit_many(items)
    assert [item.index for item in results] == list(range(5))
    assert pool.max_active >= 2


def test_submit_many_keeps_other_items_when_one_item_fails():
    pool = make_pool(lambda item: (_ for _ in ()).throw(ValueError("bad"))
                       if item.index == 2 else item.index)
    results = pool.submit_many([BatchWorkItem(i, Path(f"{i}.png")) for i in range(4)])
    assert [item.index for item in results] == [0, 1, 2, 3]
    assert isinstance(results[2].error, ValueError)
    assert [item.value for item in results if item.error is None] == [0, 1, 3]


def test_submit_many_rejects_empty_input_and_invalid_worker_count():
    with pytest.raises(ValueError, match="worker_count"):
        BatchInferencePool(0, lambda _: object(), lambda *_: None)
    pool = make_pool(lambda item: item.index)
    with pytest.raises(ValueError, match="at least one"):
        pool.submit_many([])
~~~

- [ ] **Step 2: Run the tests and confirm RED**

Run:

~~~
pytest tests/test_parallel_inference.py -q
~~~

Expected: collection failure because src.parallel_inference is absent.

- [ ] **Step 3: Implement the minimal bounded pool**

Construct one concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) and one worker context per pool slot in the pool constructor; reuse both for every submit_many call so a warm batch does not recreate threads. Submit one item per task, catch exceptions into BatchItemResult, sort by index, and shut down the executor and worker contexts in close. Reject duplicate indices and empty input; do not silently trim a batch. Keep the pool generic so Paddle lifecycle remains in Task 3.

- [ ] **Step 4: Run the tests and confirm GREEN**

Run:

~~~
pytest tests/test_parallel_inference.py -q
~~~

Expected: all pool tests pass repeatedly.

- [ ] **Step 5: Commit**

~~~
git add src/parallel_inference.py tests/test_parallel_inference.py
git commit -m "feat: add bounded batch execution pool"
~~~

### Task 3: Bind independent Paddle sessions to the fast classifier

**Files:**
- Modify: src/orientation_classifier.py
- Modify: src/fast_orientation.py (only when the worker-specific engine factory is placed there)
- Modify: src/model_execution_gate.py (only when a per-session gate is required by the tested implementation)
- Test: tests/test_orientation_classifier.py
- Test: tests/test_fast_orientation.py

**Interfaces:**
- OrientationClassifier.batch_capabilities() -> dict[str, object].
- OrientationClassifier.prepare_batch_pool(*, worker_count: int | None = None, threads_per_worker: int = 1) -> dict[str, object].
- OrientationClassifier.close_batch_pool() -> None.
- OrientationClassifier.predict_many_with_cache(cache: TemplateCache, image_paths: Sequence[Path], *, library_revision: int | None = None) -> list[dict[str, object]].

- [ ] **Step 1: Write failing classifier tests**

Add a fake predictor whose predict blocks on a barrier and returns deterministic embeddings. Verify overlap, stable order, and that existing single prediction still uses the original predictor. Add a failed clone test that reports batch_ready=false without removing the existing inference lock.

~~~
def test_predict_many_uses_independent_sessions_and_preserves_order():
    classifier, cache, predictor_factory = fast_classifier_with_blocking_predictor()
    status = classifier.prepare_batch_pool(worker_count=2, threads_per_worker=1)
    assert status["batch_ready"] is True
    results = classifier.predict_many_with_cache(
        cache, [Path("a.png"), Path("b.png")], library_revision=1)
    assert [result["index"] for result in results] == [0, 1]
    assert predictor_factory.max_active >= 2


def test_failed_session_creation_reports_fallback_without_unlocking_shared_predictor():
    classifier, cache = classifier_with_uncloneable_predictor()
    status = classifier.prepare_batch_pool(worker_count=4, threads_per_worker=1)
    assert status["batch_ready"] is False
    assert "fallback" in status
    classifier.predict_fast_with_cache(cache, Path("a.png"))
~~~

- [ ] **Step 2: Run the tests and confirm RED**

Run:

~~~
pytest tests/test_orientation_classifier.py::test_predict_many_uses_independent_sessions_and_preserves_order tests/test_orientation_classifier.py::test_failed_session_creation_reports_fallback_without_unlocking_shared_predictor -q
~~~

Expected: failure because the batch methods and session factory do not exist.

- [ ] **Step 3: Implement the fast CPU session factory**

During OrientationClassifier.load, retain the immutable Paddle configuration needed to create or clone a predictor. For CPU fast_geometry, create worker-specific RecPredictor wrappers around independent raw predictors, preferring clone or PredictorPool and using a fresh configured predictor as the safe fallback. Set each worker's CPU math threads to threads_per_worker; cap workers at the detected physical-core limit and at 4. Build a worker-specific FastOrientationEngine whose embed_batch is bound to that worker predictor, sharing only immutable geometry and cache data. Do not call the main classifier's _global_embeddings from a worker.

prepare_batch_pool prewarms every session with the existing zero-image self-check. On any creation or prewarm failure, close partial sessions, leave single inference untouched, and return batch_ready=false, worker_count, and fallback. For non-CPU or non-fast modes, return batch_ready=false and use the serial path. predict_many_with_cache captures one caller-provided cache snapshot, dispatches each image through the pool, adds index, image_path, and library_revision, and preserves all existing prediction fields.

- [ ] **Step 4: Run classifier and fast-path regression tests**

Run:

~~~
pytest tests/test_orientation_classifier.py tests/test_fast_orientation.py -q
~~~

Expected: all existing tests plus concurrency and fallback tests pass; GPU and legacy paths remain unchanged.

- [ ] **Step 5: Commit**

~~~
git add src/orientation_classifier.py src/fast_orientation.py src/model_execution_gate.py tests/test_orientation_classifier.py tests/test_fast_orientation.py
git commit -m "feat: add independent CPU inference sessions"
~~~

### Task 4: Add an immutable-snapshot catalog batch API and backend command

**Files:**
- Modify: src/workpiece_catalog.py
- Modify: src/orientation_tcp_service.py
- Test: tests/test_workpiece_catalog.py
- Test: tests/test_orientation_tcp_service.py

**Interfaces:**
- WorkpieceCatalog.predict_many(workpiece_id: str, image_paths: Sequence[Path]) -> list[dict[str, object]].
- OrientationCommandDispatcher.dispatch accepts command predict_batch with workpiece_id and image_paths.
- hello response adds capabilities with predict_batch, batch_ready, batch_workers, and batch_threads_per_worker.

- [ ] **Step 1: Write failing catalog and protocol tests**

Add tests for one immutable snapshot, ordered results, partial item failure, empty list, non-string paths, unknown workpiece, and capability metadata:

~~~
def test_predict_batch_captures_one_snapshot_and_returns_input_order():
    catalog, classifier = catalog_with_batch_classifier()
    result = catalog.predict_many("m7", [Path("a.png"), Path("b.png"), Path("c.png")])
    assert [item["index"] for item in result] == [0, 1, 2]
    assert classifier.snapshot_calls == 1


def test_dispatch_predict_batch_reports_item_error_without_dropping_other_items():
    response = ready_dispatcher().dispatch({
        "version": 1, "request_id": "r1", "command": "predict_batch",
        "workpiece_id": "m7", "image_paths": ["a.png", "bad.png", "c.png"],
    })
    assert response["ok"] is True
    assert [item["index"] for item in response["items"]] == [0, 1, 2]
    assert response["items"][1]["ok"] is False


@pytest.mark.parametrize("paths", [[], [1], ["a.png", None]])
def test_dispatch_predict_batch_rejects_invalid_lists(paths):
    response = ready_dispatcher().dispatch({
        "version": 1, "request_id": "r1", "command": "predict_batch",
        "workpiece_id": "m7", "image_paths": paths,
    })
    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_REQUEST"
~~~

- [ ] **Step 2: Run tests and confirm RED**

Run:

~~~
pytest tests/test_workpiece_catalog.py::test_predict_batch_captures_one_snapshot_and_returns_input_order tests/test_orientation_tcp_service.py -q
~~~

Expected: new tests fail because no batch catalog method or dispatcher command exists; existing tests must remain green.

- [ ] **Step 3: Implement catalog snapshot and dispatcher response**

Capture ActiveWorkpieceSnapshot once, call classifier.predict_many_with_cache when batch_ready, and use scalar predict_with_cache in deterministic input order as fallback. Normalize each item to:

~~~
{
    "index": index,
    "image_path": str(path),
    "ok": True,
    "prediction": prediction,
}
~~~

or the same shape with ok false and a stable code/message error. Add wall-clock batch_timings_ms for decode, inference, postprocess, and total, plus worker_count and fallback. Extend _fast_prediction_error only for top-level failures; one bad image must not abort other items. Add capability data to hello from runtime.classifier.batch_capabilities() without changing readiness semantics.

- [ ] **Step 4: Run focused backend tests and full Python regression**

Run:

~~~
pytest tests/test_workpiece_catalog.py tests/test_orientation_tcp_service.py -q
~~~

Expected: all tests pass, including old predict, old library recovery, and existing protocol errors.

- [ ] **Step 5: Commit**

~~~
git add src/workpiece_catalog.py src/orientation_tcp_service.py tests/test_workpiece_catalog.py tests/test_orientation_tcp_service.py
git commit -m "feat: expose ordered batch prediction protocol"
~~~

### Task 5: Teach BackendClient to negotiate batch capabilities

**Files:**
- Modify: qt_app/backendclient.h
- Modify: qt_app/backendclient.cpp
- Test: qt_app/tests/test_backendclient.cpp

**Interfaces:**
- BackendClient::supportsBatchPrediction() const -> bool.
- BackendClient::batchPredictionReady() const -> bool.
- BackendClient::batchWorkerCount() const -> int.
- Existing handshakeSucceeded(quint64, QJsonObject) signal remains unchanged.

- [ ] **Step 1: Write failing Qt handshake tests**

Add a response with capabilities.predict_batch=true and assert the accessors; add a response with no capabilities and assert compatibility fallback. Verify reconnect clears stale capability data.

~~~
void batchCapabilitiesAreStoredFromHandshake() {
    BackendClient client;
    QJsonObject capabilities{{"predict_batch", true}, {"batch_ready", true},
                             {"batch_workers", 4}, {"batch_threads_per_worker", 1}};
    // The test connects to FakeTcpServer, replies to hello with the JSON above,
    // waits for handshakeSucceeded, then executes the accessors.
    QVERIFY(client.supportsBatchPrediction());
    QVERIFY(client.batchPredictionReady());
    QCOMPARE(client.batchWorkerCount(), 4);
}
~~~

- [ ] **Step 2: Run the test and confirm RED**

Run:

~~~
.\scripts\run_qt5_tests.ps1 -Targets test_backendclient
~~~

Expected: compile failure because the accessors and stored metadata do not exist.

- [ ] **Step 3: Implement capability storage**

Add a QJsonObject handshakeMetadata_ member. Store the accepted hello response before emitting handshakeSucceeded; clear it in connectToService, disconnectFromService, and transport failure. Accessors require both predict_batch=true and batch_ready=true, and return 0 for invalid worker counts.

- [ ] **Step 4: Run the focused Qt test**

Run:

~~~
.\scripts\run_qt5_tests.ps1 -Targets test_backendclient
~~~

Expected: all BackendClient tests pass.

- [ ] **Step 5: Commit**

~~~
git add qt_app/backendclient.h qt_app/backendclient.cpp qt_app/tests/test_backendclient.cpp
git commit -m "feat: negotiate batch inference capabilities"
~~~

### Task 6: Switch Qt batch detection to one ordered request with compatibility fallback

**Files:**
- Modify: qt_app/inspectionpage.h
- Modify: qt_app/inspectionpage.cpp
- Modify: qt_app/mainwindow.h
- Modify: qt_app/mainwindow.cpp
- Test: qt_app/tests/test_inspectionpage.cpp
- Test: qt_app/tests/test_mainwindow.cpp

**Interfaces:**
- InspectionPage::setBatchPredictionCapabilities(bool supported, bool ready, int workers).
- InspectionPage::handleBackendResponse handles predict and predict_batch.
- InspectionPage::handleBackendFailure handles both commands and can restart the scalar compatibility sequence.

- [ ] **Step 1: Write failing Qt batch-flow tests**

Extend BatchPredictionServer with a predict_batch branch that records image_paths and returns ordered items. Add tests asserting one batch request for a capable backend, table/image selection updates for every returned item, old server behavior still sends one predict per image, and partial-error handling.

~~~
void capableBackendReceivesOneBatchRequestAndPreservesRows() {
    // Configure BatchPredictionServer to advertise predict_batch/batch_ready,
    // submit three paths, and let the Qt event loop process the response.
    QTRY_COMPARE(server.batchRequestCount(), 1);
    QCOMPARE(server.lastBatchPaths(), paths);
    waitForBatchCompletion(window, 3);
    QCOMPARE(table->rowCount(), 3);
    table->setCurrentCell(1, 0);
    QVERIFY(currentImageLabel->text().contains(QStringLiteral("batch-1.png")));
}
~~~

- [ ] **Step 2: Run focused Qt tests and confirm RED**

Run:

~~~
.\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage
.\scripts\run_qt5_tests.ps1 -Targets test_mainwindow
~~~

Expected: new tests fail because capability setters and predict_batch routing do not exist; existing scalar tests should still pass.

- [ ] **Step 3: Implement page-level batch response handling**

Store a batch request state separate from currentBatchRequestId_. On predict_batch, validate that items is an array, map each index to batchRecordOrder_, update label/review/elapsed/error/disposition, and rebuild the table once. Preserve manual selection while responses are applied. Read batch_timings_ms.total, worker_count, and fallback into the batch summary so the operator can see the actual backend mode and elapsed time. If the response is malformed, mark the whole batch failed with a visible protocol error; do not silently mark all rows successful.

- [ ] **Step 4: Implement MainWindow capability routing and fallback**

When onBackendReady runs, pass BackendClient capability accessors to InspectionPage. In submitBatchPrediction, issue predict_batch with one image_paths array when ready; otherwise retain requestNextBatchPrediction and scalar predict. Route response/failure ownership for both command names. Keep queued refresh and stop behavior intact; a batch request failure falls back only when no item result was received, and displays the fallback reason.

- [ ] **Step 5: Run all relevant Qt tests**

Run:

~~~
.\scripts\run_qt5_tests.ps1 -Targets test_inspectionpage
.\scripts\run_qt5_tests.ps1 -Targets test_mainwindow
~~~

Expected: new batch tests and all pre-existing UI, selection, review, reconnect, and compatibility tests pass.

- [ ] **Step 6: Commit**

~~~
git add qt_app/inspectionpage.h qt_app/inspectionpage.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_inspectionpage.cpp qt_app/tests/test_mainwindow.cpp
git commit -m "feat: route Qt batch detection through parallel protocol"
~~~

### Task 7: Add a reproducible batch benchmark and frozen-package smoke coverage

**Files:**
- Create: scripts/benchmark_batch_inference.py
- Modify: scripts/smoke_portable_package.py
- Modify: tests/test_benchmark_portable_service.py
- Test: tests/test_smoke_portable_package.py
- Create after measurement: docs/verification/cpu-batch-parallel-inference-results.md

**Interfaces:**
- CLI scripts/benchmark_batch_inference.py --package-root PATH --workpiece-id ID --image PATH (repeatable) --warmup INT --iterations INT --workers INT --threads-per-worker INT --report PATH.
- JSON report fields: hardware, configuration, warmup, iterations, batch_size, timings_ms, throughput_images_per_second, accuracy, fallback.

- [ ] **Step 1: Write failing benchmark/report tests**

Test percentile calculation, ordered batch payload generation, and report validation. Require warmup/measured separation and P50/P95/P99 plus worker/thread configuration.

~~~
def test_batch_report_has_percentiles_and_configuration(tmp_path):
    report = run_synthetic_batch_benchmark([20.0, 30.0, 40.0], batch_size=5,
                                           workers=4, threads_per_worker=1)
    assert report["batch_size"] == 5
    assert report["configuration"] == {"workers": 4, "threads_per_worker": 1}
    assert report["timings_ms"]["p95"] >= 30.0
~~~

- [ ] **Step 2: Run the tests and confirm RED**

Run:

~~~
pytest tests/test_benchmark_portable_service.py tests/test_smoke_portable_package.py -q
~~~

Expected: new report assertions fail because no batch benchmark/report fields exist.

- [ ] **Step 3: Implement the benchmark and smoke command**

Use one persistent TCP connection, send hello, verify predict_batch capability, issue exactly one 5-image request per measured iteration, and record wall-clock response time. Run both 4×1 and 2×2 configurations by setting backend environment/configuration before launch; never include model startup in warm timings. The smoke test also launches the frozen backend from a temporary working directory and fails if src.orientation_classifier is missing.

- [ ] **Step 4: Run benchmark/report tests and static checks**

Run:

~~~
pytest tests/test_benchmark_portable_service.py tests/test_smoke_portable_package.py -q
python -m py_compile scripts/benchmark_batch_inference.py
~~~

Expected: all tests pass and the script compiles without warnings.

- [ ] **Step 5: Commit**

~~~
git add scripts/benchmark_batch_inference.py scripts/smoke_portable_package.py tests/test_benchmark_portable_service.py tests/test_smoke_portable_package.py
git commit -m "test: add CPU batch throughput benchmark"
~~~

### Task 8: Build, measure on i5-8250U, and close the release loop

**Files:**
- Create: docs/verification/cpu-batch-parallel-inference-results.md
- No source changes are planned in this task; a defect returns to its owning task with a new regression test before any source edit.

**Interfaces:**
- Release artifact remains WorkpieceOrientation-CPU-x64-<version>.zip.
- Verification report records commit, package SHA-256, OS, CPU, model fingerprint, worker/thread configurations, warmup count, sample count, P50/P95/P99/max, and fallback reason.

- [ ] **Step 1: Run the complete Python regression suite before packaging**

Run:

~~~
pytest -q
~~~

Expected: all Python tests pass. If a failure is found, add a focused regression test before changing production code.

- [ ] **Step 2: Build the CPU frozen backend and audit its archive**

Run from the repository root:

~~~
.\scripts\build_portable_backend.ps1 -Edition cpu -Python E:\python\anaconda3\envs\shitu\python.exe
~~~

Expected: build succeeds, no project Python files are loose in the frozen backend, and the archive audit confirms every production_src_modules entry.

- [ ] **Step 3: Run Qt 5.14.2 tests and build the release UI**

Run:

~~~
.\scripts\run_qt5_tests.ps1
~~~

Expected: all Qt tests pass. Build the portable Qt release with the existing Qt 5.14.2 toolchain after the tests are green.

- [ ] **Step 4: Measure both worker configurations on the target machine**

Run the benchmark with 50 warmup batches and at least 200 measured 5-image batches for 4×1 and 2×2. Save both JSON reports and compare P50/P95/P99, throughput, CPU utilization, and sustained behavior after thermal stabilization. Use P95 ≤100 ms as the primary target; report a miss honestly and retain the faster measured configuration as the default.

- [ ] **Step 5: Write the verification report and run the extracted-package smoke test**

Document whether the target is met, separate cold startup from warm throughput, and include the exact package SHA-256. Extract the ZIP into a temporary directory, launch from a different current directory, complete hello, run one batch, and verify the Qt capability path. Delete only the temporary extraction directory created by the test.

- [ ] **Step 6: Commit the final verification evidence**

~~~
git add docs/verification/cpu-batch-parallel-inference-results.md
git commit -m "docs: record CPU batch throughput verification"
~~~
