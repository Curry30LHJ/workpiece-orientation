# CPU 推理路径优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** 在现有离线 CPU 版本上消除快速路径中完全相同槽位的重复 PP-ShiTu 推理，并用可验证的 OneDNN 线程配置降低 CPU 热推理延迟，同时保持分类结果和 GPU 行为不变。

**Architecture:** FastGeometryProcessor 继续生成 raw/front/back 三个语义槽位。CPU 侧在 FastOrientationEngine 中对槽位做 shape、dtype 和像素内容的精确去重，提取唯一 embedding 后恢复原槽位顺序；GPU 侧默认关闭去重。OrientationClassifier.load() 读取 CPU 线程配置和优化开关，并将其传递给 PaddleClas 与快速引擎。便携包基准提供线程覆盖入口，统一输出准确率和 P50/P95/P99 指标。

**Tech Stack:** Python 3.10、NumPy、OpenCV、PaddlePaddle/PaddleClas、OneDNN/MKLDNN、pytest、现有离线 TCP 后端和便携包基准脚本。

## Global Constraints

- 不重新训练、微调或替换 PP-ShiTu 权重。
- 不改变 PP-ShiTu 的输入尺寸、预处理语义或特征向量格式。
- 不修改几何规则拟合、边界偏移、内部/外部忽略语义和正反面融合阈值。
- 不在本变更中引入 ONNX/OpenVINO/INT8 或更小模型。
- 不删除 ALIKED、LightGlue 或 ORB 的兼容/回退代码。
- 不在后端启动时自动执行线程搜索。
- Global.cpu_num_threads 是线程配置入口，WORKPIECE_CPU_THREADS 优先级高于配置文件。
- P95 <= 25 ms 是持续目标，第一阶段不得在未实测时宣称达标。
- 旧版 5+5 工件库、正反面不等模板数量和已有缓存格式必须继续兼容。
- GPU 默认保持当前三槽位 batch size 3 行为。

---

### Task 1: 为快速引擎增加精确槽位去重

**Files:**
- Modify: src/fast_orientation.py（构造函数和 predict 路径）
- Test: tests/test_fast_orientation.py

**Interfaces:**
- Consumes: 现有 FastGeometryVariants.images（三个 np.ndarray）和 embed_batch(images) 回调。
- Produces: FastOrientationEngine(..., deduplicate_identical_slots: bool = False)；私有 _embed_query_slots(images: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]，返回三个 embedding 和唯一图像数量。

- [ ] **Step 1: 写失败测试**

为测试 fake 引擎增加一个可控的槽位几何 stub，并覆盖以下代码语义：

~~~python
class SlotGeometry:
    def __init__(self, markers):
        self.markers = tuple(markers)

    def build_variants(self, image, compiled):
        images = tuple(np.full_like(image, marker) for marker in self.markers)
        return FastGeometryVariants(
            images, "not_configured", False, (), {},
            {"geometry_context": 0.0, "geometry_fit": 0.0, "mask_build": 0.0},
        )


def make_small_cache(engine, tmp_path):
    front = write_markers(tmp_path, "front", 20, base=10)
    back = write_markers(tmp_path, "back", 20, base=180)
    return engine.build_cache(
        front, back, geometry_profile=None, library_revision=1,
        model_fingerprint="m",
    )


def test_cpu_dedup_reuses_identical_query_slots(tmp_path):
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 10, 10)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    cache = make_small_cache(engine, tmp_path)
    embedder.batch_sizes.clear()

    result = engine.predict(marker_image(10), cache)

    assert embedder.batch_sizes == [1]
    assert result["timings_ms"]["global_unique_slots"] == 1


def test_dedup_keeps_three_calls_when_geometry_slots_differ(tmp_path):
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 30)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    cache = make_small_cache(engine, tmp_path)

    engine.predict(marker_image(10), cache)

    assert embedder.batch_sizes == [3]


def test_dedup_maps_two_equal_slots_back_to_original_order():
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 20)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    slots = (marker_image(10), marker_image(20), marker_image(20))

    embeddings, unique_count = engine._embed_query_slots(slots)

    assert embedder.batch_sizes == [2]
    assert unique_count == 2
    assert np.array_equal(embeddings[1], embeddings[2])
~~~

所有 stub 和 helper 都在本测试文件中定义，不使用真实模型。当前实现应让三个新增测试失败，而原有 batch size 3 测试继续通过。

- [ ] **Step 2: 运行测试确认先红**

运行：

~~~powershell
python -m pytest tests/test_fast_orientation.py -k "dedup or predict_calls_one_three_image_batch" -q
~~~

预期：新增测试失败，现有 test_predict_calls_one_three_image_batch... 仍通过。

- [ ] **Step 3: 实现最小逻辑**

在 FastOrientationEngine 保存布尔开关，新增 _embed_query_slots。比较必须同时满足 shape、dtype 和 np.array_equal 像素完全相同，按首次出现顺序生成 unique_images 与 slot_to_unique；只把 unique_images 送入 embed_batch，再按映射恢复三个 embedding。dedup 开关关闭时保持原调用。比较阶段只捕获比较异常并回退原三槽位路径；embedder 返回数量错误或推理异常必须原样抛出，不能在模型异常后重复调用。

predict 调用 _embed_query_slots，并在 timings_ms 中增加 global_input_slots=3 和 global_unique_slots；global_batch 的现有含义不变。

- [ ] **Step 4: 运行定向测试确认变绿**

~~~powershell
python -m pytest tests/test_fast_orientation.py -k "dedup or predict_calls_one_three_image_batch" -q
~~~

预期：新增测试和原有 fast orientation 测试通过。

- [ ] **Step 5: 提交**

~~~powershell
git add src/fast_orientation.py tests/test_fast_orientation.py
git commit -m "perf: reuse identical CPU query slots"
~~~

### Task 2: 接入 CPU 开关、线程配置和运行时诊断

**Files:**
- Modify: src/orientation_classifier.py（Paddle 配置和 FastOrientationEngine 创建）
- Test: tests/test_orientation_classifier.py
- Test: tests/test_orientation_service_environment.py（仅在服务入口增加配置覆盖测试时修改）

**Interfaces:**
- Consumes: compute_device、Global.cpu_num_threads 和进程环境。
- Produces: CPU_THREADS_ENV = WORKPIECE_CPU_THREADS、CPU_SLOT_DEDUP_ENV = WORKPIECE_CPU_DEDUPLICATE_SLOTS；私有 _resolve_cpu_num_threads(global_config) -> int 和 _resolve_cpu_slot_dedup(compute_device) -> bool；加载后的 classifier 暴露 cpu_num_threads 与 fast_engine.deduplicate_identical_slots。

- [ ] **Step 1: 写失败测试**

使用现有 fake Paddle/PaddleClas 安装辅助函数增加：

~~~python
def test_cpu_load_applies_thread_override_and_enables_slot_dedup(tmp_path, monkeypatch):
    captured = _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    monkeypatch.setenv("WORKPIECE_CPU_THREADS", "4")
    monkeypatch.delenv("WORKPIECE_CPU_DEDUPLICATE_SLOTS", raising=False)

    loaded = OrientationClassifier.load(
        tmp_path, tmp_path / "model", compute_device="cpu",
        inference_mode="fast_geometry",
    )

    assert captured["config"].Global.cpu_num_threads == 4
    assert loaded.cpu_num_threads == 4
    assert loaded.fast_engine.deduplicate_identical_slots is True


def test_gpu_load_ignores_cpu_slot_dedup_and_cpu_thread_override(tmp_path, monkeypatch):
    captured = _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    monkeypatch.setenv("WORKPIECE_CPU_THREADS", "1")
    monkeypatch.setenv("WORKPIECE_CPU_DEDUPLICATE_SLOTS", "1")

    loaded = OrientationClassifier.load(
        tmp_path, tmp_path / "model", compute_device="gpu",
        inference_mode="fast_geometry",
    )

    assert loaded.fast_engine.deduplicate_identical_slots is False
    assert captured["config"].Global.use_gpu is True
    assert captured["config"].Global.enable_mkldnn is False


def test_invalid_cpu_thread_override_keeps_yaml_value_and_logs_warning(tmp_path, monkeypatch, caplog):
    captured = _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    monkeypatch.setenv("WORKPIECE_CPU_THREADS", "not-an-int")

    OrientationClassifier.load(
        tmp_path, tmp_path / "model", compute_device="cpu",
        inference_mode="fast_geometry",
    )

    assert captured["config"].Global.cpu_num_threads == 2
    assert "WORKPIECE_CPU_THREADS" in caplog.text
~~~

- [ ] **Step 2: 运行测试确认红**

~~~powershell
python -m pytest tests/test_orientation_classifier.py -k "cpu_load_applies or gpu_load_ignores or invalid_cpu_thread" -q
~~~

预期：新测试失败，因为当前加载器没有读取环境变量或设置引擎去重开关。

- [ ] **Step 3: 实现配置解析和传递**

在 orientation_classifier.py 中：

1. 从 config.Global.cpu_num_threads 读取默认值；当前部署配置和代码回退默认值均为 2。WORKPIECE_CPU_THREADS 存在时要求为大于 0 的十进制整数，否则 warning 并保留有效 YAML 值或回退到 2。
2. 仅 compute_device == cpu 时写回 config.Global.cpu_num_threads；GPU 不受该环境变量影响。
3. 解析 WORKPIECE_CPU_DEDUPLICATE_SLOTS 的 0/1、true/false、yes/no；CPU 默认启用，GPU 强制关闭；非法值 warning 后采用设备默认值。
4. 将 deduplicate_identical_slots 传给 FastOrientationEngine，并保存 classifier.cpu_num_threads。
5. 保持已有 Paddle 设备自检、模型指纹和 use_gpu/enable_mkldnn 逻辑不变。

- [ ] **Step 4: 运行 CPU 配置与 fast orientation 测试**

~~~powershell
python -m pytest tests/test_orientation_classifier.py tests/test_fast_orientation.py -q
~~~

预期：全部通过。

- [ ] **Step 5: 提交**

~~~powershell
git add src/orientation_classifier.py tests/test_orientation_classifier.py tests/test_orientation_service_environment.py
git commit -m "perf: configure CPU fast inference runtime"
~~~

### Task 3: 为便携 CPU 基准增加线程覆盖和去重指标

**Files:**
- Modify: release_tools/portable_benchmark.py（run_benchmark 参数和子进程环境）
- Modify: scripts/benchmark_portable_service.py（CLI 参数）
- Test: tests/test_benchmark_portable_service.py

**Interfaces:**
- Consumes: run_benchmark(..., cpu_threads: int | None = None) 和 app_config.json 的 compute_device。
- Produces: --cpu-threads CLI 参数、报告 settings.cpu_threads、CPU 子进程环境 WORKPIECE_CPU_THREADS。

- [ ] **Step 1: 写失败测试**

扩展现有 FakeProcess 测试捕获 Popen 的 env，并增加：

~~~python
def test_run_benchmark_passes_cpu_thread_override_to_backend(tmp_path, monkeypatch):
    captured = {}

    class CapturingProcess(FakeProcess):
        def __init__(self, args, **kwargs):
            captured["env"] = kwargs["env"]
            super().__init__(args, **kwargs)

    monkeypatch.setattr(benchmark.subprocess, "Popen", CapturingProcess)
    report = run_benchmark(package, spec, iterations=1000, warmup=0, cpu_threads=4)

    assert report["settings"]["cpu_threads"] == 4
    assert captured["env"]["WORKPIECE_CPU_THREADS"] == "4"
~~~

复用本文件现有最小 CPU package/config/acceptance fixture 和 FakeClient；GPU 测试应增加断言证明不会注入 CPU 线程环境变量。

- [ ] **Step 2: 运行测试确认红**

~~~powershell
python -m pytest tests/test_benchmark_portable_service.py -k "cpu_thread_override" -q
~~~

预期：失败，因为 run_benchmark 尚无 cpu_threads 参数或 env 设置。

- [ ] **Step 3: 实现 CLI 和环境透传**

在 run_benchmark 中校验 cpu_threads 为 None 或正整数；构造 os.environ.copy()，仅当包的 compute_device == cpu 且参数非空时写入 WORKPIECE_CPU_THREADS，并把 environment 传给 subprocess.Popen；报告 settings 记录 cpu_threads。GPU 包即使传入参数也不能改变其环境配置。

在 scripts/benchmark_portable_service.py 增加 --cpu-threads 正整数参数，并传入 run_benchmark。

- [ ] **Step 4: 运行基准单元测试**

~~~powershell
python -m pytest tests/test_benchmark_portable_service.py tests/test_benchmark_fast_geometry_inference.py -q
~~~

预期：全部通过。

- [ ] **Step 5: 提交**

~~~powershell
git add release_tools/portable_benchmark.py scripts/benchmark_portable_service.py tests/test_benchmark_portable_service.py
git commit -m "test: benchmark CPU thread configurations"
~~~

### Task 4: 完成回归、实机基准和结果报告

**Files:**
- Create: docs/verification/cpu-inference-optimization-results.md
- Create: docs/verification/cpu-inference-optimization-results.json
- Test: tests/test_fast_orientation.py、tests/test_orientation_classifier.py、tests/test_orientation_tcp_service.py、tests/test_workpiece_catalog.py

**Interfaces:**
- Consumes: Task 1–3 的去重开关、CPU 配置和便携基准命令。
- Produces: 可复现的 CPU 性能/准确率报告和最终分支验收记录。

- [ ] **Step 1: 运行完整自动化测试**

~~~powershell
python -m pytest -q
~~~

预期：所有现有测试与新增测试通过；与本变更无关的既有失败只记录名称和错误，不修改无关代码。

- [ ] **Step 2: 运行 CPU 线程候选基准**

同一 CPU 便携包、模型和验收集分别运行；发布流程生成的包路径为 release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip，验收集路径为 runtime_reports/fast-geometry-acceptance.json：

~~~powershell
python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 1 --warmup 50 --iterations 1000 --output runtime_reports/cpu-threads-1.json
python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 2 --warmup 50 --iterations 1000 --output runtime_reports/cpu-threads-2.json
python scripts/benchmark_portable_service.py --package-zip release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip --acceptance-spec runtime_reports/fast-geometry-acceptance.json --cpu-threads 4 --warmup 50 --iterations 1000 --output runtime_reports/cpu-threads-4.json
~~~

若目标机器物理核心数不是上述值，再增加该候选。选择 P95 最低且准确率、复核率不下降的线程数；不在后端启动时自动搜索。

- [ ] **Step 3: 运行优化前后对照**

使用同一包数据和验收图片，对 WORKPIECE_CPU_DEDUPLICATE_SLOTS=0 与默认开启各运行 1000 次，记录 backend_elapsed_ms、round_trip_ms、准确率、复核率、global_unique_slots 和模型调用次数。冷启动单独记录 startup.ready_ms，不混入热推理分布。

- [ ] **Step 4: 写入报告**

JSON 必须包含分支、提交、模型指纹、Paddle/PaddleClas 版本、CPU（逻辑/物理核心）、线程配置、样本指纹、冷启动、mean/P50/P95/P99/max、准确率、复核率、去重比例、实测 PP-ShiTu 调用次数和失败信息。Markdown 表格列出优化前、优化后和各线程候选；若未达到 25 ms，明确写出实测 P95 和 PP-ShiTu 仍占主导的结论，不宣称达标。

- [ ] **Step 5: 提交验收报告**

~~~powershell
git add docs/verification/cpu-inference-optimization-results.md docs/verification/cpu-inference-optimization-results.json
git commit -m "docs: record CPU inference optimization benchmark"
~~~

### Task 5: 最终分支检查

**Files:**
- Modify only if needed: docs/superpowers/specs/2026-08-30-cpu-inference-optimization-design.md

- [ ] **Step 1: 检查工作树和提交范围**

~~~powershell
git status --short --branch
git log --oneline --decorate -6
git diff feature/20260828/offline-portable-packages...HEAD --stat
~~~

确认 runtime_reports/ 等用户已有未跟踪文件没有被加入提交，改动只涉及规格列出的 CPU 路径、测试和报告。

- [ ] **Step 2: 检查优化开关可回退**

运行 CPU 基准时设置 WORKPIECE_CPU_DEDUPLICATE_SLOTS=0，确认输出与优化前路径一致；恢复默认值，确认无规则输入的 global_unique_slots 为 1。

- [ ] **Step 3: 提交最终状态**

如前述任务均已提交且无额外修改，保持分支 feature/20260830/cpu-inference-optimization，不合并或删除用户已有分支。
