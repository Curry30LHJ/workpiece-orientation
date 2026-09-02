# Offline GPU/CPU Portable Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce two completely offline Windows 10/11 x64 ZIP packages—GPU and CPU—whose Qt launcher silently owns a frozen backend EXE, starts with an empty workpiece library, shares one portable `data` format, and preserves the validated `fast_geometry` behavior.

**Architecture:** Keep the current Qt/TCP split, but add packaged-executable launch mode, relative runtime paths, device-aware Paddle initialization, versioned writable data, and structured startup phases. Freeze only the fast backend with PyInstaller `onedir`, deploy Qt with `windeployqt`, assemble each edition from a clean staging directory, and reject packages containing source, development paths, user libraries, or the wrong device runtime.

**Tech Stack:** Windows 10/11 x64, Qt 5.14.2 Widgets/Network, MSVC 2019, Python 3.10.20, PyInstaller 6.22.2, PaddlePaddle 3.2.2 CPU/GPU (CUDA 11.8 GPU wheel), PaddleClas 2.6.0, NumPy 1.24.4, OpenCV 4.6.0.66, pytest, PowerShell.

**Approved design:** [`docs/superpowers/specs/2026-08-28-offline-portable-gpu-cpu-packaging-design.md`](../specs/2026-08-28-offline-portable-gpu-cpu-packaging-design.md)

## Global Constraints

- Create `feature/20260828/offline-portable-packages` from commit `1559bff`; never add the existing untracked `runtime_reports/` directory.
- Initial portable version is `1.0.0`; both ZIP names and `version.json` use that exact version.
- Runtime target is Windows 10/11 x64. Build Windows artifacts on Windows; PyInstaller is not a cross-compiler.
- GPU runtime requires only a compatible NVIDIA driver. CPU runtime has no NVIDIA requirement. Neither runtime may require Python, Conda, Qt, CUDA Toolkit, network access, or downloads.
- Pin Python 3.10.20, PyInstaller 6.22.2, Paddle/Paddle GPU 3.2.2, PaddleClas 2.6.0, NumPy 1.24.4, and OpenCV 4.6.0.66.
- GPU uses the Paddle CUDA 11.8 wheel; CPU uses `paddlepaddle==3.2.2` with MKLDNN.
- Both release configurations fix `inference_mode=fast_geometry`; packaged requests must not import or execute ALIKED, LightGlue, Torch, or ORB local matching.
- Do not retrain PP-ShiTu, change model files, alter geometry-rule semantics, change Ridge decisions, or modify fusion thresholds.
- Ship an empty `data/workpieces`; exclude current templates, rules, caches, detections, reports, and test images.
- GPU and CPU use the same data layout and business schemas. Never silently merge two non-empty data roots.
- The user launches only `WorkpieceOrientation.exe`; its backend has no visible console and exits with its owning Qt process.
- GPU gate: warmed full-backend single-image `P95 <= 25 ms` on RTX 4060 Ti, current fixed corpus, at least 1000 measured requests.
- CPU has no 25 ms gate, but report mean/P50/P95/P99/max and every classification difference from GPU.
- Fail packaging on missing DLL/model, wrong device dependency, `.py`/`.pyi`, development path, non-empty user data, failed smoke, or version mismatch.
- Use TDD, focused tests before broad suites, `apply_patch` for repository edits, and one reviewable commit per task.

Primary build references:

- [PyInstaller Windows bundling](https://pyinstaller.org/en/stable/)
- [Paddle Windows CPU/CUDA 11.8 indexes](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/pip/windows-pip_en.html)

---

## File Map

### Runtime

- Create `src/model_fingerprint.py`, `src/runtime_data.py`, and focused tests: shared model hashing, data layout, first run, migration, backup, cleanup, writability.
- Create `src/windows_parent_watchdog.py` and `tests/test_windows_parent_watchdog.py`: stop backend when Qt disappears.
- Modify `src/orientation_classifier.py`, `src/orientation_tcp_service.py` and their existing tests: device, Paddle YAML, data paths, package identity, startup phases.

### Qt

- Modify `qt_app/appconfig.*`: development/packaged configuration and relative paths.
- Modify `qt_app/backendclient.*`: structured same-socket loading handshake.
- Modify `qt_app/backendprocessmanager.*`, `processlauncher.cpp`: select EXE/Python, validate identity, own shutdown.
- Modify `qt_app/mainwindow.*`, `main.cpp`: stable phase presentation and hidden package smoke mode.
- Modify corresponding `qt_app/tests/test_*.cpp`, `.pro`, and `workpiece_orientation.pro`.

### Release tooling

- Create `release_tools/backend_bundle.py`, `release_tools/portable_package.py`, `release_tools/portable_smoke.py`, `release_tools/portable_benchmark.py`, and focused tests.
- Create `deploy/orientation_backend.spec`, pinned requirement files, third-party notices, and `deploy/使用说明.txt`.
- Create `scripts/create_packaging_envs.ps1`, `build_portable_backend.ps1`, `build_portable_release.ps1`, `smoke_portable_package.py`, and `benchmark_portable_service.py`.
- Modify `scripts/build_qt5.ps1` and `.gitignore` for clean, isolated outputs.
- Create `docs/verification/offline-portable-gpu-cpu-results.{json,md}` after real verification.

---

### Task 1: Make PP-ShiTu Device Selection Explicit

**Files:**
- Modify: `src/orientation_classifier.py:215-330`
- Create: `src/model_fingerprint.py`
- Modify: `tests/test_orientation_classifier.py:1080-1140`

**Interfaces:**
- Produces: pure `model_directory_sha256(path)`, `COMPUTE_DEVICES`, `ComputeDeviceError`, `ModelFingerprintError`; `OrientationClassifier.load` gains keyword-only `paddle_config_path: Path | None`, `compute_device: str`, and `expected_model_fingerprint: str | None` parameters.
- Produces: `OrientationClassifier.compute_device` containing exactly `gpu` or `cpu`.
- Consumed by: Task 3 service startup and both frozen editions.

- [ ] **Step 1: Create the branch without touching local reports**

```powershell
git switch -c feature/20260828/offline-portable-packages
git status --short --branch
```

Expected: new branch is active; `runtime_reports/` remains untracked.

- [ ] **Step 2: Write failing device/config tests**

Add this fake Paddle helper and focused cases to `tests/test_orientation_classifier.py` (extend the existing fake PaddleClas helper to capture the YAML path and config object):

```python
def _install_fake_paddle_runtime(monkeypatch, *, compiled=True, gpu_count=1):
    selected = []
    paddle = types.ModuleType("paddle")
    paddle.is_compiled_with_cuda = lambda: compiled
    paddle.set_device = lambda value: selected.append(value) or value
    paddle.device = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: gpu_count),
        get_device=lambda: selected[-1] if selected else "cpu",
    )
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    return selected


def _install_fake_paddleclas(monkeypatch):
    captured = {}
    paddleclas = types.ModuleType("paddleclas")
    deploy = types.ModuleType("paddleclas.deploy")
    python_module = types.ModuleType("paddleclas.deploy.python")
    predict_rec = types.ModuleType("paddleclas.deploy.python.predict_rec")
    utils = types.ModuleType("paddleclas.deploy.utils")
    config_module = types.ModuleType("paddleclas.deploy.utils.config")

    class FakeRecPredictor:
        def __init__(self, config):
            captured["config"] = config

        def predict(self, images):
            captured.setdefault("predict_calls", []).append(len(images))
            return [np.asarray([1.0, 0.0], np.float32) for _ in images]

    def get_config(path, show=False):
        captured["config_path"] = path
        config = SimpleNamespace(Global=SimpleNamespace())
        captured["config"] = config
        return config

    predict_rec.RecPredictor = FakeRecPredictor
    config_module.get_config = get_config
    for name, module in {
        "paddleclas": paddleclas,
        "paddleclas.deploy": deploy,
        "paddleclas.deploy.python": python_module,
        "paddleclas.deploy.python.predict_rec": predict_rec,
        "paddleclas.deploy.utils": utils,
        "paddleclas.deploy.utils.config": config_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


@pytest.mark.parametrize(
    "device,use_gpu,mkldnn,selected_device",
    [("gpu", True, False, "gpu:0"), ("cpu", False, True, "cpu")],
)
def test_load_configures_requested_paddle_device(
    tmp_path, monkeypatch, device, use_gpu, mkldnn, selected_device
):
    captured = _install_fake_paddleclas(monkeypatch)
    selected = _install_fake_paddle_runtime(monkeypatch)
    yaml_path = tmp_path / "部署 配置.yaml"
    yaml_path.write_text("Global: {}\n", encoding="utf-8")
    loaded = OrientationClassifier.load(
        tmp_path, tmp_path / "模型",
        paddle_config_path=yaml_path,
        compute_device=device,
        inference_mode="fast_geometry",
    )
    assert captured["config_path"] == str(yaml_path)
    assert captured["config"].Global.use_gpu is use_gpu
    assert captured["config"].Global.enable_mkldnn is mkldnn
    assert selected == [selected_device]
    assert captured["predict_calls"] == [1]
    assert loaded.compute_device == device


def test_gpu_load_refuses_missing_cuda_without_cpu_fallback(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch, compiled=False, gpu_count=0)
    with pytest.raises(ComputeDeviceError) as error:
        OrientationClassifier.load(
            tmp_path, tmp_path / "model",
            compute_device="gpu", inference_mode="fast_geometry",
        )
    assert error.value.code == "GPU_UNAVAILABLE"


def test_load_rejects_unknown_compute_device(tmp_path):
    with pytest.raises(ValueError, match="compute_device"):
        OrientationClassifier.load(
            tmp_path, tmp_path / "model",
            compute_device="automatic", inference_mode="fast_geometry",
        )


def test_cpu_load_never_probes_cuda(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    paddle = sys.modules["paddle"]
    def unexpected_cuda_probe():
        raise AssertionError("CPU edition probed CUDA")
    paddle.is_compiled_with_cuda = unexpected_cuda_probe
    paddle.device.cuda.device_count = unexpected_cuda_probe
    loaded = OrientationClassifier.load(
        tmp_path, tmp_path / "model",
        compute_device="cpu", inference_mode="fast_geometry",
    )
    assert loaded.compute_device == "cpu"


def test_load_rejects_wrong_expected_model_fingerprint(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference.pdmodel").write_bytes(b"model")
    with pytest.raises(ModelFingerprintError) as error:
        OrientationClassifier.load(
            tmp_path, model_dir,
            compute_device="gpu",
            expected_model_fingerprint="0" * 64,
            inference_mode="fast_geometry",
        )
    assert error.value.code == "MODEL_FINGERPRINT_MISMATCH"
```

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py -q -p no:cacheprovider -k "requested_paddle_device or missing_cuda or unknown_compute_device"
```

Expected: missing arguments/error/state cause failures.

- [ ] **Step 4: Implement the minimal device contract**

Add:

```python
COMPUTE_DEVICES = ("gpu", "cpu")
DEFAULT_COMPUTE_DEVICE = "gpu"


class ComputeDeviceError(OrientationClassifierError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelFingerprintError(OrientationClassifierError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_compute_device(value: str) -> str:
    device = str(value).strip().lower()
    if device not in COMPUTE_DEVICES:
        raise ValueError("compute_device must be 'gpu' or 'cpu'")
    return device


def _select_paddle_device(paddle: Any, device: str) -> str:
    if device == "gpu":
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
            raise ComputeDeviceError("GPU_UNAVAILABLE", "未检测到可用的 NVIDIA GPU/Paddle GPU 运行时")
        paddle.set_device("gpu:0")
        if not str(paddle.device.get_device()).startswith("gpu"):
            raise ComputeDeviceError("DEVICE_MISMATCH", "Paddle 未实际使用 GPU")
        return "gpu"
    paddle.set_device("cpu")
    if not str(paddle.device.get_device()).startswith("cpu"):
        raise ComputeDeviceError("DEVICE_MISMATCH", "Paddle 未实际使用 CPU")
    return "cpu"
```

Move the existing path-plus-content SHA-256 loop into dependency-free `src/model_fingerprint.py` as `model_directory_sha256(path)` and retain `_model_directory_fingerprint` as a delegating compatibility wrapper. Store `compute_device` on the classifier. Resolve `paddle_config_path` explicitly when supplied; otherwise preserve the current development default. Compute the deterministic directory fingerprint before constructing Paddle and reject a supplied non-matching 64-character SHA-256 with `MODEL_FINGERPRINT_MISMATCH`. Set `use_gpu=device == "gpu"`, `enable_mkldnn=device == "cpu"`, `enable_benchmark=False`, and `gpu_mem=1024`. After creating `RecPredictor`, run one constant 512×512 RGB image through `predict`, require exactly one finite non-empty embedding, and map failure to `RUNTIME_SELF_CHECK_FAILED`. Preserve lazy local-model imports and `fast_geometry` behavior.

- [ ] **Step 5: Run classifier regressions**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_classifier.py -q -p no:cacheprovider
```

Expected: all pass, including the no-ALIKED/LightGlue fast-load test.

- [ ] **Step 6: Commit**

```powershell
git add src/model_fingerprint.py src/orientation_classifier.py tests/test_orientation_classifier.py
git commit -m "feat: configure Paddle device explicitly"
```

---

### Task 2: Add a Versioned Writable Data Layout

**Files:**
- Create: `src/runtime_data.py`
- Create: `tests/test_runtime_data.py`

**Interfaces:**
- Produces: `DATA_LAYOUT_VERSION = 1`.
- Produces: immutable `RuntimeDataPaths(root, workpieces, rules, cache, logs, temp, metadata, backup)`.
- Produces: `prepare_runtime_data(root, *, clock=None)` and `legacy_runtime_data(library_dir)`.
- Consumed by: Task 3 service and Task 7 package assembler.

- [ ] **Step 1: Write failing creation/migration/rollback tests**

Create `tests/test_runtime_data.py`:

```python
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.runtime_data import DATA_LAYOUT_VERSION, RuntimeDataError, prepare_runtime_data


def fixed_clock():
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_first_run_creates_empty_portable_layout(tmp_path: Path):
    paths = prepare_runtime_data(tmp_path / "甲方 数据", clock=fixed_clock)
    assert paths.workpieces.is_dir()
    assert paths.rules.is_dir()
    assert paths.cache.is_dir()
    assert paths.logs.is_dir()
    assert paths.temp.is_dir()
    assert list(paths.workpieces.iterdir()) == []
    assert json.loads(paths.metadata.read_text(encoding="utf-8")) == {
        "layout_version": DATA_LAYOUT_VERSION
    }


def test_unversioned_legacy_library_is_backed_up_and_migrated(tmp_path: Path):
    root = tmp_path / "data"
    (root / "piece-a").mkdir(parents=True)
    (root / "piece-a" / "manifest.json").write_text(
        json.dumps({"id": "piece-a", "name": "M1"}), encoding="utf-8"
    )
    (root / ".recycled").mkdir()
    (root / "diagnostics").mkdir()
    (root / "diagnostics" / "old.log").write_text("old", encoding="utf-8")
    paths = prepare_runtime_data(root, clock=fixed_clock)
    assert (paths.workpieces / "piece-a" / "manifest.json").is_file()
    assert (paths.workpieces / ".recycled").is_dir()
    assert (paths.logs / "old.log").read_text(encoding="utf-8") == "old"
    assert paths.backup == tmp_path / "data.backup-20260828-120000"
    assert (paths.backup / "piece-a" / "manifest.json").is_file()


def test_unknown_legacy_entry_is_rejected_without_mutation(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "unknown.bin").write_bytes(b"keep")
    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root, clock=fixed_clock)
    assert error.value.code == "DATA_LAYOUT_AMBIGUOUS"
    assert (root / "unknown.bin").read_bytes() == b"keep"
    assert not (tmp_path / "data.backup-20260828-120000").exists()


def test_newer_layout_is_rejected_without_rewrite(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    marker = root / "data_layout.json"
    marker.write_text(json.dumps({"layout_version": 99}), encoding="utf-8")
    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root)
    assert error.value.code == "DATA_VERSION_UNSUPPORTED"
    assert json.loads(marker.read_text(encoding="utf-8"))["layout_version"] == 99


def test_owned_temp_directory_is_cleared_on_start(tmp_path: Path):
    paths = prepare_runtime_data(tmp_path / "data")
    (paths.temp / "stale.tmp").write_text("stale", encoding="utf-8")
    prepare_runtime_data(paths.root)
    assert list(paths.temp.iterdir()) == []
```

Add a fault-injection test that makes the final `os.replace` fail; the original root must be restored and staging removed.

- [ ] **Step 2: Run and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_runtime_data.py -q -p no:cacheprovider
```

Expected: `ModuleNotFoundError` for `src.runtime_data`.

- [ ] **Step 3: Implement public types and first-run creation**

```python
DATA_LAYOUT_VERSION = 1


class RuntimeDataError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeDataPaths:
    root: Path
    workpieces: Path
    rules: Path
    cache: Path
    logs: Path
    temp: Path
    metadata: Path
    backup: Path | None = None
```

Resolve paths; create `workpieces/rules/cache/logs/temp`; atomically write `data_layout.json` via a sibling temporary file. Verify writability by creating and deleting a hidden probe whose name is `.write-probe-` plus a generated UUID; map failure to `DATA_DIRECTORY_NOT_WRITABLE`.

- [ ] **Step 4: Implement the only supported legacy migration transaction**

For an unversioned root:

1. If empty or containing only `workpieces`, `rules`, `cache`, `logs`, `temp`, initialize v1 in place.
2. If it contains workpiece directories with `manifest.json` plus only `.recycled`, `.geometry-mask-jobs`, `.evolution`, and `diagnostics`, copy them into a sibling staging layout (`diagnostics` becomes `logs`; everything else becomes `workpieces`), write metadata, rename original to `data.backup-YYYYMMDD-HHMMSS`, then rename staging to `data`.
3. Reject every other entry with `DATA_LAYOUT_AMBIGUOUS`.
4. If final rename fails, restore backup to the original path and raise `DATA_MIGRATION_FAILED`.

Do not merge two non-empty versioned roots. `legacy_runtime_data(library_dir)` retains the old library path and maps diagnostics/temp beneath it without moving data.

- [ ] **Step 5: Run focused and old-library recovery tests**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_runtime_data.py tests/test_workpiece_library.py tests/test_workpiece_catalog.py -q -p no:cacheprovider
```

Expected: all pass, including old 5+5 and unequal-template recovery.

- [ ] **Step 6: Commit**

```powershell
git add src/runtime_data.py tests/test_runtime_data.py
git commit -m "feat: add portable runtime data layout"
```

---

### Task 3: Expose Structured Backend Startup and Package Identity

**Files:**
- Create: `src/windows_parent_watchdog.py`
- Create: `tests/test_windows_parent_watchdog.py`
- Modify: `src/orientation_tcp_service.py:97-270,360-410,1104-1195`
- Modify: `tests/test_orientation_tcp_service.py:1500-1660`
- Modify: `tests/test_orientation_service_environment.py`
- Modify: `tests/test_orientation_service_integration.py`

**Interfaces:**
- Produces: `ServiceRuntime.update_loading(phase, message, progress)`.
- Adds hello fields: `phase`, `message`, `progress`, `package_version`, `edition`, `compute_device`, `model_fingerprint`, `instance_token`, `error_action`, and `log_path`.
- Adds CLI: mutually exclusive `--data-root`/`--library-dir`, plus `--paddle-config`, `--compute-device`, `--model-sha256`, `--package-version`, `--edition`, `--instance-token`, and `--parent-pid`.
- Requires the matching packaged `instance_token` on the `shutdown` command.
- Produces: `start_parent_watchdog(parent_pid, on_parent_exit, *, wait_for_exit=None) -> threading.Thread | None`.
- Consumed by: Task 5 Qt handshake and Task 6 frozen backend.

- [ ] **Step 1: Write failing parser, phase, identity, and watchdog tests**

Add:

```python
def test_packaged_parser_accepts_identity_device_and_parent(tmp_path):
    args = service_module._build_argument_parser().parse_args([
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path / "models"),
        "--data-root", str(tmp_path / "data"),
        "--paddle-config", str(tmp_path / "inference.yaml"),
        "--compute-device", "cpu",
        "--model-sha256", "a" * 64,
        "--package-version", "1.0.0",
        "--edition", "cpu",
        "--instance-token", "launch-123",
        "--parent-pid", "4321",
        "--inference-mode", "fast_geometry",
    ])
    assert args.data_root == tmp_path / "data"
    assert args.library_dir is None
    assert (args.compute_device, args.package_version, args.edition,
            args.instance_token, args.parent_pid) == (
        "cpu", "1.0.0", "cpu", "launch-123", 4321
    )


def test_parser_rejects_data_root_and_library_together(tmp_path):
    with pytest.raises(SystemExit):
        service_module._build_argument_parser().parse_args([
            "--project-root", str(tmp_path),
            "--model-dir", str(tmp_path / "models"),
            "--data-root", str(tmp_path / "data"),
            "--library-dir", str(tmp_path / "library"),
        ])


def test_hello_reports_loading_phase_and_identity():
    runtime = ServiceRuntime(
        package_version="1.0.0", edition="gpu", compute_device="gpu",
        model_fingerprint="a" * 64, instance_token="launch-123",
    )
    runtime.update_loading("loading_model", "正在加载 PP-ShiTu 模型", 35)
    response = OrientationCommandDispatcher(runtime).dispatch({
        "version": 1, "request_id": "hello-1", "command": "hello"
    })
    assert response["ready"] is False
    assert response["phase"] == "loading_model"
    assert response["progress"] == 35
    assert response["package_version"] == "1.0.0"
    assert response["edition"] == "gpu"
    assert response["compute_device"] == "gpu"
    assert response["model_fingerprint"] == "a" * 64
    assert response["instance_token"] == "launch-123"


def test_loading_phase_and_progress_never_regress():
    runtime = ServiceRuntime()
    runtime.update_loading("loading_model", "正在加载模型", 35)
    runtime.update_loading("preparing_data", "迟到的数据检查", 15)
    snapshot = runtime.snapshot()
    assert (snapshot.phase, snapshot.progress) == ("loading_model", 35)


def test_packaged_shutdown_rejects_wrong_instance_token():
    runtime = ServiceRuntime(
        package_version="1.0.0", edition="gpu", compute_device="gpu",
        model_fingerprint="a" * 64, instance_token="owned-token",
    )
    response = OrientationCommandDispatcher(runtime).dispatch({
        "version": 1, "request_id": "stop-1", "command": "shutdown",
        "instance_token": "foreign-token",
    })
    assert response["ok"] is False
    assert response["error"]["code"] == "INSTANCE_TOKEN_MISMATCH"
```

Create `tests/test_windows_parent_watchdog.py`:

```python
from src.windows_parent_watchdog import start_parent_watchdog


def test_parent_exit_requests_shutdown_once():
    calls = []
    thread = start_parent_watchdog(
        4321,
        lambda: calls.append("shutdown"),
        wait_for_exit=lambda pid: calls.append(f"wait:{pid}"),
    )
    thread.join(timeout=1)
    assert calls == ["wait:4321", "shutdown"]
```

Update the existing loader fake to accept `paddle_config_path`/`compute_device` and record `preparing_data`, `loading_model`, `restoring_library`, then ready.

- [ ] **Step 2: Run and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_windows_parent_watchdog.py tests/test_orientation_service_environment.py tests/test_orientation_tcp_service.py -q -p no:cacheprovider -k "packaged_parser or data_root_and_library or identity or parent_exit or runtime_loader"
```

Expected: missing module, arguments, and metadata fail.

- [ ] **Step 3: Extend runtime state additively**

Use `dataclasses.replace` and retain protocol version 1:

```python
STARTUP_PHASE_ORDER = {
    "loading_runtime": 0,
    "preparing_data": 1,
    "loading_model": 2,
    "restoring_library": 3,
    "ready": 4,
}


@dataclass(frozen=True)
class RuntimeSnapshot:
    status: str
    phase: str = "loading_runtime"
    message: str = "正在加载运行环境"
    progress: int = 5
    package_version: str = "dev"
    edition: str = "dev"
    compute_device: str = "gpu"
    model_fingerprint: str = ""
    instance_token: str = "external"
    error_action: str = ""
    log_path: str = ""
    # retain classifier/library/error/catalog/evolution/geometry fields


def update_loading(self, phase: str, message: str, progress: int) -> None:
    if not 0 <= progress < 100:
        raise ValueError("loading progress must be between 0 and 99")
    with self._lock:
        if self._snapshot.status != "loading" or self._shutdown_requested:
            return
        if STARTUP_PHASE_ORDER[phase] < STARTUP_PHASE_ORDER[self._snapshot.phase]:
            return
        if progress < self._snapshot.progress:
            return
        self._snapshot = replace(
            self._snapshot, phase=phase, message=message, progress=progress
        )
```

Every hello includes identity/phase/progress. Ready replaces the expected model fingerprint with the fingerprint actually computed by the loaded classifier and sets `phase=ready`, `progress=100`; failed startup keeps the last phase/progress and stable code. Add one centralized service-side mapping from model/fingerprint, GPU/device, CPU initialization, unwritable data, migration, and cache-rebuild startup codes to a Chinese next action, and include the resolved log path without template-image content. Parameterized tests cover each service-side mapping; Task 5 owns missing-backend, port-conflict, and timeout actions. Packaged startup rejects a missing/blank instance token and invalid model SHA before binding, and its dispatcher rejects `shutdown` unless the request token matches. Development mode with the default `external` token preserves the existing protocol.

- [ ] **Step 4: Wire packaged arguments, data paths, logging, and loader phases**

Parser defaults remain development-compatible: version/edition `dev`, compute device `gpu`, no parent PID. Bind the listener before model/library loading. Create only the log directory before bind; perform migration in the loader:

```python
runtime.update_loading("preparing_data", "正在检查数据目录", 15)
paths = prepare_runtime_data(data_root) if data_root else legacy_runtime_data(library_dir)
runtime.update_loading("loading_model", "正在加载 PP-ShiTu 模型", 35)
classifier = OrientationClassifier.load(
    project_root, model_dir,
    paddle_config_path=paddle_config_path,
    compute_device=compute_device,
    expected_model_fingerprint=model_sha256,
    local_search_mode=local_search_mode,
    inference_mode=inference_mode,
)
runtime.update_loading("restoring_library", "正在恢复工件库和快速缓存", 80)
library = WorkpieceLibrary(paths.workpieces)
```

Map `RuntimeDataError.code`, `ComputeDeviceError.code`, and `ModelFingerprintError.code` directly; unknown failures remain `MODEL_LOAD_FAILED`. Change `configure_diagnostic_logging` to accept the exact `paths.logs` directory, rotate at local midnight, retain at most 14 daily files, and prune oldest files until the directory total is at most 30 MiB. Add a test with dated synthetic log files proving pruning never touches files outside `paths.logs`. An empty `paths.workpieces` proceeds to Ready and is never converted to a startup error.

- [ ] **Step 5: Implement the Windows parent watcher**

Use `ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE)`, `WaitForSingleObject`, and `CloseHandle`. Invalid/absent PID or an unopenable handle returns `None` and logs a warning. Once the TCP server exists, start the watcher with `server.request_shutdown`; never terminate a PID directly.

- [ ] **Step 6: Run service regressions and production-model integration**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_runtime_data.py tests/test_windows_parent_watchdog.py tests/test_orientation_service_environment.py tests/test_orientation_tcp_service.py -q -p no:cacheprovider
```

Then:

```powershell
$env:WORKPIECE_ORIENTATION_RUN_INTEGRATION='1'
$env:WORKPIECE_ORIENTATION_PROJECT_ROOT='E:\Project\wang\pp_813'
$env:WORKPIECE_ORIENTATION_MODEL_DIR='E:\Project\wang\pp_813\third_party\models\shiru_rec\general_PPLCNetV2_base_pretrained_v1.0_infer'
$env:WORKPIECE_ORIENTATION_PYTHON='E:\python\anaconda3\envs\shitu\python.exe'
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_orientation_service_integration.py -q -p no:cacheprovider
```

Expected: phases precede ready; explicit fast GPU service works; no local matcher imports.

- [ ] **Step 7: Commit**

```powershell
git add src/orientation_tcp_service.py src/windows_parent_watchdog.py tests/test_windows_parent_watchdog.py tests/test_orientation_tcp_service.py tests/test_orientation_service_environment.py tests/test_orientation_service_integration.py
git commit -m "feat: expose packaged backend startup state"
```

---

### Task 4: Parse Development and Packaged Qt Configurations

**Files:**
- Modify: `qt_app/appconfig.h`
- Modify: `qt_app/appconfig.cpp`
- Modify: `qt_app/app_config.json.example`
- Modify: `qt_app/tests/test_appconfig.cpp`

**Interfaces:**
- Produces: `BackendLaunchMode { PythonScript, PackagedExecutable }`.
- Adds `backendExecutable`, `paddleConfigPath`, `dataRoot`, `modelSha256`, `computeDevice`, `edition`, and `packageVersion`.
- Preserves existing Python/script/project/model/library development fields.
- Consumed by: Task 5 manager and Task 7 generated config.

- [ ] **Step 1: Write failing packaged-config tests**

Add a package-tree helper to `test_appconfig.cpp`:

```cpp
static QJsonObject validPackagedConfig(const QTemporaryDir &temporary,
                                       const QString &edition) {
    QDir root(temporary.path());
    root.mkpath(QStringLiteral("backend/resources"));
    root.mkpath(QStringLiteral("models/shitu_rec"));
    QFile(root.filePath(QStringLiteral("backend/orientation_backend.exe"))).open(QIODevice::WriteOnly);
    QFile(root.filePath(QStringLiteral("backend/resources/inference_general.yaml"))).open(QIODevice::WriteOnly);
    return {
        {QStringLiteral("launch_mode"), QStringLiteral("packaged_executable")},
        {QStringLiteral("backend_executable"), QStringLiteral("backend/orientation_backend.exe")},
        {QStringLiteral("project_root"), QStringLiteral(".")},
        {QStringLiteral("paddle_config"), QStringLiteral("backend/resources/inference_general.yaml")},
        {QStringLiteral("model_dir"), QStringLiteral("models/shitu_rec")},
        {QStringLiteral("data_root"), QStringLiteral("data")},
        {QStringLiteral("model_sha256"), QString(64, QLatin1Char('a'))},
        {QStringLiteral("compute_device"), edition},
        {QStringLiteral("edition"), edition},
        {QStringLiteral("package_version"), QStringLiteral("1.0.0")},
        {QStringLiteral("inference_mode"), QStringLiteral("fast_geometry")},
        {QStringLiteral("host"), QStringLiteral("127.0.0.1")},
        {QStringLiteral("port"), 37651},
        // Frozen Paddle initialization can exceed one minute offline; the
        // packaged watchdog is a ten-minute ceiling for both editions.
        {QStringLiteral("startup_timeout_ms"), 600000},
        {QStringLiteral("request_timeout_ms"), 120000},
    };
}
```

Add cases that accept a Chinese/space-containing relative package root whose absolute path is at least 180 characters; reject packaged `legacy`; reject edition/device mismatch; and still accept the existing development JSON shape with `launchMode=PythonScript`.

- [ ] **Step 2: Run and verify RED**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_appconfig
```

Expected: compile fails on missing launch-mode and portable fields.

- [ ] **Step 3: Implement launch modes and config-relative paths**

Add:

```cpp
enum class BackendLaunchMode { PythonScript, PackagedExecutable };

struct AppConfig {
    BackendLaunchMode launchMode = BackendLaunchMode::PythonScript;
    QString pythonExecutable;
    QString backendScript;
    QString backendExecutable;
    QString projectRoot;
    QString paddleConfigPath;
    QString modelDir;
    QString libraryDir;
    QString dataRoot;
    QString modelSha256;
    QString computeDevice = QStringLiteral("gpu");
    QString edition = QStringLiteral("dev");
    QString packageVersion = QStringLiteral("dev");
    // retain modes, host, port, and timeouts
};
```

Resolve every relative filesystem field against `QFileInfo(configPath).absoluteDir()`:

```cpp
const auto resolvedPath = [&configDir](const QString &value) {
    return QDir::cleanPath(QFileInfo(value).isAbsolute()
        ? value : configDir.absoluteFilePath(value));
};
```

Missing `launch_mode` remains development mode. Packaged mode requires backend EXE, Paddle YAML, model directory, data root, a lowercase 64-character model SHA-256, edition/device/version, and `fast_geometry`; the data root may not exist yet, but its parent must exist. Require edition/device equality and loopback host. Development mode leaves `modelSha256` optional so existing developer configs remain compatible.

- [ ] **Step 4: Update the checked-in development example**

Include explicit fields while retaining all existing host/port/timeouts:

```json
{
  "launch_mode": "python_script",
  "python_executable": "E:/python/anaconda3/envs/shitu/python.exe",
  "backend_script": "E:/Project/wang/pp_813/src/orientation_tcp_service.py",
  "project_root": "E:/Project/wang/pp_813",
  "paddle_config": "E:/Project/wang/pp_813/third_party/PaddleClas/deploy/configs/inference_general.yaml",
  "model_dir": "E:/Project/wang/pp_813/third_party/models/shiru_rec/general_PPLCNetV2_base_pretrained_v1.0_infer",
  "library_dir": "E:/Project/wang/pp_813/runtime_library",
  "compute_device": "gpu",
  "edition": "dev",
  "package_version": "dev",
  "inference_mode": "fast_geometry"
}
```

- [ ] **Step 5: Run config/foundation tests**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_appconfig,test_appfoundation
```

Expected: both pass; development compatibility remains.

- [ ] **Step 6: Commit**

```powershell
git add qt_app/appconfig.h qt_app/appconfig.cpp qt_app/app_config.json.example qt_app/tests/test_appconfig.cpp
git commit -m "feat: load portable Qt runtime config"
```

---

### Task 5: Launch the Frozen Backend and Stabilize Qt Startup

**Files:**
- Modify: `qt_app/backendclient.h`, `qt_app/backendclient.cpp`
- Modify: `qt_app/backendprocessmanager.h`, `qt_app/backendprocessmanager.cpp`
- Modify: `qt_app/processlauncher.cpp`
- Modify: `qt_app/mainwindow.h`, `qt_app/mainwindow.cpp`
- Modify: `qt_app/tests/test_backendclient.cpp`
- Modify: `qt_app/tests/test_backendprocessmanager.cpp`
- Modify: `qt_app/tests/test_mainwindow.cpp`

**Interfaces:**
- Changes `connectToService` to accept a startup generation and produces generation-bearing `handshakeLoading`, `handshakeSucceeded`, and `transportFailed` signals.
- Changes `handshakeSucceeded()` to `handshakeSucceeded(quint64, const QJsonObject&)`.
- Changes manager loading signal to `backendLoading(const QString &phase, const QString &message, int progress)`.
- Produces exact Python-script or packaged-EXE command line, a per-launch UUID instance token, stale-generation rejection, and identity validation before Ready.

- [ ] **Step 1: Write failing same-socket loading tests**

Extend the existing `FakeTcpServer` with `int connectionCount() const`, incrementing a private `connectionCount_` in `acceptConnection()`. Return two loading hellos and one ready hello from its existing `requestReceived` signal, then assert:

```cpp
void loadingHandshakeStaysConnectedUntilReady() {
    FakeTcpServer server;
    QVERIFY(server.start());
    int helloCount = 0;
    QObject::connect(&server, &FakeTcpServer::requestReceived, &server,
                     [&](const QJsonObject &request) {
        if (request.value(QStringLiteral("command")).toString() != QStringLiteral("hello")) {
            return;
        }
        ++helloCount;
        const QString id = request.value(QStringLiteral("request_id")).toString();
        QJsonObject response{{"version", 1}, {"request_id", id}, {"ok", true},
                             {"service", "workpiece-orientation"},
                             {"package_version", "1.0.0"}, {"edition", "gpu"},
                             {"compute_device", "gpu"},
                             {"model_fingerprint", QString(64, QLatin1Char('a'))},
                             {"instance_token", "launch-123"}};
        response.insert(QStringLiteral("ready"), helloCount > 2);
        response.insert(QStringLiteral("status"), helloCount > 2 ? "ready" : "loading");
        response.insert(QStringLiteral("phase"), helloCount > 1 ? "loading_model" : "loading_runtime");
        response.insert(QStringLiteral("message"), "正在加载");
        response.insert(QStringLiteral("progress"), helloCount > 2 ? 100 : helloCount * 30);
        server.sendJson(response);
    });
    BackendClient client;
    QSignalSpy loadingSpy(&client, &BackendClient::handshakeLoading);
    QSignalSpy readySpy(&client, &BackendClient::handshakeSucceeded);
    QSignalSpy transportSpy(&client, &BackendClient::transportFailed);
    client.connectToService(QHostAddress::LocalHost, server.port(), 1000, 7);
    QTRY_COMPARE_WITH_TIMEOUT(loadingSpy.count(), 2, 2500);
    QTRY_COMPARE_WITH_TIMEOUT(readySpy.count(), 1, 2500);
    QCOMPARE(server.connectionCount(), 1);
    QCOMPARE(transportSpy.count(), 0);
    QCOMPARE(readySpy.at(0).at(0).toULongLong(), quint64(7));
    QCOMPARE(readySpy.at(0).at(1).toJsonObject()
                 .value(QStringLiteral("package_version")).toString(),
             QStringLiteral("1.0.0"));
}
```

- [ ] **Step 2: Write failing EXE-launch and foreign-service tests**

Extend `HandshakeServer` with `setIdentity(version, edition, device, modelSha256, instanceToken)` and include those five values in every hello. Add these exact helpers to `test_backendprocessmanager.cpp`:

```cpp
static AppConfig packagedConfigFor(quint16 port, int startupTimeoutMs = 1000) {
    AppConfig config = configFor(port, startupTimeoutMs);
    config.launchMode = BackendLaunchMode::PackagedExecutable;
    config.backendExecutable = QStringLiteral("backend/orientation_backend.exe");
    config.paddleConfigPath = QStringLiteral("backend/resources/inference_general.yaml");
    config.dataRoot = QStringLiteral("data");
    config.modelSha256 = QString(64, QLatin1Char('a'));
    config.computeDevice = QStringLiteral("gpu");
    config.edition = QStringLiteral("gpu");
    config.packageVersion = QStringLiteral("1.0.0");
    config.inferenceMode = QStringLiteral("fast_geometry");
    return config;
}

static QString argumentValue(const QStringList &arguments, const QString &name) {
    const int index = arguments.indexOf(name);
    return index >= 0 && index + 1 < arguments.size() ? arguments.at(index + 1) : QString();
}
```

Then add:

```cpp
void packagedModeLaunchesBackendExeWithoutScriptArgument() {
    const quint16 port = unusedPort();
    HandshakeServer server;
    BackendClient client;
    FakeProcessLauncher launcher;
    AppConfig config = packagedConfigFor(port);
    BackendProcessManager manager(config, &client, &launcher);
    QObject::connect(&launcher, &FakeProcessLauncher::startRequested, &server, [&]() {
        server.setIdentity(
            config.packageVersion, config.edition, config.computeDevice,
            config.modelSha256,
            argumentValue(launcher.lastArguments, QStringLiteral("--instance-token")));
        QVERIFY(server.listen(port));
    });
    manager.start();
    QTRY_COMPARE_WITH_TIMEOUT(launcher.startCalls, 1, 1000);
    QCOMPARE(launcher.lastProgram, config.backendExecutable);
    QVERIFY(!launcher.lastArguments.contains(config.backendScript));
    QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--data-root")), config.dataRoot);
    QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--compute-device")), QStringLiteral("gpu"));
    QCOMPARE(argumentValue(launcher.lastArguments, QStringLiteral("--model-sha256")), config.modelSha256);
    QVERIFY(!argumentValue(launcher.lastArguments, QStringLiteral("--instance-token")).isEmpty());
}


void mismatchedExternalPackageIsRejectedWithoutTermination() {
    HandshakeServer server;
    server.setIdentity(QStringLiteral("0.9.0"), QStringLiteral("gpu"),
                       QStringLiteral("gpu"), QString(64, QLatin1Char('a')),
                       QStringLiteral("foreign-token"));
    QVERIFY(server.listen(0));
    BackendClient client;
    FakeProcessLauncher launcher;
    BackendProcessManager manager(packagedConfigFor(server.port()), &client, &launcher);
    QSignalSpy unavailableSpy(&manager, &BackendProcessManager::backendUnavailable);
    manager.start();
    QTRY_COMPARE_WITH_TIMEOUT(unavailableSpy.count(), 1, 1000);
    QCOMPARE(launcher.startCalls, 0);
    QCOMPARE(launcher.terminateCalls, 0);
    QCOMPARE(launcher.killCalls, 0);
}
```

Add one packaged-mode test in which an otherwise compatible service is already listening with `foreign-token`; require `BACKEND_INSTANCE_CONFLICT`, zero launches, and zero terminate/kill calls. Add one stale-generation test: start generation 1, restart into generation 2, inject generation-1 Loading/Ready signals, and assert the manager remains in generation 2 state. Add a MainWindow test that emits runtime/model/library phases and ready, and asserts no “未连接” or Error presentation occurs between phases. Add a ready-with-empty-library response and assert it presents a normal empty-library message rather than Error.

- [ ] **Step 3: Run and verify RED**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_backendclient,test_backendprocessmanager,test_mainwindow
```

Expected: new signals, EXE launch, identity, and phase tests fail.

- [ ] **Step 4: Keep loading handshakes on one socket**

Add a 500 ms single-shot `handshakeRetryTimer_`, store the generation passed to `connectToService`, and factor hello sending into `sendHandshake()`. For a valid loading response:

```cpp
requestTimer_->stop();
handshakeRequestId_.clear();
setState(State::Handshaking, message);
emit handshakeLoading(connectionGeneration_, response);
handshakeRetryTimer_->start();
return;
```

Do not abort, set Error, or emit transport failure. Retry hello only while connected/Handshaking. Ready emits its generation and metadata object; failed hello remains terminal. All timers are stopped and their generation invalidated on disconnect.

- [ ] **Step 5: Select program/arguments and validate identity**

Add:

```cpp
QString backendProgram() const;
QStringList backendArguments() const;
bool identityMatches(const QJsonObject &metadata, QString *reason) const;
```

Development program is Python and arguments start with the script. Packaged program is `backendExecutable` and arguments contain no script. Packaged arguments are exactly:

```text
--host --port --project-root --model-dir --paddle-config --data-root
--compute-device --model-sha256 --edition --package-version --instance-token --parent-pid
--local-search-mode --inference-mode
```

Use `QCoreApplication::applicationPid()` for parent PID and `QUuid::createUuid().toString(QUuid::WithoutBraces)` once per owned launch. Require exact version, edition, device, and model fingerprint before Ready. In packaged mode, any service found before this launch is an instance conflict even when its other identity fields match; development mode retains compatible external-service reuse. After launch, require the returned token to equal the owned token. Mismatch disconnects and reports incompatibility but never terminates an external service.

On normal Qt exit, send `shutdown` only when `ownedByThisSession()` and the ready hello token equals `launchInstanceToken_`; include that token in the request. If graceful shutdown times out, terminate/kill only the same owned `QProcess` handle. Never send shutdown or terminate/kill for a reused development service or a token mismatch.

- [ ] **Step 6: Make the manager the single UI startup authority**

Increment `startupGeneration_` for every start, restart, stop, and terminal failure; pass it into each client connection and ignore every Loading/Ready/Failure signal whose generation differs. Emit `starting_process` at 0 before an owned launch and `loading_runtime` after the socket handshake. When a manager exists, remove the direct client-ready → MainWindow-ready connection. Map phases:

```cpp
starting_process  -> QStringLiteral("正在启动后端")
loading_runtime   -> QStringLiteral("正在加载运行环境")
preparing_data    -> QStringLiteral("正在检查数据目录")
loading_model     -> QStringLiteral("正在加载 PP-ShiTu 模型")
restoring_library -> QStringLiteral("正在恢复工件库和缓存")
```

Remain in `BackendUiState::Loading` through all phases and show monotonic backend progress; clamp only for display and reject a protocol value outside 0–100. Only manager ready enters Ready at 100 and terminal failure enters Error. Preserve Recovering after a service that was previously ready disconnects.

For missing backend EXE/dependency, occupied port, identity conflict, and startup timeout, the manager supplies a stable code, Chinese next action, and the configured `data/logs` path. For service-reported failures, preserve its code/action/log path verbatim. Add focused assertions that the detail panel exposes these fields while the top-right status remains a single stable phase label.

- [ ] **Step 7: Prevent hidden output from blocking**

Before QProcess start, route stdout/stderr to `QProcess::nullDevice()`. Keep asynchronous `errorOccurred` and `finished` signals. PyInstaller also uses `console=False` in Task 6.

- [ ] **Step 8: Run affected Qt suites**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_appconfig,test_backendclient,test_backendprocessmanager,test_appfoundation,test_mainwindow
```

Expected: all pass; one connection handles loading; foreign service is not killed.

- [ ] **Step 9: Commit**

```powershell
git add qt_app/backendclient.h qt_app/backendclient.cpp qt_app/backendprocessmanager.h qt_app/backendprocessmanager.cpp qt_app/processlauncher.cpp qt_app/mainwindow.h qt_app/mainwindow.cpp qt_app/tests/test_backendclient.cpp qt_app/tests/test_backendprocessmanager.cpp qt_app/tests/test_mainwindow.cpp
git commit -m "feat: manage frozen backend lifecycle"
```

---

### Task 6: Freeze Minimal GPU and CPU Backend Editions

**Files:**
- Create: `release_tools/__init__.py`
- Create: `release_tools/backend_bundle.py`
- Create: `tests/test_backend_bundle.py`
- Create: `deploy/orientation_backend.spec`
- Create: `deploy/requirements/common.txt`
- Create: `deploy/requirements/gpu.txt`
- Create: `deploy/requirements/cpu.txt`
- Create: `scripts/create_packaging_envs.ps1`
- Create: `scripts/build_portable_backend.ps1`

**Interfaces:**
- Produces `BundleEdition(name, paddle_distribution, paddle_version, compute_device, cuda_namespaces)`.
- Produces `edition_for`, `validate_installed_distributions`, and `pyinstaller_excludes`.
- Produces `release_staging/backend-{edition}/orientation_backend/`.
- Consumed by: Task 7 assembler.

- [ ] **Step 1: Write failing bundle metadata tests**

Create `tests/test_backend_bundle.py`:

```python
import pytest

from release_tools.backend_bundle import (
    BundleEnvironmentError, edition_for,
    pyinstaller_excludes, validate_installed_distributions,
)


def test_gpu_edition_is_cuda118_paddle_322():
    edition = edition_for("gpu")
    assert edition.paddle_distribution == "paddlepaddle-gpu"
    assert edition.paddle_version == "3.2.2"
    assert edition.compute_device == "gpu"
    assert "nvidia.cudnn" in edition.cuda_namespaces


def test_cpu_rejects_gpu_paddle_in_same_environment():
    with pytest.raises(BundleEnvironmentError, match="paddlepaddle-gpu"):
        validate_installed_distributions(
            edition_for("cpu"),
            {"paddlepaddle": "3.2.2", "paddlepaddle-gpu": "3.2.2",
             "paddleclas": "2.6.0", "pyinstaller": "6.22.2"},
        )


def test_fast_bundle_excludes_local_matcher_stack():
    excluded = set(pyinstaller_excludes())
    assert {"torch", "lightglue", "src.aliked_lightglue_matcher",
            "src.local_sift_matcher", "faiss", "visualdl"} <= excluded
```

- [ ] **Step 2: Run and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_backend_bundle.py -q -p no:cacheprovider
```

Expected: missing `release_tools.backend_bundle`.

- [ ] **Step 3: Implement edition metadata and environment validation**

```python
from collections.abc import Sequence


@dataclass(frozen=True)
class BundleEdition:
    name: str
    paddle_distribution: str
    paddle_version: str
    compute_device: str
    cuda_namespaces: Sequence[str]


EDITIONS = {
    "gpu": BundleEdition(
        "gpu", "paddlepaddle-gpu", "3.2.2", "gpu",
        ("nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cuda_runtime",
         "nvidia.cudnn", "nvidia.cufft", "nvidia.curand",
         "nvidia.cusolver", "nvidia.cusparse"),
    ),
    "cpu": BundleEdition("cpu", "paddlepaddle", "3.2.2", "cpu", ()),
}
```

Require 64-bit CPython 3.10, PyInstaller 6.22.2, PaddleClas 2.6.0, the exact edition Paddle distribution, and absence of the opposite Paddle package. Excludes include Torch/LightGlue/local matchers plus `faiss`, `sklearn`, `visualdl`, pytest, training, and benchmark modules unused by fast production requests.

- [ ] **Step 4: Pin direct build requirements**

`deploy/requirements/common.txt`:

```text
PyInstaller==6.22.2
paddleclas==2.6.0
numpy==1.24.4
opencv-python==4.6.0.66
easydict==1.13
gast==0.3.3
Pillow==12.1.0
prettytable==3.18.0
PyYAML==6.0.3
scipy==1.15.3
tqdm==4.70.0
ujson==5.13.0
```

Edition files:

```text
# gpu.txt
-r common.txt
paddlepaddle-gpu==3.2.2
```

```text
# cpu.txt
-r common.txt
paddlepaddle==3.2.2
```

`create_packaging_envs.ps1` creates `workpiece-package-gpu` and `workpiece-package-cpu` with Python 3.10.20, installs GPU Paddle from the official `cu118` index and CPU Paddle from the official `cpu` index, then validates no opposite Paddle package is present.

The script uses these exact Paddle commands after installing each edition's common requirements:

```powershell
conda run -n workpiece-package-gpu python -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
conda run -n workpiece-package-cpu python -m pip install paddlepaddle==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

- [ ] **Step 5: Write the shared PyInstaller onedir spec**

Read `WORKPIECE_PACKAGE_EDITION` and `WORKPIECE_PROJECT_ROOT`; validate through `backend_bundle`. Core spec:

```python
paddle_binaries = collect_dynamic_libs("paddle")
paddle_datas = collect_data_files("paddle", include_py_files=False)
cuda_binaries = []
for namespace in edition.cuda_namespaces:
    cuda_binaries += collect_dynamic_libs(namespace)

a = Analysis(
    [str(project_root / "src" / "orientation_tcp_service.py")],
    pathex=[str(project_root)],
    binaries=paddle_binaries + cuda_binaries,
    datas=paddle_datas,
    hiddenimports=[
        "paddle.base.core", "paddle.inference",
        "paddleclas.deploy.python.predict_rec",
        "paddleclas.deploy.utils.config",
        "paddleclas.deploy.utils.predictor",
    ],
    excludes=list(pyinstaller_excludes()),
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True,
          name="orientation_backend", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="orientation_backend")
```

Do not add source directories as data. Model and inference YAML remain external.

- [ ] **Step 6: Implement backend build scripts**

`build_portable_backend.ps1` accepts edition, Python, project root, and output root; validates distributions; clears only its own work/dist folders; sets spec environment variables; runs:

```powershell
& $Python -m PyInstaller --clean --noconfirm `
    --distpath $DistPath --workpath $WorkPath `
    (Join-Path $ProjectRoot 'deploy\orientation_backend.spec')
```

Fail unless the edition output contains `orientation_backend.exe`.

- [ ] **Step 7: Test helpers, create environments, and build both EXEs**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_backend_bundle.py -q -p no:cacheprovider
.\scripts\create_packaging_envs.ps1
.\scripts\build_portable_backend.ps1 -Edition gpu -Python E:\python\anaconda3\envs\workpiece-package-gpu\python.exe
.\scripts\build_portable_backend.ps1 -Edition cpu -Python E:\python\anaconda3\envs\workpiece-package-cpu\python.exe
```

Dependency installation requires user approval when executed. Expected: GPU output contains NVIDIA runtime DLLs; CPU output contains no NVIDIA directory/CUDA DLL.

- [ ] **Step 8: Smoke both frozen services**

Launch each EXE with temporary data, external model/YAML, the computed model SHA-256, correct edition/device, a generated instance token, `fast_geometry`, and version `1.0.0`; poll hello through every loading response to Ready, verify returned device/fingerprint/token, run one prediction self-check, and send shutdown. Expected identities match; no console opens.

- [ ] **Step 9: Commit**

```powershell
git add release_tools/__init__.py release_tools/backend_bundle.py tests/test_backend_bundle.py deploy/orientation_backend.spec deploy/requirements/common.txt deploy/requirements/gpu.txt deploy/requirements/cpu.txt scripts/create_packaging_envs.ps1 scripts/build_portable_backend.ps1
git commit -m "build: freeze GPU and CPU backend editions"
```

---

### Task 7: Assemble, Audit, and Zip Each Portable Edition

**Files:**
- Create: `release_tools/portable_package.py`
- Create: `tests/test_portable_package.py`
- Create: `deploy/使用说明.txt`
- Create: `deploy/THIRD_PARTY-NOTICES.txt`
- Create: `scripts/build_portable_release.ps1`
- Modify: `scripts/build_qt5.ps1`
- Modify: `qt_app/workpiece_orientation.pro`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `PackageLayout`, `stage_package`, injectable `audit_package(root, *, edition, version, forbidden_roots, dependency_checker=run_dumpbin)`, `write_manifest`, `collect_licenses`, `zip_package`, `write_sha256`.
- Produces relative packaged `app_config.json` with `launch_mode=packaged_executable` and fixed edition/device/fast mode.
- Produces `release_artifacts/WorkpieceOrientation-{GPU|CPU}-x64-1.0.0.zip` and sibling `.sha256`.
- Consumes: Task 6 frozen backend directories.

- [ ] **Step 1: Write failing staging/audit tests**

Create `tests/test_portable_package.py`:

```python
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from release_tools.portable_package import (
    PackageAuditError, audit_package, build_release_config,
    write_manifest, write_sha256, zip_package,
)
from src.model_fingerprint import model_directory_sha256


def minimal_stage(root: Path, edition: str = "gpu") -> Path:
    root.mkdir()
    (root / "backend").mkdir()
    (root / "backend" / "orientation_backend.exe").write_bytes(b"MZ-backend")
    if edition == "gpu":
        for name in ("paddle_inference.dll", "cudnn64_8.dll", "cublas64_11.dll", "cudart64_110.dll"):
            (root / "backend" / name).write_bytes(b"MZ-runtime")
    (root / "backend" / "resources").mkdir()
    (root / "backend" / "resources" / "inference_general.yaml").write_text("Global: {}\n")
    (root / "models" / "shitu_rec").mkdir(parents=True)
    (root / "models" / "shitu_rec" / "inference.pdmodel").write_bytes(b"model")
    (root / "models" / "shitu_rec" / "inference.pdiparams").write_bytes(b"params")
    (root / "models" / "shitu_rec" / "inference.pdiparams.info").write_bytes(b"info")
    (root / "data" / "workpieces").mkdir(parents=True)
    for name in ("rules", "cache", "logs", "temp"):
        (root / "data" / name).mkdir()
    (root / "data" / "data_layout.json").write_text(
        json.dumps({"layout_version": 1}), encoding="utf-8"
    )
    (root / "WorkpieceOrientation.exe").write_bytes(b"MZ-qt")
    (root / "Qt5Core.dll").write_bytes(b"MZ-qtcore")
    (root / "platforms").mkdir()
    (root / "platforms" / "qwindows.dll").write_bytes(b"MZ-platform")
    (root / "qt.conf").write_text("[Paths]\nPlugins=.\n", encoding="utf-8")
    (root / "third_party_licenses").mkdir()
    (root / "third_party_licenses" / "index.txt").write_text("licenses", encoding="utf-8")
    model_sha = model_directory_sha256(root / "models" / "shitu_rec")
    config = build_release_config(
        edition=edition, version="1.0.0", model_sha256=model_sha
    )
    (root / "app_config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "version.json").write_text(json.dumps({
        "version": "1.0.0", "edition": edition,
        "git_commit": "0" * 40, "build_utc": "2026-08-28T12:00:00Z",
        "python": "3.10.20", "paddle": "3.2.2", "paddleclas": "2.6.0",
        "pyinstaller": "6.22.2", "qt": "5.14.2",
        "model_sha256": model_sha,
    }), encoding="utf-8")
    (root / "THIRD_PARTY-NOTICES.txt").write_text("notices", encoding="utf-8")
    (root / "使用说明.txt").write_text("离线使用说明", encoding="utf-8")
    return root


def audit_synthetic(root: Path, edition: str = "gpu", forbidden_roots=()):
    return audit_package(
        root, edition=edition, version="1.0.0",
        forbidden_roots=list(forbidden_roots),
        dependency_checker=lambda executable, package_root: [],
    )


def test_generated_configs_are_relative_fast_and_device_specific():
    model_sha = "a" * 64
    gpu = build_release_config(edition="gpu", version="1.0.0", model_sha256=model_sha)
    cpu = build_release_config(edition="cpu", version="1.0.0", model_sha256=model_sha)
    assert gpu["backend_executable"] == "backend/orientation_backend.exe"
    assert gpu["compute_device"] == gpu["edition"] == "gpu"
    assert cpu["compute_device"] == cpu["edition"] == "cpu"
    assert gpu["inference_mode"] == cpu["inference_mode"] == "fast_geometry"
    assert gpu["model_sha256"] == cpu["model_sha256"] == model_sha
    assert gpu["startup_timeout_ms"] == 600000
    assert cpu["startup_timeout_ms"] == 600000
    assert all("E:/" not in str(value) for value in gpu.values())


@pytest.mark.parametrize("bad_name", ["leak.py", "stub.pyi"])
def test_audit_rejects_visible_python_source(tmp_path: Path, bad_name: str):
    root = minimal_stage(tmp_path / "package")
    (root / "backend" / bad_name).write_text("secret", encoding="utf-8")
    with pytest.raises(PackageAuditError, match="Python source"):
        audit_synthetic(root)


def test_audit_rejects_nonempty_workpiece_library(tmp_path: Path):
    root = minimal_stage(tmp_path / "package")
    (root / "data" / "workpieces" / "existing-piece").mkdir()
    with pytest.raises(PackageAuditError, match="workpieces"):
        audit_synthetic(root)


def test_audit_rejects_development_absolute_path(tmp_path: Path):
    root = minimal_stage(tmp_path / "package")
    (root / "bad.json").write_text(r'{"path":"E:\\Project\\wang\\pp_813"}')
    with pytest.raises(PackageAuditError, match="absolute path"):
        audit_synthetic(root, forbidden_roots=[Path(r"E:\Project\wang\pp_813")])


def test_cpu_audit_rejects_cuda_runtime(tmp_path: Path):
    root = minimal_stage(tmp_path / "package", edition="cpu")
    (root / "backend" / "cudnn64_8.dll").write_bytes(b"MZ")
    with pytest.raises(PackageAuditError, match="CUDA"):
        audit_synthetic(root, edition="cpu")


def test_manifest_and_zip_use_one_versioned_root(tmp_path: Path):
    root = minimal_stage(tmp_path / "WorkpieceOrientation-GPU")
    manifest = write_manifest(root, edition="gpu", version="1.0.0")
    assert manifest["version"] == "1.0.0"
    assert all("sha256" in item for item in manifest["files"])
    archive = zip_package(root, tmp_path / "WorkpieceOrientation-GPU-x64-1.0.0.zip")
    assert archive.is_file()
    with ZipFile(archive) as zipped:
        assert {Path(name).parts[0] for name in zipped.namelist()} == {
            "WorkpieceOrientation-GPU"
        }
    checksum = write_sha256(archive)
    digest, filename = checksum.read_text(encoding="ascii").strip().split("  ", 1)
    assert len(digest) == 64
    assert filename == archive.name
```

- [ ] **Step 2: Run and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_portable_package.py -q -p no:cacheprovider
```

Expected: missing portable-package module.

- [ ] **Step 3: Implement config generation and clean staging**

`build_release_config` returns only relative paths:

```python
def build_release_config(
    *, edition: str, version: str, model_sha256: str
) -> dict[str, object]:
    return {
        "launch_mode": "packaged_executable",
        "backend_executable": "backend/orientation_backend.exe",
        "project_root": ".",
        "paddle_config": "backend/resources/inference_general.yaml",
        "model_dir": "models/shitu_rec",
        "data_root": "data",
        "model_sha256": model_sha256,
        "compute_device": edition,
        "edition": edition,
        "package_version": version,
        "local_search_mode": "adaptive",
        "inference_mode": "fast_geometry",
        "host": "127.0.0.1",
        "port": 37651,
        "startup_timeout_ms": 600000,
        "request_timeout_ms": 120000,
    }
```

`stage_package` always creates a new edition-specific staging directory; copy the Qt EXE/runtime, frozen backend, exactly `inference.pdmodel`, `inference.pdiparams`, and `inference.pdiparams.info`, Paddle inference YAML, guide, and notices. Generate `qt.conf` with package-local plugin paths and a `third_party_licenses/index.txt`. Compute `model_directory_sha256` only after copying, write that exact value into both release config and `version.json`, and create empty data subdirectories plus a v1 `data_layout.json`. Never copy the repository `runtime_library`, `data`, or `runtime_reports`.

- [ ] **Step 4: Implement exhaustive package audit**

Audit must:

1. Verify required EXE/config/model/data/license files.
2. Validate edition/device/version/fast configuration, recompute and compare the model fingerprint, and reject absolute config paths.
3. Require `data/workpieces` empty and reject manifest/rule/cache files anywhere under shipped data.
4. Reject `.py`, `.pyi`, `.pdb`, `.obj`, test/report/manual fixture paths, and unresolved template tokens.
5. Search every file's bytes for normalized UTF-8/UTF-16 representations of project root, Python prefix, Qt root, and username profile paths.
6. For CPU, reject filenames containing CUDA/cuDNN/cuBLAS/NVIDIA runtime tokens. For GPU, require Paddle GPU and required NVIDIA DLL families in backend.
7. Invoke `dumpbin /HEADERS` and `/DEPENDENTS` for both EXEs, require x64 machine type, and fail if a non-system dependency is absent from the package.

`manifest.json` lists every file except itself using POSIX relative path, byte size, and SHA-256. `version.json` records version, edition, Git commit, build UTC time, Python, Paddle, PaddleClas, PyInstaller, Qt, and model SHA-256.

- [ ] **Step 5: Collect third-party notices without shipping package source**

`collect_licenses` uses `importlib.metadata.distribution(name)` to copy `LICENSE*`/`COPYING*` files for the distributions present in each build environment, copies `third_party/PaddleClas/LICENSE`, and writes a component/version/license-index file. `deploy/THIRD_PARTY-NOTICES.txt` names Qt, Python, PyInstaller, Paddle, PaddleClas, NumPy, OpenCV, and edition-specific NVIDIA redistributables. Missing required license text is a build failure.

- [ ] **Step 6: Make the Qt build staging-safe**

Change `build_qt5.ps1` to accept `-BuildDir` and `-SkipRuntimeConfig`. It must never copy `qt_app/app_config.json` when packaging. Set in `workpiece_orientation.pro`:

```qmake
TARGET = WorkpieceOrientation
VERSION = 1.0.0
QMAKE_TARGET_PRODUCT = Workpiece Orientation
QMAKE_TARGET_DESCRIPTION = Workpiece front/back inspection
```

Preserve current Release flags and resources.

- [ ] **Step 7: Implement the release orchestration script**

`build_portable_release.ps1` accepts `-Edition gpu|cpu|all`, `-Version 1.0.0`, both Python paths, Qt paths, model path, and output root. For each edition it:

1. calls Task 6 backend build;
2. builds Qt once in a clean package build directory;
3. runs `windeployqt --release --compiler-runtime --no-translations` on `WorkpieceOrientation.exe`;
4. calls `release_tools.portable_package stage`;
5. runs audit before and after manifest generation;
6. runs Task 8 smoke;
7. creates the versioned ZIP and `.sha256`.

No command may enumerate arbitrary paths and pass them to another shell for deletion; clear only resolved `release_staging/gpu-1.0.0` or `release_staging/cpu-1.0.0` and verify the selected path remains under the repository release-staging root.

- [ ] **Step 8: Add the Chinese offline guide and ignore outputs**

The guide must include edition choice, unzip/start, GPU driver requirement, first library creation, data backup/copy rules, update procedure, log location, common startup errors, and uninstall-by-folder-delete. Explicitly state that non-empty data roots cannot be merged and the program must be closed during copy.

Add to `.gitignore`:

```gitignore
release_artifacts/
release_staging/
deploy/pyinstaller-work/
```

- [ ] **Step 9: Run release-tool tests and build-time audit on synthetic packages**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_backend_bundle.py tests/test_portable_package.py -q -p no:cacheprovider
```

Expected: all pass; every injected leak is rejected.

- [ ] **Step 10: Commit**

```powershell
git add release_tools/portable_package.py tests/test_portable_package.py deploy/使用说明.txt deploy/THIRD_PARTY-NOTICES.txt scripts/build_portable_release.ps1 scripts/build_qt5.ps1 qt_app/workpiece_orientation.pro .gitignore
git commit -m "build: assemble audited portable packages"
```

---

### Task 8: Add Packaged Startup, Functional Smoke, and Data-Portability Checks

**Files:**
- Create: `qt_app/startupsmokecontroller.h`
- Create: `qt_app/startupsmokecontroller.cpp`
- Create: `qt_app/tests/test_startupsmokecontroller.cpp`
- Create: `qt_app/tests/test_startupsmokecontroller.pro`
- Modify: `qt_app/main.cpp`
- Modify: `qt_app/workpiece_orientation.pro`
- Modify: `scripts/run_qt5_tests.ps1`
- Create: `release_tools/portable_smoke.py`
- Create: `scripts/smoke_portable_package.py`
- Create: `tests/test_smoke_portable_package.py`

**Interfaces:**
- Produces hidden Qt flag `--package-smoke-test`; exit 0 on backend Ready, 2 on terminal failure, 3 on timeout.
- Produces injectable `release_tools.portable_smoke.run_smoke(options: SmokeOptions, *, process_factory, socket_factory, temp_root_factory) -> SmokeReport` and a thin CLI `smoke_portable_package.py --package-root --dataset-root --front-template-count --back-template-count --seed` using the repository's deterministic split helper.
- Produces a copied-data interoperability check between built GPU and CPU package roots.
- Consumed by: Task 7 release script and Task 9 clean-machine acceptance.

- [ ] **Step 1: Write failing Qt smoke-controller tests**

Create a controller test with a real `BackendProcessManager` signal source and this local launcher/config fixture (all virtual methods mirror the existing manager-test fake):

```cpp
class NoopProcessLauncher : public ProcessLauncher {
    Q_OBJECT
public:
    using ProcessLauncher::ProcessLauncher;
    bool start(const QString &, const QStringList &, const QString &) override { return true; }
    void terminate() override {}
    void kill() override {}
    bool isRunning() const override { return false; }
};

static AppConfig smokeConfig() {
    AppConfig config;
    config.pythonExecutable = QStringLiteral("python.exe");
    config.backendScript = QStringLiteral("service.py");
    config.projectRoot = QStringLiteral(".");
    config.modelDir = QStringLiteral("models");
    config.libraryDir = QStringLiteral("runtime_library");
    config.host = QHostAddress::LocalHost;
    config.port = 37651;
    config.startupTimeoutMs = 1000;
    return config;
}
```

Then add:

```cpp
void readyFinishesWithZeroOnce() {
    BackendClient client;
    NoopProcessLauncher launcher;
    BackendProcessManager manager(smokeConfig(), &client, &launcher);
    StartupSmokeController controller(&manager, 1000);
    QSignalSpy finishedSpy(&controller, &StartupSmokeController::finished);
    emit manager.backendReady();
    emit manager.backendReady();
    QCOMPARE(finishedSpy.count(), 1);
    QCOMPARE(finishedSpy.at(0).at(0).toInt(), 0);
}


void failureAndTimeoutUseDistinctExitCodes() {
    BackendClient client;
    NoopProcessLauncher launcher;
    BackendProcessManager manager(smokeConfig(), &client, &launcher);
    StartupSmokeController failed(&manager, 1000);
    QSignalSpy failedSpy(&failed, &StartupSmokeController::finished);
    emit manager.backendUnavailable(QStringLiteral("模型缺失"));
    QCOMPARE(failedSpy.at(0).at(0).toInt(), 2);

    StartupSmokeController timedOut(&manager, 1);
    QSignalSpy timeoutSpy(&timedOut, &StartupSmokeController::finished);
    QTRY_COMPARE_WITH_TIMEOUT(timeoutSpy.count(), 1, 100);
    QCOMPARE(timeoutSpy.at(0).at(0).toInt(), 3);
}
```

- [ ] **Step 2: Write failing pure-Python smoke-client tests**

Create `tests/test_smoke_portable_package.py`; import `run_smoke` from `release_tools.portable_smoke` and inject a fake process factory, free-port provider, socket client, and temporary-root factory. Assert the client performs, in order:

```python
assert commands == [
    "hello", "list_workpieces", "register", "list_workpieces",
    "predict", "predict", "get_geometry_mask_profile",
    "recycle_workpiece", "list_recycled_workpieces", "restore_workpiece",
    "shutdown",
]
```

Assert the first list is empty, registered front/back template counts equal every supplied image, both expected labels are returned, restored workpiece is listed, and process exits within the configured smoke budget (including any asynchronous cache shutdown).

- [ ] **Step 3: Run focused tests and verify RED**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_startupsmokecontroller
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_smoke_portable_package.py -q -p no:cacheprovider
```

Expected: missing controller/project/script failures.

- [ ] **Step 4: Implement the hidden Qt startup probe**

`StartupSmokeController` connects once to manager Ready/Unavailable and owns a single-shot timeout timer. It emits exactly one `finished(int)` and disconnects/halts the timer afterward.

In `main.cpp`, load config and construct client/manager exactly as normal. If arguments contain `--package-smoke-test`, do not show MainWindow; construct the controller using `startupTimeoutMs + 5000`, call `manager.start()`, and exit the application with the controller code. Normal user startup is unchanged.

- [ ] **Step 5: Implement the packaged backend protocol smoke**

`release_tools.portable_smoke` owns the testable implementation and `scripts/smoke_portable_package.py` only parses arguments and calls its `main`. The implementation must:

1. validate `app_config.json` and resolve only package-relative paths;
2. copy the package into a fresh temporary directory whose path includes Chinese, spaces, and an absolute length of at least 180 characters so shipped `data` remains untouched;
3. start `backend/orientation_backend.exe` hidden with the same arguments as Qt and a free loopback port;
4. poll structured hello through loading phases to Ready within edition timeout;
5. prove initial list is empty;
6. register every provided front/back image without truncation;
7. predict one known front and back image and require expected labels;
8. request geometry profile to prove geometry modules are frozen in;
9. recycle, list recycle bin, restore, and list the workpiece;
10. request shutdown with the generated matching instance token and assert process exit.

Use only external fixture paths; never copy fixtures into the original package or final ZIP. Capture backend log and command results into the caller-provided report path.

- [ ] **Step 6: Implement explicit GPU-to-CPU data portability smoke**

Add `--source-package-root` plus `--destination-package-root` portability mode: after successful registration in a temporary source package copy, stop it; copy its complete `data` to a fresh destination package copy whose data is empty; start the destination edition, require the same workpiece ID/template counts, and predict the same two queries. Run once GPU-to-CPU and once CPU-to-GPU. If destination is non-empty, fail before copying.

- [ ] **Step 7: Run Qt and script tests**

```powershell
.\scripts\run_qt5_tests.ps1 -Targets test_startupsmokecontroller,test_appconfig,test_backendclient,test_backendprocessmanager
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_smoke_portable_package.py tests/test_portable_package.py -q -p no:cacheprovider
```

Expected: all pass, with no GUI shown in smoke mode.

- [ ] **Step 8: Run real smoke against both staged packages**

```powershell
E:\python\anaconda3\envs\shitu\python.exe .\scripts\smoke_portable_package.py `
  --package-root .\release_staging\gpu-1.0.0 `
  --dataset-root .\data\1_M1 --front-template-count 5 --back-template-count 10 --seed 20260813 `
  --report .\release_staging\reports\gpu-smoke.json
E:\python\anaconda3\envs\shitu\python.exe .\scripts\smoke_portable_package.py `
  --package-root .\release_staging\cpu-1.0.0 `
  --dataset-root .\data\1_M1 --front-template-count 5 --back-template-count 10 --seed 20260813 `
  --report .\release_staging\reports\cpu-smoke.json
E:\python\anaconda3\envs\shitu\python.exe .\scripts\smoke_portable_package.py `
  --source-package-root .\release_staging\gpu-1.0.0 `
  --destination-package-root .\release_staging\cpu-1.0.0 `
  --dataset-root .\data\1_M1 --front-template-count 5 --back-template-count 10 --seed 20260813 `
  --report .\release_staging\reports\gpu-to-cpu.json
E:\python\anaconda3\envs\shitu\python.exe .\scripts\smoke_portable_package.py `
  --source-package-root .\release_staging\cpu-1.0.0 `
  --destination-package-root .\release_staging\gpu-1.0.0 `
  --dataset-root .\data\1_M1 --front-template-count 5 --back-template-count 10 --seed 20260813 `
  --report .\release_staging\reports\cpu-to-gpu.json
```

The script selects held-out queries with the same deterministic split helper used by `test_orientation_service_integration.py` and rejects a query whose content hash equals a selected template. Expected: all four commands succeed; original staged package data remains empty; no backend remains running.

- [ ] **Step 9: Commit**

```powershell
git add qt_app/startupsmokecontroller.h qt_app/startupsmokecontroller.cpp qt_app/tests/test_startupsmokecontroller.cpp qt_app/tests/test_startupsmokecontroller.pro qt_app/main.cpp qt_app/workpiece_orientation.pro scripts/run_qt5_tests.ps1 release_tools/portable_smoke.py scripts/smoke_portable_package.py tests/test_smoke_portable_package.py
git commit -m "test: add portable package smoke coverage"
```

---

### Task 9: Run Full Offline Acceptance and Produce Release Evidence

**Files:**
- Create: `release_tools/portable_benchmark.py`
- Create: `scripts/benchmark_portable_service.py`
- Create: `tests/test_benchmark_portable_service.py`
- Create after real runs: `docs/verification/offline-portable-gpu-cpu-results.json`
- Create after real runs: `docs/verification/offline-portable-gpu-cpu-results.md`
- Modify if evidence reveals only packaging defects: files owned by Tasks 1–8 plus their focused tests.

**Interfaces:**
- Produces a benchmark JSON containing edition, hardware, versions, startup phases/times, per-request labels/timings, accuracy, differences, and mean/P50/P95/P99/max.
- Produces final two ZIPs and SHA-256 files under `release_artifacts/`.
- Consumes the current fixed M1/M2/M7 acceptance set externally; no acceptance image enters a ZIP.

- [ ] **Step 1: Write failing benchmark-statistics and comparison tests**

Create `tests/test_benchmark_portable_service.py` and import the pure functions from `release_tools.portable_benchmark`:

```python
from release_tools.portable_benchmark import summarize, compare_predictions


def test_summarize_reports_required_percentiles():
    summary = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert set(summary) == {"count", "mean", "p50", "p95", "p99", "max"}
    assert summary["count"] == 5
    assert summary["mean"] == 30.0
    assert summary["max"] == 50.0


def test_compare_predictions_lists_every_cpu_gpu_difference():
    gpu = {"a.png": "front", "b.png": "back"}
    cpu = {"a.png": "back", "b.png": "back"}
    assert compare_predictions(gpu, cpu) == [
        {"image": "a.png", "gpu": "front", "cpu": "back"}
    ]
```

- [ ] **Step 2: Run and verify RED**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_benchmark_portable_service.py -q -p no:cacheprovider
```

Expected: missing benchmark module/functions.

- [ ] **Step 3: Implement production-boundary benchmark collection**

`release_tools.portable_benchmark` owns the testable implementation; `scripts/benchmark_portable_service.py` is a thin argument parser. It starts the frozen backend from a package copy, waits for Ready, copies the externally prepared acceptance `data` root named by the acceptance spec only into that temporary package, warms with 50 requests, then executes at least 1000 TCP `predict` requests. It refuses a non-empty temporary destination and never modifies the ZIP, staging tree, external source data, or acceptance spec. Use NumPy percentile method `linear` and round only when rendering Markdown:

```python
def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(array.max()),
    }
```

Record both client round-trip and backend `elapsed_ms/timings_ms.total`. GPU gate applies to the existing full-backend timing field and the same corpus/definition used by `benchmark_fast_geometry_inference.py`; round-trip is reported separately. Capture `platform`, CPU, memory, GPU name, driver, Paddle actual device, package manifest hashes, and startup phase durations.

- [ ] **Step 4: Re-run and commit the benchmark tooling**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest tests/test_benchmark_portable_service.py -q -p no:cacheprovider
git add release_tools/portable_benchmark.py scripts/benchmark_portable_service.py tests/test_benchmark_portable_service.py
git commit -m "test: benchmark portable backend releases"
```

Expected: statistics and full CPU/GPU difference reporting pass before any expensive package run.

- [ ] **Step 5: Run every source-level test before packaging**

```powershell
E:\python\anaconda3\envs\shitu\python.exe -m pytest -q -p no:cacheprovider
.\scripts\run_qt5_tests.ps1
.\scripts\build_qt5.ps1 -SkipRuntimeConfig
```

Expected: all existing and new Python/Qt tests pass; Release Qt build succeeds.

- [ ] **Step 6: Build and re-audit both final ZIPs**

```powershell
.\scripts\build_portable_release.ps1 -Edition all -Version 1.0.0 `
  -GpuPython E:\python\anaconda3\envs\workpiece-package-gpu\python.exe `
  -CpuPython E:\python\anaconda3\envs\workpiece-package-cpu\python.exe
```

Expected artifacts:

```text
release_artifacts/WorkpieceOrientation-GPU-x64-1.0.0.zip
release_artifacts/WorkpieceOrientation-GPU-x64-1.0.0.zip.sha256
release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip
release_artifacts/WorkpieceOrientation-CPU-x64-1.0.0.zip.sha256
```

Extract each ZIP to a new directory and rerun `audit_package` on extracted contents so ZIP layout—not only staging—is verified.

- [ ] **Step 7: Run clean Windows offline startup and functional acceptance**

Use Windows Sandbox or a clean Windows 10/11 x64 VM with network disabled and no Python/Conda/Qt/CUDA Toolkit on PATH. GPU image may have only its compatible NVIDIA driver. For each ZIP:

1. verify SHA-256;
2. extract to a Chinese path containing spaces whose absolute path is at least 180 characters;
3. run `WorkpieceOrientation.exe --package-smoke-test` and require exit 0;
4. open the normal Qt app and verify no console appears;
5. create a workpiece, run single/batch detection, confirm asynchronous entry, delete/restore, create/publish/rollback a geometry rule, exit, and reopen;
6. confirm data persists and exactly one backend exists while Qt runs;
7. close Qt and confirm no backend remains;
8. disconnect network for the complete run.

Record Windows build, hardware, driver, start phase durations, screenshots of Ready and the empty first-run library, and smoke JSON. A machine that already has required developer runtimes does not count as clean acceptance.

- [ ] **Step 8: Run GPU accuracy and 1000-request latency gate**

```powershell
E:\python\anaconda3\envs\shitu\python.exe .\scripts\benchmark_portable_service.py `
  --package-zip .\release_artifacts\WorkpieceOrientation-GPU-x64-1.0.0.zip `
  --acceptance-spec .\runtime_reports\fast-geometry-acceptance.json `
  --warmup 50 --iterations 1000 `
  --output .\release_staging\reports\gpu-portable-benchmark.json
```

Expected: no error or accuracy regression relative to the current fast baseline; review-rate gate remains satisfied; backend full-detection P95 is at most 25 ms. If packaging alone causes regression, fix the responsible Task 1–8 unit with a failing regression test, rebuild, and rerun the full gate; do not loosen the metric or remove slow samples.

- [ ] **Step 9: Run CPU accuracy/performance and cross-edition comparison**

```powershell
E:\python\anaconda3\envs\shitu\python.exe .\scripts\benchmark_portable_service.py `
  --package-zip .\release_artifacts\WorkpieceOrientation-CPU-x64-1.0.0.zip `
  --acceptance-spec .\runtime_reports\fast-geometry-acceptance.json `
  --warmup 50 --iterations 1000 `
  --compare .\release_staging\reports\gpu-portable-benchmark.json `
  --output .\release_staging\reports\cpu-portable-benchmark.json
```

Expected: CPU functional run completes with MKLDNN active. Report all label/review differences; do not impose 25 ms.

- [ ] **Step 10: Write machine-readable and human-readable verification artifacts**

Construct the JSON from actual command results using these variables; the serializer writes their concrete values:

```python
result = {
    "version": "1.0.0",
    "git_commit": git_commit,
    "packages": {
        "gpu": {"zip_sha256": gpu_sha256, "content_audit": "passed"},
        "cpu": {"zip_sha256": cpu_sha256, "content_audit": "passed"},
    },
    "tests": test_results,
    "gpu": gpu_results,
    "cpu": cpu_results,
    "cpu_gpu_differences": compare_predictions(gpu_predictions, cpu_predictions),
}
```

Validate `git_commit` as 40 lowercase hexadecimal characters and both SHA values as 64 lowercase hexadecimal characters. Markdown links to the design/plan, names exact hardware/software, lists pass/fail gates, package sizes and hashes, confirms empty shipped data/no source/no absolute paths, and documents any CPU/GPU differences.

- [ ] **Step 11: Run final review commands and commit evidence**

```powershell
git diff --check
git status --short --branch
git add docs/verification/offline-portable-gpu-cpu-results.json docs/verification/offline-portable-gpu-cpu-results.md
git commit -m "docs: verify offline portable packages"
```

Expected: only intended tracked changes are committed; `runtime_reports/`, `release_staging/`, and `release_artifacts/` remain untracked/ignored. Deliver the two ZIP paths, sizes, hashes, GPU gate, CPU timings, clean-environment results, and any remaining hardware compatibility limitation to the user.
