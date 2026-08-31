import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZipFile, ZipInfo

import cv2
import numpy as np
import pytest

import release_tools.portable_benchmark as benchmark

from release_tools.portable_benchmark import (
    _latency_gate,
    _validate_config,
    _acceptance_rows,
    compare_predictions,
    run_benchmark,
    summarize,
)


def _batch_benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_batch_inference.py"
    spec = importlib.util.spec_from_file_location("batch_benchmark_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_benchmark_percentiles_are_linear_and_stable_for_small_samples():
    module = _batch_benchmark_module()

    assert module.percentiles_ms([10.0]) == {"p50": 10.0, "p95": 10.0, "p99": 10.0, "max": 10.0}
    assert module.percentiles_ms([10.0, 20.0, 30.0]) == {
        "p50": 20.0, "p95": 29.0, "p99": 29.8, "max": 30.0,
    }


def test_batch_benchmark_builds_ordered_five_path_payloads_with_unique_request_ids(tmp_path):
    module = _batch_benchmark_module()
    images = [tmp_path / f"{index}.png" for index in range(5)]

    first = module.batch_request("workpiece-7", images, request_id="first")
    second = module.batch_request("workpiece-7", images, request_id="second")

    assert first == {
        "version": 1, "request_id": "first", "command": "predict_batch",
        "workpiece_id": "workpiece-7", "image_paths": [str(path) for path in images],
    }
    assert second["request_id"] != first["request_id"]
    with pytest.raises(ValueError, match="exactly 5"):
        module.batch_request("workpiece-7", images[:4], request_id="short")


def test_batch_benchmark_excludes_warmups_and_reports_required_schema():
    module = _batch_benchmark_module()

    report = module.build_report(
        samples_ms=[20.0, 30.0, 40.0], warmup=2, iterations=3,
        workers=4, threads_per_worker=1, hardware={"cpu": "test"},
        accuracy={"checked": 3, "passed": 3}, fallback={"used": False},
    )

    assert set(report) == {
        "hardware", "configuration", "warmup", "iterations", "batch_size", "timings_ms",
        "throughput_images_per_second", "accuracy", "fallback",
    }
    assert report["configuration"] == {"workers": 4, "threads_per_worker": 1}
    assert report["warmup"] == 2
    assert report["timings_ms"]["samples"] == [20.0, 30.0, 40.0]
    assert report["timings_ms"]["p50"] == 30.0
    assert module.build_report([10.0], 0, 1, 2, 2, {}, {}, {})["configuration"] == {
        "workers": 2, "threads_per_worker": 2,
    }


def test_batch_benchmark_validates_protocol_order_and_item_errors():
    module = _batch_benchmark_module()
    paths = [f"{index}.png" for index in range(5)]
    valid = {"ok": True, "request_id": "batch-1", "command": "predict_batch", "items": [
        {"index": index, "image_path": path, "ok": True, "prediction": {"label": "front"}}
        for index, path in enumerate(paths)
    ]}

    assert module.validate_batch_response(valid, "batch-1", paths) == ["front"] * 5
    invalid = {**valid, "items": list(reversed(valid["items"]))}
    with pytest.raises(RuntimeError, match="item order"):
        module.validate_batch_response(invalid, "batch-1", paths)
    item_error = {**valid, "items": [{**valid["items"][0], "ok": False, "error": {"message": "bad image"}}, *valid["items"][1:]]}
    with pytest.raises(RuntimeError, match="bad image"):
        module.validate_batch_response(item_error, "batch-1", paths)


def test_batch_benchmark_closes_persistent_socket_and_backend_on_failure(tmp_path):
    module = _batch_benchmark_module()
    images = [tmp_path / f"{index}.png" for index in range(5)]
    closed = []

    class Client:
        def __init__(self): self.calls = []
        def request(self, payload):
            self.calls.append(payload)
            if payload["command"] == "hello":
                return {"ok": True, "request_id": payload["request_id"], "command": "hello", "ready": True,
                        "capabilities": {"predict_batch": True, "batch_ready": True}}
            return {"ok": False, "request_id": payload["request_id"], "command": "predict_batch", "error": {"message": "bad batch"}}
        def close(self): closed.append("socket")

    class Process:
        def poll(self): return None
        def terminate(self): closed.append("terminate")
        def wait(self, timeout=None): closed.append("wait")

    with pytest.raises(RuntimeError, match="bad batch"):
        module.run_batch_measurements(Client(), Process(), "wp", images, warmup=0, iterations=1)
    assert closed == ["socket", "terminate", "wait"]


def test_batch_benchmark_reuses_one_connection_and_discards_warmup_samples(tmp_path):
    module = _batch_benchmark_module()
    images = [tmp_path / f"{index}.png" for index in range(5)]

    class Client:
        def __init__(self): self.requests = []
        def request(self, payload):
            self.requests.append(payload)
            if payload["command"] == "hello":
                return {"ok": True, "request_id": payload["request_id"], "command": "hello", "ready": True,
                        "capabilities": {"predict_batch": True, "batch_ready": True}}
            return {"ok": True, "request_id": payload["request_id"], "command": "predict_batch", "items": [
                {"index": index, "image_path": str(path), "ok": True, "prediction": {"label": "front"}}
                for index, path in enumerate(images)
            ]}

    class Process:
        def poll(self): return 0

    client = Client()
    samples, labels, _backend = module.run_batch_measurements(
        client, Process(), "wp", images, warmup=2, iterations=3,
    )
    assert len(samples) == 3
    assert labels == ["front"] * 5
    assert [request["command"] for request in client.requests] == [
        "hello", "predict_batch", "predict_batch", "predict_batch", "predict_batch", "predict_batch",
    ]
    assert len({request["request_id"] for request in client.requests}) == 6


def _write_image(path: Path, value: int) -> tuple[int, int, str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((8, 9, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    data = encoded.tobytes()
    path.write_bytes(data)
    return image.shape[1], image.shape[0], hashlib.sha256(data).hexdigest(), len(data)


def _acceptance_fixture(root: Path) -> Path:
    """Create a small schema-faithful M1/M2/M7 acceptance corpus."""

    inventory = {}
    selections = {}
    for case_index, case in enumerate(("M1", "M2", "M7"), start=1):
        dataset_root = root / "data" / f"1_{case}"
        details = {"dataset_path": str(dataset_root), "dataset_counts": {"front": 1, "back": 1}}
        fp_case = {}
        for direction, orientation_dir in (("front", "0"), ("back", "1")):
            inventory_rows = []
            fp_rows = []
            for role, name, value in (("template", "template.png", case_index + (0 if direction == "front" else 10)), ("query", "query.png", case_index + 20 + (0 if direction == "front" else 10))):
                source = dataset_root / orientation_dir / name
                width, height, digest, size = _write_image(source, value)
                inventory_rows.append({
                    "case": case, "dataset": f"1_{case}", "role": role,
                    "expected_orientation": direction, "image_path": str(source),
                    "sha256": digest, "width": width, "height": height,
                    **({"query_identity": f"{case}-{direction}"} if role == "query" else {}),
                })
                fp_rows.append({
                    "path": f"data/1_{case}/{orientation_dir}/{name}",
                    "size": size, "sha256": digest,
                })
            details[direction] = {
                "templates": [inventory_rows[0]], "queries": [inventory_rows[1]]
            }
            fp_case[direction] = {"templates": [fp_rows[0]], "queries": [fp_rows[1]]}
        fp_case["template_query_overlap_count"] = 0
        fp_case["aggregate_sha256"] = benchmark._canonical_sha256(fp_case)
        inventory[case] = details
        selections[case] = fp_case
    fingerprints = {"selections": selections}
    fingerprints["overall_sha256"] = benchmark._canonical_sha256(fingerprints)
    spec = {"schema_version": 1, "passed": True, "selection_inventory": inventory,
            "fingerprints": {"fast_geometry": fingerprints}}
    path = root / "acceptance.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _recompute_acceptance_fingerprints(payload: dict) -> None:
    selections = payload["fingerprints"]["fast_geometry"]["selections"]
    for case, body in selections.items():
        body["aggregate_sha256"] = benchmark._canonical_sha256(
            {key: value for key, value in body.items() if key != "aggregate_sha256"}
        )
    fast = payload["fingerprints"]["fast_geometry"]
    fast["overall_sha256"] = benchmark._canonical_sha256(
        {key: value for key, value in fast.items() if key != "overall_sha256"}
    )


def _write_zip(path: Path, files: dict[str, bytes]) -> Path:
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def _minimal_config() -> dict[str, str]:
    return {
        "backend_executable": "backend/orientation_backend.exe", "project_root": ".",
        "model_dir": "models", "paddle_config": "backend/config.yaml", "data_root": "data",
        "model_sha256": "a" * 64, "edition": "gpu", "compute_device": "gpu",
        "package_version": "1.0.0",
    }


def test_run_benchmark_passes_cpu_thread_override_to_backend(tmp_path: Path, monkeypatch):
    config = {**_minimal_config(), "edition": "cpu", "compute_device": "cpu"}
    package = tmp_path / "package.zip"
    _write_zip(package, {"app_config.json": json.dumps(config).encode(), "backend/config.yaml": b"x"})
    template = tmp_path / "template.png"
    query = tmp_path / "query.png"
    _write_image(template, 1)
    _write_image(query, 2)
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "run"))
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(benchmark, "_environment", lambda config: {})
    monkeypatch.setattr(benchmark, "_acceptance_rows", lambda spec, root: (
        {"M1:front": [template], "M1:back": [template]},
        [{"case": "M1", "expected_orientation": "front", "image_path": str(query), "query_identity": "q1"}],
    ))
    token = "12345678-1234-1234-1234-123456789abc"
    monkeypatch.setattr(benchmark.uuid, "uuid4", lambda: token)
    captured = {}

    class CapturingProcess:
        returncode = None
        def __init__(self, *args, **kwargs):
            captured["env"] = kwargs["env"]
        def poll(self): return None if self.returncode is None else self.returncode
        def wait(self, timeout=None): self.returncode = 0; return 0
        def kill(self): self.returncode = -9

    class FakeClient:
        def __init__(self, *_): pass
        def request(self, command, **fields):
            if command == "hello":
                return {"ready": True, "status": "ready", "phase": "ready", "edition": "cpu",
                        "compute_device": "cpu", "package_version": "1.0.0", "model_fingerprint": "a" * 64,
                        "instance_token": token}
            if command == "register": return {"template_counts": {"front": 1, "back": 1}, "workpiece": {"id": "wp1"}}
            if command == "predict":
                return {
                    "label": "front",
                    "elapsed_ms": 1.0,
                    "timings_ms": {
                        "total": 1.0,
                        "global_model_calls": 1,
                        "global_input_slots": 3,
                        "global_unique_slots": 1,
                    },
                    "needs_review": False,
                }
            if command == "shutdown": return {"ok": True}
            raise AssertionError(command)
        def close(self): pass

    monkeypatch.setattr(benchmark.subprocess, "Popen", CapturingProcess)
    monkeypatch.setattr(benchmark.socket, "create_connection", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "_Client", FakeClient)
    report = run_benchmark(package, tmp_path / "unused.json", iterations=1000, warmup=0, cpu_threads=4)
    assert report["settings"]["cpu_threads"] == 4
    assert captured["env"]["WORKPIECE_CPU_THREADS"] == "4"
    assert report["model_calls"]["status"] == "measured"
    assert report["model_calls"]["total"] == 1000
    assert report["model_calls"]["mean"] == 1.0
    assert all(row["global_model_calls"] == 1 for row in report["requests"])
    assert report["deduplication"]["status"] == "measured"
    assert report["deduplication"]["total_input_slots"] == 3000
    assert report["deduplication"]["total_unique_slots"] == 1000
    assert report["deduplication"]["dedup_ratio"] == pytest.approx(2 / 3)
    assert report["startup"]["cold_start_ms"]["count"] == 1
    assert report["startup"]["cold_start_model_calls"]["status"] == "not_applicable"


def test_environment_reports_physical_cores_from_psutil(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=123),
        cpu_count=lambda logical=True: 8 if logical else 4,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    environment = benchmark._environment({"compute_device": "cpu"})

    assert environment["memory_bytes"] == 123
    assert environment["physical_cores"] == 4


def test_summarize_reports_required_percentiles():
    summary = summarize([10.0, 20.0, 30.0, 40.0, 50.0])
    assert set(summary) == {"count", "mean", "p50", "p95", "p99", "max"}
    assert summary["count"] == 5
    assert summary["mean"] == 30.0
    assert summary["max"] == 50.0
    assert summary["p50"] == 30.0


def test_cpu_latency_target_is_monitoring_not_a_false_pass():
    gate = _latency_gate("cpu", 60.0)

    assert gate["status"] == "monitoring"
    assert gate["target_met"] is False
    assert gate["passed"] is None


def test_gpu_latency_target_remains_a_hard_gate():
    gate = _latency_gate("gpu", 26.0)

    assert gate["status"] == "gate"
    assert gate["target_met"] is False
    assert gate["passed"] is False


def test_compare_predictions_lists_every_cpu_gpu_difference():
    gpu = {"a.png": "front", "b.png": "back"}
    cpu = {"a.png": "back", "b.png": "back"}
    assert compare_predictions(gpu, cpu) == [
        {"image": "a.png", "gpu": "front", "cpu": "back"}
    ]


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1.0], "not-a-list"])
def test_summarize_rejects_empty_nonfinite_negative_or_wrong_input(values):
    with pytest.raises(ValueError):
        summarize(values)


@pytest.mark.parametrize("gpu,cpu", [([], {}), ({"a": "front"}, []), ({"a": 1}, {"a": "front"})])
def test_compare_predictions_rejects_invalid_maps(gpu, cpu):
    with pytest.raises(ValueError):
        compare_predictions(gpu, cpu)


def test_compare_predictions_is_sorted_and_reports_missing_entries():
    gpu = {"z": "front", "a": "back"}
    cpu = {"z": "front", "b": "back"}
    assert compare_predictions(gpu, cpu) == [
        {"image": "a", "gpu": "back", "cpu": None},
        {"image": "b", "gpu": None, "cpu": "back"},
    ]


def test_run_benchmark_requires_at_least_1000_measured_requests():
    with pytest.raises(ValueError, match="at least 1000"):
        run_benchmark("missing.zip", "missing.json", iterations=999)


def test_validate_config_rejects_absolute_and_parent_paths():
    config = {key: "ok" for key in ("backend_executable", "project_root", "model_dir", "paddle_config", "data_root")}
    with pytest.raises(ValueError):
        _validate_config("C:/package", {**config, "model_dir": "C:/outside"})
    with pytest.raises(ValueError):
        _validate_config("C:/package", {**config, "data_root": "../outside"})


def test_acceptance_rows_accepts_real_schema_without_inventory_size(tmp_path: Path):
    spec = _acceptance_fixture(tmp_path / "corpus")
    templates, queries = _acceptance_rows(spec, tmp_path / "extract")
    assert {key: len(value) for key, value in templates.items()} == {
        "M1:front": 1, "M1:back": 1, "M2:front": 1,
        "M2:back": 1, "M7:front": 1, "M7:back": 1,
    }
    assert len(queries) == 6


@pytest.mark.parametrize("mutation", ["sha256", "size", "path"])
def test_acceptance_rows_rejects_fingerprint_multiset_or_size_mutation(tmp_path: Path, mutation: str):
    root = tmp_path / "corpus"
    spec = _acceptance_fixture(root)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    fp_row = payload["fingerprints"]["fast_geometry"]["selections"]["M1"]["front"]["templates"][0]
    if mutation == "sha256":
        fp_row["sha256"] = "f" * 64
    elif mutation == "size":
        fp_row["size"] += 1
    else:
        fp_row["path"] = "data/1_M1/0/missing.png"
    _recompute_acceptance_fingerprints(payload)
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="fingerprint|size|hash|source"):
        _acceptance_rows(spec, tmp_path / "extract")


def test_acceptance_rows_rejects_inventory_size_and_dimension_corruption(tmp_path: Path):
    root = tmp_path / "corpus"
    spec = _acceptance_fixture(root)
    payload = json.loads(spec.read_text(encoding="utf-8"))
    row = payload["selection_inventory"]["M2"]["back"]["queries"][0]
    row["size"] = 1
    row["width"] = 99
    spec.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="size|dimensions"):
        _acceptance_rows(spec, tmp_path / "extract")


def test_acceptance_rows_rejects_nested_reparse_component(tmp_path: Path, monkeypatch):
    root = tmp_path / "corpus"
    spec = _acceptance_fixture(root)
    # Simulate a junction/reparse point without requiring elevated Windows
    # privileges to create one in the test workspace.
    monkeypatch.setattr(benchmark, "_is_reparse_point", lambda path: path.name == "0")
    with pytest.raises(ValueError, match="reparse point"):
        _acceptance_rows(spec, tmp_path / "extract")


def test_client_resets_connect_timeout_for_long_requests():
    class FakeFile:
        def __init__(self):
            self.payload = b'{"request_id":"ignored","ok":true}\n'
        def write(self, data):
            return len(data)
        def flush(self):
            return None
        def readline(self):
            request = json.loads(self.payload.decode())
            request["request_id"] = json.loads(self.last.decode())["request_id"]
            self.payload = (json.dumps(request) + "\n").encode()
            return self.payload

    class FakeSocket:
        def __init__(self):
            self.timeouts = []
            self.file = FakeFile()
        def makefile(self, *_):
            original = self.file.write
            def write(data):
                self.file.last = data
                return original(data)
            self.file.write = write
            return self.file
        def settimeout(self, value):
            self.timeouts.append(value)

    sock = FakeSocket()
    client = benchmark._Client(sock, request_timeout=120.0)
    assert client.request("register", timeout=45.0)["ok"] is True
    assert sock.timeouts == [45.0]


def test_benchmark_temp_root_uses_explicit_uuid_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("managed ACL")))
    root = benchmark._explicit_temp_dir(tmp_path, "benchmark-test")
    try:
        assert root.parent == tmp_path
        assert root.name.startswith(".benchmark-test-")
    finally:
        root.rmdir()


def test_run_benchmark_accepts_versioned_top_level_zip_and_reports_lifecycle(tmp_path: Path, monkeypatch):
    config = _minimal_config()
    package = tmp_path / "package.zip"
    root_name = "WorkpieceOrientation-GPU"
    _write_zip(package, {
        f"{root_name}/app_config.json": json.dumps(config).encode(),
        f"{root_name}/backend/config.yaml": b"Global: {}\n",
        f"{root_name}/version.json": json.dumps({"version": "1.0.0", "python": "3.10", "paddle": "3.2.2", "paddleclas": "2.6.0", "pyinstaller": "6.22.2", "qt": "5.14.2"}).encode(),
    })
    template = tmp_path / "template.png"
    query = tmp_path / "query.png"
    _write_image(template, 1)
    _write_image(query, 2)
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "run"))
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(benchmark, "_environment", lambda config: {"paddle_device_actual": "gpu"})
    monkeypatch.setattr(benchmark, "_acceptance_rows", lambda spec, root: ({"M1:front": [template], "M1:back": [template]}, [{
        "case": "M1", "expected_orientation": "front", "image_path": str(query), "query_identity": "q1"
    }]))
    token = "12345678-1234-1234-1234-123456789abc"
    monkeypatch.setattr(benchmark.uuid, "uuid4", lambda: token)
    captured_environment = {}

    class FakeProcess:
        returncode = None
        def __init__(self, *args, **kwargs):
            captured_environment["env"] = kwargs["env"]
            kwargs["stdout"].write("fake backend log\n")
            kwargs["stdout"].flush()
        def poll(self): return None if self.returncode is None else self.returncode
        def wait(self, timeout=None): self.returncode = 0; return 0
        def kill(self): self.returncode = -9

    class FakeClient:
        def __init__(self, *_): self.closed = False
        def request(self, command, **fields):
            if command == "hello":
                return {"ready": True, "status": "ready", "phase": "ready", "edition": "gpu",
                        "compute_device": "gpu", "package_version": "1.0.0", "model_fingerprint": "a" * 64,
                        "instance_token": token}
            if command == "register":
                return {"template_counts": {"front": 1, "back": 1}, "workpiece": {"id": "wp1"}}
            if command == "predict":
                return {"label": "front", "elapsed_ms": 1.0, "timings_ms": {"total": 1.0}, "needs_review": False}
            if command == "shutdown":
                return {"ok": True}
            raise AssertionError(command)
        def close(self): self.closed = True

    monkeypatch.setattr(benchmark.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(benchmark.socket, "create_connection", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark, "_Client", FakeClient)
    report = run_benchmark(package, tmp_path / "unused.json", iterations=1000, warmup=0, cpu_threads=4)
    assert report["passed"] is True
    assert report["lifecycle"]["shutdown_confirmed"] is True
    assert report["lifecycle"]["process_exit"] == 0
    assert "fake backend log" in report["backend_log"]
    assert "WORKPIECE_CPU_THREADS" not in captured_environment["env"]
    assert report["versions"] == {"package": "1.0.0", "python": "3.10", "paddle": "3.2.2", "paddleclas": "2.6.0", "pyinstaller": "6.22.2", "qt": "5.14.2"}
    assert report["model_calls"]["status"] == "unavailable"
    assert report["model_calls"]["total"] is None
    assert all(row["global_model_calls"] is None for row in report["requests"])
    assert report["deduplication"]["status"] == "unavailable"
    assert report["deduplication"]["dedup_ratio"] is None
    assert report["startup"]["cold_start_ms"]["count"] == 1


def test_run_benchmark_timeout_preserves_process_exit_log_and_lifecycle(tmp_path: Path, monkeypatch):
    config = _minimal_config()
    package = tmp_path / "package.zip"
    _write_zip(package, {"app_config.json": json.dumps(config).encode(), "backend/config.yaml": b"x"})
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "run"))
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(benchmark, "_environment", lambda config: {})
    template = tmp_path / "template.png"
    query = tmp_path / "query.png"
    _write_image(template, 1)
    _write_image(query, 2)
    monkeypatch.setattr(benchmark, "_acceptance_rows", lambda spec, root: ({"M1:front": [template], "M1:back": [template]}, [{
        "case": "M1", "expected_orientation": "front", "image_path": str(query), "query_identity": "q1"
    }]))

    class FakeProcess:
        returncode = None
        def __init__(self, *args, **kwargs): kwargs["stdout"].write("startup failed\n")
        def poll(self): return None
        def wait(self, timeout=None): raise subprocess.TimeoutExpired("fake", timeout)
        def kill(self): self.returncode = -9

    monkeypatch.setattr(benchmark.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(benchmark.socket, "create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(TimeoutError) as error:
        run_benchmark(package, tmp_path / "unused.json", iterations=1000, timeout_seconds=0)
    assert error.value.process_exit == -9
    assert "startup failed" in error.value.backend_log
    assert error.value.lifecycle["forced_termination"] is True


def test_run_benchmark_rejects_zip_path_traversal_and_symlink(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "run"))
    (tmp_path / "run").mkdir()
    traversal = tmp_path / "traversal.zip"
    _write_zip(traversal, {"../escape.txt": b"bad"})
    with pytest.raises(ValueError, match="escapes extraction root"):
        run_benchmark(traversal, tmp_path / "unused.json", iterations=1000)
    symlink = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16) | 0x1FF
    with ZipFile(symlink, "w") as archive:
        archive.writestr(info, b"target")
    with pytest.raises(ValueError, match="symlink"):
        run_benchmark(symlink, tmp_path / "unused.json", iterations=1000)
