import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

import release_tools.portable_smoke as smoke
from release_tools.portable_smoke import SmokeOptions, run_portability, run_smoke


def _smoke_wrapper_module():
    path = Path(__file__).parents[1] / "scripts" / "smoke_portable_package.py"
    spec = importlib.util.spec_from_file_location("batch_smoke_wrapper_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_batch_smoke_uses_different_cwd_and_runs_ordered_five_image_batch(tmp_path):
    module = _smoke_wrapper_module()
    package = _package_fixture(tmp_path / "package", edition="cpu")
    images = [tmp_path / f"image-{index}.png" for index in range(5)]
    launched = {}

    class Process:
        returncode = 0
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
        def terminate(self): self.returncode = 0

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
        def close(self): pass

    client = Client()
    result = module.run_frozen_batch_smoke(
        package, "wp-1", images,
        process_factory=lambda args, cwd: launched.update(args=args, cwd=cwd) or Process(),
        client_factory=lambda host, port: client,
        temp_dir_factory=lambda: tmp_path / "different cwd",
    )

    assert Path(launched["cwd"]).resolve() != package.resolve()
    assert result["labels"] == ["front"] * 5
    assert [request["command"] for request in client.requests] == ["hello", "predict_batch"]
    assert client.requests[1]["image_paths"] == [str(path) for path in images]
    assert not (tmp_path / "different cwd").exists()


def test_frozen_batch_smoke_reports_missing_orientation_classifier_clearly(tmp_path):
    module = _smoke_wrapper_module()
    package = _package_fixture(tmp_path / "package", edition="cpu")
    images = [tmp_path / f"image-{index}.png" for index in range(5)]

    class Process:
        returncode = 1
        stdout = None
        def poll(self): return self.returncode
        def wait(self, timeout=None): return self.returncode
        def terminate(self): pass

    with pytest.raises(RuntimeError, match=r"src\.orientation_classifier"):
        module.run_frozen_batch_smoke(
            package, "wp-1", images,
            process_factory=lambda args, cwd: Process(),
            client_factory=lambda host, port: (_ for _ in ()).throw(OSError("unavailable")),
            temp_dir_factory=lambda: tmp_path / "different cwd",
        )


def test_json_socket_applies_configured_read_timeout():
    class RawSocket:
        def __init__(self):
            self.timeout = None

        def settimeout(self, value):
            self.timeout = value

        def makefile(self, *_args):
            return object()

    raw = RawSocket()
    smoke._JsonSocket(raw, timeout_seconds=37.5)
    assert raw.timeout == 37.5


def test_smoke_enforces_one_shared_deadline_across_protocol_requests(tmp_path, monkeypatch):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    clock = [0.0]
    budgets = []

    class SlowSocket(FakeSocket):
        def set_timeout(self, value):
            budgets.append(value)

        def request(self, command, **fields):
            response = super().request(command, **fields)
            clock[0] += 0.6
            return response

    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
    ]
    monkeypatch.setattr(smoke.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError, match="deadline|timed out"):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset,
                         front_template_count=2, back_template_count=2,
                         timeout_seconds=1.0),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: SlowSocket([], responses),
            temp_root_factory=lambda: tmp_path / "deadline-copy",
        )
    assert budgets
    assert budgets == sorted(budgets, reverse=True)
    assert budgets[-1] <= 1.0


def test_smoke_wrapper_runs_directly_outside_repo_cwd():
    script = Path(__file__).parents[1] / "scripts" / "smoke_portable_package.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=script.parents[1].parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


class FakeSocket:
    def __init__(self, events, responses):
        self.events, self.responses = events, iter(responses)
        self.requests = []

    def request(self, command, **fields):
        self.events.append(command)
        self.requests.append((command, fields))
        response = next(self.responses)
        return response(command, fields) if callable(response) else response

    def close(self):
        self.events.append("close")


class FakeProcess:
    returncode = None
    stdout = None
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode = 0; return 0
    def terminate(self): self.returncode = 0


def test_copy_to_long_temp_does_not_depend_on_mkdtemp_acl(tmp_path: Path):
    source = tmp_path / "package"
    source.mkdir()
    (source / "app_config.json").write_text("{}", encoding="utf-8")
    target = smoke._copy_to_long_temp(source, None)
    try:
        assert target.is_dir()
        assert target.parent.name.startswith(".便携 smoke package-")
        assert " " in str(target)
        assert any(ord(character) > 127 for character in str(target))
    finally:
        import shutil
        shutil.rmtree(target.parent)


def test_copy_to_long_temp_leaves_headroom_for_frozen_native_extensions(tmp_path: Path):
    source = tmp_path / "package"
    nested = source / "backend" / "_internal" / "numpy" / "core"
    nested.mkdir(parents=True)
    (nested / "_multiarray_umath.cp310-win_amd64.pyd").write_bytes(b"x")
    target = smoke._copy_to_long_temp(source, None)
    try:
        extension = target / "backend" / "_internal" / "numpy" / "core" / "_multiarray_umath.cp310-win_amd64.pyd"
        assert len(str(target.resolve())) >= 180
        assert len(str(extension.resolve())) < 260
    finally:
        import shutil
        shutil.rmtree(target.parent)


@pytest.mark.parametrize("factory", ["default", "provided"])
def test_copy_to_long_temp_removes_partial_copy_on_failure(tmp_path: Path, monkeypatch, factory):
    source = tmp_path / "package"
    source.mkdir()
    (source / "app_config.json").write_text("{}", encoding="utf-8")
    target = (tmp_path / "provided-copy") if factory == "provided" else None

    def fail_copytree(_source, destination):
        destination = Path(destination)
        destination.mkdir(parents=True)
        (destination / "partial.bin").write_bytes(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr(smoke.shutil, "copytree", fail_copytree)
    root_factory = (lambda: target) if target is not None else None
    with pytest.raises(OSError, match="copy failed"):
        smoke._copy_to_long_temp(source, root_factory)

    assert target is None or not target.exists()
    if target is None:
        assert not list(tmp_path.glob(".便携 smoke package-*"))


class FailingWaitProcess(FakeProcess):
    def wait(self, timeout=None):
        self.returncode = 7
        return self.returncode


def _package_fixture(root: Path, *, edition: str = "gpu") -> Path:
    (root / "backend").mkdir(parents=True)
    (root / "backend" / "orientation_backend.exe").write_bytes(b"x")
    (root / "models").mkdir()
    (root / "data" / "workpieces").mkdir(parents=True)
    for name in ("rules", "cache", "logs", "temp"):
        (root / "data" / name).mkdir()
    (root / "data" / "data_layout.json").write_text('{"layout_version":1}', encoding="utf-8")
    (root / "app_config.json").write_text(
        '{"backend_executable":"backend/orientation_backend.exe","project_root":".",'
        '"model_dir":"models","paddle_config":"backend/config.yaml",'
        f'"data_root":"data","compute_device":"{edition}","edition":"{edition}",'
        '"package_version":"1.0.0","model_sha256":"' + 'a' * 64 + '"}', encoding="utf-8"
    )
    return root


def _dataset_fixture(root: Path) -> Path:
    for label in ("0", "1"):
        (root / label).mkdir(parents=True)
        for index in range(3):
            (root / label / f"{index}.png").write_bytes(f"{label}-{index}".encode())
    return root


def test_smoke_runs_protocol_in_order_and_reports_portable_result(tmp_path):
    dataset = tmp_path / "dataset"
    package = tmp_path / "package"
    (package / "backend" / "resources").mkdir(parents=True)
    (package / "backend" / "orientation_backend.exe").write_bytes(b"x")
    (package / "backend" / "resources" / "inference_general.yaml").write_text("x")
    (package / "models" / "shitu_rec").mkdir(parents=True)
    (package / "data").mkdir()
    (package / "app_config.json").write_text(
        '{"backend_executable":"backend/orientation_backend.exe","project_root":".",'
        '"model_dir":"models/shitu_rec","paddle_config":"backend/resources/inference_general.yaml",'
        '"data_root":"data","compute_device":"gpu","edition":"gpu","package_version":"1.0.0",'
        '"model_sha256":"' + 'a' * 64 + '"}'
    )
    for label in ("0", "1"):
        (dataset / label).mkdir(parents=True)
        for index in range(3):
            (dataset / label / f"{index}.png").write_bytes(f"{label}-{index}".encode())
    events = []
    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {"library_revision": 1}},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True},
    ]
    process = FakeProcess()
    options = SmokeOptions(package_root=package, dataset_root=dataset,
                            front_template_count=2, back_template_count=2, seed=7,
                            report_path=tmp_path / "report.json")
    report = run_smoke(options, process_factory=lambda *a, **k: process,
                       socket_factory=lambda *a, **k: FakeSocket(events, responses),
                       temp_root_factory=lambda *a, **k: tmp_path / "tmp copy")
    assert events[:12] == ["hello", "list_workpieces", "register", "list_workpieces",
        "predict", "predict", "get_geometry_mask_profile", "recycle_workpiece",
        "list_recycled_workpieces", "restore_workpiece", "list_workpieces", "shutdown"]
    assert report["ok"] is True
    assert report["template_counts"] == {"front": 2, "back": 2}
    assert process.returncode == 0
    assert options.report_path.is_file()
    assert not (tmp_path / "tmp copy").exists()


def test_run_smoke_writes_failure_report_and_preserves_nonzero_exit(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    process = FailingWaitProcess()
    process.stdout = type("FakeStdout", (), {"read": lambda self: "backend output"})()
    report_path = tmp_path / "failure.json"
    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": False, "error": "register failed"},
    ]
    with pytest.raises(RuntimeError, match="register failed"):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset,
                         front_template_count=2, back_template_count=2,
                         report_path=report_path),
            process_factory=lambda *a, **k: process,
            socket_factory=lambda *a, **k: FakeSocket([], responses),
            temp_root_factory=lambda: tmp_path / "source-copy",
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["error"].startswith("register failed")
    assert payload["commands"][-1] == "register"
    assert payload["results"][-1]["response"]["ok"] is False
    assert payload["process_exit"] == 7
    assert payload["backend_log"] == "backend output"
    assert not (tmp_path / "source-copy").exists()


def test_run_portability_uses_destination_counts_and_records_destination_commands(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    source = _package_fixture(tmp_path / "source", edition="gpu")
    destination = _package_fixture(tmp_path / "destination", edition="cpu")
    processes = []
    sockets = []
    source_responses = [
        {"ok": True, "ready": True, "instance_token": "source-token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {}}, {"ok": True},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]}, {"ok": True},
    ]
    destination_responses = [
        {"ok": True, "ready": True, "instance_token": "destination-token"},
        {"ok": True, "workpieces": [{"id": "wp-1", "template_counts": {"front": 2, "back": 2}}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {}}, {"ok": True},
    ]
    def process_factory(*args, **kwargs):
        process = FakeProcess()
        process.stdout = type("FakeStdout", (), {"read": lambda self: "destination backend output"})()
        processes.append(process); return process
    def socket_factory(*args, **kwargs):
        socket = FakeSocket([], source_responses if not sockets else destination_responses)
        sockets.append(socket); return socket
    report_path = tmp_path / "portability.json"
    report = run_portability(
        SmokeOptions(source_package_root=source, destination_package_root=destination,
                     dataset_root=dataset, front_template_count=2, back_template_count=2,
                     report_path=report_path),
        process_factory=process_factory, socket_factory=socket_factory,
        temp_root_factory=lambda: tmp_path / "copy",
    )
    assert report["destination"]["template_counts"] == {"front": 2, "back": 2}
    assert report["destination"]["backend_log"] == "destination backend output"
    assert report["destination"]["commands"] == [
        "hello", "list_workpieces", "predict", "predict", "get_geometry_mask_profile", "shutdown"
    ]
    shutdown_fields = [fields for command, fields in sockets[1].requests if command == "shutdown"][-1]
    assert shutdown_fields["instance_token"] == "destination-token"
    assert all({"workpiece_id", "image_path"} <= set(fields)
               for command, fields in sockets[1].requests if command == "predict")
    assert report_path.is_file()
    assert not (tmp_path / "copy").exists()
    assert not (tmp_path / "copy-destination").exists()


def test_run_portability_rejects_nonempty_destination_before_copy(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    source = _package_fixture(tmp_path / "source")
    destination = _package_fixture(tmp_path / "destination")
    (destination / "data" / "workpieces" / "already-there").mkdir()
    with pytest.raises(RuntimeError, match="empty"):
        run_portability(
            SmokeOptions(source_package_root=source, destination_package_root=destination,
                         dataset_root=dataset),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: FakeSocket([], []),
            temp_root_factory=lambda: tmp_path / "copy",
        )


def test_run_portability_report_write_error_preserves_primary_error(tmp_path, monkeypatch):
    import release_tools.portable_smoke as portable_smoke

    source = _package_fixture(tmp_path / "source")
    destination = _package_fixture(tmp_path / "destination")
    (destination / "data" / "workpieces" / "already-there").mkdir()
    def fail_persist(*args, **kwargs):
        raise OSError("report failed")
    monkeypatch.setattr(portable_smoke, "_persist_report", fail_persist)
    with pytest.raises(RuntimeError, match="empty") as caught:
        run_portability(
            SmokeOptions(source_package_root=source, destination_package_root=destination,
                         report_path=tmp_path / "report.json"),
        )
    assert caught.value.smoke_report["report_persist_error"] == "report failed"


def test_run_smoke_requires_explicit_empty_workpiece_list(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    for index, malformed in enumerate(({}, {"workpieces": None})):
        responses = [
            {"ok": True, "ready": True, "instance_token": "hello-token"},
            {"ok": True, **malformed},
        ]
        with pytest.raises(RuntimeError, match="workpieces"):
            run_smoke(
                SmokeOptions(package_root=package, dataset_root=dataset,
                             front_template_count=2, back_template_count=2),
                process_factory=lambda *a, **k: FakeProcess(),
                socket_factory=lambda *a, responses=responses, **k: FakeSocket([], responses),
                temp_root_factory=lambda index=index: tmp_path / f"copy-{index}",
            )


def test_run_smoke_rejects_missing_hello_instance_token(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    with pytest.raises(RuntimeError, match="instance token"):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset,
                         front_template_count=2, back_template_count=2),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: FakeSocket([], [{"ok": True, "ready": True}]),
            temp_root_factory=lambda: tmp_path / "copy",
        )


def test_run_smoke_report_write_error_preserves_primary_error(tmp_path, monkeypatch):
    import release_tools.portable_smoke as portable_smoke

    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    config_path = package / "app_config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8").replace(
        '"backend_executable":"backend/orientation_backend.exe"',
        '"backend_executable":"../backend/orientation_backend.exe"'), encoding="utf-8")
    def fail_persist(*args, **kwargs):
        raise OSError("report failed")
    monkeypatch.setattr(portable_smoke, "_persist_report", fail_persist)
    with pytest.raises(ValueError, match="relative") as caught:
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset, report_path=tmp_path / "report.json"),
        )
    assert caught.value.smoke_report["report_persist_error"] == "report failed"


def test_run_portability_rejects_extra_destination_workpiece(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    source = _package_fixture(tmp_path / "source")
    destination = _package_fixture(tmp_path / "destination", edition="cpu")
    source_responses = [
        {"ok": True, "ready": True, "instance_token": "source-token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {}}, {"ok": True},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]}, {"ok": True},
    ]
    destination_responses = [
        {"ok": True, "ready": True, "instance_token": "destination-token"},
        {"ok": True, "workpieces": [
            {"id": "wp-1", "template_counts": {"front": 2, "back": 2}},
            {"id": "unexpected", "template_counts": {"front": 1, "back": 1}},
        ]},
    ]
    sockets = []
    processes = []
    def process_factory(*args, **kwargs):
        process = FakeProcess()
        process.stdout = type("FakeStdout", (), {"read": lambda self: "destination failure output"})()
        processes.append(process)
        return process
    def socket_factory(*args, **kwargs):
        responses = source_responses if not sockets else destination_responses
        sock = FakeSocket([], responses); sockets.append(sock); return sock
    report_path = tmp_path / "extra-destination.json"
    with pytest.raises(RuntimeError, match="destination workpiece set"):
        run_portability(
            SmokeOptions(source_package_root=source, destination_package_root=destination,
                         dataset_root=dataset, front_template_count=2, back_template_count=2,
                         report_path=report_path),
            process_factory=process_factory, socket_factory=socket_factory,
            temp_root_factory=lambda: tmp_path / "copy",
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["destination"]["backend_log"] == "destination failure output"
    assert not (tmp_path / "copy").exists()
    assert not (tmp_path / "copy-destination").exists()


def test_run_portability_rejects_missing_destination_hello_token(tmp_path, monkeypatch):
    import release_tools.portable_smoke as portable_smoke

    source = _package_fixture(tmp_path / "source")
    destination = _package_fixture(tmp_path / "destination", edition="cpu")
    source_report = {
        "workpiece_id": "wp-1", "template_counts": {"front": 2, "back": 2},
        "temp_package": str(source), "front_query": "front.png", "back_query": "back.png",
        "commands": [], "results": [],
    }
    monkeypatch.setattr(portable_smoke, "run_smoke", lambda *args, **kwargs: source_report)
    destination_responses = [{"ok": True, "ready": True}]
    with pytest.raises(RuntimeError, match="destination hello omitted instance token"):
        run_portability(
            SmokeOptions(source_package_root=source, destination_package_root=destination,
                         dataset_root=tmp_path),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: FakeSocket([], destination_responses),
            temp_root_factory=lambda: tmp_path / "copy",
        )


def test_run_smoke_uses_hello_token_for_shutdown_and_predict_fields(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    responses = [
        {"ok": True, "ready": True, "instance_token": "hello-token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {}}, {"ok": True},
        {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": [{"id": "wp-1"}]}, {"ok": True},
    ]
    sock = FakeSocket([], responses)
    process = FakeProcess(); process.token = "wrong-process-token"
    run_smoke(
        SmokeOptions(package_root=package, dataset_root=dataset, front_template_count=2, back_template_count=2),
        process_factory=lambda *a, **k: process, socket_factory=lambda *a, **k: sock,
        temp_root_factory=lambda: tmp_path / "copy",
    )
    predict_requests = [fields for command, fields in sock.requests if command == "predict"]
    assert all({"workpiece_id", "image_path"} <= set(fields) for fields in predict_requests)
    shutdown_fields = [fields for command, fields in sock.requests if command == "shutdown"][-1]
    assert shutdown_fields["instance_token"] == "hello-token"


def test_run_smoke_rejects_non_string_registered_workpiece_id(tmp_path):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": 7}, "template_counts": {"front": 2, "back": 2}},
    ]
    with pytest.raises(RuntimeError, match="register did not return workpiece id"):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset,
                         front_template_count=2, back_template_count=2),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: FakeSocket([], responses),
            temp_root_factory=lambda: tmp_path / "copy",
        )


@pytest.mark.parametrize(
    ("malformed_list", "expected_error"),
    [
        ([{"id": "wp-1"}, {"id": "extra"}], "registered workpiece missing from list"),
        ([{"id": "wp-1"}, {"id": "wp-1"}], "recycled workpiece missing from recycle list"),
        ([{"id": "wp-1"}, {"id": 7}], "restored workpiece missing from list"),
    ],
)
def test_run_smoke_rejects_non_unique_or_invalid_lifecycle_workpiece_lists(
    tmp_path, malformed_list, expected_error
):
    dataset = _dataset_fixture(tmp_path / "dataset")
    package = _package_fixture(tmp_path / "package")
    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": True, "workpiece": {"id": "wp-1"}, "template_counts": {"front": 2, "back": 2}},
        {"ok": True, "workpieces": malformed_list}
        if expected_error == "registered workpiece missing from list"
        else {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "label": "front"}, {"ok": True, "label": "back"},
        {"ok": True, "profile": {}}, {"ok": True},
        {"ok": True, "workpieces": malformed_list}
        if expected_error == "recycled workpiece missing from recycle list"
        else {"ok": True, "workpieces": [{"id": "wp-1"}]},
        {"ok": True, "workpiece": {"id": "wp-1"}},
        {"ok": True, "workpieces": malformed_list}
        if expected_error == "restored workpiece missing from list"
        else {"ok": True, "workpieces": [{"id": "wp-1"}]},
    ]
    with pytest.raises(RuntimeError, match=expected_error):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset,
                         front_template_count=2, back_template_count=2),
            process_factory=lambda *a, **k: FakeProcess(),
            socket_factory=lambda *a, **k: FakeSocket([], responses),
            temp_root_factory=lambda: tmp_path / "copy",
        )
