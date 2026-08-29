import json
from pathlib import Path
import subprocess
import sys

import pytest

import release_tools.portable_smoke as smoke
from release_tools.portable_smoke import SmokeOptions, run_portability, run_smoke


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
