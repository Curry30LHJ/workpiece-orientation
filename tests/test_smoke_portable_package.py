import json
from pathlib import Path

import pytest

from release_tools.portable_smoke import SmokeOptions, run_portability, run_smoke


class FakeSocket:
    def __init__(self, events, responses):
        self.events, self.responses = events, iter(responses)

    def request(self, command, **fields):
        self.events.append(command)
        response = next(self.responses)
        return response(command, fields) if callable(response) else response

    def close(self):
        self.events.append("close")


class FakeProcess:
    returncode = None
    def poll(self): return self.returncode
    def wait(self, timeout=None): self.returncode = 0; return 0
    def terminate(self): self.returncode = 0


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
    report_path = tmp_path / "failure.json"
    responses = [
        {"ok": True, "ready": True, "instance_token": "token"},
        {"ok": True, "workpieces": []},
        {"ok": False, "error": "register failed"},
    ]
    with pytest.raises(RuntimeError, match="register failed"):
        run_smoke(
            SmokeOptions(package_root=package, dataset_root=dataset, report_path=report_path),
            process_factory=lambda *a, **k: process,
            socket_factory=lambda *a, **k: FakeSocket([], responses),
            temp_root_factory=lambda: tmp_path / "source-copy",
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["error"] == "register failed"
    assert payload["commands"][-1] == "register"
    assert payload["results"][-1]["response"]["ok"] is False
    assert payload["process_exit"] == 7


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
        process = FakeProcess(); processes.append(process); return process
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
    assert [entry["command"] for entry in report["destination"]["commands"]] == [
        "hello", "list_workpieces", "predict", "predict", "get_geometry_mask_profile", "shutdown"
    ]
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
