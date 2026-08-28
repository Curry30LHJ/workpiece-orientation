from pathlib import Path

from release_tools.portable_smoke import SmokeOptions, run_smoke


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
