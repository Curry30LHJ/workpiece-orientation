from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
import uuid

import pytest

from src.shitu_baseline import split_labels


pytestmark = pytest.mark.integration


class JsonClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=10)
        self.file = self.sock.makefile("rwb")

    def request(self, command: str, **fields):
        request = {"version": 1, "request_id": str(uuid.uuid4()), "command": command, **fields}
        self.file.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.file.flush()
        response = json.loads(self.file.readline().decode("utf-8"))
        assert response["request_id"] == request["request_id"]
        return response

    def close(self):
        self.file.close()
        self.sock.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _copy_images(source_paths: list[Path], target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, source_path in enumerate(source_paths):
        destination = target / f"样本-{index}{source_path.suffix.lower()}"
        shutil.copy2(source_path, destination)
        copied.append(str(destination))
    return copied


@pytest.fixture(scope="session")
def integration_settings() -> dict[str, Path | str]:
    if os.environ.get("WORKPIECE_ORIENTATION_RUN_INTEGRATION") != "1":
        pytest.skip("set WORKPIECE_ORIENTATION_RUN_INTEGRATION=1 to load production models")
    root = Path(os.environ.get("WORKPIECE_ORIENTATION_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    model_dir = Path(
        os.environ.get(
            "WORKPIECE_ORIENTATION_MODEL_DIR",
            root / "third_party" / "models" / "shiru_rec" / "general_PPLCNetV2_base_pretrained_v1.0_infer",
        )
    )
    python_executable = os.environ.get("WORKPIECE_ORIENTATION_PYTHON", sys.executable)
    if not model_dir.is_dir():
        pytest.skip(f"production model directory not found: {model_dir}")
    return {"root": root, "model_dir": model_dir, "python": python_executable}


@pytest.fixture
def running_service(integration_settings, tmp_path: Path):
    root = Path(integration_settings["root"])
    port = _free_port()
    library_dir = tmp_path / "运行时工件库"
    command = [
        str(integration_settings["python"]),
        "-m",
        "src.orientation_tcp_service",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--project-root",
        str(root),
        "--model-dir",
        str(integration_settings["model_dir"]),
        "--library-dir",
        str(library_dir),
    ]
    process = subprocess.Popen(command, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = None
    deadline = time.monotonic() + 120
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"orientation service exited with code {process.returncode}")
            try:
                client = JsonClient("127.0.0.1", port)
                hello = client.request("hello")
                if hello.get("ok") is True:
                    break
            except OSError:
                if client is not None:
                    client.close()
                client = None
                time.sleep(0.25)
        else:
            raise TimeoutError("orientation service did not become ready")
        yield client, tmp_path
    finally:
        if client is not None:
            try:
                client.request("shutdown")
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            client.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)


@pytest.mark.parametrize("dataset_name", ["1_M1", "1_M2", "1_M7"])
def test_service_label_matches_dataset_orientation(dataset_name: str, running_service, integration_settings):
    client, tmp_path = running_service
    dataset_dir = Path(integration_settings["root"]) / "data" / dataset_name
    templates, held_out = split_labels(dataset_dir, template_count=5, seed=20260813)
    front = _copy_images(templates["0"], tmp_path / dataset_name / "正面")
    back = _copy_images(templates["1"], tmp_path / dataset_name / "反面")
    front_query = _copy_images([held_out["0"][0]], tmp_path / dataset_name / "待测正面")[0]
    back_query = _copy_images([held_out["1"][0]], tmp_path / dataset_name / "待测反面")[0]
    response = client.request(
        "register",
        name=dataset_name,
        replace=False,
        front_images=front[:5],
        back_images=back[:5],
    )
    assert response["ok"] is True, response
    workpiece_id = response["workpiece"]["id"]
    assert client.request("list_workpieces")["ok"] is True

    front_prediction = client.request("predict", workpiece_id=workpiece_id, image_path=front_query)
    back_prediction = client.request("predict", workpiece_id=workpiece_id, image_path=back_query)
    assert front_prediction["ok"] is True, front_prediction
    assert back_prediction["ok"] is True, back_prediction
    assert front_prediction["label"] == "front"
    assert back_prediction["label"] == "back"
