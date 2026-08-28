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
from src.model_fingerprint import model_directory_sha256


pytestmark = pytest.mark.integration


class JsonClient:
    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.file = self.sock.makefile("rwb")

    def request(self, command: str, **fields):
        request = {"version": 1, "request_id": str(uuid.uuid4()), "command": command, **fields}
        self.file.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.file.flush()
        while True:
            response = json.loads(self.file.readline().decode("utf-8"))
            assert response["request_id"] == request["request_id"]
            if response.get("event") == "progress":
                continue
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
    paddle_config = root / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml"
    if not paddle_config.is_file():
        pytest.skip(f"production Paddle config not found: {paddle_config}")
    return {
        "root": root,
        "model_dir": model_dir,
        "paddle_config": paddle_config,
        "model_sha256": model_directory_sha256(model_dir),
        "python": python_executable,
    }


@pytest.fixture(scope="session")
def running_service(integration_settings, tmp_path_factory):
    root = Path(integration_settings["root"])
    tmp_path = tmp_path_factory.mktemp("orientation-service-integration")
    port = _free_port()
    data_root = tmp_path / "data"
    instance_token = f"integration-{uuid.uuid4()}"
    service_root = Path(__file__).resolve().parents[1]
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
        "--data-root",
        str(data_root),
        "--paddle-config",
        str(integration_settings["paddle_config"]),
        "--compute-device",
        "gpu",
        "--model-sha256",
        str(integration_settings["model_sha256"]),
        "--package-version",
        "1.0.0",
        "--edition",
        "gpu",
        "--instance-token",
        instance_token,
        "--parent-pid",
        str(os.getpid()),
        "--inference-mode",
        "fast_geometry",
    ]
    process = subprocess.Popen(command, cwd=service_root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = None
    startup_hellos = []
    # The production model stack can take several minutes to initialize. The
    # service binds first and reports status=loading during that time, so keep
    # the connection and continue the hello handshake on the same socket.
    deadline = time.monotonic() + 600
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"orientation service exited with code {process.returncode}")
            try:
                client = JsonClient("127.0.0.1", port)
                while time.monotonic() < deadline:
                    hello = client.request("hello")
                    startup_hellos.append(hello)
                    if hello.get("ok") is True and hello.get("ready") is True:
                        break
                    if hello.get("ok") is True and hello.get("status") == "loading":
                        time.sleep(0.25)
                        continue
                    raise RuntimeError(f"orientation service handshake failed: {hello}")
                else:
                    raise TimeoutError("orientation service did not become ready")
                break
            except OSError:
                if client is not None:
                    client.close()
                client = None
                time.sleep(0.25)
        else:
            raise TimeoutError("orientation service did not become ready")
        loading_hellos = [hello for hello in startup_hellos if hello.get("ready") is False]
        assert loading_hellos
        assert [hello["progress"] for hello in startup_hellos] == sorted(
            hello["progress"] for hello in startup_hellos
        )
        ready = startup_hellos[-1]
        assert ready["phase"] == "ready"
        assert ready["progress"] == 100
        assert ready["package_version"] == "1.0.0"
        assert ready["edition"] == "gpu"
        assert ready["compute_device"] == "gpu"
        assert ready["model_fingerprint"] == integration_settings["model_sha256"]
        assert ready["instance_token"] == instance_token
        yield client, tmp_path
    finally:
        if client is not None:
            try:
                client.request("shutdown", instance_token=instance_token)
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
        progress_events=True,
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
