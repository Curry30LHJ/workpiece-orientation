#!/usr/bin/env python3
"""Portable smoke entry point, including a frozen five-image batch probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, Callable


# Keep this wrapper self-contained when launched by path from any working
# directory (as the portable release script does).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from release_tools.portable_smoke import _free_port, _package_config, _real_process_factory, _start_backend, main
from release_tools.backend_bundle import assert_frozen_backend_modules


class _JsonClient:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb")

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.file.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                raise RuntimeError("frozen backend closed smoke connection")
            response = json.loads(line.decode("utf-8"))
            if response.get("event") != "progress":
                return response

    def close(self) -> None:
        self.file.close()
        self.sock.close()


def _validate_response(response: dict[str, Any], request: dict[str, Any], paths: list[str] | None = None) -> list[str | None]:
    if response.get("request_id") != request["request_id"] or response.get("command") != request["command"]:
        raise RuntimeError("frozen batch smoke response id/command mismatch")
    if response.get("ok") is not True:
        raise RuntimeError(f"{request['command']} failed: {response.get('error')}")
    if request["command"] == "hello":
        capabilities = response.get("capabilities")
        if (response.get("ready") is not True or not isinstance(capabilities, dict)
                or capabilities.get("predict_batch") is not True or capabilities.get("batch_ready") is not True):
            raise RuntimeError("frozen backend hello does not advertise ready predict_batch support")
        return []
    items = response.get("items")
    if paths is None or not isinstance(items, list) or len(items) != 5:
        raise RuntimeError("frozen batch smoke received invalid batch items")
    labels = []
    for index, (path, item) in enumerate(zip(paths, items)):
        if not isinstance(item, dict) or item.get("index") != index or item.get("image_path") != path or item.get("ok") is not True:
            raise RuntimeError("frozen batch smoke received unordered or failed batch item")
        prediction = item.get("prediction")
        if not isinstance(prediction, dict):
            raise RuntimeError("frozen batch smoke item prediction must be an object")
        label = prediction.get("label")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError("frozen batch smoke item prediction must contain a non-empty label")
        labels.append(label)
    return labels


def run_frozen_batch_smoke(package_root: Path, workpiece_id: str, image_paths: list[Path] | list[str], *,
                           process_factory: Callable[..., Any] = _real_process_factory,
                           client_factory: Callable[[str, int], Any] | None = None,
                           temp_dir_factory: Callable[[], Path] | None = None) -> dict[str, Any]:
    """Launch a frozen backend from another CWD and validate one ordered batch.

    The temporary CWD is owned by this function; the supplied package is never
    copied, modified, or deleted.
    """
    package = Path(package_root).resolve()
    paths = [str(path) for path in image_paths]
    if len(paths) != 5:
        raise ValueError("frozen batch smoke requires exactly 5 image paths")
    config = _package_config(package)
    assert_frozen_backend_modules(package / config["backend_executable"])
    work_dir = Path(temp_dir_factory() if temp_dir_factory else package.parent / f".frozen-batch-smoke-{uuid.uuid4().hex}")
    if work_dir.exists():
        raise FileExistsError(f"temporary smoke CWD already exists: {work_dir}")
    work_dir.mkdir(parents=True)
    process = client = None
    try:
        port = _free_port()
        def launch(args: list[str], _package_cwd: str, token: str | None = None) -> Any:
            try:
                return process_factory(args, cwd=str(work_dir))
            except TypeError:
                return process_factory(args, str(work_dir))
        process = _start_backend(package, config, port, launch)
        factory = client_factory or (lambda host, endpoint: _JsonClient(socket.create_connection((host, endpoint), timeout=2.0)))
        deadline = time.monotonic() + 30.0
        while True:
            try:
                client = factory("127.0.0.1", port)
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("frozen backend failed to start; verify frozen package includes src.orientation_classifier")
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for frozen backend")
                time.sleep(0.1)
        hello = {"version": 1, "request_id": str(uuid.uuid4()), "command": "hello"}
        _validate_response(client.request(hello), hello)
        batch = {"version": 1, "request_id": str(uuid.uuid4()), "command": "predict_batch",
                 "workpiece_id": workpiece_id, "image_paths": paths}
        labels = _validate_response(client.request(batch), batch, paths)
        return {"ok": True, "labels": labels, "batch_size": 5}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    process.wait(timeout=10)
                except Exception:
                    pass
            except Exception:
                pass
        shutil.rmtree(work_dir, ignore_errors=True)


def _batch_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test frozen predict_batch from another working directory")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--workpiece-id", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    run_frozen_batch_smoke(args.package_root, args.workpiece_id, args.image)
    return 0


if __name__ == "__main__":
    if "--batch-workpiece-id" in sys.argv:
        arguments = sys.argv[1:]
        index = arguments.index("--batch-workpiece-id")
        arguments[index] = "--workpiece-id"
        raise SystemExit(_batch_main(arguments))
    raise SystemExit(main())
