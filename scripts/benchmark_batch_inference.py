#!/usr/bin/env python3
"""Measure warm five-image ``predict_batch`` requests against a CPU package.

Startup and capability negotiation are deliberately outside the timing loop.
Percentiles use linear interpolation: for a sorted sample of n values, the
rank is ``(n - 1) * percentile / 100`` and adjacent ranks are interpolated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def percentiles_ms(samples: Iterable[float]) -> dict[str, float]:
    """Return stable linear percentiles without optional numeric packages."""
    values = sorted(float(value) for value in samples)
    if not values or any(value < 0 for value in values):
        raise ValueError("timing samples must be non-empty and non-negative")

    def percentile(percent: float) -> float:
        rank = (len(values) - 1) * percent / 100.0
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (rank - lower)

    return {"p50": percentile(50), "p95": percentile(95), "p99": percentile(99), "max": values[-1]}


def batch_request(workpiece_id: str, image_paths: Iterable[Path | str], *, request_id: str | None = None) -> dict[str, Any]:
    paths = [str(path) for path in image_paths]
    if not isinstance(workpiece_id, str) or not workpiece_id.strip():
        raise ValueError("workpiece_id must be a non-empty string")
    if len(paths) != 5:
        raise ValueError("predict_batch benchmark requires exactly 5 image paths")
    if any(not path for path in paths):
        raise ValueError("image paths must be non-empty")
    return {"version": 1, "request_id": request_id or str(uuid.uuid4()), "command": "predict_batch",
            "workpiece_id": workpiece_id, "image_paths": paths}


def validate_batch_response(response: dict[str, Any], request_id: str, image_paths: Iterable[Path | str]) -> list[str]:
    paths = [str(path) for path in image_paths]
    if (response.get("request_id") != request_id
            or ("command" in response and response["command"] != "predict_batch")):
        raise RuntimeError("predict_batch response id/command mismatch")
    if response.get("ok") is not True:
        error = response.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(str(message or "predict_batch failed"))
    items = response.get("items")
    if not isinstance(items, list) or len(items) != len(paths):
        raise RuntimeError("predict_batch response has invalid item count")
    labels: list[str] = []
    for index, (path, item) in enumerate(zip(paths, items)):
        if not isinstance(item, dict) or item.get("index") != index or item.get("image_path") != path:
            raise RuntimeError("predict_batch response item order/path mismatch")
        if item.get("ok") is not True:
            error = item.get("error")
            message = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(str(message or f"predict_batch item {index} failed"))
        prediction = item.get("prediction")
        if not isinstance(prediction, dict):
            raise RuntimeError("predict_batch item prediction must be an object")
        label = prediction.get("label")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError("predict_batch item prediction must contain a non-empty label")
        labels.append(label)
    return labels


def build_report(samples_ms: Iterable[float], warmup: int, iterations: int, workers: int,
                 threads_per_worker: int, hardware: dict[str, Any], accuracy: dict[str, Any],
                 fallback: dict[str, Any]) -> dict[str, Any]:
    samples = [float(value) for value in samples_ms]
    if len(samples) != iterations:
        raise ValueError("measured sample count must equal iterations")
    timing_summary = {"samples": samples, **percentiles_ms(samples)}
    total_seconds = sum(samples) / 1000.0
    return {
        "hardware": hardware,
        "configuration": {"workers": workers, "threads_per_worker": threads_per_worker},
        "warmup": warmup,
        "iterations": iterations,
        "batch_size": 5,
        "timings_ms": timing_summary,
        "throughput_images_per_second": (5 * len(samples) / total_seconds) if total_seconds else 0.0,
        "accuracy": accuracy,
        "fallback": fallback,
    }


def _validate_hello(response: dict[str, Any], request_id: str) -> bool:
    if (response.get("request_id") != request_id
            or ("command" in response and response["command"] != "hello")
            or response.get("ok") is not True):
        raise RuntimeError("hello protocol validation failed")
    if response.get("ready") is not True:
        return False
    capabilities = response.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("predict_batch") is not True or capabilities.get("batch_ready") is not True:
        raise RuntimeError("backend hello does not advertise ready predict_batch support")
    return True


def _cleanup(client: Any | None, process: Any | None) -> None:
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


def run_batch_measurements(client: Any, process: Any, workpiece_id: str, image_paths: Iterable[Path | str], *,
                           warmup: int, iterations: int,
                           startup_timeout_seconds: float = 600.0) -> tuple[list[float], list[str], dict[str, Any]]:
    """Use one connected client for hello, warmups, and measured batches."""
    paths = list(image_paths)
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    last_response: dict[str, Any] = {}
    measured_labels: list[str] = []
    try:
        deadline = time.monotonic() + startup_timeout_seconds
        while True:
            hello_id = str(uuid.uuid4())
            hello = client.request({"version": 1, "request_id": hello_id, "command": "hello"})
            if _validate_hello(hello, hello_id):
                break
            if hello.get("status") != "loading":
                raise RuntimeError(f"backend hello is neither ready nor loading: {hello}")
            if process.poll() is not None:
                raise RuntimeError("packaged backend exited during hello handshake")
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for ready backend hello")
            time.sleep(0.1)
        for _ in range(warmup):
            payload = batch_request(workpiece_id, paths)
            last_response = client.request(payload)
            validate_batch_response(last_response, payload["request_id"], paths)
        samples: list[float] = []
        for _ in range(iterations):
            payload = batch_request(workpiece_id, paths)
            start = time.perf_counter()
            last_response = client.request(payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            labels = validate_batch_response(last_response, payload["request_id"], paths)
            samples.append(elapsed_ms)
            measured_labels.extend(labels)
        return samples, measured_labels, last_response
    except Exception:
        _cleanup(client, process)
        raise


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
                raise RuntimeError("backend closed persistent benchmark connection")
            response = json.loads(line.decode("utf-8"))
            if response.get("event") != "progress":
                return response

    def close(self) -> None:
        self.file.close()
        self.sock.close()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _hardware() -> dict[str, Any]:
    return {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
            "logical_processors": os.cpu_count()}


def accuracy_summary(labels: list[str], expected_labels: list[str] | None) -> dict[str, Any]:
    if expected_labels is None:
        return {"available": False, "reason": "no expected labels were supplied"}
    if len(expected_labels) != 5:
        raise ValueError("expected_labels must contain exactly five labels")
    if len(labels) % 5:
        raise ValueError("measured labels must contain complete five-image batches")
    expected = expected_labels * (len(labels) // 5)
    correct = sum(actual == wanted for actual, wanted in zip(labels, expected))
    return {"available": True, "correct": correct, "total": len(labels),
            "rate": correct / len(labels) if labels else 0.0}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _launch(package_root: Path, workers: int, threads_per_worker: int) -> tuple[Any, int]:
    from release_tools.portable_smoke import _package_config, _start_backend
    config = _package_config(package_root)
    if str(config.get("compute_device")).lower() != "cpu":
        raise ValueError("batch benchmark requires a packaged CPU backend")
    port = _free_port()
    environment = os.environ.copy()
    environment["WORKPIECE_BATCH_WORKERS"] = str(workers)
    environment["WORKPIECE_BATCH_THREADS_PER_WORKER"] = str(threads_per_worker)

    def start_with_environment(args: list[str], cwd: str, token: str | None = None) -> subprocess.Popen:
        kwargs: dict[str, Any] = {"cwd": cwd, "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                                  "text": True, "env": environment}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        return subprocess.Popen(args, **kwargs)

    return _start_backend(package_root, config, port, start_with_environment), port


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark five-image batches from a packaged CPU backend")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--workpiece-id", required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=_positive_int, default=200)
    parser.add_argument("--workers", type=_positive_int, required=True)
    parser.add_argument("--threads-per-worker", type=_positive_int, required=True)
    parser.add_argument("--expected-label", action="append",
                        help="optional expected label for each of the five ordered images")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    client = process = None
    try:
        if len(args.image) != 5:
            raise ValueError("--image must be supplied exactly five times")
        process, port = _launch(args.package_root.resolve(), args.workers, args.threads_per_worker)
        deadline = time.monotonic() + 600.0
        while True:
            try:
                client = _JsonClient(socket.create_connection(("127.0.0.1", port), timeout=2.0))
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("packaged backend exited before benchmark connection")
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for packaged backend")
                time.sleep(0.1)
        if args.expected_label is not None and len(args.expected_label) != 5:
            raise ValueError("--expected-label must be supplied exactly five times when used")
        samples, labels, backend = run_batch_measurements(client, process, args.workpiece_id, args.image,
                                                            warmup=args.warmup, iterations=args.iterations)
        fallback = backend.get("fallback")
        report = build_report(
            samples, args.warmup, args.iterations, args.workers, args.threads_per_worker,
            _hardware(), accuracy_summary(labels, args.expected_label),
            {
                "used": fallback is not None,
                "reason": fallback if isinstance(fallback, str) else None,
                "batch_mode": backend.get("batch_mode"),
                "actual_worker_count": backend.get("worker_count"),
            },
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"batch benchmark failed: {exc}", file=sys.stderr)
        return 2
    finally:
        _cleanup(client, process)


if __name__ == "__main__":
    raise SystemExit(main())
