"""Offline benchmark runner for packaged orientation backends.

The statistic and comparison helpers are intentionally independent of the
backend so they can be used by CI and by the acceptance report generator.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any
import zipfile

import numpy as np


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Return untrimmed NumPy-linear latency statistics.

    Empty, non-numeric, non-finite, negative, and multidimensional inputs are
    rejected rather than silently producing misleading acceptance results.
    """
    if isinstance(values, (str, bytes)):
        raise ValueError("values must be a one-dimensional numeric sequence")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be a one-dimensional numeric sequence") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("values must be finite and non-negative")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(array.max()),
    }


def compare_predictions(
    gpu: Mapping[str, str], cpu: Mapping[str, str]
) -> list[dict[str, str | None]]:
    """List all label differences, including entries missing from one report."""
    if not isinstance(gpu, Mapping) or not isinstance(cpu, Mapping):
        raise ValueError("predictions must be mappings")
    for name, values in (("gpu", gpu), ("cpu", cpu)):
        for image, label in values.items():
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"{name} prediction keys must be non-empty strings")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name} prediction labels must be non-empty strings")
    differences = []
    for image in sorted(set(gpu) | set(cpu)):
        gpu_label = gpu.get(image)
        cpu_label = cpu.get(image)
        if gpu_label != cpu_label:
            differences.append({"image": image, "gpu": gpu_label, "cpu": cpu_label})
    return differences


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Client:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.file = sock.makefile("rwb")

    def request(self, command: str, **fields: Any) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        self.file.write((json.dumps({"version": 1, "request_id": request_id,
                                     "command": command, **fields}) + "\n").encode())
        self.file.flush()
        while True:
            line = self.file.readline()
            if not line:
                raise RuntimeError("backend closed benchmark connection")
            response = json.loads(line.decode("utf-8"))
            if response.get("event") == "progress":
                continue
            if response.get("request_id") != request_id:
                raise RuntimeError("backend request id mismatch")
            if response.get("ok") is not True:
                raise RuntimeError(f"{command} failed: {response}")
            return response

    def close(self) -> None:
        self.file.close()
        self.sock.close()


def _environment(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "memory_bytes": None,
        "gpu_name": None,
        "gpu_driver": None,
        "paddle_device_requested": config.get("compute_device"),
    }
    try:
        import psutil  # type: ignore
        result["memory_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5, check=False)
        row = probe.stdout.strip().splitlines()[0] if probe.returncode == 0 and probe.stdout.strip() else ""
        if row:
            parts = [part.strip() for part in row.split(",", 1)]
            result["gpu_name"] = parts[0]
            if len(parts) > 1:
                result["gpu_driver"] = parts[1]
    except Exception:
        pass
    try:
        import paddle  # type: ignore
        result["paddle"] = getattr(paddle, "__version__", None)
        result["paddle_device_actual"] = paddle.get_device()
    except Exception:
        result["paddle_device_actual"] = None
    return result


def _manifest_hashes(root: Path) -> dict[str, str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(row["path"]): str(row["sha256"]) for row in payload.get("files", [])
            if isinstance(row, Mapping) and "path" in row and "sha256" in row}


def _validate_config(root: Path | str, config: Mapping[str, Any]) -> None:
    """Validate all package paths before using them as subprocess arguments."""
    root_path = Path(root).resolve()
    for key in ("backend_executable", "project_root", "model_dir", "paddle_config", "data_root"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"package config path must be a non-empty string: {key}")
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"package config path must be relative and contained: {key}")
        resolved = (root_path / candidate).resolve()
        if root_path not in resolved.parents and resolved != root_path:
            raise ValueError(f"package config path escapes package root: {key}")
        current = root_path
        for part in candidate.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"package config path traverses symlink: {key}")


def _acceptance_rows(spec_path: Path, temp_root: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    inventory = payload.get("selection_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("acceptance spec has no selection_inventory")
    if set(inventory) != {"M1", "M2", "M7"}:
        raise ValueError("acceptance spec must contain exactly M1, M2, and M7")
    fingerprints = ((payload.get("fingerprints") or {}).get("fast_geometry")
                    if isinstance(payload.get("fingerprints"), Mapping) else None)
    selections_fp = fingerprints.get("selections") if isinstance(fingerprints, Mapping) else None
    overall_fp = fingerprints.get("overall_sha256") if isinstance(fingerprints, Mapping) else None
    if not isinstance(selections_fp, Mapping) or not isinstance(overall_fp, str) or len(overall_fp) != 64:
        raise ValueError("acceptance spec is missing fast_geometry fingerprints")
    templates: dict[str, list[Path]] = {}
    queries: list[dict[str, Any]] = []
    template_hashes: set[str] = set()
    query_hashes: set[str] = set()
    overlap_by_case: dict[str, int] = {}
    for case, details in inventory.items():
        if not isinstance(details, Mapping):
            continue
        if not isinstance(details.get("dataset_path"), str) or not str(details.get("dataset_path")).strip():
            raise ValueError(f"acceptance case {case} has no dataset fingerprint/path")
        case_fp = selections_fp.get(case)
        if not isinstance(case_fp, Mapping) or not isinstance(case_fp.get("aggregate_sha256"), str) or len(str(case_fp.get("aggregate_sha256"))) != 64:
            raise ValueError(f"acceptance case {case} has no aggregate fingerprint")
        for direction in ("front", "back"):
            section = details.get(direction)
            if not isinstance(section, Mapping):
                continue
            for role in ("templates", "queries"):
                rows = section.get(role, [])
                if not isinstance(rows, list):
                    continue
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping) or not isinstance(row.get("image_path"), str):
                        continue
                    if row.get("role") != ("template" if role == "templates" else "query"):
                        raise ValueError("acceptance inventory row has invalid role")
                    source = Path(row["image_path"])
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    destination = temp_root / "data" / "benchmark" / str(case) / direction / f"{index:04d}_{source.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    if row.get("sha256") and _sha256(source) != str(row["sha256"]):
                        raise ValueError(f"acceptance source hash mismatch: {source}")
                    if int(row.get("width", 0)) <= 0 or int(row.get("height", 0)) <= 0:
                        raise ValueError("acceptance inventory row has invalid dimensions")
                    if role == "templates":
                        templates.setdefault(f"{case}:{direction}", []).append(destination)
                        if row.get("sha256"):
                            template_hashes.add(str(row["sha256"]))
                    else:
                        if row.get("expected_orientation") not in {"front", "back"}:
                            raise ValueError("acceptance query expected_orientation must be front or back")
                        queries.append({**dict(row), "case": str(case), "direction": direction,
                                        "image_path": str(destination), "query_identity": str(row.get("query_identity", destination.name))})
                        if row.get("sha256"):
                            query_hashes.add(str(row["sha256"]))
        overlap_by_case[str(case)] = 0
    if not templates or not queries:
        raise ValueError("acceptance spec contains no usable templates and queries")
    if template_hashes & query_hashes:
        raise ValueError("acceptance template/query SHA-256 overlap")
    for case, count in overlap_by_case.items():
        expected_overlap = selections_fp[case].get("template_query_overlap_count", 0)
        if int(expected_overlap) != count:
            raise ValueError(f"acceptance overlap count mismatch for {case}")
    identities = [str(row["query_identity"]) for row in queries]
    if any(not identity.strip() for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("acceptance query identities must be unique and non-empty")
    return templates, queries


def run_benchmark(package_zip: Path, acceptance_spec: Path, *, warmup: int = 50,
                  iterations: int = 1000, compare: Path | None = None,
                  timeout_seconds: float = 180.0) -> dict[str, Any]:
    """Extract an immutable package copy and benchmark its TCP backend."""
    if warmup < 0 or iterations < 1000:
        raise ValueError("warmup must be non-negative and iterations must be at least 1000")
    package_zip = Path(package_zip).resolve()
    if not package_zip.is_file():
        raise FileNotFoundError(package_zip)
    started = perf_counter()
    temp_parent = Path(tempfile.mkdtemp(prefix="portable-benchmark-"))
    process = None
    client = None
    completed = False
    report: dict[str, Any] = {"package_zip": str(package_zip), "package_sha256": _sha256(package_zip),
                              "warmup": warmup, "iterations": iterations, "startup": {}}
    try:
        package_root = temp_parent / "package-copy"
        if package_root.exists() and any(package_root.iterdir()):
            raise FileExistsError("temporary benchmark destination must be empty")
        with zipfile.ZipFile(package_zip) as archive:
            root_resolved = package_root.resolve()
            for member in archive.infolist():
                if stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF):
                    raise ValueError(f"ZIP symlink entries are not allowed: {member.filename}")
                target = (package_root / member.filename).resolve()
                if root_resolved not in target.parents and target != root_resolved:
                    raise ValueError(f"ZIP entry escapes extraction root: {member.filename}")
            archive.extractall(package_root)
        config = json.loads((package_root / "app_config.json").read_text(encoding="utf-8"))
        _validate_config(package_root, config)
        environment = _environment(config)
        report.update({"edition": config.get("edition"), "version": config.get("package_version"),
                       "versions": {"package": config.get("package_version")},
                       "manifest_hashes": _manifest_hashes(package_root), "environment": environment,
                       "hardware": environment})
        templates, queries = _acceptance_rows(Path(acceptance_spec), package_root)
        report["acceptance_source_hashes"] = {
            str(row["query_identity"]): str(row["sha256"])
            for row in queries
            if row.get("sha256")
        }
        port = _free_port()
        executable = package_root / str(config["backend_executable"])
        token = str(uuid.uuid4())
        args = [str(executable), "--host", "127.0.0.1", "--port", str(port),
                "--project-root", str(package_root / config["project_root"]),
                "--model-dir", str(package_root / config["model_dir"]),
                "--paddle-config", str(package_root / config["paddle_config"]),
                "--data-root", str(package_root / config["data_root"]), "--compute-device", str(config["compute_device"]),
                "--model-sha256", str(config["model_sha256"]), "--edition", str(config["edition"]),
                "--package-version", str(config["package_version"]), "--instance-token", token,
                "--parent-pid", str(os.getpid()), "--local-search-mode", str(config.get("local_search_mode", "adaptive")),
                "--inference-mode", str(config.get("inference_mode", "fast_geometry"))]
        process = subprocess.Popen(args, cwd=str(package_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        report["startup"]["process_spawn_ms"] = (perf_counter() - started) * 1000.0
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                client = _Client(socket.create_connection(("127.0.0.1", port), timeout=5))
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(f"backend exited with code {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("backend did not accept a connection")
                time.sleep(0.1)
        report["startup"]["connection_ms"] = (perf_counter() - started) * 1000.0
        phases: list[dict[str, Any]] = []
        hello = None
        while time.monotonic() < deadline:
            hello = client.request("hello")
            phases.append({"phase": hello.get("phase"), "status": hello.get("status"), "elapsed_ms": (perf_counter() - started) * 1000.0})
            if hello.get("ready") is True:
                break
            if hello.get("status") != "loading":
                raise RuntimeError(f"backend startup failed: {hello}")
            time.sleep(0.1)
        else:
            raise TimeoutError("backend did not become ready")
        report["startup"]["phases"] = phases
        report["startup"]["ready_ms"] = (perf_counter() - started) * 1000.0
        if hello.get("edition") != config.get("edition") or hello.get("compute_device") != config.get("compute_device"):
            raise RuntimeError("backend hello edition/device does not match app_config")
        if hello.get("package_version") != config.get("package_version"):
            raise RuntimeError("backend hello package version does not match app_config")
        if hello.get("model_fingerprint") != config.get("model_sha256"):
            raise RuntimeError("backend hello model fingerprint does not match app_config")
        if hello.get("instance_token") != token:
            raise RuntimeError("backend hello instance token mismatch")
        report["environment"]["paddle_device_actual"] = hello.get("compute_device")
        report["hardware"] = report["environment"]
        workpieces: dict[str, str] = {}
        for case in sorted({str(row["case"]) for row in queries}):
            front = templates.get(f"{case}:front", [])
            back = templates.get(f"{case}:back", [])
            response = client.request("register", name=f"benchmark-{case}", replace=False,
                                      front_images=[str(p) for p in front], back_images=[str(p) for p in back])
            counts = response.get("template_counts")
            if counts != {"front": len(front), "back": len(back)} or not front or not back:
                raise RuntimeError(f"register template counts mismatch for {case}: {counts}")
            workpiece = response.get("workpiece") or {}
            workpiece_id = workpiece.get("id")
            if not isinstance(workpiece_id, str) or not workpiece_id.strip():
                raise RuntimeError(f"register returned invalid workpiece id for {case}")
            workpieces[case] = workpiece_id
        for index in range(warmup):
            row = queries[index % len(queries)]
            client.request("predict", workpiece_id=workpieces[str(row["case"])], image_path=row["image_path"])
        samples: list[dict[str, Any]] = []
        predictions: dict[str, str] = {}
        for index in range(iterations):
            row = queries[index % len(queries)]
            expected_label = row.get("expected_orientation")
            if expected_label not in {"front", "back"}:
                raise ValueError("acceptance query expected_orientation must be front or back")
            t0 = perf_counter()
            response = client.request("predict", workpiece_id=workpieces[str(row["case"])], image_path=row["image_path"])
            round_trip_ms = (perf_counter() - t0) * 1000.0
            label = response.get("label")
            if label not in {"front", "back"}:
                raise RuntimeError("predict response label must be front or back")
            predictions[str(row["query_identity"])] = label
            backend_timings = response.get("timings_ms") if isinstance(response.get("timings_ms"), Mapping) else {}
            backend_total = response.get("elapsed_ms", backend_timings.get("total"))
            if backend_total is None:
                raise RuntimeError("predict response omitted elapsed_ms/timings_ms.total")
            samples.append({"image": str(row["query_identity"]), "expected": row.get("expected_orientation"),
                            "label": label, "round_trip_ms": round_trip_ms,
                            "backend_elapsed_ms": float(backend_total) if backend_total is not None else None,
                            "backend_timings_ms": dict(backend_timings),
                            "needs_review": bool(response.get("needs_review", False))})
        report.update({"workpieces": workpieces, "predictions": predictions, "requests": samples,
                       "timings": {"round_trip_ms": summarize([r["round_trip_ms"] for r in samples]),
                                   "backend_elapsed_ms": summarize([r["backend_elapsed_ms"] for r in samples])}})
        expected = {str(row["query_identity"]): str(row.get("expected_orientation")) for row in queries}
        correct = sum(predictions.get(k) == v for k, v in expected.items())
        review_counts = {}
        for row in samples:
            review_counts[row["image"]] = review_counts.get(row["image"], 0) + int(row["needs_review"])
        review_count = sum(int(row["needs_review"]) for row in samples)
        review_rate = float(review_count / len(samples)) if samples else 1.0
        review_by_identity = {image: count > 0 for image, count in review_counts.items()}
        report["needs_review"] = {"count": review_count, "measured_rate": review_rate,
                                   "by_image": review_by_identity, "counts_by_image": review_counts}
        accuracy_value = float(correct / len(expected)) if expected else 0.0
        identity_review_rate = float(sum(review_by_identity.values()) / len(expected)) if expected else 1.0
        report["accuracy"] = {"queries": len(expected), "correct": int(correct), "accuracy": accuracy_value,
                                "review_count": review_count, "review_rate": review_rate,
                                "per_query_review_rate": identity_review_rate}
        backend_summary = report["timings"]["backend_elapsed_ms"]
        report["gates"] = {"gpu_backend_p95_ms": {"actual": backend_summary["p95"], "limit": 25.0,
                                                      "passed": report["edition"] != "gpu" or backend_summary["p95"] <= 25.0},
                         "accuracy_complete": {"actual": len(predictions), "limit": len(expected), "passed": len(predictions) == len(expected)},
                         "correctness": {"actual": int(correct), "limit": len(expected), "passed": correct == len(expected)},
                         "review_rate": {"actual": review_rate, "limit": 0.05, "passed": review_rate <= 0.05}}
        report["passed"] = all(item["passed"] for item in report["gates"].values())
        if compare is not None:
            previous = json.loads(Path(compare).read_text(encoding="utf-8"))
            report["differences"] = compare_predictions(previous.get("predictions", {}), predictions)
            previous_review = (previous.get("needs_review") or {}).get("by_image", {})
            report["needs_review_differences"] = [
                {"image": image, "gpu": previous_review.get(image), "cpu": value}
                for image, value in sorted(review_by_identity.items())
                if previous_review.get(image) != value
            ]
        completed = True
        return report
    finally:
        active_error = sys.exc_info()[1]
        shutdown_error = None
        if client is not None:
            try:
                if process is not None and process.poll() is None:
                    client.request("shutdown", instance_token=(hello or {}).get("instance_token", ""))
            except Exception as exc:
                shutdown_error = exc
            client.close()
        if process is not None:
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait(timeout=10)
            if completed and process.returncode not in (None, 0):
                raise RuntimeError(f"backend exited with code {process.returncode}")
            if process.stdout is not None:
                try:
                    backend_log = process.stdout.read()
                except Exception as exc:
                    backend_log = f"<backend log unavailable: {exc}>"
            else:
                backend_log = ""
            if active_error is not None:
                setattr(active_error, "process_exit", process.returncode)
                setattr(active_error, "backend_log", backend_log)
        if shutdown_error is not None and completed:
            raise RuntimeError(f"backend shutdown failed: {shutdown_error}") from shutdown_error
        shutil.rmtree(temp_parent, ignore_errors=True)


__all__ = ["compare_predictions", "run_benchmark", "summarize"]
