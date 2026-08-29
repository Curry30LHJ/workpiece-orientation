"""Offline benchmark runner for packaged orientation backends.

The statistic and comparison helpers are intentionally independent of the
backend so they can be used by CI and by the acceptance report generator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any
import zipfile

import numpy as np
import cv2
from scripts.benchmark_adaptive_local_search import _canonical_sha256


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


def _read_image(path: Path) -> np.ndarray | None:
    """Read an image even when the Windows path contains non-ASCII text."""

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Client:
    def __init__(self, sock: socket.socket, request_timeout: float | None = None):
        self.sock = sock
        self.request_timeout = request_timeout
        self.file = sock.makefile("rwb")

    def request(self, command: str, *, timeout: float | None = None, **fields: Any) -> dict[str, Any]:
        # ``socket.create_connection`` uses a short connect timeout.  The
        # resulting socket retains that timeout, so reset it for every
        # protocol request; registration and CPU inference may legitimately
        # take longer than the connection handshake.
        request_timeout = self.request_timeout if timeout is None else timeout
        if request_timeout is not None:
            self.sock.settimeout(max(float(request_timeout), 0.001))
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


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ACCEPTANCE_CASES = ("M1", "M2", "M7")


def _strict_sha256(value: object, label: str) -> str:
    """Return a canonical SHA-256 string, rejecting ambiguous representations."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _strict_int(value: object, label: str, *, positive: bool = False) -> int:
    # ``bool`` is an ``int`` subclass, but accepting it in a corpus manifest
    # makes malformed JSON surprisingly easy to miss.
    if type(value) is not int or (value <= 0 if positive else value < 0):
        qualifier = "positive " if positive else "non-negative "
        raise ValueError(f"{label} must be a {qualifier}integer")
    return value


def _normalise_fingerprint_path(value: object, label: str) -> str:
    """Validate and normalize a corpus-relative fingerprint path."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path must be a non-empty string")
    text = value.replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} path must be relative and contained")
    parts = path.parts
    if not parts or parts[0] not in {"data", "runtime_library"}:
        raise ValueError(f"{label} path must start with data/ or runtime_library/")
    return "/".join(parts)


def _is_reparse_point(path: Path) -> bool:
    """Detect symlinks and Windows junctions/reparse points."""

    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        attributes = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        # ctypes defaults to a signed ``c_int`` on Windows; the documented
        # INVALID_FILE_ATTRIBUTES value therefore arrives as ``-1`` rather
        # than ``0xFFFFFFFF`` on some Python builds.
        if attributes in {-1, 0xFFFFFFFF}:
            return False
        return bool(attributes & 0x400)
    except (AttributeError, OSError):
        return False


def _assert_no_reparse_components(path: Path, floor: Path) -> None:
    """Reject a symlink/junction in any component below ``floor``.

    Calling ``resolve`` before this check would hide a junction that points to
    another directory inside the corpus.  The lexical walk therefore happens
    first and is intentionally limited to the corpus subtree, not the drive
    root (which may itself be a mount point on Windows).
    """

    path = Path(path)
    floor = Path(floor)
    try:
        relative = path.relative_to(floor)
    except ValueError as exc:
        raise ValueError(f"path is outside corpus root: {path}") from exc
    current = floor
    for component in relative.parts:
        current = current / component
        if _is_reparse_point(current):
            raise ValueError(f"reparse point is not allowed in acceptance corpus: {current}")


def _source_for_fingerprint(project_root: Path, relative: str, *, case: str,
                            direction: str) -> Path:
    """Resolve a fingerprint path and enforce its dataset/direction partition."""

    parts = relative.split("/")
    orientation_dir = "0" if direction == "front" else "1"
    if parts[0] == "data":
        if len(parts) < 4 or parts[1] != f"1_{case}" or parts[2] != orientation_dir:
            raise ValueError(f"fingerprint path is outside {case}/{direction} dataset: {relative}")
    elif len(parts) < 3 or parts[-2] != orientation_dir:
        raise ValueError(f"fingerprint path has invalid direction partition: {relative}")
    lexical_source = project_root / Path(*parts)
    _assert_no_reparse_components(lexical_source, project_root)
    source = lexical_source.resolve()
    if project_root.resolve() not in source.parents:
        raise ValueError(f"fingerprint path escapes corpus root: {relative}")
    return source


def _acceptance_rows(spec_path: Path, temp_root: Path) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    payload = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1 or payload.get("passed") is not True:
        raise ValueError("acceptance spec schema/passed gate is invalid")
    inventory = payload.get("selection_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(_ACCEPTANCE_CASES):
        raise ValueError("acceptance spec must contain exactly M1, M2, and M7")
    fingerprints = ((payload.get("fingerprints") or {}).get("fast_geometry")
                    if isinstance(payload.get("fingerprints"), Mapping) else None)
    if not isinstance(fingerprints, Mapping):
        raise ValueError("acceptance spec is missing fast_geometry fingerprints")
    selections_fp = fingerprints.get("selections")
    if not isinstance(selections_fp, Mapping) or set(selections_fp) != set(_ACCEPTANCE_CASES):
        raise ValueError("acceptance spec is missing fast_geometry selections")
    overall_fp = _strict_sha256(fingerprints.get("overall_sha256"), "fast_geometry overall fingerprint")
    templates: dict[str, list[Path]] = {}
    queries: list[dict[str, Any]] = []
    template_hashes: set[str] = set()
    query_hashes: set[str] = set()
    overlap_by_case: dict[str, int] = {}
    template_by_case: dict[str, set[str]] = {}
    query_by_case: dict[str, set[str]] = {}
    query_path_by_hash: dict[str, Path] = {}

    for case in _ACCEPTANCE_CASES:
        details = inventory.get(case)
        if not isinstance(details, Mapping):
            raise ValueError(f"acceptance case {case} details must be an object")
        expected_dataset = f"1_{case}"
        dataset_text = details.get("dataset_path")
        if not isinstance(dataset_text, str) or not dataset_text.strip():
            raise ValueError(f"acceptance case {case} has no dataset fingerprint/path")
        dataset_root = Path(dataset_text)
        if not dataset_root.is_absolute() or dataset_root.name != expected_dataset or not dataset_root.is_dir():
            raise ValueError(f"acceptance case {case} dataset_path is invalid")
        project_root = dataset_root.parent.parent
        _assert_no_reparse_components(dataset_root, project_root.parent)
        dataset_root = dataset_root.resolve()
        if dataset_root.parent.name.lower() != "data":
            raise ValueError(f"acceptance case {case} dataset_path must be under data/")
        project_root = dataset_root.parent.parent.resolve()
        case_fp = selections_fp.get(case)
        if not isinstance(case_fp, Mapping):
            raise ValueError(f"acceptance case {case} has no aggregate fingerprint")
        case_aggregate = _strict_sha256(case_fp.get("aggregate_sha256"), f"acceptance case {case} aggregate fingerprint")
        case_body = {k: v for k, v in case_fp.items() if k != "aggregate_sha256"}
        if _canonical_sha256(case_body) != case_aggregate:
            raise ValueError(f"acceptance case {case} aggregate fingerprint mismatch")
        template_by_case[case] = set()
        query_by_case[case] = set()
        for direction in ("front", "back"):
            section = details.get(direction)
            if not isinstance(section, Mapping):
                raise ValueError(f"acceptance case {case} missing {direction} section")
            fp_section = case_fp.get(direction)
            if not isinstance(fp_section, Mapping):
                raise ValueError(f"acceptance fingerprint missing {case}/{direction}")
            for role in ("templates", "queries"):
                rows = section.get(role)
                fp_rows = fp_section.get(role)
                if not isinstance(rows, list) or not isinstance(fp_rows, list):
                    raise ValueError(f"acceptance case {case}/{direction} {role} must be a list")
                if len(rows) != len(fp_rows):
                    raise ValueError(f"acceptance fingerprint row count mismatch: {case}/{direction}/{role}")
                fp_identities: list[tuple[str, int, str]] = []
                for fp_index, item in enumerate(fp_rows):
                    if not isinstance(item, Mapping):
                        raise ValueError(f"acceptance fingerprint row is malformed: {case}/{direction}/{role}/{fp_index}")
                    rel = _normalise_fingerprint_path(item.get("path"), f"{case}/{direction}/{role}/{fp_index}")
                    size = _strict_int(item.get("size"), f"{case}/{direction}/{role}/{fp_index} size")
                    digest = _strict_sha256(item.get("sha256"), f"{case}/{direction}/{role}/{fp_index} sha256")
                    _source_for_fingerprint(project_root, rel, case=case, direction=direction)
                    # Keep the canonical spelling for resolution; compare
                    # identities case-insensitively because Windows paths are
                    # case-insensitive, while the hash remains exact.
                    fp_identities.append((rel, size, digest))
                if len({(rel.lower(), size, digest) for rel, size, digest in fp_identities}) != len(fp_identities):
                    raise ValueError(f"duplicate acceptance fingerprint rows: {case}/{direction}/{role}")
                inventory_identities: list[tuple[str, int, str]] = []
                for index, row in enumerate(rows):
                    label = f"{case}/{direction}/{role}/{index}"
                    if not isinstance(row, Mapping) or not isinstance(row.get("image_path"), str) or not row.get("image_path").strip():
                        raise ValueError(f"acceptance inventory row is malformed: {label}")
                    if row.get("case") != case or row.get("dataset") != expected_dataset or row.get("role") != ("template" if role == "templates" else "query") or row.get("expected_orientation") != direction:
                        raise ValueError(f"acceptance inventory row metadata mismatch: {label}")
                    digest = _strict_sha256(row.get("sha256"), f"{label} sha256")
                    width = _strict_int(row.get("width"), f"{label} width", positive=True)
                    height = _strict_int(row.get("height"), f"{label} height", positive=True)
                    source_input = Path(row["image_path"])
                    if not source_input.is_absolute() or not source_input.is_file():
                        raise FileNotFoundError(source_input)
                    _assert_no_reparse_components(source_input, project_root)
                    source = source_input.resolve()
                    # Resolve the source to the fingerprint path, rather than
                    # trusting a suffix/filename heuristic.  This binds the
                    # acceptance run to the exact corpus selected for it.
                    rel_candidates = []
                    for rel, size, fp_digest in fp_identities:
                        expected_source = _source_for_fingerprint(project_root, rel, case=case, direction=direction)
                        if expected_source == source:
                            rel_candidates.append((rel, size, fp_digest))
                    if len(rel_candidates) != 1:
                        raise ValueError(f"acceptance inventory/fingerprint path mismatch: {source_input}")
                    rel, expected_size, expected_digest = rel_candidates[0]
                    actual_size = source.stat().st_size
                    declared_size = actual_size
                    if "size" in row:
                        declared_size = _strict_int(row.get("size"), f"{label} size")
                    if actual_size != expected_size or actual_size != declared_size:
                        raise ValueError(f"acceptance source size mismatch: {source}")
                    actual_digest = _sha256(source)
                    if actual_digest != digest or actual_digest != expected_digest:
                        raise ValueError(f"acceptance source hash mismatch: {source}")
                    image = _read_image(source)
                    if image is None or image.shape[1] != width or image.shape[0] != height:
                        raise ValueError(f"acceptance image dimensions mismatch: {source}")
                    identity = (rel.lower(), actual_size, actual_digest)
                    inventory_identities.append(identity)
                    destination = temp_root / "data" / "benchmark" / case / direction / f"{index:04d}_{source.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    if role == "templates":
                        templates.setdefault(f"{case}:{direction}", []).append(destination)
                        template_hashes.add(actual_digest)
                        template_by_case[case].add(actual_digest)
                    else:
                        query_identity = row.get("query_identity")
                        if not isinstance(query_identity, str) or not query_identity.strip():
                            raise ValueError(f"{label} query_identity must be a non-empty string")
                        if actual_digest in query_path_by_hash:
                            raise ValueError(f"duplicate query SHA-256: {source}")
                        query_path_by_hash[actual_digest] = source
                        queries.append({**dict(row), "case": case, "direction": direction,
                                        "image_path": str(destination), "query_identity": query_identity})
                        query_hashes.add(actual_digest)
                        query_by_case[case].add(actual_digest)
                if Counter(inventory_identities) != Counter(
                    (rel.lower(), size, digest) for rel, size, digest in fp_identities
                ):
                    raise ValueError(f"acceptance inventory/fingerprint multiset mismatch: {case}/{direction}/{role}")
        overlap_by_case[case] = len(template_by_case[case] & query_by_case[case])

    if not templates or not queries:
        raise ValueError("acceptance spec contains no usable templates and queries")
    if template_hashes & query_hashes:
        raise ValueError("acceptance template/query SHA-256 overlap")
    for case, count in overlap_by_case.items():
        expected_overlap = _strict_int(selections_fp[case].get("template_query_overlap_count"), f"acceptance {case} overlap count")
        if expected_overlap != count:
            raise ValueError(f"acceptance overlap count mismatch for {case}")
    body = {k: v for k, v in fingerprints.items() if k != "overall_sha256"}
    if _canonical_sha256(body) != overall_fp:
        raise ValueError("acceptance overall fingerprint mismatch")
    identities = [str(row["query_identity"]) for row in queries]
    if len(set(identities)) != len(identities):
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
    log_handle = None
    hello: Mapping[str, Any] | None = None
    completed = False
    lifecycle: dict[str, Any] = {
        "process_spawned": False,
        "connection_established": False,
        "shutdown_requested": False,
        "shutdown_confirmed": False,
        "forced_termination": False,
        "process_exit": None,
    }
    report: dict[str, Any] = {"package_zip": str(package_zip), "package_sha256": _sha256(package_zip),
                              "warmup": warmup, "iterations": iterations, "startup": {},
                              "lifecycle": lifecycle}
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
        if not (package_root / "app_config.json").is_file():
            children = [p for p in package_root.iterdir() if p.is_dir()]
            if len(children) == 1 and (children[0] / "app_config.json").is_file():
                package_root = children[0]
            else:
                raise ValueError("ZIP must contain app_config.json at root or one top-level directory")
        config = json.loads((package_root / "app_config.json").read_text(encoding="utf-8"))
        _validate_config(package_root, config)
        configured_request_timeout = config.get("request_timeout_ms", 120000)
        if (isinstance(configured_request_timeout, bool)
                or not isinstance(configured_request_timeout, (int, float))
                or not math.isfinite(float(configured_request_timeout))
                or configured_request_timeout <= 0):
            raise ValueError("package config request_timeout_ms must be a positive number")
        request_timeout_seconds = float(configured_request_timeout) / 1000.0
        environment = _environment(config)
        version_metadata: dict[str, Any] = {}
        version_path = package_root / "version.json"
        if version_path.is_file():
            try:
                metadata = json.loads(version_path.read_text(encoding="utf-8"))
                if isinstance(metadata, Mapping):
                    version_metadata = {str(key): value for key, value in metadata.items()}
            except (OSError, ValueError):
                # Package audit reports malformed metadata.  Keep benchmark
                # reports backward-compatible for older packages that lack it.
                version_metadata = {}
        report.update({"edition": config.get("edition"), "version": config.get("package_version"),
                       "versions": {"package": config.get("package_version"), **{
                           key: version_metadata[key] for key in
                           ("python", "paddle", "paddleclas", "pyinstaller", "qt")
                           if key in version_metadata
                       }},
                       "package_metadata": version_metadata,
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
        log_handle = (temp_parent / "backend.log").open("w+", encoding="utf-8")
        process = subprocess.Popen(args, cwd=str(package_root), stdout=log_handle, stderr=subprocess.STDOUT,
                                   text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lifecycle["process_spawned"] = True
        report["startup"]["process_spawn_ms"] = (perf_counter() - started) * 1000.0
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("backend did not accept a connection")
                client = _Client(socket.create_connection(("127.0.0.1", port), timeout=min(5.0, remaining)),
                                 request_timeout=request_timeout_seconds)
                lifecycle["connection_established"] = True
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError(f"backend exited with code {process.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("backend did not accept a connection")
                time.sleep(0.1)
        report["startup"]["connection_ms"] = (perf_counter() - started) * 1000.0
        phases: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("backend did not become ready")
            hello = client.request("hello", timeout=min(request_timeout_seconds, remaining))
            phases.append({"phase": hello.get("phase"), "status": hello.get("status"), "elapsed_ms": (perf_counter() - started) * 1000.0})
            if hello.get("ready") is True:
                break
            if hello.get("status") != "loading":
                raise RuntimeError(f"backend startup failed: {hello}")
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        else:
            raise TimeoutError("backend did not become ready")
        report["startup"]["phases"] = phases
        report["startup"]["ready_ms"] = (perf_counter() - started) * 1000.0
        deadline = time.monotonic() + timeout_seconds
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
        def request_with_deadline(command: str, **fields: Any) -> dict[str, Any]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"benchmark overall timeout exceeded before {command}")
            return client.request(command, timeout=min(request_timeout_seconds, remaining), **fields)

        workpieces: dict[str, str] = {}
        for case in sorted({str(row["case"]) for row in queries}):
            front = templates.get(f"{case}:front", [])
            back = templates.get(f"{case}:back", [])
            response = request_with_deadline("register", name=f"benchmark-{case}", replace=False,
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
            if time.monotonic() >= deadline:
                raise TimeoutError("benchmark overall timeout exceeded during warmup")
            row = queries[index % len(queries)]
            request_with_deadline("predict", workpiece_id=workpieces[str(row["case"])], image_path=row["image_path"])
        samples: list[dict[str, Any]] = []
        predictions: dict[str, str] = {}
        correct_samples = 0
        for index in range(iterations):
            if time.monotonic() >= deadline:
                raise TimeoutError("benchmark overall timeout exceeded")
            row = queries[index % len(queries)]
            expected_label = row.get("expected_orientation")
            if expected_label not in {"front", "back"}:
                raise ValueError("acceptance query expected_orientation must be front or back")
            t0 = perf_counter()
            response = request_with_deadline("predict", workpiece_id=workpieces[str(row["case"])], image_path=row["image_path"])
            round_trip_ms = (perf_counter() - t0) * 1000.0
            label = response.get("label")
            if label not in {"front", "back"}:
                raise RuntimeError("predict response label must be front or back")
            predictions[str(row["query_identity"])] = label
            correct_samples += int(label == expected_label)
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
        sample_accuracy = float(correct_samples / len(samples)) if samples else 0.0
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
                                "correct_samples": correct_samples, "sample_accuracy": sample_accuracy,
                                "review_count": review_count, "review_rate": review_rate,
                                "per_query_review_rate": identity_review_rate}
        backend_summary = report["timings"]["backend_elapsed_ms"]
        report["gates"] = {"gpu_backend_p95_ms": {"actual": backend_summary["p95"], "limit": 25.0,
                                                      "passed": report["edition"] != "gpu" or backend_summary["p95"] <= 25.0},
                         "accuracy_complete": {"actual": len(predictions), "limit": len(expected), "passed": len(predictions) == len(expected)},
                         "correctness": {"actual": sample_accuracy, "limit": 1.0, "passed": sample_accuracy == 1.0},
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
        close_error = None
        if client is not None:
            try:
                if process is not None and process.poll() is None:
                    lifecycle["shutdown_requested"] = True
                    client.request("shutdown", timeout=5.0,
                                   instance_token=(hello or {}).get("instance_token", ""))
                    lifecycle["shutdown_confirmed"] = True
            except Exception as exc:
                shutdown_error = exc
            try:
                client.close()
            except Exception as exc:
                close_error = exc
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired as exc:
                lifecycle["forced_termination"] = True
                try:
                    process.kill()
                    process.wait(timeout=10)
                except Exception as kill_error:
                    if active_error is None:
                        active_error = kill_error
                if shutdown_error is None:
                    shutdown_error = exc
            except Exception as exc:
                if shutdown_error is None:
                    shutdown_error = exc
            lifecycle["process_exit"] = process.returncode
        backend_log = ""
        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.seek(0)
                backend_log = log_handle.read()
            except Exception as exc:
                backend_log = f"<backend log unavailable: {exc}>"
            finally:
                try:
                    log_handle.close()
                except Exception as exc:
                    if close_error is None:
                        close_error = exc
        report["lifecycle"] = dict(lifecycle)
        report["backend_log"] = backend_log
        if process is not None:
            report["process_exit"] = process.returncode
        if active_error is not None:
            setattr(active_error, "process_exit", lifecycle.get("process_exit"))
            setattr(active_error, "backend_log", backend_log)
            setattr(active_error, "lifecycle", dict(lifecycle))
        final_error: BaseException | None = None
        if completed:
            if process is not None and process.returncode != 0:
                final_error = RuntimeError(f"backend exited with code {process.returncode}")
            elif not lifecycle["shutdown_confirmed"]:
                cause = shutdown_error or close_error
                final_error = RuntimeError(
                    f"backend shutdown failed: {cause}" if cause else "backend shutdown was not confirmed"
                )
        elif active_error is None and shutdown_error is not None and process is not None and lifecycle["forced_termination"]:
            # Preserve a useful timeout/termination error when startup failed
            # without a primary exception.
            final_error = RuntimeError(f"backend termination failed: {shutdown_error}")
        shutil.rmtree(temp_parent, ignore_errors=True)
        if final_error is not None and active_error is None:
            setattr(final_error, "process_exit", lifecycle.get("process_exit"))
            setattr(final_error, "backend_log", backend_log)
            setattr(final_error, "lifecycle", dict(lifecycle))
            raise final_error from shutdown_error


__all__ = ["compare_predictions", "run_benchmark", "summarize"]
