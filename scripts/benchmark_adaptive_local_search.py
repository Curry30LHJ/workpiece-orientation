"""Reproducible adaptive-versus-exhaustive LightGlue acceptance benchmark.

The public invocation starts two isolated worker processes.  Each worker fixes
``local_search_mode`` when it loads the classifier and never changes it while
processing M1, M2, and M7.  The parent compares rows by identity and emits both
machine-readable JSON and a concise release-gate report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


M1_WORKPIECE_ID = "31f082d1a04e486b9345846f4d585033"
RELEASE_CASES = ("M1", "M2", "M7")
DATASET_LABELS = {"0": "front", "1": "back"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
TEMPLATE_COUNT = 20
QUERY_COUNT = 20
SEED = 20260825


@dataclass(frozen=True)
class CaseSpec:
    name: str
    dataset_dir: Path
    templates: dict[str, tuple[Path, ...]]
    queries: tuple[tuple[str, Path], ...]
    dataset_counts: dict[str, int]
    cache_id: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _images(label_dir: Path) -> list[Path]:
    if not label_dir.is_dir():
        raise FileNotFoundError(f"missing dataset label directory: {label_dir}")
    paths = sorted(
        (
            path
            for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if not paths:
        raise ValueError(f"dataset label directory has no images: {label_dir}")
    return paths


def _locate_dataset(project_root: Path, case_name: str) -> Path:
    data_root = Path(project_root) / "data"
    number = case_name.removeprefix("M")
    direct_candidates = [data_root / f"1_M{number}", data_root / f"M{number}"]
    for candidate in direct_candidates:
        if candidate.is_dir():
            return candidate
    matches = sorted(
        (
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.upper().endswith(f"M{number}")
        ),
        key=lambda path: path.name.casefold(),
    ) if data_root.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"unable to locate {case_name} below {data_root}; expected 1_M{number}"
        )
    raise ValueError(
        f"ambiguous {case_name} datasets below {data_root}: "
        + ", ".join(path.name for path in matches)
    )


def _validate_case(spec: CaseSpec) -> None:
    template_hashes = {
        _sha256(path)
        for paths in spec.templates.values()
        for path in paths
    }
    query_hashes = [_sha256(path) for _, path in spec.queries]
    if len(spec.queries) != 2 * QUERY_COUNT:
        raise ValueError(
            f"{spec.name} requires exactly {2 * QUERY_COUNT} held-out images; "
            f"found {len(spec.queries)}"
        )
    if len(query_hashes) != len(set(query_hashes)):
        raise ValueError(f"{spec.name} held-out queries are not content-distinct")
    overlap = sorted(set(query_hashes) & template_hashes)
    if overlap:
        raise ValueError(
            f"{spec.name} held-out queries overlap templates by {len(overlap)} hashes"
        )
    counts = Counter(label for label, _ in spec.queries)
    if counts != Counter({"front": QUERY_COUNT, "back": QUERY_COUNT}):
        raise ValueError(
            f"{spec.name} requires {QUERY_COUNT} queries per orientation; "
            f"found {dict(counts)}"
        )


def _select_m1_case(
    project_root: Path,
    library_dir: Path,
    workpiece_id: str,
) -> CaseSpec:
    dataset_dir = _locate_dataset(project_root, "M1")
    workpiece_root = Path(library_dir) / workpiece_id
    if not workpiece_root.is_dir():
        raise FileNotFoundError(f"M1 workpiece library not found: {workpiece_id}")
    manifest_path = workpiece_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"unable to read M1 workpiece manifest: {manifest_path}") from exc
    if manifest.get("id") != workpiece_id:
        raise ValueError(f"M1 manifest id does not match {workpiece_id}")

    templates: dict[str, tuple[Path, ...]] = {}
    queries: list[tuple[str, Path]] = []
    dataset_counts: dict[str, int] = {}
    for disk_label, label in DATASET_LABELS.items():
        template_paths = tuple(_images(workpiece_root / disk_label))
        if len(template_paths) != 28:
            raise ValueError(
                f"M1 {label} cache must contain 28 templates; found {len(template_paths)}"
            )
        template_hashes = {_sha256(path) for path in template_paths}
        if len(template_hashes) != 28:
            raise ValueError(f"M1 {label} templates are not content-distinct")
        dataset_paths = _images(dataset_dir / disk_label)
        dataset_counts[label] = len(dataset_paths)
        remaining = [path for path in dataset_paths if _sha256(path) not in template_hashes]
        if len(remaining) < QUERY_COUNT:
            raise ValueError(
                f"M1 {label} has only {len(remaining)} non-template images; "
                f"requires {QUERY_COUNT}"
            )
        templates[label] = template_paths
        queries.extend((label, path) for path in remaining[:QUERY_COUNT])

    spec = CaseSpec(
        name="M1",
        dataset_dir=dataset_dir,
        templates=templates,
        queries=tuple(queries),
        dataset_counts=dataset_counts,
        cache_id=workpiece_id,
    )
    _validate_case(spec)
    return spec


def _select_in_memory_case(project_root: Path, case_name: str) -> CaseSpec:
    from src.shitu_baseline import split_labels

    dataset_dir = _locate_dataset(project_root, case_name)
    split_templates, held_out = split_labels(
        dataset_dir, template_count=TEMPLATE_COUNT, seed=SEED
    )
    templates: dict[str, tuple[Path, ...]] = {}
    queries: list[tuple[str, Path]] = []
    dataset_counts: dict[str, int] = {}
    for disk_label, label in DATASET_LABELS.items():
        dataset_counts[label] = len(_images(dataset_dir / disk_label))
        templates[label] = tuple(split_templates[disk_label])
        available_queries = held_out[disk_label]
        if len(available_queries) < QUERY_COUNT:
            raise ValueError(
                f"{case_name} {label} has only {len(available_queries)} held-out images; "
                f"requires {QUERY_COUNT} after {TEMPLATE_COUNT} templates"
            )
        queries.extend((label, path) for path in available_queries[:QUERY_COUNT])
    spec = CaseSpec(
        name=case_name,
        dataset_dir=dataset_dir,
        templates=templates,
        queries=tuple(queries),
        dataset_counts=dataset_counts,
        cache_id=f"in-memory:{case_name}:seed-{SEED}",
    )
    _validate_case(spec)
    return spec


def _build_case_specs(
    project_root: Path,
    library_dir: Path,
    m1_workpiece_id: str,
) -> list[CaseSpec]:
    specs = [
        _select_m1_case(project_root, library_dir, m1_workpiece_id),
        _select_in_memory_case(project_root, "M2"),
        _select_in_memory_case(project_root, "M7"),
    ]
    if [spec.name for spec in specs] != list(RELEASE_CASES):
        raise AssertionError("benchmark case order changed")
    return specs


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _nvidia_identity() -> tuple[str | None, str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        first_line = next(
            (line.strip() for line in completed.stdout.splitlines() if line.strip()), ""
        )
        if not first_line:
            return None, None
        name, separator, driver = first_line.rpartition(",")
        if not separator:
            return first_line, None
        return name.strip(), driver.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None


def _git_identity(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unavailable", "git_dirty": None, "git_dirty_tracked": []}
    dirty_rows = [line for line in status.splitlines() if line.strip()]
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty_rows),
        "git_dirty_tracked": dirty_rows,
    }


def _collect_environment(project_root: Path) -> dict[str, Any]:
    gpu_name, gpu_driver = _nvidia_identity()
    paddle_cuda = "unavailable"
    paddle = sys.modules.get("paddle")
    if paddle is not None:
        paddle_version = str(paddle.__version__)
        try:
            paddle_cuda = str(paddle.version.cuda() or "none")
        except Exception:
            paddle_cuda = "unknown"
        if gpu_name is None:
            try:
                gpu_name = str(paddle.device.cuda.get_device_name(0))
            except Exception:
                pass
    else:
        paddle_version = _package_version("paddlepaddle-gpu")
        if paddle_version == "not-installed":
            paddle_version = _package_version("paddlepaddle")

    torch_cuda = "unavailable"
    torch = sys.modules.get("torch")
    if torch is not None:
        torch_version = str(torch.__version__)
        torch_cuda = str(torch.version.cuda or "none")
        if gpu_name is None and torch.cuda.is_available():
            gpu_name = str(torch.cuda.get_device_name(0))
    else:
        torch_version = _package_version("torch")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "paddle": paddle_version,
        "paddle_cuda": paddle_cuda,
        "torch": torch_version,
        "torch_cuda": torch_cuda,
        "numpy": str(np.__version__),
        "gpu_name": gpu_name or "unavailable",
        "gpu_driver": gpu_driver or "unavailable",
    }
    result.update(_git_identity(project_root))
    return result


def _load_m1_cache(classifier: Any, spec: CaseSpec, library_dir: Path) -> tuple[Any, Any, dict[str, int]]:
    from src.workpiece_library import WorkpieceLibrary

    workpiece_root = Path(library_dir) / spec.cache_id
    record = WorkpieceLibrary._record_from_root(workpiece_root)
    cache = classifier.load_template_cache(record)
    if cache is None:
        raise RuntimeError(
            f"M1 persisted template cache is missing or invalid for {spec.cache_id}"
        )
    counts = {
        label: len(cache.local_features.get(label, []))
        for label in ("front", "back")
    }
    if counts != {"front": 28, "back": 28}:
        raise ValueError(f"M1 persisted cache must be 28+28; found {counts}")
    classifier.set_template_cache(record.id, cache)

    manifest = json.loads((workpiece_root / "manifest.json").read_text(encoding="utf-8"))
    active_revision = manifest.get("geometry_mask_active_revision")
    effective = cache
    if active_revision is not None:
        if type(active_revision) is not int or active_revision <= 0:
            raise ValueError(f"invalid M1 active geometry revision: {active_revision!r}")
        revision_path = workpiece_root / "geometry_masks" / "revisions" / f"{active_revision}.json"
        try:
            profile_payload = json.loads(revision_path.read_text(encoding="utf-8"))
            profile = profile_payload["profile"]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"unable to load M1 active geometry profile revision {active_revision}"
            ) from exc
        effective, _ = classifier.prepare_geometry_cache(
            record.id,
            record,
            profile,
            getattr(classifier, "geometry_calibrator", None),
            None,
            base_cache=cache,
        )
    return record, effective, counts


def _build_caches(
    classifier: Any,
    specs: Iterable[CaseSpec],
    library_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    caches: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if spec.name == "M1":
            record, cache, base_counts = _load_m1_cache(classifier, spec, library_dir)
            library_revision = int(record.revision)
            template_counts = base_counts
        else:
            cache = classifier.build_template_cache(
                spec.templates["front"], spec.templates["back"]
            )
            library_revision = None
            template_counts = {
                label: len(spec.templates[label]) for label in ("front", "back")
            }
        caches[spec.name] = (cache, library_revision)
        metadata[spec.name] = {
            "dataset": spec.dataset_dir.name,
            "dataset_counts": dict(spec.dataset_counts),
            "query_counts": {"front": QUERY_COUNT, "back": QUERY_COUNT},
            "template_counts": template_counts,
            "effective_template_counts": {
                label: len(cache.local_features[label]) for label in ("front", "back")
            },
            "cache_id": spec.cache_id,
        }
    return caches, metadata


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _predict_row(
    classifier: Any,
    cache: Any,
    library_revision: int | None,
    spec: CaseSpec,
    actual: str,
    query_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    captured: list[Any] = []
    original_search = classifier._search_local

    def capture_search(*args: Any, **kwargs: Any) -> Any:
        search = original_search(*args, **kwargs)
        captured.append(search)
        return search

    classifier._search_local = capture_search
    wall_started = time.perf_counter()
    try:
        result = classifier.predict_with_cache(
            cache,
            query_path,
            library_revision=library_revision,
        )
    finally:
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        classifier._search_local = original_search
    if len(captured) != 1:
        raise RuntimeError(
            f"{spec.name} {query_path.name} captured {len(captured)} local searches; expected 1"
        )
    search = captured[0]
    geometry_mask = result.get("geometry_mask", {})
    timings = dict(geometry_mask.get("timings_ms", {})) if isinstance(geometry_mask, dict) else {}
    timings["local_matching"] = float(search.matching_ms)
    return {
        "case": spec.name,
        "image_path": _project_relative(query_path, project_root),
        "actual": actual,
        "label": result["label"],
        "needs_review": bool(result["needs_review"]),
        "elapsed_ms": float(result["elapsed_ms"]),
        "wall_ms": float(wall_ms),
        "timings_ms": {key: float(value) for key, value in timings.items()},
        "local_search": result["local_search"],
        "evidence": {
            "global_scores": result["global_scores"],
            "global_margin": result["global_margin"],
            "local_scores": result["local_scores"],
            "local_margin": result["local_margin"],
            "decision_source": result["decision_source"],
            "trace": search.trace,
        },
    }


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    process_started = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    library_dir = Path(args.library_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not library_dir.is_dir():
        raise FileNotFoundError(f"library directory not found: {library_dir}")
    if args.warmup < 0:
        raise ValueError("warmup must be zero or greater")

    from src.orientation_classifier import OrientationClassifier

    specs = _build_case_specs(project_root, library_dir, args.m1_workpiece_id)
    classifier = OrientationClassifier.load(
        project_root,
        model_dir,
        local_search_mode=args.worker_mode,
    )
    if classifier.local_search_mode != args.worker_mode:
        raise RuntimeError("classifier changed the fixed worker local-search mode")
    caches, case_metadata = _build_caches(classifier, specs, library_dir)

    warmup_actual = 0
    for spec in specs:
        cache, revision = caches[spec.name]
        for _, query_path in spec.queries[: args.warmup]:
            classifier.predict_with_cache(cache, query_path, library_revision=revision)
            warmup_actual += 1

    rows: list[dict[str, Any]] = []
    for spec in specs:
        cache, revision = caches[spec.name]
        for actual, query_path in spec.queries:
            rows.append(
                _predict_row(
                    classifier,
                    cache,
                    revision,
                    spec,
                    actual,
                    query_path,
                    project_root,
                )
            )

    return {
        "mode": args.worker_mode,
        "rows": rows,
        "environment": _collect_environment(project_root),
        "benchmark": {
            "warmup_per_case": int(args.warmup),
            "warmup_predictions": warmup_actual,
            "worker_pid": os.getpid(),
            "worker_port": None,
            "worker_transport": "direct isolated subprocess",
            "worker_total_wall_ms": (time.perf_counter() - process_started) * 1000.0,
            "model_id": model_dir.name,
            "m1_workpiece_id": args.m1_workpiece_id,
            "cases": case_metadata,
        },
    }


def _key_rows(payload: dict[str, Any], expected_mode: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    if payload.get("mode") != expected_mode:
        raise ValueError(
            f"expected {expected_mode} worker payload, got {payload.get('mode')!r}"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{expected_mode} worker payload rows must be a list")
    keyed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row["case"]), str(row["image_path"]), str(row["actual"]))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid {expected_mode} worker row") from exc
        if key in keyed:
            raise ValueError(f"duplicate {expected_mode} row: {key}")
        keyed[key] = row
    return keyed


def _latency(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "total": 0.0}
    return {
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "total": float(np.sum(data)),
    }


def _mode_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages = Counter(str(row["local_search"].get("stage", "unknown")) for row in rows)
    stage_counts = {
        stage: int(stages.get(stage, 0)) for stage in ("top5", "top10", "full")
    }
    for stage, count in sorted(stages.items()):
        if stage not in stage_counts:
            stage_counts[stage] = int(count)
    matched_counts = [
        sum(int(value) for value in row["local_search"]["matched_counts"].values())
        for row in rows
    ]
    correct = sum(str(row["label"]) == str(row["actual"]) for row in rows)
    return {
        "images": len(rows),
        "correct": correct,
        "accuracy": float(correct / len(rows)) if rows else 0.0,
        "needs_review": sum(bool(row["needs_review"]) for row in rows),
        "elapsed_ms": _latency([float(row["elapsed_ms"]) for row in rows]),
        "wall_ms": _latency([float(row.get("wall_ms", row["elapsed_ms"])) for row in rows]),
        "local_matching_ms": _latency(
            [float(row.get("timings_ms", {}).get("local_matching", 0.0)) for row in rows]
        ),
        "stage_counts": stage_counts,
        "average_matched_count": float(np.mean(matched_counts)) if matched_counts else 0.0,
    }


def _compare_worker_payloads(exhaustive: dict, adaptive: dict) -> dict:
    exhaustive_rows = _key_rows(exhaustive, "exhaustive")
    adaptive_rows = _key_rows(adaptive, "adaptive")
    if set(exhaustive_rows) != set(adaptive_rows):
        missing_adaptive = sorted(set(exhaustive_rows) - set(adaptive_rows))
        missing_exhaustive = sorted(set(adaptive_rows) - set(exhaustive_rows))
        raise ValueError(
            "worker row keys differ: "
            f"missing adaptive={missing_adaptive}, missing exhaustive={missing_exhaustive}"
        )

    label_mismatches: list[dict[str, Any]] = []
    review_mismatches: list[dict[str, Any]] = []
    ordered_keys = sorted(exhaustive_rows)
    for key in ordered_keys:
        exhaustive_row = exhaustive_rows[key]
        adaptive_row = adaptive_rows[key]
        mismatch = {
            "case": key[0],
            "image_path": key[1],
            "actual": key[2],
            "exhaustive": exhaustive_row,
            "adaptive": adaptive_row,
        }
        if exhaustive_row.get("label") != adaptive_row.get("label"):
            label_mismatches.append(mismatch)
        if bool(exhaustive_row.get("needs_review")) != bool(
            adaptive_row.get("needs_review")
        ):
            review_mismatches.append(mismatch)

    cases = sorted({key[0] for key in ordered_keys}, key=lambda item: RELEASE_CASES.index(item) if item in RELEASE_CASES else len(RELEASE_CASES))
    case_statistics: dict[str, Any] = {}
    for case in cases:
        case_statistics[case] = {}
        for mode, keyed in (("exhaustive", exhaustive_rows), ("adaptive", adaptive_rows)):
            case_statistics[case][mode] = _mode_statistics(
                [keyed[key] for key in ordered_keys if key[0] == case]
            )

    exhaustive_all = _mode_statistics([exhaustive_rows[key] for key in ordered_keys])
    adaptive_all = _mode_statistics([adaptive_rows[key] for key in ordered_keys])
    exhaustive_wall = exhaustive_all["wall_ms"]["total"]
    adaptive_wall = adaptive_all["wall_ms"]["total"]
    exhaustive_worker_total = float(
        exhaustive.get("benchmark", {}).get("worker_total_wall_ms", exhaustive_wall)
    )
    adaptive_worker_total = float(
        adaptive.get("benchmark", {}).get("worker_total_wall_ms", adaptive_wall)
    )
    gate_passed = not label_mismatches and not review_mismatches
    missing_release_cases = sorted(set(RELEASE_CASES) - set(cases))
    release_gate_passed = gate_passed and not missing_release_cases
    return {
        "gate_passed": gate_passed,
        "release_gate_passed": release_gate_passed,
        "default_local_search_mode": "adaptive" if release_gate_passed else "exhaustive",
        "missing_release_cases": missing_release_cases,
        "label_mismatches": label_mismatches,
        "review_mismatches": review_mismatches,
        "case_statistics": case_statistics,
        "overall_statistics": {
            "exhaustive": exhaustive_all,
            "adaptive": adaptive_all,
        },
        "performance": {
            "wall_speedup": float(exhaustive_wall / adaptive_wall) if adaptive_wall else None,
            "worker_total_speedup": (
                float(exhaustive_worker_total / adaptive_worker_total)
                if adaptive_worker_total else None
            ),
            "worker_total_wall_ms": {
                "exhaustive": exhaustive_worker_total,
                "adaptive": adaptive_worker_total,
            },
        },
        "environment": adaptive.get("environment", {}),
        "worker_environments": {
            "exhaustive": exhaustive.get("environment", {}),
            "adaptive": adaptive.get("environment", {}),
        },
        "benchmark": adaptive.get("benchmark", {}),
        "workers": {
            "exhaustive": exhaustive.get("benchmark", {}),
            "adaptive": adaptive.get("benchmark", {}),
        },
    }


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "不可用"
    return f"{float(value):.{digits}f}"


def _render_markdown(report: dict) -> str:
    environment = report.get("environment", {})
    benchmark = report.get("benchmark", {})
    label_ok = not report.get("label_mismatches")
    review_ok = not report.get("review_mismatches")
    release_ok = bool(report.get("release_gate_passed"))
    lines = [
        "# Adaptive LightGlue Candidate Search 验收结果",
        "",
        f"- 逐图标签一致：{'通过' if label_ok else '失败'}",
        f"- 逐图复检状态一致：{'通过' if review_ok else '失败'}",
        f"- M1/M2/M7 发布门禁：{'通过' if release_ok else '失败'}",
        f"- 正式默认模式：`{report.get('default_local_search_mode', 'exhaustive')}`",
        "",
        "## 环境与可复现信息",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 时间戳（UTC） | {environment.get('timestamp_utc', 'unavailable')} |",
        f"| 平台 | {environment.get('platform', 'unavailable')} |",
        f"| Python | {environment.get('python', 'unavailable')} |",
        f"| GPU | {environment.get('gpu_name', 'unavailable')} |",
        f"| GPU 驱动 | {environment.get('gpu_driver', 'unavailable')} |",
        f"| Paddle / CUDA | {environment.get('paddle', 'unavailable')} / {environment.get('paddle_cuda', 'unavailable')} |",
        f"| Torch / CUDA | {environment.get('torch', 'unavailable')} / {environment.get('torch_cuda', 'unavailable')} |",
        f"| NumPy | {environment.get('numpy', 'unavailable')} |",
        f"| Git 提交 | `{environment.get('git_commit', 'unavailable')}` |",
        f"| 工作树 | {'dirty' if environment.get('git_dirty') else 'clean'} |",
        f"| 模型 ID | `{benchmark.get('model_id', 'unavailable')}` |",
        f"| M1 工件库 ID | `{benchmark.get('m1_workpiece_id', 'unavailable')}` |",
        f"| 每数据集 warmup | {benchmark.get('warmup_per_case', 'unavailable')} |",
        "",
        "### 隔离进程",
        "",
        "| 模式 | PID | 端口 | 传输 | 总耗时（ms） |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for mode in ("exhaustive", "adaptive"):
        worker = report.get("workers", {}).get(mode, {})
        port = worker.get("worker_port")
        lines.append(
            f"| {mode} | {worker.get('worker_pid', 'unavailable')} | "
            f"{'N/A' if port is None else port} | "
            f"{worker.get('worker_transport', 'unavailable')} | "
            f"{_number(worker.get('worker_total_wall_ms'))} |"
        )

    lines.extend([
        "",
        "## 固定验收集",
        "",
        "| 数据集 | 原始图片（正/反） | 查询（正/反） | 模板（正/反） | 缓存 ID |",
        "| --- | --- | --- | --- | --- |",
    ])
    for case in RELEASE_CASES:
        item = benchmark.get("cases", {}).get(case, {})
        dataset = item.get("dataset_counts", {})
        queries = item.get("query_counts", {})
        templates = item.get("template_counts", {})
        lines.append(
            f"| {case} | {dataset.get('front', 0)}/{dataset.get('back', 0)} | "
            f"{queries.get('front', 0)}/{queries.get('back', 0)} | "
            f"{templates.get('front', 0)}/{templates.get('back', 0)} | "
            f"`{item.get('cache_id', 'unavailable')}` |"
        )

    lines.extend([
        "",
        "## 每数据集结果",
        "",
        "| 数据集 | 模式 | 图片 | 准确率 | 复检 | wall P50/P95（ms） | local_matching P50/P95（ms） | top5/top10/full | 平均匹配数 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ])
    for case, case_stats in report.get("case_statistics", {}).items():
        for mode in ("exhaustive", "adaptive"):
            stats = case_stats[mode]
            stage = stats["stage_counts"]
            lines.append(
                f"| {case} | {mode} | {stats['images']} | {stats['accuracy']:.4f} | "
                f"{stats['needs_review']} | {_number(stats['wall_ms']['p50'])}/{_number(stats['wall_ms']['p95'])} | "
                f"{_number(stats['local_matching_ms']['p50'])}/{_number(stats['local_matching_ms']['p95'])} | "
                f"{stage.get('top5', 0)}/{stage.get('top10', 0)}/{stage.get('full', 0)} | "
                f"{stats['average_matched_count']:.2f} |"
            )

    performance = report.get("performance", {})
    lines.extend([
        "",
        "## 总体性能",
        "",
        f"- 查询 wall time 加速比（exhaustive/adaptive）：{_number(performance.get('wall_speedup'), 3)}x",
        f"- 含模型加载、建库与 warmup 的进程总时长加速比：{_number(performance.get('worker_total_speedup'), 3)}x",
        f"- 标签不一致：{len(report.get('label_mismatches', []))} 张",
        f"- 复检状态不一致：{len(report.get('review_mismatches', []))} 张",
    ])
    missing = report.get("missing_release_cases", [])
    if missing:
        lines.append(f"- 缺少发布数据集：{', '.join(missing)}")
    if report.get("label_mismatches") or report.get("review_mismatches"):
        lines.extend([
            "",
            "## 差异说明",
            "",
            "完整逐图证据（含候选排序 trace）保存在同名 JSON 报告中。",
        ])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _run_parent(args: argparse.Namespace) -> int:
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    report: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(prefix="adaptive-lightglue-benchmark-") as temporary:
            temporary_root = Path(temporary)
            payloads: dict[str, dict[str, Any]] = {}
            for worker_mode in ("exhaustive", "adaptive"):
                worker_output = temporary_root / f"{worker_mode}.json"
                worker_command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-mode", worker_mode,
                    "--project-root", str(args.project_root),
                    "--model-dir", str(args.model_dir),
                    "--library-dir", str(args.library_dir),
                    "--m1-workpiece-id", args.m1_workpiece_id,
                    "--warmup", str(args.warmup),
                    "--worker-output", str(worker_output),
                ]
                subprocess.run(worker_command, check=True)
                payloads[worker_mode] = json.loads(worker_output.read_text(encoding="utf-8"))
            report = _compare_worker_payloads(
                payloads["exhaustive"], payloads["adaptive"]
            )
    except Exception as exc:
        report = {
            "gate_passed": False,
            "release_gate_passed": False,
            "default_local_search_mode": "exhaustive",
            "missing_release_cases": list(RELEASE_CASES),
            "label_mismatches": [],
            "review_mismatches": [],
            "case_statistics": {},
            "performance": {},
            "environment": _collect_environment(Path(args.project_root)),
            "benchmark": {
                "model_id": Path(args.model_dir).name,
                "m1_workpiece_id": args.m1_workpiece_id,
                "warmup_per_case": args.warmup,
                "cases": {},
            },
            "workers": {},
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    _write_json(output_json, report)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(report)
    if report.get("error"):
        markdown += (
            "\n## 执行错误\n\n"
            f"`{report['error']['type']}: {report['error']['message']}`\n"
        )
    output_markdown.write_text(markdown, encoding="utf-8")
    return 0 if report.get("release_gate_passed") else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare adaptive and exhaustive LightGlue candidate search"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--m1-workpiece-id", default=M1_WORKPIECE_ID)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--worker-mode", choices=("adaptive", "exhaustive"))
    parser.add_argument("--worker-output", type=Path)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.worker_mode is not None:
        if args.worker_output is None:
            raise SystemExit("--worker-output is required in worker mode")
        try:
            payload = _run_worker(args)
        except Exception as exc:
            payload = {
                "mode": args.worker_mode,
                "rows": [],
                "environment": _collect_environment(Path(args.project_root)),
                "benchmark": {
                    "worker_pid": os.getpid(),
                    "worker_port": None,
                    "worker_transport": "direct isolated subprocess",
                    "model_id": Path(args.model_dir).name,
                    "m1_workpiece_id": args.m1_workpiece_id,
                    "warmup_per_case": args.warmup,
                    "cases": {},
                },
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            _write_json(args.worker_output, payload)
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        _write_json(args.worker_output, payload)
        return 0
    if args.output_json is None or args.output_markdown is None:
        raise SystemExit("--output-json and --output-markdown are required")
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
