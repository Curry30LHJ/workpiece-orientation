"""Reproducible legacy-versus-fast-geometry acceptance benchmark.

The public process validates the fixed M1/M2/M7 selection and then starts two
short-lived workers.  The legacy worker produces accuracy evidence and exits.
The fast worker is loaded permanently in ``fast_geometry`` mode, produces its
accuracy evidence, performs warmups, records every requested latency sample,
and exits.  Only the parent joins results and evaluates release gates.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

if __package__ in {None, ""}:
    project_import_root = str(Path(__file__).resolve().parents[1])
    if project_import_root not in sys.path:
        sys.path.insert(0, project_import_root)

from scripts.benchmark_adaptive_local_search import (
    RELEASE_CASES,
    CaseSpec,
    _build_case_specs,
    _build_input_fingerprint,
    _collect_environment,
    _ensure_project_import_path,
    _fingerprint_validation_issues,
    _json_default,
    _load_m1_cache,
    _sha256,
)


ENGINE_MODES = ("legacy", "fast_geometry")
EFFECT_DISTANCE_EPSILON = 1e-6


def summarize_latencies(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize every supplied sample; no outlier is removed or trimmed."""
    values = np.asarray(list(samples), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("latency samples must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("latency samples must be finite and non-negative")
    return {
        "samples": int(values.size),
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(np.max(values)),
    }


def evaluate_rule_effects(
    enabled_targets: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require a real non-empty mask and changed embedding for every case/direction."""
    ordered: list[dict[str, str]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for raw in enabled_targets:
        if not isinstance(raw, Mapping):
            raise ValueError("enabled rule target must be an object")
        case = str(raw.get("case", ""))
        direction = str(raw.get("direction", ""))
        workpiece_id = raw.get("workpiece_id")
        if not case or direction not in {"front", "back"}:
            raise ValueError("enabled rule target requires case and direction")
        key = (case, direction, None if workpiece_id is None else str(workpiece_id))
        if key in seen:
            continue
        seen.add(key)
        target = {"case": case, "direction": direction}
        if workpiece_id is not None:
            target["workpiece_id"] = str(workpiece_id)
        ordered.append(target)

    selected: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for target in ordered:
        match = next(
            (
                dict(row)
                for row in evidence_rows
                if row.get("case") == target["case"]
                and row.get("direction") == target["direction"]
                and (
                    "workpiece_id" not in target
                    or row.get("workpiece_id") == target["workpiece_id"]
                )
                and int(row.get("mask_pixels", 0)) > 0
                and float(row.get("embedding_distance", 0.0)) > EFFECT_DISTANCE_EPSILON
            ),
            None,
        )
        if match is None:
            missing.append(dict(target))
        else:
            selected.append(match)
    return {
        "passed": not missing,
        "enabled_rule_targets": ordered,
        "missing_targets": missing,
        "evidence": selected,
        "all_attempts": [dict(row) for row in evidence_rows],
    }


def evaluate_gates(
    *,
    added_errors: int,
    review_rate: float,
    p95_ms: float,
    rule_effects_passed: bool,
    max_added_errors: int = 0,
    max_review_rate: float = 0.05,
    max_p95_ms: float = 25.0,
) -> dict[str, Any]:
    checks = [
        ("added_errors", int(added_errors), int(max_added_errors), int(added_errors) <= int(max_added_errors)),
        ("review_rate", float(review_rate), float(max_review_rate), float(review_rate) <= float(max_review_rate)),
        ("p95_ms", float(p95_ms), float(max_p95_ms), float(p95_ms) <= float(max_p95_ms)),
        ("rule_effects", bool(rule_effects_passed), True, bool(rule_effects_passed)),
    ]
    results = {
        name: {"actual": actual, "limit": limit, "passed": passed}
        for name, actual, limit, passed in checks
    }
    failures = [
        {"name": name, "actual": actual, "limit": limit}
        for name, actual, limit, passed in checks
        if not passed
    ]
    return {"passed": not failures, "results": results, "failures": failures}


def pairwise_accuracy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    legacy_correct = fast_correct = 0
    added: list[str] = []
    changed: list[str] = []
    for row in rows:
        expected = row.get("expected_orientation")
        legacy_label = (row.get("legacy") or {}).get("label")
        fast_label = (row.get("fast") or {}).get("label")
        legacy_ok = legacy_label == expected
        fast_ok = fast_label == expected
        legacy_correct += int(legacy_ok)
        fast_correct += int(fast_ok)
        identity = str(row.get("query_identity"))
        if legacy_label != fast_label:
            changed.append(identity)
        if legacy_ok and not fast_ok:
            added.append(identity)
    total = len(rows)
    return {
        "queries": total,
        "legacy_correct": legacy_correct,
        "legacy_accuracy": float(legacy_correct / total) if total else 0.0,
        "fast_correct": fast_correct,
        "fast_accuracy": float(fast_correct / total) if total else 0.0,
        "changed_predictions": len(changed),
        "changed_prediction_identities": changed,
        "added_errors": len(added),
        "added_error_identities": added,
    }


def _normalize_prediction_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("predictor result must be an object")
    timings = result.get("timings_ms")
    if not isinstance(timings, Mapping):
        timings = {}
    if "total" not in timings:
        if "elapsed_ms" not in result:
            raise ValueError("predictor result must include timings_ms.total or elapsed_ms")
        timings = {**timings, "total": result["elapsed_ms"]}
    cleaned = dict(result)
    cleaned["timings_ms"] = {key: float(value) for key, value in timings.items()}
    return cleaned


def run_fast_measurements(
    queries: Sequence[Mapping[str, Any]],
    predict: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *,
    warmup: int,
    repeats: int,
    minimum_measured_samples: int,
) -> dict[str, Any]:
    """Run deterministic warmup/measurement cycles using a model-free callback."""
    if not queries:
        raise ValueError("at least one query is required")
    if warmup < 0 or repeats <= 0 or minimum_measured_samples <= 0:
        raise ValueError("warmup/repeats/minimum samples are invalid")
    identities = [str(query.get("query_identity", "")) for query in queries]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        raise ValueError("queries require unique non-empty identities")

    for index in range(warmup):
        predict(queries[index % len(queries)])

    measured_count = max(len(queries) * repeats, minimum_measured_samples)
    latency_rows: list[dict[str, Any]] = []
    accuracy_by_identity: dict[str, dict[str, Any]] = {}
    for measurement_index in range(measured_count):
        query = queries[measurement_index % len(queries)]
        identity = identities[measurement_index % len(queries)]
        started = perf_counter()
        prediction = predict(query)
        elapsed_ms = (perf_counter() - started) * 1000.0
        result = _normalize_prediction_result(prediction)
        result["timings_ms"]["total"] = elapsed_ms
        if identity not in accuracy_by_identity:
            accuracy_by_identity[identity] = {**dict(query), "result": result}
        latency_rows.append({
            "query_identity": identity,
            "measurement_index": measurement_index,
            "repeat_index": measurement_index // len(queries),
            "timings_ms": dict(result["timings_ms"]),
        })
    totals = [row["timings_ms"]["total"] for row in latency_rows]
    return {
        "warmup_predictions": int(warmup),
        "accuracy_rows": [accuracy_by_identity[identity] for identity in identities],
        "latency_rows": latency_rows,
        "latency_summary": summarize_latencies(totals),
    }


def _decode_image(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"unreadable image: {path.resolve()}")
    return int(image.shape[1]), int(image.shape[0])


def _inventory_row(
    path: Path,
    *,
    dataset: str,
    expected_orientation: str,
    role: str,
    case: str,
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"missing {role} image: {source}")
    width, height = _decode_image(source)
    digest = _sha256(source)
    result = {
        "case": case,
        "dataset": dataset,
        "role": role,
        "expected_orientation": expected_orientation,
        "image_path": str(source),
        "sha256": digest,
        "width": width,
        "height": height,
    }
    if role == "query":
        result["query_identity"] = hashlib.sha256(
            f"{case}\0{expected_orientation}\0{digest}".encode("utf-8")
        ).hexdigest()
    return result


def audit_case_specs(
    specs: Iterable[CaseSpec],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Decode and content-audit the exact release selection before inference."""
    ordered_specs = list(specs)
    if [spec.name for spec in ordered_specs] != list(RELEASE_CASES):
        raise ValueError("release cases must be exactly M1, M2, M7 in that order")
    inventory: dict[str, Any] = {}
    queries: list[dict[str, Any]] = []
    all_template_hashes: set[str] = set()
    all_query_hashes: set[str] = set()
    for spec in ordered_specs:
        case_inventory: dict[str, Any] = {
            "dataset_path": str(Path(spec.dataset_dir).resolve()),
            "dataset_counts": dict(spec.dataset_counts),
            "cache_id": spec.cache_id,
        }
        case_template_hashes: set[str] = set()
        case_query_hashes: set[str] = set()
        for direction in ("front", "back"):
            templates = [
                _inventory_row(
                    path,
                    dataset=spec.dataset_dir.name,
                    expected_orientation=direction,
                    role="template",
                    case=spec.name,
                )
                for path in spec.templates[direction]
            ]
            selected_queries = [
                _inventory_row(
                    path,
                    dataset=spec.dataset_dir.name,
                    expected_orientation=actual,
                    role="query",
                    case=spec.name,
                )
                for actual, path in spec.queries
                if actual == direction
            ]
            case_inventory[direction] = {
                "templates": templates,
                "queries": selected_queries,
            }
            case_template_hashes.update(row["sha256"] for row in templates)
            case_query_hashes.update(row["sha256"] for row in selected_queries)
        overlap = sorted(case_template_hashes & case_query_hashes)
        if overlap:
            raise ValueError(f"{spec.name} template/query SHA-256 overlap: {len(overlap)}")
        case_inventory["template_query_overlap_count"] = 0
        inventory[spec.name] = case_inventory
        all_template_hashes.update(case_template_hashes)
        for actual, path in spec.queries:
            resolved = str(Path(path).resolve())
            row = next(
                item
                for item in case_inventory[actual]["queries"]
                if item["image_path"] == resolved
            )
            queries.append(dict(row))
            if row["sha256"] in all_query_hashes:
                raise ValueError(f"duplicate query content: {resolved}")
            all_query_hashes.add(row["sha256"])
    global_overlap = sorted(all_template_hashes & all_query_hashes)
    if global_overlap:
        raise ValueError(f"release template/query SHA-256 overlap: {len(global_overlap)}")
    if len({row["query_identity"] for row in queries}) != len(queries):
        raise ValueError("query identities are not unique")
    return inventory, queries


def _shared_input_fingerprint(
    legacy_payload: Mapping[str, Any], fast_payload: Mapping[str, Any]
) -> dict[str, Any]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for mode, payload in (
        ("legacy", legacy_payload),
        ("fast_geometry", fast_payload),
    ):
        fingerprint = payload.get("input_fingerprint")
        issues = _fingerprint_validation_issues(fingerprint)
        if issues:
            raise ValueError(
                f"{mode} input fingerprint is invalid: " + "; ".join(issues)
            )
        fingerprints[mode] = dict(fingerprint)
    legacy = fingerprints["legacy"]
    fast = fingerprints["fast_geometry"]
    if legacy != fast:
        raise ValueError("input fingerprints differ")
    return legacy


def _key_accuracy_rows(payload: Mapping[str, Any], mode: str) -> dict[str, dict[str, Any]]:
    if payload.get("mode") != mode:
        raise ValueError(f"expected {mode} worker payload")
    rows = payload.get("accuracy_rows")
    if not isinstance(rows, list):
        raise ValueError(f"{mode} accuracy_rows must be a list")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{mode} accuracy row must be an object")
        identity = str(raw.get("query_identity", ""))
        if not identity:
            raise ValueError(f"{mode} accuracy row has no query identity")
        if identity in result:
            raise ValueError(f"duplicate {mode} accuracy identity: {identity}")
        result[identity] = dict(raw)
    return result


def _timing_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        timings = row.get("timings_ms")
        if not isinstance(timings, Mapping):
            raise ValueError("latency row timings_ms must be an object")
        for name, value in timings.items():
            stages[str(name)].append(float(value))
    if "total" not in stages:
        raise ValueError("latency rows contain no total timing")
    return {name: summarize_latencies(values) for name, values in sorted(stages.items())}


def _accuracy_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    memberships: dict[str, set[str]] = defaultdict(set)
    by_identity = {str(row["query_identity"]): row for row in rows}
    for row in rows:
        identity = str(row["query_identity"])
        for rule in row.get("applied_rules", []) or []:
            memberships[
                f"interference:{rule.get('direction')}:{rule.get('mode')}:{rule.get('name')}"
            ].add(identity)
        angle = row.get("fitted_angle_deg")
        if angle is not None:
            absolute = abs(float(angle))
            memberships[f"rotation:{'0-15' if absolute < 15 else '15-45' if absolute < 45 else '45+'}"].add(identity)
        offset = row.get("fitted_center_offset_norm")
        if offset is not None:
            value = float(offset)
            memberships[f"offset:{'0-0.05' if value < 0.05 else '0.05-0.15' if value < 0.15 else '0.15+'}"].add(identity)
        max_side = max(int(row.get("width", 0)), int(row.get("height", 0)))
        memberships[f"crop:{'<=256' if max_side <= 256 else '257-512' if max_side <= 512 else '>512'}"].add(identity)
    return {
        name: pairwise_accuracy([by_identity[identity] for identity in sorted(identities)])
        for name, identities in sorted(memberships.items())
    }


def combine_worker_payloads(
    legacy_payload: Mapping[str, Any],
    fast_payload: Mapping[str, Any],
    *,
    max_p95_ms: float = 25.0,
    max_added_errors: int = 0,
    max_review_rate: float = 0.05,
) -> dict[str, Any]:
    shared_fingerprint = _shared_input_fingerprint(legacy_payload, fast_payload)
    legacy = _key_accuracy_rows(legacy_payload, "legacy")
    fast = _key_accuracy_rows(fast_payload, "fast_geometry")
    if set(legacy) != set(fast):
        raise ValueError("legacy and fast query identity sets differ")
    if legacy_payload.get("selection_inventory") != fast_payload.get("selection_inventory"):
        raise ValueError("worker selection inventories differ")

    ordered_identities = [str(row["query_identity"]) for row in fast_payload["accuracy_rows"]]
    accuracy_rows: list[dict[str, Any]] = []
    compared_fields = (
        "case", "dataset", "expected_orientation", "image_path", "sha256", "width", "height"
    )
    for identity in ordered_identities:
        legacy_row = legacy[identity]
        fast_row = fast[identity]
        if any(legacy_row.get(field) != fast_row.get(field) for field in compared_fields):
            raise ValueError(f"worker metadata differs for query identity {identity}")
        merged = {
            key: value
            for key, value in fast_row.items()
            if key != "result"
        }
        merged["legacy"] = dict(legacy_row.get("result") or {})
        merged["fast"] = dict(fast_row.get("result") or {})
        for field in (
            "applied_rules",
            "fitted_angle_deg",
            "fitted_center_x",
            "fitted_center_y",
            "fitted_center_offset_norm",
        ):
            if field in merged["fast"]:
                merged[field] = merged["fast"][field]
        accuracy_rows.append(merged)

    latency_rows = [dict(row) for row in fast_payload.get("latency_rows", [])]
    known = set(fast)
    seen_measurements: set[int] = set()
    for row in latency_rows:
        if row.get("query_identity") not in known:
            raise ValueError("latency row references an unknown query identity")
        index = int(row.get("measurement_index", -1))
        if index < 0 or index in seen_measurements:
            raise ValueError("latency measurement indices must be unique and non-negative")
        seen_measurements.add(index)
    timings = _timing_summaries(latency_rows)
    pairwise = pairwise_accuracy(accuracy_rows)
    review_rows = [row for row in accuracy_rows if bool(row["fast"].get("needs_review"))]
    review_rate = float(len(review_rows) / len(accuracy_rows)) if accuracy_rows else 0.0
    review_reasons = Counter(
        code
        for row in review_rows
        for code in row["fast"].get("review_reason_codes", []) or ["unspecified"]
    )
    rule_effects = evaluate_rule_effects(
        fast_payload.get("enabled_rule_targets", []),
        fast_payload.get("rule_effects", []),
    )
    gates = evaluate_gates(
        added_errors=pairwise["added_errors"],
        review_rate=review_rate,
        p95_ms=float(timings["total"]["p95_ms"]),
        rule_effects_passed=bool(rule_effects["passed"]),
        max_added_errors=max_added_errors,
        max_review_rate=max_review_rate,
        max_p95_ms=max_p95_ms,
    )
    by_case = {
        case: pairwise_accuracy([row for row in accuracy_rows if row.get("case") == case])
        for case in RELEASE_CASES
    }
    return {
        "schema_version": 1,
        "passed": gates["passed"],
        "settings": {
            "max_p95_ms": float(max_p95_ms),
            "max_added_errors": int(max_added_errors),
            "max_review_rate": float(max_review_rate),
        },
        "selection_inventory": fast_payload.get("selection_inventory"),
        "accuracy_rows": accuracy_rows,
        "latency_rows": latency_rows,
        "accuracy": {"overall": pairwise, "by_case": by_case},
        "review": {
            "count": len(review_rows),
            "rate": review_rate,
            "reasons": dict(sorted(review_reasons.items())),
        },
        "timings_ms": timings,
        "groups": _accuracy_groups(accuracy_rows),
        "rule_effects": rule_effects,
        "gates": gates,
        "workers": {
            "legacy": dict(legacy_payload.get("worker") or {}),
            "fast_geometry": dict(fast_payload.get("worker") or {}),
        },
        "fingerprints": {
            "legacy": shared_fingerprint,
            "fast_geometry": shared_fingerprint,
        },
        "environment": fast_payload.get("environment") or legacy_payload.get("environment"),
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _non_negative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return result


def _rate(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


class WorkerProcessError(RuntimeError):
    """A worker failed after optionally publishing structured error evidence."""

    def __init__(self, mode: str, return_code: int, payload: Mapping[str, Any]):
        self.mode = mode
        self.return_code = int(return_code)
        self.payload = dict(payload)
        error = self.payload.get("error")
        detail = error.get("message") if isinstance(error, Mapping) else str(error or "unknown error")
        super().__init__(f"{mode} worker exited with code {self.return_code}: {detail}")


def _worker_command(args: argparse.Namespace, mode: str, output: Path) -> list[str]:
    if mode not in ENGINE_MODES:
        raise ValueError(f"unsupported worker mode: {mode}")
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode", mode,
        "--worker-output", str(output),
        "--project-root", str(Path(args.project_root).resolve()),
        "--model-dir", str(Path(args.model_dir).resolve()),
        "--library-dir", str(Path(args.library_dir).resolve()),
        "--m1-workpiece-id", str(args.m1_workpiece_id),
        "--warmup", str(args.warmup),
        "--repeats", str(args.repeats),
        "--minimum-measured-samples", str(args.minimum_measured_samples),
    ]


def run_isolated_workers(
    args: argparse.Namespace,
    *,
    run_command: Callable[..., Any] = subprocess.run,
) -> dict[str, dict[str, Any]]:
    """Launch sequential, separate processes with an immutable worker mode."""
    payloads: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="fast-geometry-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for mode in ENGINE_MODES:
            output = temporary_root / f"{mode}.json"
            command = _worker_command(args, mode, output)
            completed = run_command(command, check=False)
            return_code = int(getattr(completed, "returncode", 0))
            if output.is_file():
                try:
                    payload = json.loads(output.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {
                        "mode": mode,
                        "error": {
                            "type": type(exc).__name__,
                            "message": f"worker output is not valid JSON: {exc}",
                        },
                    }
            else:
                payload = {
                    "mode": mode,
                    "error": {
                        "type": "WorkerOutputMissing",
                        "message": "worker produced no JSON output",
                    },
                }
            if return_code != 0 or payload.get("error"):
                raise WorkerProcessError(mode, return_code, payload)
            if payload.get("mode") != mode:
                raise ValueError(f"{mode} worker changed its immutable mode")
            payloads[mode] = payload
    return payloads


def _build_worker_caches(
    classifier: Any,
    specs: Sequence[CaseSpec],
    library_dir: Path,
) -> dict[str, tuple[Any, int | None]]:
    caches: dict[str, tuple[Any, int | None]] = {}
    for spec in specs:
        if spec.name == "M1":
            if classifier.inference_mode == "fast_geometry":
                cache, revision = _load_fast_m1_cache(classifier, spec, library_dir)
            else:
                record, cache, _ = _load_m1_cache(classifier, spec, library_dir)
                revision = int(record.revision)
        else:
            revision = 1
            cache = classifier.build_template_cache(
                spec.templates["front"], spec.templates["back"], library_revision=revision
            )
        caches[spec.name] = (cache, revision)
    return caches


def _load_fast_m1_cache(
    classifier: Any,
    spec: CaseSpec,
    library_dir: Path,
) -> tuple[Any, int]:
    """Load only the local-feature-free M1 artifact in a fast worker."""
    from src.orientation_classifier import TemplateCache
    from src.workpiece_library import WorkpieceLibrary

    workpiece_root = Path(library_dir) / spec.cache_id
    record = WorkpieceLibrary._record_from_root(workpiece_root)
    manifest = json.loads((workpiece_root / "manifest.json").read_text(encoding="utf-8"))
    active_revision = manifest.get("geometry_mask_active_revision")
    profile = None
    if active_revision is not None:
        if type(active_revision) is not int or active_revision <= 0:
            raise ValueError(f"invalid M1 active geometry revision: {active_revision!r}")
        revision_path = workpiece_root / "geometry_masks" / "revisions" / f"{active_revision}.json"
        payload = json.loads(revision_path.read_text(encoding="utf-8"))
        profile = payload.get("profile")
        if not isinstance(profile, Mapping):
            raise ValueError(f"M1 geometry profile revision {active_revision} is invalid")

    runtime = classifier.load_fast_runtime_cache(record)
    if (
        runtime is None
        or runtime.library_revision != int(record.revision)
        or runtime.geometry_profile_revision != active_revision
    ):
        runtime = classifier.build_fast_runtime_cache(record, profile)
    cache = TemplateCache(
        global_vectors={
            "front": np.empty((0, 0), dtype=np.float32),
            "back": np.empty((0, 0), dtype=np.float32),
        },
        local_features={"front": [], "back": []},
        geometry_profile=dict(profile) if profile is not None else None,
        geometry_profile_revision=active_revision,
        fast_runtime=runtime,
        fast_template_signature=classifier._fast_template_signature(
            record.front_images, record.back_images
        ),
    )
    return cache, int(record.revision)


def _assert_fast_worker_is_local_feature_free() -> None:
    forbidden_roots = (
        "torch",
        "lightglue",
        "aliked",
        "src.aliked_lightglue_matcher",
        "src.soft_center_matcher",
    )
    forbidden = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in forbidden_roots)
    )
    if forbidden:
        raise RuntimeError(
            "fast worker loaded the local-feature stack: " + ", ".join(forbidden[:10])
        )


class _FastWorkerLocalFeatureViolation(RuntimeError):
    def __init__(self, entry_point: str, counts: Mapping[str, int]):
        self.entry_point = entry_point
        self.local_feature_call_counts = dict(counts)
        super().__init__(
            f"fast worker attempted forbidden {entry_point} processing"
        )


class _FastWorkerLocalGuard:
    """Make every forbidden local-feature entry point executable evidence."""

    def __init__(self, classifier: Any = None):
        self.classifier = classifier
        self._counts = {"ALIKED": 0, "LightGlue": 0, "ORB": 0}
        self._original_extract_features: Any = None
        self._original_score_feature_pair: Any = None
        self._original_orb_create: Any = None
        self._installed = False
        self._classifier_guarded = False

    def _blocker(self, name: str) -> Callable[..., Any]:
        def blocked(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            self._counts[name] += 1
            raise _FastWorkerLocalFeatureViolation(name, self._counts)

        return blocked

    def install(self) -> None:
        _assert_fast_worker_is_local_feature_free()
        if not self._installed:
            self._original_orb_create = cv2.ORB_create
            cv2.ORB_create = self._blocker("ORB")
            self._installed = True
        if self.classifier is not None and not self._classifier_guarded:
            try:
                self.bind_classifier(self.classifier)
            except Exception:
                self.close()
                raise

    def bind_classifier(self, classifier: Any) -> None:
        if not self._installed:
            raise RuntimeError("fast worker ORB guard must be installed before model load")
        _assert_fast_worker_is_local_feature_free()
        for field in ("extractor", "matcher", "device"):
            if getattr(classifier, field, None) is not None:
                self.close()
                raise RuntimeError(
                    f"fast worker classifier {field} must be absent"
                )
        self.classifier = classifier
        self._original_extract_features = classifier._extract_features
        self._original_score_feature_pair = classifier._score_feature_pair
        self._classifier_guarded = True
        try:
            classifier._extract_features = self._blocker("ALIKED")
            classifier._score_feature_pair = self._blocker("LightGlue")
        except Exception:
            self.close()
            raise

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def assert_clean(self) -> None:
        _assert_fast_worker_is_local_feature_free()
        if self.classifier is None or not self._classifier_guarded:
            raise RuntimeError("fast worker classifier guard is missing")
        for field in ("extractor", "matcher", "device"):
            if getattr(self.classifier, field, None) is not None:
                raise RuntimeError(
                    f"fast worker classifier {field} must remain absent"
                )
        attempted = {name: count for name, count in self._counts.items() if count}
        if attempted:
            raise RuntimeError(f"fast worker attempted local-feature processing: {attempted}")

    def close(self) -> None:
        try:
            if self._classifier_guarded and self.classifier is not None:
                self.classifier._extract_features = self._original_extract_features
                self.classifier._score_feature_pair = self._original_score_feature_pair
                self._classifier_guarded = False
        finally:
            if self._installed:
                cv2.ORB_create = self._original_orb_create
                self._installed = False


def _result_geometry_metadata(
    query: Mapping[str, Any], result: Mapping[str, Any], cache: Any
) -> dict[str, Any]:
    applied_rules: list[dict[str, Any]] = []
    fitted_shape: Mapping[str, Any] | None = None
    directions = result.get("directions")
    compiled = getattr(getattr(cache, "fast_runtime", None), "compiled_geometry", None)
    compiled_directions = getattr(compiled, "directions", {}) if compiled is not None else {}
    if isinstance(directions, Mapping):
        for direction in ("front", "back"):
            diagnostic = directions.get(direction)
            configured = compiled_directions.get(direction) if isinstance(compiled_directions, Mapping) else None
            if not isinstance(diagnostic, Mapping) or diagnostic.get("status") != "active":
                continue
            if isinstance(configured, Mapping):
                for rule in configured.get("rules", []):
                    if isinstance(rule, Mapping) and rule.get("enabled", True):
                        applied_rules.append({
                            "direction": direction,
                            "name": str(rule.get("name", rule.get("rule_id", "unnamed"))),
                            "mode": str(rule.get("mode", "unknown")),
                        })
            candidate = diagnostic.get("effective_shape") or diagnostic.get("fitted_shape")
            if not isinstance(candidate, Mapping):
                rules = diagnostic.get("rules")
                if isinstance(rules, Sequence):
                    candidate = next(
                        (
                            rule.get("fitted_shape")
                            for rule in rules
                            if isinstance(rule, Mapping) and isinstance(rule.get("fitted_shape"), Mapping)
                        ),
                        None,
                    )
            if fitted_shape is None and isinstance(candidate, Mapping):
                fitted_shape = candidate
    metadata: dict[str, Any] = {"applied_rules": applied_rules}
    if fitted_shape is not None:
        width = int(query["width"])
        height = int(query["height"])
        scale = min(1.0, 256.0 / max(width, height))
        context_width, context_height = width * scale, height * scale
        cx = float(fitted_shape.get("cx", context_width / 2.0))
        cy = float(fitted_shape.get("cy", context_height / 2.0))
        metadata.update({
            "fitted_angle_deg": float(fitted_shape.get("angle_deg", 0.0)),
            "fitted_center_x": cx,
            "fitted_center_y": cy,
            "fitted_center_offset_norm": float(
                math.hypot(cx - context_width / 2.0, cy - context_height / 2.0)
                / max(math.hypot(context_width, context_height), 1.0)
            ),
        })
    return metadata


def _predict_query(
    classifier: Any,
    caches: Mapping[str, tuple[Any, int | None]],
    query_paths: Mapping[str, Path],
    query: Mapping[str, Any],
) -> dict[str, Any]:
    cache, revision = caches[str(query["case"])]
    result = classifier.predict_with_cache(
        cache, query_paths[str(query["query_identity"])], library_revision=revision
    )
    cleaned = _normalize_prediction_result(result)
    cleaned.update(_result_geometry_metadata(query, cleaned, cache))
    return cleaned


def _legacy_accuracy_rows(
    classifier: Any,
    caches: Mapping[str, tuple[Any, int | None]],
    query_paths: Mapping[str, Path],
    queries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **dict(query),
            "result": _predict_query(classifier, caches, query_paths, query),
        }
        for query in queries
    ]


def _enabled_rule_targets(
    caches: Mapping[str, tuple[Any, int | None]],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for case in RELEASE_CASES:
        cache, _ = caches[case]
        compiled = getattr(getattr(cache, "fast_runtime", None), "compiled_geometry", None)
        directions = getattr(compiled, "directions", {}) if compiled is not None else {}
        for direction in ("front", "back"):
            profile = directions.get(direction) if isinstance(directions, Mapping) else None
            if isinstance(profile, Mapping) and any(
                isinstance(rule, Mapping) and rule.get("enabled", True)
                for rule in profile.get("rules", [])
            ):
                result.append({"case": case, "direction": direction})
    return result


def _rule_effect_rows(
    classifier: Any,
    caches: Mapping[str, tuple[Any, int | None]],
    query_paths: Mapping[str, Path],
    queries: Sequence[Mapping[str, Any]],
    enabled_targets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pending = {
        (str(target.get("case", "")), str(target.get("direction", "")))
        for target in enabled_targets
    }
    evidence: list[dict[str, Any]] = []
    for query in queries:
        if not pending:
            break
        cache, _ = caches[str(query["case"])]
        runtime = getattr(cache, "fast_runtime", None)
        if runtime is None:
            continue
        path = query_paths[str(query["query_identity"])]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable image during rule-effect audit: {path}")
        variants = classifier.fast_engine.geometry.build_variants(
            image, runtime.compiled_geometry
        )
        case = str(query["case"])
        for slot, direction in ((1, "front"), (2, "back")):
            target = (case, direction)
            if target not in pending:
                continue
            diagnostic = variants.directions.get(direction, {})
            ratio = float(diagnostic.get("ignored_ratio", 0.0)) if isinstance(diagnostic, Mapping) else 0.0
            mask_pixels = int(round(ratio * image.shape[0] * image.shape[1]))
            distance = 0.0
            if diagnostic.get("status") == "active" and mask_pixels > 0:
                raw_embedding, masked_embedding = classifier._global_embeddings(
                    [variants.images[0], variants.images[slot]]
                )
                distance = float(
                    np.linalg.norm(
                        np.asarray(raw_embedding, dtype=np.float64)
                        - np.asarray(masked_embedding, dtype=np.float64)
                    )
                )
            row = {
                "case": case,
                "direction": direction,
                "query_identity": query["query_identity"],
                "image_path": query["image_path"],
                "mask_pixels": mask_pixels,
                "mask_ratio": ratio,
                "embedding_distance": distance,
                "geometry_status": diagnostic.get("status", "not_configured"),
            }
            evidence.append(row)
            if mask_pixels > 0 and distance > EFFECT_DISTANCE_EPSILON:
                pending.remove(target)
    return evidence


def _run_worker(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    library_dir = Path(args.library_dir).resolve()
    _ensure_project_import_path(project_root)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not library_dir.is_dir():
        raise FileNotFoundError(f"library directory not found: {library_dir}")
    specs = _build_case_specs(project_root, library_dir, args.m1_workpiece_id)
    inventory, queries = audit_case_specs(specs)
    query_paths = {
        query["query_identity"]: Path(query["image_path"])
        for query in queries
    }

    from src.orientation_classifier import OrientationClassifier

    input_fingerprint = _build_input_fingerprint(
        project_root, model_dir, library_dir, args.m1_workpiece_id, specs
    )
    fixed_mode = str(args.worker_mode)
    guard = _FastWorkerLocalGuard() if fixed_mode == "fast_geometry" else None
    try:
        if guard is not None:
            guard.install()
        classifier = OrientationClassifier.load(
            project_root, model_dir, inference_mode=fixed_mode
        )
        if classifier.inference_mode != fixed_mode:
            raise RuntimeError("classifier changed the immutable worker mode")
        if guard is not None:
            guard.bind_classifier(classifier)
        caches = _build_worker_caches(classifier, specs, library_dir)
        worker = {
            "pid": os.getpid(),
            "mode": fixed_mode,
            "transport": "isolated subprocess",
            "local_feature_call_counts": None,
        }
        common = {
            "mode": fixed_mode,
            "selection_inventory": inventory,
            "input_fingerprint": input_fingerprint,
            "environment": _collect_environment(project_root),
            "worker": worker,
        }
        if fixed_mode == "legacy":
            return {
                **common,
                "accuracy_rows": _legacy_accuracy_rows(
                    classifier, caches, query_paths, queries
                ),
            }

        measured = run_fast_measurements(
            queries,
            lambda query: _predict_query(classifier, caches, query_paths, query),
            warmup=args.warmup,
            repeats=args.repeats,
            minimum_measured_samples=args.minimum_measured_samples,
        )
        enabled = _enabled_rule_targets(caches)
        rule_effects = _rule_effect_rows(
            classifier, caches, query_paths, queries, enabled
        )
        if guard is None:  # Defensive: fixed mode is immutable above.
            raise RuntimeError("fast worker local-feature guard is missing")
        guard.assert_clean()
        worker["local_feature_call_counts"] = guard.counts()
        return {
            **common,
            **measured,
            "enabled_rule_targets": enabled,
            "rule_effects": rule_effects,
        }
    finally:
        if guard is not None:
            guard.close()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _run_parent(args: argparse.Namespace) -> int:
    output = Path(args.output)
    try:
        project_root = Path(args.project_root).resolve()
        specs = _build_case_specs(
            project_root, Path(args.library_dir).resolve(), args.m1_workpiece_id
        )
        audit_case_specs(specs)
        payloads = run_isolated_workers(args)
        report = combine_worker_payloads(
            payloads["legacy"],
            payloads["fast_geometry"],
            max_p95_ms=args.max_p95_ms,
            max_added_errors=args.max_added_errors,
            max_review_rate=args.max_review_rate,
        )
        report["settings"].update({
            "project_root": str(project_root),
            "model_dir": str(Path(args.model_dir).resolve()),
            "library_dir": str(Path(args.library_dir).resolve()),
            "m1_workpiece_id": args.m1_workpiece_id,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "minimum_measured_samples": args.minimum_measured_samples,
            "output": str(output.resolve()),
        })
    except Exception as exc:
        report = {
            "schema_version": 1,
            "passed": False,
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if not key.startswith("worker_")
            },
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "gates": {
                "passed": False,
                "failures": [{"name": "benchmark_execution", "actual": type(exc).__name__, "limit": "success"}],
            },
        }
        if isinstance(exc, WorkerProcessError):
            report["worker_failure"] = {
                "mode": exc.mode,
                "return_code": exc.return_code,
                "payload": exc.payload,
            }
    _write_json(output, report)
    return 0 if report.get("passed") else 2


def _build_parser(*, worker: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark isolated legacy and lightweight geometry orientation inference"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--m1-workpiece-id", required=True)
    parser.add_argument("--warmup", type=_non_negative_int, default=50)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--minimum-measured-samples", type=_positive_int, default=1000)
    if worker:
        parser.add_argument(
            "--worker-mode", choices=ENGINE_MODES, required=True, help=argparse.SUPPRESS
        )
        parser.add_argument(
            "--worker-output", type=Path, required=True, help=argparse.SUPPRESS
        )
    else:
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--max-p95-ms", type=_positive_float, default=25.0)
        parser.add_argument("--max-added-errors", type=_non_negative_int, default=0)
        parser.add_argument("--max-review-rate", type=_rate, default=0.05)
    return parser


def main() -> int:
    worker = any(
        argument == "--worker-mode" or argument.startswith("--worker-mode=")
        for argument in sys.argv[1:]
    )
    args = _build_parser(worker=worker).parse_args()
    if not worker:
        return _run_parent(args)
    try:
        payload = _run_worker(args)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        local_counts = getattr(exc, "local_feature_call_counts", None)
        if isinstance(local_counts, Mapping):
            error["local_feature_call_counts"] = dict(local_counts)
        payload = {
            "mode": args.worker_mode,
            "error": error,
            "environment": _collect_environment(Path(args.project_root)),
        }
        _write_json(args.worker_output, payload)
        return 1
    _write_json(args.worker_output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
