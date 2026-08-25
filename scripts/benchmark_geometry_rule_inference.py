"""Benchmark baseline and geometry-rule query paths on one recovered workpiece.

The serial v2 path exists only in this benchmark: production always keeps the
single PP-ShiTu batch containing the front-processed and back-processed query.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry_mask_profiles import GeometryMaskProfiles
from src.geometry_profile_schema import materialize_runtime_profile
from src.orientation_classifier import OrientationClassifier, TemplateCache
from src.workpiece_catalog import WorkpieceCatalog
from src.workpiece_library import WorkpieceLibrary


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "mean_ms": float(np.mean(array)),
    }


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row})
    return {
        key: _summary([float(row.get(key, 0.0)) for row in rows])
        for key in keys
    }


def _baseline_cache(cache: TemplateCache) -> TemplateCache:
    raw_global = cache.raw_global_vectors or cache.global_vectors
    raw_local = cache.raw_local_features or cache.local_features
    return replace(
        cache,
        global_vectors=raw_global,
        local_features=raw_local,
        raw_global_vectors=raw_global,
        raw_local_features=raw_local,
        geometry_profile=None,
        geometry_profile_revision=None,
        geometry_template_report=None,
        geometry_template_indices=None,
        geometry_unsafe=False,
    )


def _v1_cache(cache: TemplateCache) -> TemplateCache:
    if cache.geometry_profile is None:
        raise RuntimeError("active geometry profile is required for v1/v2 comparison")
    return replace(cache, geometry_profile=materialize_runtime_profile(cache.geometry_profile))


def _benchmark_mode(
    classifier: OrientationClassifier,
    cache: TemplateCache,
    queries: list[Path],
    *,
    mode: str,
    warmup: int,
    repeats: int,
    embedding_strategy: str,
) -> dict[str, Any]:
    original_embeddings = classifier._global_embeddings
    predictor_batch_sizes: list[int] = []

    if embedding_strategy == "serial":
        def measured_embeddings(images):
            predictor_batch_sizes.extend([1] * len(images))
            return [original_embeddings([image])[0] for image in images]
    else:
        def measured_embeddings(images):
            predictor_batch_sizes.append(len(images))
            return original_embeddings(images)

    classifier._global_embeddings = measured_embeddings
    elapsed_values: list[float] = []
    timing_rows: list[dict[str, Any]] = []
    labels: dict[str, int] = {}
    try:
        for index in range(warmup):
            classifier.predict_with_cache(cache, queries[index % len(queries)])
        predictor_batch_sizes.clear()
        for _ in range(repeats):
            for query in queries:
                started = time.perf_counter()
                result = classifier.predict_with_cache(cache, query)
                elapsed_values.append((time.perf_counter() - started) * 1000.0)
                label = str(result.get("label", "unknown"))
                labels[label] = labels.get(label, 0) + 1
                geometry = result.get("geometry_mask", {})
                timings = geometry.get("timings_ms", {}) if isinstance(geometry, dict) else {}
                timing_rows.append(dict(timings) if isinstance(timings, dict) else {})
    finally:
        classifier._global_embeddings = original_embeddings

    result = {
        "mode": mode,
        "samples": len(elapsed_values),
        "warmup": warmup,
        "repeats": repeats,
        **_summary(elapsed_values),
        "timings_ms": _timing_summary(timing_rows),
        "predictor_batch_sizes": predictor_batch_sizes,
        "labels": labels,
    }
    return result


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark geometry-rule inference modes")
    parser.add_argument("--workpiece-id", required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--library-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be >= 0 and repeats must be > 0")

    project_root = args.project_root.resolve()
    model_dir = (args.model_dir or project_root / "third_party" / "models" / "shiru_rec"
                 / "general_PPLCNetV2_base_pretrained_v1.0_infer").resolve()
    library_dir = (args.library_dir or project_root / "runtime_library").resolve()
    query_dir = args.query_dir.resolve()
    queries = sorted(
        path for path in query_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not queries:
        raise SystemExit(f"no query images found: {query_dir}")

    classifier = OrientationClassifier.load(project_root, model_dir)
    library = WorkpieceLibrary(library_dir)
    catalog = WorkpieceCatalog(library, classifier)
    profiles = GeometryMaskProfiles(
        catalog,
        getattr(classifier, "geometry_calibrator", None),
        storage_dir=library_dir / ".geometry-mask-jobs",
        start_worker=False,
    )
    catalog.set_geometry_profiles(profiles)
    try:
        catalog.recover()
        snapshot = catalog.capture_snapshot(args.workpiece_id)
        cache = snapshot.cache
        if cache.geometry_profile is None:
            raise SystemExit(
                "the selected workpiece has no published active geometry profile; "
                "publish a complete front/back v2 profile before benchmarking"
            )

        modes: list[tuple[str, TemplateCache, str]] = [
            ("baseline", _baseline_cache(cache), "batch"),
            ("v1_active", _v1_cache(cache), "batch"),
            ("v2_serial_reference", cache, "serial"),
            ("v2_batch", cache, "batch"),
        ]
        results = [
            _benchmark_mode(
                classifier, mode_cache, queries,
                mode=name, warmup=args.warmup, repeats=args.repeats,
                embedding_strategy=strategy,
            )
            for name, mode_cache, strategy in modes
        ]
        payload = {
            "workpiece_id": args.workpiece_id,
            "library_revision": snapshot.record.revision,
            "template_counts": {
                "front": len(snapshot.record.front_images),
                "back": len(snapshot.record.back_images),
            },
            "query_count": len(queries),
            "queries": [str(path) for path in queries],
            "environment": {
                "platform": platform.platform(),
                "python": sys.version,
                "paddle": _version("paddle"),
                "torch": _version("torch"),
                "numpy": np.__version__,
            },
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        profiles.shutdown()


if __name__ == "__main__":
    main()
