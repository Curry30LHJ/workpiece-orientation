"""Compare native PP-ShiTu recognition output with the Python baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def read_f32_matrix(path: Path, rows: int, columns: int) -> np.ndarray:
    """Read an exact row-major float32 matrix from *path*."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    values = np.fromfile(Path(path), dtype=np.float32)
    expected = rows * columns
    if values.size != expected:
        raise ValueError(
            f"expected {expected} float32 values, found {values.size}"
        )
    return values.reshape(rows, columns)


def compare_embeddings(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    min_cosine: float = 0.9999,
    max_abs_error: float = 0.001,
) -> dict[str, object]:
    """Return row-wise cosine and element-wise error parity metrics."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[0] == 0
        or reference.shape[1] == 0
    ):
        raise ValueError(
            "embedding matrices must be non-empty and have identical 2-D shapes"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("embedding matrices must contain only finite values")

    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    if np.any(reference_norm == 0.0) or np.any(candidate_norm == 0.0):
        raise ValueError("embedding rows must have non-zero norm")

    cosines = np.sum(reference * candidate, axis=1) / (
        reference_norm * candidate_norm
    )
    minimum_cosine = float(np.min(cosines))
    maximum_absolute_error = float(np.max(np.abs(reference - candidate)))
    return {
        "rows": int(reference.shape[0]),
        "columns": int(reference.shape[1]),
        "minimum_cosine": minimum_cosine,
        "maximum_absolute_error": maximum_absolute_error,
        "minimum_cosine_required": float(min_cosine),
        "maximum_absolute_error_allowed": float(max_abs_error),
        "passed": (
            minimum_cosine >= min_cosine
            and maximum_absolute_error <= max_abs_error
        ),
    }


def project_pipeline_p95(
    *,
    python_pipeline_p95_ms: float,
    python_feature_p95_ms: float,
    native_feature_p95_ms: float,
) -> float:
    """Replace the measured Python feature stage with its native equivalent."""
    values = (
        python_pipeline_p95_ms,
        python_feature_p95_ms,
        native_feature_p95_ms,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("latencies must be finite and non-negative")
    if python_feature_p95_ms > python_pipeline_p95_ms:
        raise ValueError("Python feature P95 cannot exceed pipeline P95")
    return (
        float(python_pipeline_p95_ms)
        - float(python_feature_p95_ms)
        + float(native_feature_p95_ms)
    )


def evaluate_integration_gate(
    *,
    embedding_metrics: Mapping[str, object],
    labels_match: bool,
    review_decisions_match: bool,
    python_feature_p95_ms: float,
    native_feature_p95_ms: float,
    python_pipeline_p95_ms: float,
) -> dict[str, object]:
    """Evaluate the approved correctness and 15% performance gates."""
    if python_feature_p95_ms <= 0.0 or python_pipeline_p95_ms <= 0.0:
        raise ValueError("Python latency baselines must be positive")

    projected_pipeline_p95_ms = project_pipeline_p95(
        python_pipeline_p95_ms=python_pipeline_p95_ms,
        python_feature_p95_ms=python_feature_p95_ms,
        native_feature_p95_ms=native_feature_p95_ms,
    )
    native_feature_improvement_ratio = (
        python_feature_p95_ms - native_feature_p95_ms
    ) / python_feature_p95_ms
    projected_pipeline_improvement_ratio = (
        python_pipeline_p95_ms - projected_pipeline_p95_ms
    ) / python_pipeline_p95_ms
    embedding_gate_passed = bool(embedding_metrics.get("passed", False))
    feature_gate_passed = native_feature_improvement_ratio >= 0.15
    projected_pipeline_gate_passed = projected_pipeline_improvement_ratio >= 0.15

    return {
        "embedding_metrics": dict(embedding_metrics),
        "labels_match": bool(labels_match),
        "review_decisions_match": bool(review_decisions_match),
        "python_feature_p95_ms": float(python_feature_p95_ms),
        "native_feature_p95_ms": float(native_feature_p95_ms),
        "python_pipeline_p95_ms": float(python_pipeline_p95_ms),
        "projected_pipeline_p95_ms": projected_pipeline_p95_ms,
        "native_feature_improvement_ratio": native_feature_improvement_ratio,
        "projected_pipeline_improvement_ratio": (
            projected_pipeline_improvement_ratio
        ),
        "embedding_gate_passed": embedding_gate_passed,
        "feature_gate_passed": feature_gate_passed,
        "projected_pipeline_gate_passed": projected_pipeline_gate_passed,
        "passed": (
            embedding_gate_passed
            and bool(labels_match)
            and bool(review_decisions_match)
            and feature_gate_passed
            and projected_pipeline_gate_passed
        ),
    }
