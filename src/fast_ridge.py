"""Deterministic, per-workpiece linear Ridge orientation head."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


RIDGE_REGULARIZATION_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True)
class RidgeHead:
    weights: np.ndarray
    bias: float
    regularization: float
    review_threshold: float
    feature_dim: int
    training_summary: dict[str, Any]


@dataclass(frozen=True)
class RidgeDecision:
    label: str
    margin: float
    needs_review: bool


def pack_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """L2-normalize each embedding and concatenate them in supplied order."""
    packed = []
    for embedding in embeddings:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not np.isfinite(norm):
            raise ValueError("embedding features must be finite")
        packed.append(vector / norm if norm > 0 else vector)
    return np.concatenate(packed).astype(np.float32, copy=False) if packed else np.empty(0, np.float32)


def _as_features(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.isfinite(result).all():
        raise ValueError("features must be finite")
    return result


def _fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray, regularization: float) -> tuple[np.ndarray, float]:
    mean_x = np.sum(weight[:, None] * x, axis=0)
    mean_y = float(np.sum(weight * y))
    xw = (x - mean_x) * np.sqrt(weight[:, None])
    yw = (y - mean_y) * np.sqrt(weight)
    alpha = np.linalg.solve(xw @ xw.T + regularization * np.eye(len(x)), yw)
    weights = xw.T @ alpha
    bias = mean_y - float(mean_x @ weights)
    scale = max(float(np.linalg.norm(weights)), 1e-12)
    return weights / scale, bias / scale


def _metadata(values: Sequence[Any] | None, count: int, default: Any) -> list[Any]:
    if values is None:
        return [default(index) if callable(default) else default for index in range(count)]
    if len(values) != count:
        raise ValueError("metadata length must match feature rows")
    return list(values)


def fit_ridge_head(
    front_features: np.ndarray,
    back_features: np.ndarray,
    *,
    front_source_ids: Sequence[int] | None = None,
    back_source_ids: Sequence[int] | None = None,
    front_original_rows: Sequence[bool] | None = None,
    back_original_rows: Sequence[bool] | None = None,
    regularization_grid: Sequence[float] = RIDGE_REGULARIZATION_GRID,
) -> RidgeHead:
    front = _as_features(front_features, "front_features")
    back = _as_features(back_features, "back_features")
    if front.shape[1] != back.shape[1]:
        raise ValueError("front and back feature dimensions must match")
    nf, nb = len(front), len(back)
    fs = _metadata(front_source_ids, nf, lambda i: ("front", i))
    bs = _metadata(back_source_ids, nb, lambda i: ("back", i))
    fo = _metadata(front_original_rows, nf, True)
    bo = _metadata(back_original_rows, nb, True)

    x = np.vstack((front, back))
    y = np.concatenate((np.ones(nf), -np.ones(nb)))
    weights = np.concatenate((np.full(nf, 0.5 / nf), np.full(nb, 0.5 / nb)))
    sources = fs + bs
    originals = [bool(v) for v in fo + bo]
    classes = ["front"] * nf + ["back"] * nb
    groups = []
    for index, (source, original) in enumerate(zip(sources, originals)):
        if original:
            key = (classes[index], source)
            if key not in groups:
                groups.append(key)

    candidates = []
    for regularization in regularization_grid:
        regularization = float(regularization)
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization values must be positive and finite")
        margins: list[tuple[float, float]] = []
        for class_name, source in groups:
            held = np.array([c == class_name and s == source for c, s in zip(classes, sources)])
            held_original = held & np.array(originals)
            train = ~np.array([c == class_name and s == source for c, s in zip(classes, sources)])
            if len(set(y[train])) < 2:
                continue
            fold_y = y[train]
            fold_weights = np.empty(len(fold_y), dtype=np.float64)
            for target in (1.0, -1.0):
                target_rows = fold_y == target
                fold_weights[target_rows] = 0.5 / np.count_nonzero(target_rows)
            w, b = _fit(x[train], fold_y, fold_weights, regularization)
            for row, target in zip(x[held_original], y[held_original]):
                margins.append((float(row @ w + b), float(target)))
        if margins:
            errors = sum((margin >= 0) != (target >= 0) for margin, target in margins)
            correct = [abs(margin) for margin, target in margins if (margin >= 0) == (target >= 0)]
            minimum = min(correct) if correct else 0.0
            threshold = float(np.clip(0.5 * minimum, 0.01, 0.25))
            review_count = sum(abs(margin) < threshold for margin, _ in margins)
            candidates.append((errors, review_count, -minimum, regularization, margins, True))
        else:
            candidates.append((0, 0, 0.0, regularization, [], False))
    if not candidates:
        raise ValueError("regularization_grid must not be empty")
    usable_candidates = [candidate for candidate in candidates if candidate[5]]
    selected = min(usable_candidates, key=lambda candidate: candidate[:4]) if usable_candidates else None
    regularization = selected[3] if selected is not None else 1.0
    final_weights, final_bias = _fit(x, y, weights, regularization)
    valid_margins = selected[4] if selected is not None else []
    if valid_margins:
        correct = [abs(margin) for margin, target in valid_margins if (margin >= 0) == (target >= 0)]
        review_threshold = float(np.clip(0.5 * min(correct), 0.01, 0.25)) if correct else 0.25
        validation_status = "validated" if selected[0] == 0 else "cross_validation_failed"
    else:
        training_margins = x @ final_weights + final_bias
        review_threshold = float(np.clip(0.25 * min(abs(training_margins)), 0.01, 0.25))
        validation_status = "few_shot_unverified"
    summary = {
        "class_counts": {"front": nf, "back": nb},
        "validation_folds": len(groups),
        "validation_grouping": "leave_one_source_out",
        "validation_status": validation_status,
    }
    return RidgeHead(final_weights.astype(np.float32), float(final_bias), regularization, review_threshold, x.shape[1], summary)


def predict_ridge(head: RidgeHead, feature: np.ndarray) -> RidgeDecision:
    vector = np.asarray(feature, dtype=np.float64).reshape(-1)
    if vector.size != head.feature_dim or not np.isfinite(vector).all():
        raise ValueError("feature must be finite and match the head feature dimension")
    margin = float(vector @ head.weights.astype(np.float64) + head.bias)
    return RidgeDecision("front" if margin >= 0 else "back", margin, abs(margin) < head.review_threshold)
