"""Native-coordinate interference rectangles and local-feature filtering."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

import cv2
import numpy as np


class InvalidMaskError(ValueError):
    """Raised when an operator-supplied rectangle is malformed."""


def build_active_mask_map(front_count: int, back_count: int, groups: Sequence[dict]) -> dict[str, list[list[dict[str, float]]]]:
    """Collect only active, enabled annotation regions by template."""
    masks: dict[str, list[list[dict[str, float]]]] = {
        "front": [[] for _ in range(front_count)],
        "back": [[] for _ in range(back_count)],
    }
    for group in groups:
        if not group.get("enabled", True):
            continue
        if group.get("propagation", {}).get("state") != "active":
            continue
        for annotation in group.get("annotations", []):
            if annotation.get("status", "active") != "active":
                continue
            orientation = annotation.get("orientation")
            index = annotation.get("index")
            if orientation not in masks or type(index) is not int or not 0 <= index < len(masks[orientation]):
                continue
            for region in annotation.get("regions", []):
                validate_region(region)
                masks[orientation][index].append(deepcopy(region))
    return masks


def _rect_corners(region: dict[str, float]) -> np.ndarray:
    validate_region(region)
    x = float(region["x"])
    y = float(region["y"])
    width = float(region["width"])
    height = float(region["height"])
    if width <= 0 or height <= 0:
        raise InvalidMaskError("mask rectangle must have positive width and height")
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32)


def validate_region(region: dict[str, float]) -> None:
    if not isinstance(region, dict) or any(key not in region for key in ("x", "y", "width", "height")):
        raise InvalidMaskError("mask rectangle requires x, y, width and height")
    try:
        values = [float(region[key]) for key in ("x", "y", "width", "height")]
    except (TypeError, ValueError) as exc:
        raise InvalidMaskError("mask rectangle coordinates must be numeric") from exc
    if not np.all(np.isfinite(values)) or values[2] <= 0 or values[3] <= 0:
        raise InvalidMaskError("mask rectangle must be finite and positive")


def _inside(points: np.ndarray, region: dict[str, float]) -> np.ndarray:
    x0 = float(region["x"])
    y0 = float(region["y"])
    x1 = x0 + float(region["width"])
    y1 = y0 + float(region["height"])
    return (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)


def _select_axis(value: Any, axis: int, keep: np.ndarray) -> Any:
    indices = np.flatnonzero(keep)
    try:
        import torch
        if torch.is_tensor(value):
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            return value.index_select(axis, index)
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return np.take(value, indices, axis=axis)
    return value


def filter_template_features(features: dict[str, Any], regions: Sequence[dict[str, float]]) -> dict[str, Any]:
    """Remove keypoints and aligned local tensors inside template-side masks."""
    if not regions or "keypoints" not in features:
        return deepcopy(features)
    result = dict(features)
    keypoints = features["keypoints"]
    raw = keypoints[0] if getattr(keypoints, "ndim", 0) == 3 else keypoints
    try:
        points = raw.detach().cpu().numpy()  # torch tensor
    except AttributeError:
        points = np.asarray(raw)
    keep = np.ones(len(points), dtype=bool)
    for region in regions:
        keep &= ~_inside(points, region)
    if getattr(keypoints, "ndim", 0) == 3:
        result["keypoints"] = _select_axis(keypoints, 1, keep)
    else:
        result["keypoints"] = _select_axis(keypoints, 0, keep)
    count = len(points)
    for key, value in list(features.items()):
        if key == "keypoints" or not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)
        if len(shape) >= 2 and shape[0] == 1 and shape[1] == count:
            result[key] = _select_axis(value, 1, keep)
        elif len(shape) >= 3 and shape[0] == 1 and shape[-1] == count:
            result[key] = _select_axis(value, len(shape) - 1, keep)
        elif len(shape) == 1 and shape[0] == count:
            result[key] = _select_axis(value, 0, keep)
    return result


def project_region(region: dict[str, float], source_points: np.ndarray, target_points: np.ndarray) -> dict[str, float]:
    """Project a native rectangle through a correspondence-derived homography."""
    source = np.asarray(source_points, dtype=np.float32).reshape(-1, 1, 2)
    target = np.asarray(target_points, dtype=np.float32).reshape(-1, 1, 2)
    if len(source) < 4 or len(target) != len(source):
        raise ValueError("at least four point correspondences are required")
    matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None:
        raise ValueError("unable to estimate correspondence geometry")
    projected = cv2.perspectiveTransform(_rect_corners(region).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    x0, y0 = projected.min(axis=0)
    x1, y1 = projected.max(axis=0)
    return {"x": float(x0), "y": float(y0), "width": float(x1 - x0), "height": float(y1 - y0)}


def resolve_propagated_region(regions: Sequence[dict[str, float]], image_shape: tuple[int, int],
                              *, tolerance: float = 3.0) -> dict[str, Any]:
    """Resolve two-source projected rectangles into active or review state."""
    if len(regions) < 2:
        return {
            "status": "needs_review",
            "confidence": 0.0,
            "reason_code": "insufficient_sources",
            "reason": "at least two trusted sources required",
            "max_spread_px": 0.0,
            "area_ratio": 0.0,
        }
    corners = np.stack([_rect_corners(region) for region in regions])
    spread = float(np.max(np.linalg.norm(corners - corners[0], axis=2)))
    if spread > tolerance:
        return {
            "status": "needs_review",
            "confidence": 0.0,
            "reason_code": "source_disagreement",
            "reason": "disagreement between trusted sources",
            "max_spread_px": spread,
            "area_ratio": 0.0,
        }
    average = corners.mean(axis=0)
    x0, y0 = average.min(axis=0)
    x1, y1 = average.max(axis=0)
    area = max(0.0, float(x1 - x0) * float(y1 - y0))
    height, width = image_shape
    area_ratio = area / max(float(height * width), 1.0)
    if area <= 0 or area > float(height * width) * 0.5:
        return {
            "status": "needs_review",
            "confidence": 0.0,
            "reason_code": "area_unsafe",
            "reason": "mask area is unsafe",
            "max_spread_px": spread,
            "area_ratio": area_ratio,
            "region": {"x": float(x0), "y": float(y0), "width": float(x1 - x0), "height": float(y1 - y0)},
        }
    return {
        "status": "active",
        "confidence": max(0.0, 1.0 - spread / max(tolerance, 1e-6)),
        "reason_code": "sources_agree",
        "reason": "two trusted sources agree geometrically",
        "max_spread_px": spread,
        "area_ratio": area_ratio,
        "region": {"x": float(x0), "y": float(y0), "width": float(x1 - x0), "height": float(y1 - y0)},
    }
