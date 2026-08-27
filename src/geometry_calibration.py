"""Object-relative geometric fitting and ignore-mask construction.

The public surface deliberately stays small.  A profile contains one coarse
object anchor and zero or more rules expressed in that anchor's coordinate
frame.  Fitting is performed against the current image; no coordinates from a
previous image are reused.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import cos, hypot, radians, sin, sqrt
from time import perf_counter
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ANCHOR_SEARCH_MARGIN_RATIO = 0.35
RULE_SEARCH_BAND_RATIO = 0.08
MIN_EDGE_SUPPORT = 0.35
MIN_VISIBLE_RATIO = 0.45
MAX_FIT_RESIDUAL_RATIO = 0.03
TOPOLOGY_TOLERANCE_RATIO = 0.02

SUPPORTED_SHAPES = {"circle", "ellipse", "rotated_rectangle"}
SUPPORTED_MODES = {"inside", "outside"}


class GeometryCalibrationError(ValueError):
    """Raised when a geometry profile or image cannot be interpreted."""


@dataclass(frozen=True)
class GeometryFitContext:
    """Image-owned contour snapshot reusable across direction fits."""

    image_shape: tuple[int, int]
    contours: tuple[np.ndarray, ...]


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryCalibrationError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise GeometryCalibrationError(f"{name} must be finite")
    return result


def _gray_edges(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise GeometryCalibrationError("image must be a BGR HxWx3 array")
    if image.shape[0] < 16 or image.shape[1] < 16:
        raise GeometryCalibrationError("image is too small for geometric fitting")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 35, 125)


def _contours(image: np.ndarray) -> list[np.ndarray]:
    edges = _gray_edges(image)
    found = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = found[0] if len(found) == 2 else found[1]
    return [contour.reshape(-1, 2).astype(np.float32)
            for contour in contours if len(contour) >= 8]


def _normalize_angle(angle: float) -> float:
    result = (float(angle) + 180.0) % 180.0
    return result if result < 90.0 else result - 180.0


def _angle_delta(first: float, second: float) -> float:
    delta = abs(_normalize_angle(first) - _normalize_angle(second))
    return min(delta, 180.0 - delta)


def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = radians(angle_deg)
    rotation = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ rotation.T


def _shape_scale(anchor: Mapping[str, Any]) -> tuple[float, float]:
    if anchor["shape"] == "rotated_rectangle":
        return float(anchor["half_width"]), float(anchor["half_height"])
    return float(anchor["rx"]), float(anchor["ry"])


def _shape_extents(shape: Mapping[str, Any]) -> tuple[float, float]:
    """Return exact axis-aligned x/y half-extents for a fitted shape."""
    sx, sy = _shape_scale(shape)
    angle = radians(float(shape.get("angle_deg", 0.0)))
    cos_angle, sin_angle = abs(cos(angle)), abs(sin(angle))
    if shape["shape"] != "rotated_rectangle":
        return (
            max(1.0, sqrt((float(sx) * cos_angle) ** 2 + (float(sy) * sin_angle) ** 2)),
            max(1.0, sqrt((float(sx) * sin_angle) ** 2 + (float(sy) * cos_angle) ** 2)),
        )
    return (
        max(1.0, float(sx) * cos_angle + float(sy) * sin_angle),
        max(1.0, float(sx) * sin_angle + float(sy) * cos_angle),
    )


def public_shape_extents(shape: Mapping[str, Any]) -> tuple[float, float]:
    """Expose the established rotated shape extents for fast consumers."""
    return _shape_extents(shape)


def _shape_has_meaningful_angle(shape: Mapping[str, Any]) -> bool:
    if shape["shape"] == "rotated_rectangle":
        return True
    sx, sy = _shape_scale(shape)
    return abs(float(sx) - float(sy)) / max(float(sx), float(sy), 1.0) > 0.05


def _shape_relation(
    candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
    mode: str,
    tolerance: float = TOPOLOGY_TOLERANCE_RATIO,
) -> tuple[bool, str, float]:
    """Check whether a rule candidate has the required object-relative topology."""
    if mode not in SUPPORTED_MODES:
        raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
    tolerance = _finite(tolerance, "topology tolerance")
    if tolerance < 0.0 or tolerance >= 0.5:
        raise GeometryCalibrationError("topology tolerance must be in [0, 0.5)")
    candidate_x, candidate_y = _shape_extents(candidate)
    anchor_x, anchor_y = _shape_extents(anchor)
    center_dx = abs(float(candidate.get("cx", 0.0)) - float(anchor.get("cx", 0.0)))
    center_dy = abs(float(candidate.get("cy", 0.0)) - float(anchor.get("cy", 0.0)))
    if mode == "inside":
        valid = (
            center_dx + candidate_x <= anchor_x * (1.0 - tolerance)
            and center_dy + candidate_y <= anchor_y * (1.0 - tolerance)
        )
        relation = "contained" if valid else "invalid"
        scale_ratio = max(candidate_x / max(anchor_x, 1.0), candidate_y / max(anchor_y, 1.0))
    else:
        valid = (
            center_dx + anchor_x <= candidate_x * (1.0 + tolerance)
            and center_dy + anchor_y <= candidate_y * (1.0 + tolerance)
        )
        relation = "encloses" if valid else "invalid"
        scale_ratio = max(anchor_x / max(candidate_x, 1.0), anchor_y / max(candidate_y, 1.0))
    return bool(valid), relation, float(scale_ratio)


def _shape_perimeter(shape: Mapping[str, Any]) -> float:
    if shape["shape"] == "rotated_rectangle":
        return 4.0 * (float(shape["half_width"]) + float(shape["half_height"]))
    rx = float(shape["rx"])
    ry = float(shape["ry"])
    return float(np.pi * (3.0 * (rx + ry) - np.sqrt((3.0 * rx + ry) * (rx + 3.0 * ry))))


def _ellipse_candidate(contour: np.ndarray) -> dict[str, Any] | None:
    if len(contour) < 5:
        return None
    try:
        (cx, cy), (width, height), angle = cv2.fitEllipse(contour.reshape(-1, 1, 2))
    except cv2.error:
        return None
    rx, ry = float(width) / 2.0, float(height) / 2.0
    if rx < ry:
        rx, ry = ry, rx
        angle += 90.0
    angle_deg = _normalize_angle(angle)
    if abs(rx - ry) / max(rx, ry, 1.0) <= 0.05:
        angle_deg = 0.0
    return {
        "shape": "ellipse",
        "cx": float(cx),
        "cy": float(cy),
        "rx": rx,
        "ry": ry,
        "angle_deg": angle_deg,
        "contour": contour,
    }


def _rectangle_candidate(contour: np.ndarray) -> dict[str, Any] | None:
    try:
        (cx, cy), (width, height), angle = cv2.minAreaRect(contour.reshape(-1, 1, 2))
    except cv2.error:
        return None
    half_width, half_height = float(width) / 2.0, float(height) / 2.0
    if half_width < half_height:
        half_width, half_height = half_height, half_width
        angle += 90.0
    return {
        "shape": "rotated_rectangle",
        "cx": float(cx),
        "cy": float(cy),
        "half_width": half_width,
        "half_height": half_height,
        "angle_deg": _normalize_angle(angle),
        "contour": contour,
    }


def _candidate_for_shape(contour: np.ndarray, shape: str) -> dict[str, Any] | None:
    if shape in {"circle", "ellipse"}:
        candidate = _ellipse_candidate(contour)
        if candidate is None:
            return None
        candidate["shape"] = shape
        return candidate
    if shape == "rotated_rectangle":
        return _rectangle_candidate(contour)
    raise GeometryCalibrationError(f"unsupported shape: {shape}")


def _shape_error(candidate: Mapping[str, Any], expected: Mapping[str, Any], image_shape: tuple[int, int],
                 margin_ratio: float) -> float:
    height, width = image_shape
    diagonal = max(hypot(width, height), 1.0)
    center_error = hypot(float(candidate["cx"]) - float(expected["cx"]),
                         float(candidate["cy"]) - float(expected["cy"])) / diagonal
    scale_x, scale_y = _shape_scale(candidate)
    expected_x, expected_y = _shape_scale(expected)
    scale_error = max(abs(scale_x - expected_x) / max(expected_x, 1.0),
                      abs(scale_y - expected_y) / max(expected_y, 1.0))
    angle_error = 0.0
    if _shape_has_meaningful_angle(candidate) and _shape_has_meaningful_angle(expected):
        angle_error = _angle_delta(float(candidate.get("angle_deg", 0.0)),
                                   float(expected.get("angle_deg", 0.0))) / 90.0
    allowed = max(float(margin_ratio), 1e-3)
    return center_error / allowed + scale_error / allowed + angle_error / max(allowed * 2.0, 1e-3)


def _candidate_quality(candidate: Mapping[str, Any], expected: Mapping[str, Any],
                       image_shape: tuple[int, int], margin_ratio: float) -> tuple[float, float, float, float]:
    contour = np.asarray(candidate["contour"], dtype=np.float32)
    perimeter = max(_shape_perimeter(candidate), 1.0)
    edge_support = min(1.0, float(len(contour)) / max(perimeter * 0.45, 1.0))
    visible_ratio = min(1.0, float(len(contour)) / max(perimeter * 0.20, 1.0))
    error = _shape_error(candidate, expected, image_shape, margin_ratio)
    if candidate["shape"] == "rotated_rectangle":
        local = _rotate_points(contour - np.array([candidate["cx"], candidate["cy"]], dtype=np.float32),
                               -float(candidate["angle_deg"]))
        half_width = max(float(candidate["half_width"]), 1.0)
        half_height = max(float(candidate["half_height"]), 1.0)
        distances = np.minimum(np.abs(np.abs(local[:, 0]) - half_width),
                               np.abs(np.abs(local[:, 1]) - half_height))
        residual = min(1.0, float(np.mean(distances)) / max(min(half_width, half_height), 1.0))
    else:
        local = _rotate_points(contour - np.array([candidate["cx"], candidate["cy"]], dtype=np.float32),
                               -float(candidate["angle_deg"]))
        radial = np.sqrt((local[:, 0] / max(float(candidate["rx"]), 1.0)) ** 2
                         + (local[:, 1] / max(float(candidate["ry"]), 1.0)) ** 2)
        residual = min(1.0, float(np.mean(np.abs(radial - 1.0))))
    score = error - edge_support * 0.25 - visible_ratio * 0.10
    return score, edge_support, visible_ratio, residual


def _filter_rule_candidates(
    candidates: Sequence[Mapping[str, Any]],
    anchor: Mapping[str, Any],
    mode: str,
) -> list[dict[str, Any]]:
    """Keep only candidates whose geometry is valid for the rule direction."""
    filtered: list[dict[str, Any]] = []
    for candidate in candidates:
        valid, relation, scale_ratio = _shape_relation(candidate, anchor, mode)
        if not valid:
            continue
        enriched = dict(candidate)
        enriched.update({
            "relation_to_anchor": relation,
            "scale_ratio": float(scale_ratio),
            "topology_valid": True,
        })
        filtered.append(enriched)
    return filtered


def _coarse_shape(anchor: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
    height, width = image_shape
    coarse = anchor.get("coarse")
    if not isinstance(coarse, Mapping):
        raise GeometryCalibrationError("anchor.coarse is required")
    shape = anchor.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise GeometryCalibrationError(f"unsupported anchor shape: {shape}")
    result: dict[str, Any] = {
        "shape": shape,
        "cx": _finite(coarse.get("cx"), "anchor.coarse.cx") * width,
        "cy": _finite(coarse.get("cy"), "anchor.coarse.cy") * height,
        "angle_deg": _finite(coarse.get("angle_deg", 0.0), "anchor.coarse.angle_deg"),
    }
    if shape == "rotated_rectangle":
        result["half_width"] = _finite(coarse.get("half_width"), "anchor.coarse.half_width") * width
        result["half_height"] = _finite(coarse.get("half_height"), "anchor.coarse.half_height") * height
    elif shape == "circle":
        result["rx"] = _finite(coarse.get("r", coarse.get("rx")), "anchor.coarse.r") * min(width, height)
        result["ry"] = result["rx"]
    else:
        result["rx"] = _finite(coarse.get("rx"), "anchor.coarse.rx") * width
        result["ry"] = _finite(coarse.get("ry"), "anchor.coarse.ry") * height
    if any(value <= 0 for key, value in result.items() if key in {"rx", "ry", "half_width", "half_height"}):
        raise GeometryCalibrationError("anchor dimensions must be positive")
    return result


def public_coarse_shape(anchor: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
    """Expose the established coarse-anchor projection for fast consumers."""
    return _coarse_shape(anchor, image_shape)


def _rule_expected(rule: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    shape = rule.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise GeometryCalibrationError(f"unsupported rule shape: {shape}")
    geometry = rule.get("geometry")
    if not isinstance(geometry, Mapping):
        raise GeometryCalibrationError("rule.geometry is required")
    anchor_x, anchor_y = _shape_scale(anchor)
    local_center = np.array([[
        _finite(geometry.get("cx", 0.0), "rule.geometry.cx") * anchor_x,
        _finite(geometry.get("cy", 0.0), "rule.geometry.cy") * anchor_y,
    ]], dtype=np.float32)
    rotated_center = _rotate_points(local_center, float(anchor.get("angle_deg", 0.0)))[0]
    result: dict[str, Any] = {
        "shape": shape,
        "cx": float(anchor["cx"]) + float(rotated_center[0]),
        "cy": float(anchor["cy"]) + float(rotated_center[1]),
        "angle_deg": float(anchor.get("angle_deg", 0.0)) + _finite(geometry.get("angle_deg", 0.0), "rule.geometry.angle_deg"),
    }
    if shape == "circle":
        radius = _finite(geometry.get("r"), "rule.geometry.r")
        result["rx"], result["ry"] = abs(radius) * anchor_x, abs(radius) * anchor_y
    elif shape == "ellipse":
        result["rx"] = abs(_finite(geometry.get("rx"), "rule.geometry.rx")) * anchor_x
        result["ry"] = abs(_finite(geometry.get("ry"), "rule.geometry.ry")) * anchor_y
    else:
        result["shape"] = "rotated_rectangle"
        result["half_width"] = abs(_finite(geometry.get("half_width"), "rule.geometry.half_width")) * anchor_x
        result["half_height"] = abs(_finite(geometry.get("half_height"), "rule.geometry.half_height")) * anchor_y
    return result


def public_rule_expected(rule: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the established object-relative rule projection for fast consumers."""
    return _rule_expected(rule, anchor)


def _offset_shape(shape: Mapping[str, Any], margin_ratio: float) -> dict[str, Any]:
    margin = _finite(margin_ratio, "margin_ratio")
    if margin < -0.94 or margin > 0.94:
        raise GeometryCalibrationError("margin_ratio must be in [-0.94, 0.94]")
    result = dict(shape)
    factor = 1.0 + margin
    if shape["shape"] == "rotated_rectangle":
        result["half_width"] = max(1.0, float(shape["half_width"]) * factor)
        result["half_height"] = max(1.0, float(shape["half_height"]) * factor)
    else:
        result["rx"] = max(1.0, float(shape["rx"]) * factor)
        result["ry"] = max(1.0, float(shape["ry"]) * factor)
    return result


def public_offset_shape(shape: Mapping[str, Any], margin_ratio: float) -> dict[str, Any]:
    """Expose the established signed margin projection for fast consumers."""
    return _offset_shape(shape, margin_ratio)


def _shape_mask(shape: Mapping[str, Any], image_shape: tuple[int, int], mode: str, margin_ratio: float) -> np.ndarray:
    if mode not in SUPPORTED_MODES:
        raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    effective = _offset_shape(shape, margin_ratio)
    if effective["shape"] == "rotated_rectangle":
        half_width = effective["half_width"]
        half_height = effective["half_height"]
        rect = ((float(shape["cx"]), float(shape["cy"])),
                (2.0 * half_width, 2.0 * half_height), float(shape["angle_deg"]))
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        axes = (max(1, int(round(effective["rx"]))),
                max(1, int(round(effective["ry"]))))
        cv2.ellipse(mask, (int(round(shape["cx"])), int(round(shape["cy"]))), axes,
                    float(shape["angle_deg"]), 0, 360, 255, -1)
    if mode == "outside":
        mask = cv2.bitwise_not(mask)
    return mask


def public_shape_mask(
    shape: Mapping[str, Any], image_shape: tuple[int, int], mode: str, margin_ratio: float,
) -> np.ndarray:
    """Expose the established shape-mask semantics for fast consumers."""
    return _shape_mask(shape, image_shape, mode, margin_ratio)


def _select_aligned(value: Any, keep: np.ndarray, axis: int) -> Any:
    indices = np.flatnonzero(keep)
    if isinstance(value, np.ndarray):
        return np.take(value, indices, axis=axis)
    try:
        import torch
        if torch.is_tensor(value):
            index = torch.as_tensor(indices, dtype=torch.long, device=value.device)
            return value.index_select(axis, index)
    except ImportError:
        pass
    return value


def _keypoints_numpy(keypoints: Any) -> tuple[np.ndarray, int]:
    if isinstance(keypoints, np.ndarray):
        raw = keypoints[0] if keypoints.ndim == 3 else keypoints
        return np.asarray(raw), 1 if keypoints.ndim == 3 else 0
    try:
        import torch
        if torch.is_tensor(keypoints):
            raw = keypoints[0] if keypoints.ndim == 3 else keypoints
            return raw.detach().cpu().numpy(), 1 if keypoints.ndim == 3 else 0
    except ImportError:
        pass
    raw = keypoints[0] if getattr(keypoints, "ndim", 0) == 3 else keypoints
    return np.asarray(raw), 1 if getattr(keypoints, "ndim", 0) == 3 else 0


def filter_features_by_mask(features: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    """Remove keypoints inside a native-coordinate binary ignore mask."""
    if not isinstance(features, dict) or "keypoints" not in features:
        return deepcopy(features)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise GeometryCalibrationError("mask must be a single-channel array")
    points, keypoint_axis = _keypoints_numpy(features["keypoints"])
    height, width = mask.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    keep = mask[y, x] == 0
    result = dict(features)
    result["keypoints"] = _select_aligned(features["keypoints"], keep, keypoint_axis)
    count = len(points)
    for key, value in features.items():
        if key == "keypoints" or not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)
        if len(shape) >= 2 and shape[0] == 1 and shape[1] == count:
            result[key] = _select_aligned(value, keep, 1)
        elif len(shape) >= 3 and shape[-1] == count:
            result[key] = _select_aligned(value, keep, len(shape) - 1)
        elif len(shape) == 1 and shape[0] == count:
            result[key] = _select_aligned(value, keep, 0)
    return result


def apply_ignore_mask(image: np.ndarray, mask: np.ndarray, fill_bgr: Sequence[int]) -> np.ndarray:
    """Return a copy with only ignored pixels filled by a neutral BGR value."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise GeometryCalibrationError("image must be a BGR HxWx3 array")
    if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
        raise GeometryCalibrationError("mask shape must match image height and width")
    if len(fill_bgr) != 3:
        raise GeometryCalibrationError("fill_bgr must have three channels")
    fill = np.asarray([_finite(value, "fill_bgr") for value in fill_bgr], dtype=np.float32)
    if np.any(fill < 0) or np.any(fill > 255):
        raise GeometryCalibrationError("fill_bgr values must be in [0, 255]")
    result = image.copy()
    result[mask != 0] = np.rint(fill).astype(np.uint8)
    return result


def _geometry_rule_masks(fit: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(fit, Mapping):
        raise GeometryCalibrationError("geometry fit must be a mapping")
    inside: np.ndarray | None = None
    outside: np.ndarray | None = None
    for rule in fit.get("rules", []):
        if not isinstance(rule, Mapping) or "ignore_mask" not in rule:
            continue
        mask = rule["ignore_mask"]
        if not isinstance(mask, np.ndarray) or mask.ndim != 2:
            raise GeometryCalibrationError("geometry rule mask must be a single-channel array")
        if rule.get("mode") == "inside":
            inside = mask.copy() if inside is None else cv2.bitwise_or(inside, mask)
        elif rule.get("mode") == "outside":
            outside = mask.copy() if outside is None else cv2.bitwise_or(outside, mask)
        else:
            raise GeometryCalibrationError(f"unsupported ignore mode: {rule.get('mode')}")
    if inside is None and outside is None:
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty, empty.copy()
    if inside is None:
        inside = np.zeros_like(outside)
    if outside is None:
        outside = np.zeros_like(inside)
    if inside.shape != outside.shape:
        raise GeometryCalibrationError("geometry rule masks must have matching shapes")
    return inside, outside


def apply_geometry_fit(
    image: np.ndarray,
    fit: Mapping[str, Any],
    fill_bgr: Sequence[int],
) -> np.ndarray:
    """Apply calibrated outside masking and texture-preserving inside repair."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise GeometryCalibrationError("image must be a BGR HxWx3 array")
    inside, outside = _geometry_rule_masks(fit)
    if inside.size == 0:
        return image.copy()
    if inside.shape != image.shape[:2]:
        raise GeometryCalibrationError("geometry rule mask shape must match image")
    result = apply_ignore_mask(image, outside, fill_bgr)
    repair_mask = cv2.bitwise_and(inside, cv2.bitwise_not(outside))
    if np.any(repair_mask):
        try:
            result = cv2.inpaint(result, repair_mask, 3, cv2.INPAINT_TELEA)
        except cv2.error as exc:
            raise GeometryCalibrationError(f"inside geometry repair failed: {exc}") from exc
    return result


def geometry_feature_mask(fit: Mapping[str, Any]) -> np.ndarray:
    """Return every fitted ignore area that must not contribute local keypoints."""
    mask = fit.get("ignore_mask") if isinstance(fit, Mapping) else None
    if isinstance(mask, np.ndarray):
        if mask.ndim != 2:
            raise GeometryCalibrationError("geometry ignore mask must be a single-channel array")
        return mask
    inside, outside = _geometry_rule_masks(fit)
    if inside.size == 0:
        return inside
    return cv2.bitwise_or(inside, outside)


class GeometryCalibrator:
    """Fit an object anchor and its rules in the current image."""

    def prepare_context(self, image: np.ndarray) -> GeometryFitContext:
        image_shape = image.shape[:2] if isinstance(image, np.ndarray) else ()
        if len(image_shape) != 2:
            raise GeometryCalibrationError("image must be a BGR HxWx3 array")
        return GeometryFitContext(
            image_shape=(int(image_shape[0]), int(image_shape[1])),
            contours=tuple(_contours(image)),
        )

    def fit_reference(
        self,
        image: np.ndarray,
        seed_shape: Mapping[str, Any],
        *,
        mode: str,
        margin_ratio: float = 0.02,
        anchor_candidate_index: int | None = None,
        rule_candidate_index: int | None = None,
    ) -> dict[str, Any]:
        """Fit a coarse editor gesture to the current image.

        The editor deliberately submits only a semantic seed.  The seed is
        never reused as a fixed pixel mask: an outer object contour is first
        ranked in the reference image and the seed is then expressed in that
        fitted anchor's coordinate frame.
        """
        started = perf_counter()
        if mode not in SUPPORTED_MODES:
            raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
        margin = _finite(margin_ratio, "margin_ratio")
        if margin < -0.94 or margin > 0.94:
            raise GeometryCalibrationError("margin_ratio must be in [-0.94, 0.94]")
        seed = self._normalize_seed(seed_shape)
        image_shape = image.shape[:2] if isinstance(image, np.ndarray) else ()
        if len(image_shape) != 2:
            raise GeometryCalibrationError("image must be a BGR HxWx3 array")
        contours = _contours(image)

        anchor_expected = self._anchor_expected(seed, image_shape)
        anchor_candidates = self._rank_candidates(
            contours, anchor_expected, image_shape, ANCHOR_SEARCH_MARGIN_RATIO * 2.0,
            prefer_larger=True,
        )[:3]
        if not anchor_candidates:
            if anchor_candidate_index is not None:
                raise GeometryCalibrationError("anchor_candidate_index is out of range")
            return {
                "status": "low_confidence",
                "anchor_fit": {"candidates": [], "selected_candidate_index": None},
                "rule_fit": {"candidates": [], "selected_candidate_index": None},
                "fit_duration_ms": (perf_counter() - started) * 1000.0,
            }
        anchor_index = self._candidate_index(anchor_candidate_index, len(anchor_candidates), "anchor")
        anchor = anchor_candidates[anchor_index]

        rule_expected = self._seed_expected(seed, anchor)
        quality_rule_candidates = self._rank_candidates(
            contours, rule_expected, image_shape, RULE_SEARCH_BAND_RATIO * 2.0,
            prefer_larger=False,
        )
        topology_rule_candidates = _filter_rule_candidates(quality_rule_candidates, anchor, mode)
        rule_candidates = topology_rule_candidates[:3]
        if not rule_candidates:
            if rule_candidate_index is not None:
                raise GeometryCalibrationError("rule_candidate_index is out of range")
            return {
                "status": "low_confidence",
                "anchor_fit": self._candidate_report(anchor_candidates, anchor_index),
                "rule_fit": {
                    "candidates": [],
                    "selected_candidate_index": None,
                    "reason_code": (
                        "topology_constraint_failed"
                        if quality_rule_candidates else "boundary_candidate_not_found"
                    ),
                    "quality_candidate_count": len(quality_rule_candidates),
                    "topology_valid_candidate_count": len(topology_rule_candidates),
                },
                "fit_duration_ms": (perf_counter() - started) * 1000.0,
            }
        rule_index = self._candidate_index(rule_candidate_index, len(rule_candidates), "rule")
        rule = rule_candidates[rule_index]
        profile_anchor = self._normalized_anchor(anchor, image_shape)
        fitted_geometry = self._relative_geometry(rule, anchor)
        seed_geometry = self._relative_geometry(seed, anchor)
        effective_rule = _offset_shape(rule, margin)
        rule_report = self._candidate_report(rule_candidates, rule_index)
        rule_report["effective_shape"] = self._public_shape(effective_rule)
        return {
            "status": "active",
            "margin_ratio": margin,
            "anchor_fit": self._candidate_report(anchor_candidates, anchor_index),
            "rule_fit": rule_report,
            "profile_patch": {
                "anchor": {"mode": "auto", **profile_anchor},
                "geometry": fitted_geometry,
                "seed_geometry": seed_geometry,
                "mode": mode,
                "margin_ratio": margin,
                "margin_semantics": "signed_boundary_v2",
            },
            "fit_duration_ms": (perf_counter() - started) * 1000.0,
        }

    def fit(
        self,
        image: np.ndarray,
        direction_profile: Mapping[str, Any],
        *,
        context: GeometryFitContext | None = None,
    ) -> dict[str, Any]:
        if not isinstance(direction_profile, Mapping):
            raise GeometryCalibrationError("direction_profile must be a mapping")
        anchor_profile = direction_profile.get("anchor")
        rules = direction_profile.get("rules", [])
        if anchor_profile is None or not rules:
            return {"status": "not_configured", "rules": []}
        if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
            raise GeometryCalibrationError("rules must be a list")
        try:
            image_shape = image.shape[:2]
            fit_context = context or self.prepare_context(image)
            if tuple(image_shape) != fit_context.image_shape:
                raise GeometryCalibrationError("geometry fit context does not match image shape")
            coarse = _coarse_shape(anchor_profile, image_shape)
            anchor = self._fit_shape(
                image, coarse, image_shape, ANCHOR_SEARCH_MARGIN_RATIO,
                contours=fit_context.contours,
            )
            if anchor is None:
                return {"status": "low_confidence", "anchor": {"status": "low_confidence"}, "rules": []}
            fitted_rules = []
            for rule in rules:
                if not rule.get("enabled", True):
                    continue
                fitted_rules.append(
                    self._fit_rule(
                        image, anchor, rule, image_shape, contours=fit_context.contours
                    )
                )
            if any(item["status"] != "active" for item in fitted_rules):
                return {"status": "low_confidence", "anchor": self._public_shape(anchor), "rules": fitted_rules}
            mask = np.zeros(image_shape, dtype=np.uint8)
            for item in fitted_rules:
                mask = cv2.bitwise_or(mask, item["ignore_mask"])
            return {
                "status": "active",
                "anchor": self._public_shape(anchor),
                "rules": fitted_rules,
                "ignore_mask": mask,
                "ignored_ratio": float(np.count_nonzero(mask) / max(mask.size, 1)),
            }
        except (AttributeError, cv2.error, TypeError, ValueError) as exc:
            raise GeometryCalibrationError(str(exc)) from exc

    def _fit_shape(
        self,
        image: np.ndarray,
        expected: Mapping[str, Any],
        image_shape: tuple[int, int],
        margin_ratio: float,
        *,
        contours: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any] | None:
        candidates = self._rank_candidates(
            contours if contours is not None else _contours(image),
            expected, image_shape, margin_ratio, prefer_larger=False,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _normalize_seed(seed_shape: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(seed_shape, Mapping):
            raise GeometryCalibrationError("seed_shape must be a mapping")
        shape = seed_shape.get("shape")
        if shape not in SUPPORTED_SHAPES:
            raise GeometryCalibrationError(f"unsupported seed shape: {shape}")
        result: dict[str, Any] = {
            "shape": shape,
            "cx": _finite(seed_shape.get("cx"), "seed.cx"),
            "cy": _finite(seed_shape.get("cy"), "seed.cy"),
            "angle_deg": _finite(seed_shape.get("angle_deg", 0.0), "seed.angle_deg"),
        }
        if shape == "circle":
            radius = _finite(seed_shape.get("r", seed_shape.get("radius")), "seed.r")
            if radius <= 0:
                raise GeometryCalibrationError("seed.r must be positive")
            result["r"] = radius
        elif shape == "ellipse":
            result["rx"] = _finite(seed_shape.get("rx"), "seed.rx")
            result["ry"] = _finite(seed_shape.get("ry"), "seed.ry")
            if result["rx"] <= 0 or result["ry"] <= 0:
                raise GeometryCalibrationError("seed ellipse axes must be positive")
        else:
            result["half_width"] = _finite(seed_shape.get("half_width"), "seed.half_width")
            result["half_height"] = _finite(seed_shape.get("half_height"), "seed.half_height")
            if result["half_width"] <= 0 or result["half_height"] <= 0:
                raise GeometryCalibrationError("seed rectangle dimensions must be positive")
        return result

    @staticmethod
    def _anchor_expected(seed: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
        height, width = image_shape
        minimum = float(min(width, height))
        if seed["shape"] == "rotated_rectangle":
            return {
                "shape": "rotated_rectangle",
                "cx": float(seed["cx"]), "cy": float(seed["cy"]),
                "half_width": max(float(seed["half_width"]) * 1.8, width * 0.20),
                "half_height": max(float(seed["half_height"]) * 1.8, height * 0.16),
                "angle_deg": float(seed.get("angle_deg", 0.0)),
            }
        if seed["shape"] == "ellipse":
            seed_x, seed_y = float(seed["rx"]), float(seed["ry"])
        else:
            seed_x = seed_y = float(seed["r"])
        return {
            "shape": "ellipse",
            "cx": float(seed["cx"]), "cy": float(seed["cy"]),
            "rx": max(seed_x * 1.8, minimum * 0.25),
            "ry": max(seed_y * 1.55, minimum * 0.20),
            "angle_deg": float(seed.get("angle_deg", 0.0)),
        }

    @staticmethod
    def _seed_expected(seed: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
        """Return the pixel-space candidate expectation from an editor seed.

        ``fit_reference`` receives the shape drawn on the reference image in
        image pixels.  The anchor is only used later when the selected result
        is converted to object-relative profile geometry; applying its scale
        here would multiply the seed dimensions a second time.
        """
        result: dict[str, Any] = {
            "shape": seed["shape"],
            "cx": float(seed["cx"]),
            "cy": float(seed["cy"]),
            "angle_deg": float(seed.get("angle_deg", 0.0)),
        }
        if seed["shape"] == "circle":
            result["rx"] = result["ry"] = float(seed["r"])
        elif seed["shape"] == "ellipse":
            result["rx"] = float(seed["rx"])
            result["ry"] = float(seed["ry"])
        else:
            result["half_width"] = float(seed["half_width"])
            result["half_height"] = float(seed["half_height"])
        return result

    @staticmethod
    def _rank_candidates(
        contours: Sequence[np.ndarray], expected: Mapping[str, Any], image_shape: tuple[int, int],
        margin_ratio: float, *, prefer_larger: bool,
    ) -> list[dict[str, Any]]:
        ranked: list[tuple[float, float, dict[str, Any]]] = []
        for contour in contours:
            candidate = _candidate_for_shape(contour, expected["shape"])
            if candidate is None:
                continue
            score, support, visible, residual = _candidate_quality(candidate, expected, image_shape, margin_ratio)
            if support < MIN_EDGE_SUPPORT or visible < MIN_VISIBLE_RATIO or residual > MAX_FIT_RESIDUAL_RATIO:
                continue
            enriched = dict(candidate)
            enriched.update({"score": float(score), "edge_support": support,
                             "visible_ratio": visible, "fit_residual": residual})
            area = float(cv2.contourArea(contour))
            # The anchor is the outer object boundary in the reference image;
            # a small area prior breaks ties between nested edge contours.
            sort_score = float(score) - (0.15 * area / max(image_shape[0] * image_shape[1], 1)
                                         if prefer_larger else 0.0)
            ranked.append((sort_score, -area if prefer_larger else 0.0, enriched))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked]

    @staticmethod
    def _candidate_index(index: int | None, count: int, label: str) -> int:
        selected = 0 if index is None else int(index)
        if selected < 0 or selected >= count:
            raise GeometryCalibrationError(f"{label}_candidate_index is out of range")
        return selected

    def _candidate_report(self, candidates: Sequence[Mapping[str, Any]], selected: int) -> dict[str, Any]:
        return {
            "selected_candidate_index": int(selected),
            "candidates": [self._public_shape(candidate) for candidate in candidates],
            "fitted_shape": self._public_shape(candidates[selected]),
        }

    @staticmethod
    def _normalized_anchor(anchor: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
        height, width = image_shape
        coarse: dict[str, Any] = {"cx": float(anchor["cx"]) / width,
                                  "cy": float(anchor["cy"]) / height,
                                  "angle_deg": float(anchor.get("angle_deg", 0.0))}
        if anchor["shape"] == "rotated_rectangle":
            coarse["half_width"] = float(anchor["half_width"]) / width
            coarse["half_height"] = float(anchor["half_height"]) / height
        else:
            coarse["rx"] = float(anchor["rx"]) / width
            coarse["ry"] = float(anchor["ry"]) / height
            if anchor["shape"] == "circle":
                coarse.pop("rx", None)
                coarse.pop("ry", None)
                coarse["r"] = float(anchor["rx"]) / min(width, height)
        return {"shape": anchor["shape"], "mode": "auto", "coarse": coarse}

    @staticmethod
    def _relative_geometry(seed: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
        ax, ay = _shape_scale(anchor)
        dx = (float(seed["cx"]) - float(anchor["cx"])) / max(ax, 1.0)
        dy = (float(seed["cy"]) - float(anchor["cy"])) / max(ay, 1.0)
        local = _rotate_points(np.array([[dx, dy]], dtype=np.float32), -float(anchor.get("angle_deg", 0.0)))[0]
        result: dict[str, Any] = {
            "shape": seed["shape"], "cx": float(local[0]), "cy": float(local[1]),
            "angle_deg": _normalize_angle(float(seed.get("angle_deg", 0.0))
                                           - float(anchor.get("angle_deg", 0.0))),
        }
        if seed["shape"] == "circle":
            if "r" in seed:
                result["r"] = float(seed["r"]) / max(min(ax, ay), 1.0)
            else:
                # Circle candidates are fitted by OpenCV as ellipses so that
                # contour scoring can use both image-space axes.  Convert the
                # selected fit back to the rule's object-relative circle
                # representation instead of requiring an editor-only ``r``.
                normalized_rx = float(seed["rx"]) / max(ax, 1.0)
                normalized_ry = float(seed["ry"]) / max(ay, 1.0)
                result["r"] = (normalized_rx + normalized_ry) / 2.0
        elif seed["shape"] == "ellipse":
            result["rx"] = float(seed["rx"]) / max(ax, 1.0)
            result["ry"] = float(seed["ry"]) / max(ay, 1.0)
        else:
            result["half_width"] = float(seed["half_width"]) / max(ax, 1.0)
            result["half_height"] = float(seed["half_height"]) / max(ay, 1.0)
        return result

    def _fit_rule(
        self,
        image: np.ndarray,
        anchor: Mapping[str, Any],
        rule: Mapping[str, Any],
        image_shape: tuple[int, int],
        *,
        contours: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        mode = rule.get("mode")
        if mode not in SUPPORTED_MODES:
            raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
        expected = _rule_expected(rule, anchor)
        margin = _finite(rule.get("margin_ratio", 0.02), "rule.margin_ratio")
        if margin < -0.94 or margin > 0.94:
            raise GeometryCalibrationError("rule.margin_ratio must be in [-0.94, 0.94]")
        same_as_anchor = (
            mode == "outside"
            and expected["shape"] == anchor["shape"]
            and abs(float(expected.get("rx", 0.0)) - float(anchor.get("rx", 0.0))) <= 2.0
            and abs(float(expected.get("ry", 0.0)) - float(anchor.get("ry", 0.0))) <= 2.0
        )
        if same_as_anchor:
            fitted = dict(anchor)
            fitted.update({
                "relation_to_anchor": "encloses",
                "scale_ratio": 1.0,
                "topology_valid": True,
            })
        else:
            candidates = self._rank_candidates(
                contours if contours is not None else _contours(image),
                expected, image_shape, RULE_SEARCH_BAND_RATIO,
                prefer_larger=False,
            )
            candidates = _filter_rule_candidates(candidates, anchor, mode)
            fitted = candidates[0] if candidates else None
        if fitted is None:
            return {
                "rule_id": rule.get("rule_id"),
                "shape": rule.get("shape"),
                "mode": mode,
                "status": "low_confidence",
                "reason_code": "topology_constraint_failed",
            }
        public = self._public_shape(fitted)
        effective = _offset_shape(fitted, margin)
        public.update({
            "rule_id": rule.get("rule_id"),
            "mode": mode,
            "status": "active",
            "fitted_shape": self._public_shape(fitted),
            "effective_shape": self._public_shape(effective),
            "margin_ratio": margin,
            "ignore_mask": _shape_mask(fitted, image_shape, mode, margin),
            "edge_support": float(fitted.get("edge_support", 1.0)),
            "visible_ratio": float(fitted.get("visible_ratio", 1.0)),
            "fit_residual": float(fitted.get("fit_residual", 0.0)),
            "relation_to_anchor": fitted.get("relation_to_anchor", "encloses" if same_as_anchor else "invalid"),
            "scale_ratio": float(fitted.get("scale_ratio", 1.0)),
            "topology_valid": bool(fitted.get("topology_valid", same_as_anchor)),
        })
        return public

    @staticmethod
    def _public_shape(shape: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in shape.items() if key != "contour"}
