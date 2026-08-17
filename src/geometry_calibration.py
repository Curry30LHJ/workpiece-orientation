"""Object-relative geometric fitting and ignore-mask construction.

The public surface deliberately stays small.  A profile contains one coarse
object anchor and zero or more rules expressed in that anchor's coordinate
frame.  Fitting is performed against the current image; no coordinates from a
previous image are reused.
"""

from __future__ import annotations

from copy import deepcopy
from math import cos, hypot, radians, sin
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ANCHOR_SEARCH_MARGIN_RATIO = 0.35
RULE_SEARCH_BAND_RATIO = 0.08
MIN_EDGE_SUPPORT = 0.35
MIN_VISIBLE_RATIO = 0.45
MAX_FIT_RESIDUAL_RATIO = 0.03

SUPPORTED_SHAPES = {"circle", "ellipse", "rotated_rectangle"}
SUPPORTED_MODES = {"inside", "outside"}


class GeometryCalibrationError(ValueError):
    """Raised when a geometry profile or image cannot be interpreted."""


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
    return {
        "shape": "ellipse",
        "cx": float(cx),
        "cy": float(cy),
        "rx": rx,
        "ry": ry,
        "angle_deg": _normalize_angle(angle),
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


def _rule_expected(rule: Mapping[str, Any], anchor: Mapping[str, Any]) -> dict[str, Any]:
    shape = rule.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise GeometryCalibrationError(f"unsupported rule shape: {shape}")
    geometry = rule.get("geometry")
    if not isinstance(geometry, Mapping):
        raise GeometryCalibrationError("rule.geometry is required")
    anchor_x, anchor_y = _shape_scale(anchor)
    result: dict[str, Any] = {
        "shape": shape,
        "cx": float(anchor["cx"]) + _finite(geometry.get("cx", 0.0), "rule.geometry.cx") * anchor_x,
        "cy": float(anchor["cy"]) + _finite(geometry.get("cy", 0.0), "rule.geometry.cy") * anchor_y,
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


def _shape_mask(shape: Mapping[str, Any], image_shape: tuple[int, int], mode: str, margin_ratio: float) -> np.ndarray:
    if mode not in SUPPORTED_MODES:
        raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
    margin = _finite(margin_ratio, "rule.margin_ratio")
    if margin < 0 or margin >= 0.95:
        raise GeometryCalibrationError("rule.margin_ratio must be in [0, 0.95)")
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    factor = 1.0 + margin if mode == "inside" else 1.0 - margin
    if shape["shape"] == "rotated_rectangle":
        half_width = max(1.0, float(shape["half_width"]) * factor)
        half_height = max(1.0, float(shape["half_height"]) * factor)
        rect = ((float(shape["cx"]), float(shape["cy"])),
                (2.0 * half_width, 2.0 * half_height), float(shape["angle_deg"]))
        points = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(mask, [points], 255)
    else:
        axes = (max(1, int(round(float(shape["rx"]) * factor))),
                max(1, int(round(float(shape["ry"]) * factor))))
        cv2.ellipse(mask, (int(round(shape["cx"])), int(round(shape["cy"]))), axes,
                    float(shape["angle_deg"]), 0, 360, 255, -1)
    if mode == "outside":
        mask = cv2.bitwise_not(mask)
    return mask


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


class GeometryCalibrator:
    """Fit an object anchor and its rules in the current image."""

    def fit(self, image: np.ndarray, direction_profile: Mapping[str, Any]) -> dict[str, Any]:
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
            coarse = _coarse_shape(anchor_profile, image_shape)
            anchor = self._fit_shape(image, coarse, image_shape, ANCHOR_SEARCH_MARGIN_RATIO)
            if anchor is None:
                return {"status": "low_confidence", "anchor": {"status": "low_confidence"}, "rules": []}
            fitted_rules = []
            for rule in rules:
                if not rule.get("enabled", True):
                    continue
                fitted_rules.append(self._fit_rule(image, anchor, rule, image_shape))
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

    def _fit_shape(self, image: np.ndarray, expected: Mapping[str, Any], image_shape: tuple[int, int],
                   margin_ratio: float) -> dict[str, Any] | None:
        contours = _contours(image)
        best: tuple[float, dict[str, Any]] | None = None
        for contour in contours:
            candidate = _candidate_for_shape(contour, expected["shape"])
            if candidate is None:
                continue
            score, support, visible, residual = _candidate_quality(candidate, expected, image_shape, margin_ratio)
            if support < MIN_EDGE_SUPPORT or visible < MIN_VISIBLE_RATIO or residual > MAX_FIT_RESIDUAL_RATIO:
                continue
            if best is None or score < best[0]:
                candidate = dict(candidate)
                candidate.update({"edge_support": support, "visible_ratio": visible,
                                  "fit_residual": residual})
                best = (score, candidate)
        return best[1] if best is not None else None

    def _fit_rule(self, image: np.ndarray, anchor: Mapping[str, Any], rule: Mapping[str, Any],
                  image_shape: tuple[int, int]) -> dict[str, Any]:
        mode = rule.get("mode")
        if mode not in SUPPORTED_MODES:
            raise GeometryCalibrationError(f"unsupported ignore mode: {mode}")
        expected = _rule_expected(rule, anchor)
        margin = _finite(rule.get("margin_ratio", 0.02), "rule.margin_ratio")
        if margin < 0 or margin >= 0.95:
            raise GeometryCalibrationError("rule.margin_ratio must be in [0, 0.95)")
        same_as_anchor = (
            mode == "outside"
            and expected["shape"] == anchor["shape"]
            and abs(float(expected.get("rx", 0.0)) - float(anchor.get("rx", 0.0))) <= 2.0
            and abs(float(expected.get("ry", 0.0)) - float(anchor.get("ry", 0.0))) <= 2.0
        )
        fitted = dict(anchor) if same_as_anchor else self._fit_shape(image, expected, image_shape, RULE_SEARCH_BAND_RATIO)
        if fitted is None:
            return {
                "rule_id": rule.get("rule_id"),
                "shape": rule.get("shape"),
                "mode": mode,
                "status": "low_confidence",
                "reason_code": "boundary_not_found",
            }
        public = self._public_shape(fitted)
        public.update({
            "rule_id": rule.get("rule_id"),
            "mode": mode,
            "status": "active",
            "ignore_mask": _shape_mask(fitted, image_shape, mode, margin),
            "edge_support": float(fitted.get("edge_support", 1.0)),
            "visible_ratio": float(fitted.get("visible_ratio", 1.0)),
            "fit_residual": float(fitted.get("fit_residual", 0.0)),
        })
        return public

    @staticmethod
    def _public_shape(shape: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in shape.items() if key != "contour"}
