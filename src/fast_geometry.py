"""Shared low-resolution geometry fitting for the fast orientation path."""

from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.geometry_calibration import (
    GeometryFitContext,
    public_coarse_shape,
    public_rule_expected,
    public_shape_extents,
)
from src.geometry_profile_schema import materialize_runtime_profile


FAST_GEOMETRY_FORMAT_VERSION = 1
FAST_GEOMETRY_MAX_SIDE = 256
_LOW_CONFIDENCE = "FAST_GEOMETRY_LOW_CONFIDENCE"
_CACHE_INVALID = "FAST_CACHE_REVISION_MISMATCH"


class FastGeometryCacheError(ValueError):
    """Raised when a compiled fast-geometry cache cannot be used safely."""

    code = _CACHE_INVALID

    def __init__(self, message: str = "compiled geometry type or format is invalid") -> None:
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class CompiledFastGeometry:
    format_version: int
    profile_revision: int | None
    directions: dict[str, dict[str, Any] | None]


@dataclass(frozen=True)
class FastGeometryVariants:
    images: tuple[np.ndarray, np.ndarray, np.ndarray]
    status: str
    needs_review: bool
    review_reasons: Sequence[str]
    directions: dict[str, dict[str, Any]]
    timings_ms: dict[str, float]


def compile_fast_geometry(profile: Mapping[str, Any] | None) -> CompiledFastGeometry:
    """Copy canonical direction profiles into the compact runtime contract."""
    if profile is None:
        return CompiledFastGeometry(FAST_GEOMETRY_FORMAT_VERSION, None, {"front": None, "back": None})
    runtime = materialize_runtime_profile(profile)
    directions = runtime.get("directions", {}) if isinstance(runtime, Mapping) else {}
    compiled: dict[str, dict[str, Any] | None] = {}
    for label in ("front", "back"):
        direction = directions.get(label) if isinstance(directions, Mapping) else None
        configured = isinstance(direction, Mapping) and direction.get("anchor") is not None and bool(direction.get("rules"))
        compiled[label] = deepcopy(dict(direction)) if configured else None
    revision = profile.get("profile_revision")
    return CompiledFastGeometry(FAST_GEOMETRY_FORMAT_VERSION, revision if isinstance(revision, int) else None, compiled)


class FastGeometryProcessor:
    def __init__(self, calibrator: Any, *, max_side: int = FAST_GEOMETRY_MAX_SIDE) -> None:
        self.calibrator = calibrator
        self.max_side = int(max_side)

    def build_variants(self, image: np.ndarray, compiled: CompiledFastGeometry) -> FastGeometryVariants:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR HxWx3 array")
        self._validate_compiled(compiled)
        raw = image.copy()
        configured = {
            label: compiled.directions[label]
            for label in ("front", "back")
            if compiled.directions[label] is not None
        }
        timings = {"geometry_context": 0.0, "geometry_fit": 0.0, "mask_build": 0.0}
        diagnostics: dict[str, dict[str, Any]] = {
            label: {"status": "not_configured"} for label in ("front", "back")
        }
        if not configured:
            return FastGeometryVariants((raw, raw.copy(), raw.copy()), "not_configured", False, (), diagnostics, timings)

        started = perf_counter()
        resized = self._resize_context(image)
        context = self.calibrator.prepare_context(resized)
        timings["geometry_context"] = (perf_counter() - started) * 1000.0
        outputs = {"front": raw.copy(), "back": raw.copy()}
        failed = False
        for label in ("front", "back"):
            direction = compiled.directions.get(label)
            if direction is None:
                continue
            fit_started = perf_counter()
            mask_started: float | None = None
            try:
                fit = self.calibrator.fit(resized, direction, context=self._filtered_context(context, direction))
                timings["geometry_fit"] += (perf_counter() - fit_started) * 1000.0
                mask_started = perf_counter()
                mask = self._fit_mask(fit, resized.shape[:2])
                source_mask = (
                    cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if mask is not None else None
                )
                ignored_ratio = (
                    float(np.count_nonzero(source_mask) / max(source_mask.size, 1))
                    if source_mask is not None else 1.0
                )
                if fit.get("status") != "active" or source_mask is None or ignored_ratio >= 0.55:
                    diagnostics[label] = self._diagnostic(
                        fit, ignored_ratio=ignored_ratio, status="low_confidence",
                    )
                    failed = True
                else:
                    outputs[label] = self._fast_fill(
                        image, source_mask, direction.get("fill_bgr", [0, 0, 0]),
                    )
                    diagnostics[label] = self._diagnostic(fit, ignored_ratio=ignored_ratio)
                timings["mask_build"] += (perf_counter() - mask_started) * 1000.0
            except Exception:
                if mask_started is None:
                    timings["geometry_fit"] += (perf_counter() - fit_started) * 1000.0
                else:
                    timings["mask_build"] += (perf_counter() - mask_started) * 1000.0
                diagnostics[label] = {
                    "status": "low_confidence",
                    "reason_code": "FAST_GEOMETRY_PROCESSING_FAILED",
                }
                failed = True
        return FastGeometryVariants(
            (raw, outputs["front"], outputs["back"]),
            "low_confidence" if failed else "active",
            failed,
            (_LOW_CONFIDENCE,) if failed else (),
            diagnostics,
            timings,
        )

    def _resize_context(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(1.0, float(self.max_side) / max(height, width))
        if scale == 1.0:
            return image
        return cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _validate_compiled(compiled: CompiledFastGeometry) -> None:
        if not isinstance(compiled, CompiledFastGeometry):
            raise FastGeometryCacheError()
        if compiled.format_version != FAST_GEOMETRY_FORMAT_VERSION:
            raise FastGeometryCacheError()
        if compiled.profile_revision is not None and type(compiled.profile_revision) is not int:
            raise FastGeometryCacheError()
        if not isinstance(compiled.directions, dict):
            raise FastGeometryCacheError()
        for label in ("front", "back"):
            if label not in compiled.directions:
                raise FastGeometryCacheError()
            direction = compiled.directions[label]
            if direction is not None and not isinstance(direction, Mapping):
                raise FastGeometryCacheError()

    @staticmethod
    def _filtered_context(context: GeometryFitContext, direction: Mapping[str, Any]) -> GeometryFitContext:
        """Keep contours within the direction's coarse center/scale windows."""
        anchor = direction.get("anchor")
        if not isinstance(anchor, Mapping) or not isinstance(anchor.get("coarse"), Mapping):
            return context
        height, width = context.image_shape
        expected: list[tuple[float, float, float, float]] = []

        def add_window(shape: Mapping[str, Any]) -> None:
            extent_x, extent_y = public_shape_extents(shape)
            expected.append((
                float(shape["cx"]), float(shape["cy"]),
                max(float(extent_x), 4.0), max(float(extent_y), 4.0),
            ))

        coarse = public_coarse_shape(anchor, (height, width))
        add_window(coarse)
        for rule in direction.get("rules", []):
            if not isinstance(rule, Mapping) or not rule.get("enabled", True):
                continue
            add_window(public_rule_expected(rule, coarse))
        kept = []
        for contour in context.contours:
            x, y, contour_width, contour_height = cv2.boundingRect(np.asarray(contour, dtype=np.float32))
            cx, cy = x + contour_width / 2.0, y + contour_height / 2.0
            if any(
                abs(cx - expected_cx) <= expected_rx * 0.75 + contour_width / 2.0
                and abs(cy - expected_cy) <= expected_ry * 0.75 + contour_height / 2.0
                and 0.30 <= max(contour_width / (2.0 * expected_rx), contour_height / (2.0 * expected_ry)) <= 2.75
                for expected_cx, expected_cy, expected_rx, expected_ry in expected
            ):
                kept.append(contour)
        return GeometryFitContext(context.image_shape, tuple(kept))

    @staticmethod
    def _fit_mask(fit: Mapping[str, Any], image_shape: tuple[int, int]) -> np.ndarray | None:
        masks = []
        rules = fit.get("rules")
        if isinstance(rules, SequenceABC) and not isinstance(rules, (str, bytes, bytearray)):
            for rule in rules:
                mask = rule.get("ignore_mask") if isinstance(rule, Mapping) and rule.get("status") == "active" else None
                if isinstance(mask, np.ndarray) and mask.shape == image_shape:
                    masks.append(mask)
        elif rules is None and fit.get("status") == "active":
            # Older adapters may expose only an active aggregate; once rule
            # results exist, their active masks are the authoritative source.
            top_level = fit.get("ignore_mask")
            if isinstance(top_level, np.ndarray) and top_level.shape == image_shape:
                masks.append(top_level)
        if not masks:
            return None
        result = np.zeros(image_shape, dtype=np.uint8)
        for mask in masks:
            result = cv2.bitwise_or(result, mask)
        return result

    @staticmethod
    def _fast_fill(image: np.ndarray, mask: np.ndarray, fill_bgr: Sequence[int]) -> np.ndarray:
        binary = (mask != 0).astype(np.uint8)
        sigma = max(0.5, min(image.shape[:2]) * 0.005)
        alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigmaX=sigma)
        alpha *= binary
        fill = np.asarray(fill_bgr, dtype=np.float32).reshape(1, 1, 3)
        return np.rint(image.astype(np.float32) * (1.0 - alpha[..., None]) + fill * alpha[..., None]).astype(np.uint8)

    @staticmethod
    def _diagnostic(fit: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return None
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, Mapping):
                return {
                    str(key): clean(item)
                    for key, item in value.items()
                    if not isinstance(item, np.ndarray)
                }
            if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
                return [clean(item) for item in value if not isinstance(item, np.ndarray)]
            return value

        result = clean(fit)
        result.update(updates)
        return result
