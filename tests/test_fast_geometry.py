from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import cv2
import numpy as np
import pytest

from src.fast_geometry import FastGeometryCacheError, FastGeometryProcessor, compile_fast_geometry
from src.geometry_calibration import GeometryCalibrator, GeometryFitContext


def marker_image(marker: int, *, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), marker, dtype=np.uint8)


def configured_profile() -> dict:
    directions = {}
    for label in ("front", "back"):
        directions[label] = {
            "side": label,
            "anchor": {
                "shape": "circle",
                "coarse": {"cx": 0.5, "cy": 0.5, "r": 0.4, "angle_deg": 0.0},
            },
            "rules": [{
                "rule_id": f"{label}-inner",
                "name": f"{label} inner",
                "shape": "circle",
                "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5, "angle_deg": 0.0},
                "mode": "inside",
                "margin_ratio": -0.04,
                "margin_semantics": "signed_boundary_v2",
                "enabled": True,
            }],
            "fill_bgr": [7, 7, 7],
        }
    return {"schema_version": 1, "profile_revision": 12, "directions": directions}


class FakeCalibrator:
    def __init__(self, *, fail_side: str | None = None, mask_center: bool = False,
                 rule_masks: tuple[np.ndarray, ...] = (), include_top_level_mask: bool = True) -> None:
        self.fail_side = fail_side
        self.mask_center = mask_center
        self.rule_masks = rule_masks
        self.include_top_level_mask = include_top_level_mask
        self.prepare_calls = 0
        self.fit_labels: list[str] = []
        self.context_shapes: list[tuple[int, int]] = []

    def prepare_context(self, image: np.ndarray) -> GeometryFitContext:
        self.prepare_calls += 1
        return GeometryFitContext(image_shape=image.shape[:2], contours=())

    def fit(self, image: np.ndarray, direction_profile: dict, *, context: GeometryFitContext):
        label = direction_profile["side"]
        self.fit_labels.append(label)
        self.context_shapes.append(context.image_shape)
        if label == self.fail_side:
            return {"status": "low_confidence", "reason_code": "boundary_candidate_not_found", "rules": []}
        masks = self.rule_masks
        if self.mask_center:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            height, width = image.shape[:2]
            mask[height // 3:height * 2 // 3, width // 3:width * 2 // 3] = 255
            masks = (mask,)
        union = np.zeros(image.shape[:2], dtype=np.uint8)
        rules = []
        for index, mask in enumerate(masks):
            union = cv2.bitwise_or(union, mask)
            rules.append({"rule_id": f"rule-{index}", "mode": "inside", "status": "active", "ignore_mask": mask})
        result = {
            "status": "active",
            "rules": rules,
            "ignored_ratio": float(np.count_nonzero(union) / union.size),
        }
        if self.include_top_level_mask:
            result["ignore_mask"] = union
        return result


def processor_with_fake_calibrator(**kwargs):
    calibrator = FakeCalibrator(**kwargs)
    return FastGeometryProcessor(calibrator), calibrator


def test_no_rules_produce_three_raw_slots_without_review():
    processor, calibrator = processor_with_fake_calibrator()
    image = marker_image(17)

    variants = processor.build_variants(image, compile_fast_geometry(None))

    assert len(variants.images) == 3
    assert all(np.array_equal(item, image) for item in variants.images)
    assert variants.status == "not_configured"
    assert variants.needs_review is False
    assert calibrator.prepare_calls == 0


def test_front_and_back_share_one_resized_context():
    processor, calibrator = processor_with_fake_calibrator()

    variants = processor.build_variants(marker_image(17, size=512), compile_fast_geometry(configured_profile()))

    assert calibrator.prepare_calls == 1
    assert calibrator.fit_labels == ["front", "back"]
    assert calibrator.context_shapes == [(256, 256), (256, 256)]
    assert variants.timings_ms.keys() >= {"geometry_context", "geometry_fit", "mask_build"}


def test_inside_and_outside_masks_use_fast_fill_not_telea(monkeypatch):
    monkeypatch.setattr(cv2, "inpaint", lambda *args, **kwargs: pytest.fail("Telea called"))
    processor, _ = processor_with_fake_calibrator(mask_center=True)

    variants = processor.build_variants(marker_image(17), compile_fast_geometry(configured_profile()))

    assert variants.images[1][32, 32].tolist() == [7, 7, 7]


def test_one_failed_direction_uses_raw_slot_and_requests_review():
    processor, _ = processor_with_fake_calibrator(fail_side="back")
    image = marker_image(17)

    variants = processor.build_variants(image, compile_fast_geometry(configured_profile()))

    assert np.array_equal(variants.images[2], image)
    assert variants.needs_review is True
    assert variants.review_reasons == ("FAST_GEOMETRY_LOW_CONFIDENCE",)


def test_compilation_preserves_circle_ellipse_rotated_rectangle_and_signed_margin():
    profile = configured_profile()
    front = profile["directions"]["front"]
    front["anchor"] = {
        "shape": "ellipse",
        "coarse": {"cx": 0.42, "cy": 0.58, "rx": 0.31, "ry": 0.22, "angle_deg": 15.0},
    }
    front["rules"] = [
        {"rule_id": "circle", "name": "circle", "shape": "circle", "geometry": {"cx": 0.1, "cy": 0.0, "r": 0.4, "angle_deg": 0.0}, "mode": "inside", "margin_ratio": -0.04, "margin_semantics": "signed_boundary_v2", "enabled": True},
        {"rule_id": "ellipse", "name": "ellipse", "shape": "ellipse", "geometry": {"cx": 0.0, "cy": -0.1, "rx": 0.5, "ry": 0.3, "angle_deg": 9.0}, "mode": "inside", "margin_ratio": 0.0, "margin_semantics": "signed_boundary_v2", "enabled": True},
        {"rule_id": "rectangle", "name": "rectangle", "shape": "rotated_rectangle", "geometry": {"cx": 0.0, "cy": 0.0, "half_width": 0.6, "half_height": 0.2, "angle_deg": -8.0}, "mode": "outside", "margin_ratio": 0.02, "margin_semantics": "signed_boundary_v2", "enabled": True},
    ]
    expected = deepcopy(profile["directions"])

    compiled = compile_fast_geometry(profile)

    assert compiled.format_version == 1
    assert compiled.profile_revision == 12
    assert compiled.directions == expected


def test_multiple_rules_union_their_ignore_masks():
    first = np.zeros((64, 64), dtype=np.uint8)
    second = np.zeros((64, 64), dtype=np.uint8)
    first[8:16, 8:16] = 255
    second[40:48, 40:48] = 255
    processor, _ = processor_with_fake_calibrator(
        rule_masks=(first, second), include_top_level_mask=False,
    )

    variants = processor.build_variants(marker_image(17), compile_fast_geometry(configured_profile()))

    masked = variants.images[1]
    assert masked[10, 10].tolist() == [7, 7, 7]
    assert masked[42, 42].tolist() == [7, 7, 7]
    assert masked[24, 24].tolist() == [17, 17, 17]


def test_candidate_filter_keeps_distinct_inner_and_outer_boundaries():
    image = marker_image(30, size=512)
    cv2.circle(image, (256, 256), 180, (230, 230, 230), 5)
    cv2.circle(image, (256, 256), 104, (170, 170, 170), 5)
    profile = configured_profile()
    profile["directions"]["front"] = {
        "side": "front",
        "anchor": {"shape": "circle", "coarse": {"cx": 0.5, "cy": 0.5, "r": 180 / 512, "angle_deg": 0.0}},
        "rules": [{"rule_id": "inner", "name": "inner", "shape": "circle", "geometry": {"cx": 0.0, "cy": 0.0, "r": 104 / 180, "angle_deg": 0.0}, "mode": "inside", "margin_ratio": 0.0, "enabled": True}],
        "fill_bgr": [7, 7, 7],
    }
    profile["directions"]["back"] = {
        "side": "back",
        "anchor": {"shape": "circle", "coarse": {"cx": 0.5, "cy": 0.5, "r": 104 / 512, "angle_deg": 0.0}},
        "rules": [{"rule_id": "outer", "name": "outer", "shape": "circle", "geometry": {"cx": 0.0, "cy": 0.0, "r": 180 / 104, "angle_deg": 0.0}, "mode": "outside", "margin_ratio": 0.0, "enabled": True}],
        "fill_bgr": [7, 7, 7],
    }
    variants = FastGeometryProcessor(GeometryCalibrator()).build_variants(image, compile_fast_geometry(profile))

    inner = variants.directions["front"]["rules"][0]["fitted_shape"]["rx"]
    outer = variants.directions["back"]["rules"][0]["fitted_shape"]["rx"]
    assert inner == pytest.approx(52.0, rel=0.05)
    assert outer == pytest.approx(90.0, rel=0.05)


def test_candidate_filter_keeps_slender_rotated_ellipse_boundaries():
    profile = configured_profile()
    profile["directions"]["front"] = {
        "side": "front",
        "anchor": {
            "shape": "ellipse",
            "coarse": {"cx": 0.5, "cy": 0.5, "rx": 180 / 512, "ry": 32 / 512, "angle_deg": 45.0},
        },
        "rules": [{
            "rule_id": "inner", "name": "inner", "shape": "ellipse",
            "geometry": {"cx": 0.0, "cy": 0.0, "rx": 0.5, "ry": 0.5, "angle_deg": 0.0},
            "mode": "inside", "margin_ratio": 0.0, "enabled": True,
        }],
        "fill_bgr": [7, 7, 7],
    }
    profile["directions"]["back"] = {"side": "back", "anchor": None, "rules": []}
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
    local = np.column_stack((90.0 * np.cos(angles), 8.0 * np.sin(angles)))
    rotation = np.array([[np.sqrt(0.5), -np.sqrt(0.5)], [np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float32)
    contour = (local @ rotation.T + np.array([128.0, 128.0], dtype=np.float32)).astype(np.float32)
    context = GeometryFitContext((256, 256), (contour,))

    filtered = FastGeometryProcessor._filtered_context(
        context, compile_fast_geometry(profile).directions["front"],
    )

    assert len(filtered.contours) == 1
    assert np.array_equal(filtered.contours[0], contour)
    assert GeometryCalibrator._rank_candidates(
        filtered.contours,
        {"shape": "ellipse", "cx": 128.0, "cy": 128.0, "rx": 90.0, "ry": 8.0, "angle_deg": 45.0},
        context.image_shape,
        0.08,
        prefer_larger=False,
    )


class StaleRatioCalibrator(FakeCalibrator):
    def fit(self, image: np.ndarray, direction_profile: dict, *, context: GeometryFitContext):
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        return {
            "status": "active",
            "rules": [{"status": "active", "ignore_mask": mask}],
            "ignore_mask": mask,
            "ignored_ratio": 0.01,
        }


def test_final_resized_union_ratio_rejects_stale_fit_ratio():
    image = marker_image(17, size=512)
    variants = FastGeometryProcessor(StaleRatioCalibrator()).build_variants(
        image, compile_fast_geometry(configured_profile()),
    )

    assert variants.status == "low_confidence"
    assert variants.needs_review is True
    assert np.array_equal(variants.images[1], image)


@pytest.mark.parametrize("compiled", [
    object(),
    replace(compile_fast_geometry(configured_profile()), format_version=999),
])
def test_malformed_compiled_geometry_raises_stable_cache_error(compiled):
    processor, _ = processor_with_fake_calibrator()

    with pytest.raises(FastGeometryCacheError, match="FAST_CACHE_REVISION_MISMATCH"):
        processor.build_variants(marker_image(17), compiled)


def test_post_fit_processing_failure_isolated_to_one_direction():
    processor, _ = processor_with_fake_calibrator(mask_center=True)
    profile = configured_profile()
    profile["directions"]["back"]["fill_bgr"] = [7, 7]
    image = marker_image(17)

    variants = processor.build_variants(image, compile_fast_geometry(profile))

    assert variants.images[1][32, 32].tolist() == [7, 7, 7]
    assert np.array_equal(variants.images[2], image)
    assert variants.needs_review is True


class MixedRuleMaskCalibrator(FakeCalibrator):
    def fit(self, image: np.ndarray, direction_profile: dict, *, context: GeometryFitContext):
        active = np.zeros(image.shape[:2], dtype=np.uint8)
        failed = np.zeros(image.shape[:2], dtype=np.uint8)
        active[8:16, 8:16] = 255
        failed[40:48, 40:48] = 255
        return {
            "status": "active",
            "rules": [
                {"status": "active", "ignore_mask": active},
                {"status": "low_confidence", "ignore_mask": failed},
            ],
            "ignore_mask": np.full(image.shape[:2], 255, dtype=np.uint8),
            "ignored_ratio": 0.01,
        }


def test_failed_rule_masks_and_stale_aggregate_do_not_contribute():
    image = marker_image(17)
    variants = FastGeometryProcessor(MixedRuleMaskCalibrator()).build_variants(
        image, compile_fast_geometry(configured_profile()),
    )

    masked = variants.images[1]
    assert variants.status == "active"
    assert masked[10, 10].tolist() == [7, 7, 7]
    assert masked[42, 42].tolist() == [17, 17, 17]
    assert masked[24, 24].tolist() == [17, 17, 17]


class NestedDiagnosticsCalibrator(FakeCalibrator):
    def fit(self, image: np.ndarray, direction_profile: dict, *, context: GeometryFitContext):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[8:16, 8:16] = 255
        return {
            "status": "active",
            "rules": [{"status": "active", "ignore_mask": mask}],
            "diagnostics": {
                "nested": (
                    np.int64(3),
                    {"score": np.float32(1.25), "mask": np.ones((2, 2), dtype=np.uint8)},
                ),
            },
        }


def test_diagnostic_cleanup_serializes_nested_numpy_values():
    variants = FastGeometryProcessor(NestedDiagnosticsCalibrator()).build_variants(
        marker_image(17), compile_fast_geometry(configured_profile()),
    )

    serialized = json.dumps(variants.directions)
    decoded = json.loads(serialized)
    assert decoded["front"]["diagnostics"]["nested"] == [3, {"score": 1.25}]
