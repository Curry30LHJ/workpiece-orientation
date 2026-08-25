from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import geometry_calibration
from src.geometry_calibration import (
    GeometryCalibrator,
    _filter_rule_candidates,
    _shape_error,
    _shape_extents,
    _shape_mask,
    apply_geometry_fit,
    apply_ignore_mask,
    filter_features_by_mask,
    geometry_feature_mask,
)


def _ellipse_profile(*, mode: str = "inside", rule_shape: str = "circle") -> dict:
    return {
        "anchor": {
            "shape": "ellipse",
            "coarse": {
                "cx": 0.50,
                "cy": 0.50,
                "rx": 0.31,
                "ry": 0.25,
                "angle_deg": 18.0,
            },
        },
        "rules": [
            {
                "rule_id": "inner",
                "name": "inner",
                "shape": rule_shape,
                "geometry": (
                    {"cx": 0.0, "cy": 0.0, "r": 0.56}
                    if rule_shape == "circle"
                    else {"cx": 0.0, "cy": 0.0, "rx": 0.58, "ry": 0.44, "angle_deg": 8.0}
                ),
                "mode": mode,
                "margin_ratio": 0.02,
                "enabled": True,
            }
        ],
    }


def _circle_profile(*, mode: str = "outside", shape: str = "circle") -> dict:
    return {
        "anchor": {
            "shape": shape,
            "coarse": {
                "cx": 0.50,
                "cy": 0.50,
                "r": 0.34,
                "rx": 0.34,
                "ry": 0.30,
                "angle_deg": 0.0,
            },
        },
        "rules": [
            {
                "rule_id": "outer",
                "name": "outer",
                "shape": "circle" if shape == "circle" else "rotated_rectangle",
                "geometry": (
                    {"cx": 0.0, "cy": 0.0, "r": 1.0}
                    if shape == "circle"
                    else {"cx": 0.0, "cy": 0.0, "half_width": 1.0,
                          "half_height": 1.0, "angle_deg": 0.0}
                ),
                "mode": mode,
                "margin_ratio": 0.02,
                "enabled": True,
            }
        ],
    }


def _synthetic_ellipse(
    *, center: tuple[int, int] = (155, 92), axes: tuple[int, int] = (72, 54), angle: float = 24.0
) -> np.ndarray:
    image = np.full((220, 320, 3), 35, dtype=np.uint8)
    cv2.ellipse(image, center, axes, angle, 0, 360, (210, 210, 210), 3)
    inner_axes = (max(4, int(axes[0] * 0.56)), max(4, int(axes[1] * 0.56)))
    cv2.ellipse(image, center, inner_axes, angle, 0, 360, (125, 125, 125), 2)
    cv2.ellipse(image, center, (max(3, int(axes[0] * 0.42)), max(3, int(axes[1] * 0.32))),
                angle + 8.0, 0, 360, (165, 165, 165), 2)
    return image


def _synthetic_circle_with_corner_intrusions() -> np.ndarray:
    image = np.full((256, 256, 3), 240, dtype=np.uint8)
    cv2.circle(image, (128, 128), 85, (30, 30, 30), 4)
    cv2.circle(image, (128, 128), 78, (100, 100, 100), 2)
    cv2.rectangle(image, (0, 0), (32, 38), (15, 15, 15), -1)
    cv2.rectangle(image, (225, 219), (255, 255), (15, 15, 15), -1)
    return image


def _synthetic_rotated_rectangle() -> np.ndarray:
    image = np.full((260, 300, 3), 230, dtype=np.uint8)
    rect = ((150.0, 130.0), (170.0, 100.0), 28.0)
    corners = cv2.boxPoints(rect).astype(np.int32)
    cv2.drawContours(image, [corners], 0, (25, 25, 25), 4)
    return image


def _synthetic_nested_circles() -> np.ndarray:
    image = np.full((360, 360, 3), 245, dtype=np.uint8)
    cv2.circle(image, (180, 180), 150, (35, 35, 35), 4)
    cv2.circle(image, (180, 180), 108, (105, 105, 105), 4)
    return image


def _circle_anchor(cx: float = 180.0, cy: float = 180.0, radius: float = 150.0) -> dict:
    return {
        "shape": "circle",
        "cx": cx,
        "cy": cy,
        "rx": radius,
        "ry": radius,
        "angle_deg": 0.0,
    }


def _circle_contour(radius: float, *, center: tuple[float, float] = (180.0, 180.0)) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
    return np.column_stack((
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    )).astype(np.float32)


def _ellipse_contour(
    rx: float,
    ry: float,
    *,
    angle_deg: float,
    center: tuple[float, float] = (180.0, 180.0),
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False, dtype=np.float32)
    points = np.column_stack((rx * np.cos(angles), ry * np.sin(angles)))
    rotation = np.deg2rad(angle_deg)
    matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)],
    ], dtype=np.float32)
    return (points @ matrix.T + np.asarray(center, dtype=np.float32)).astype(np.float32)


def test_circle_extents_do_not_change_with_fit_angle():
    circle = {
        "shape": "circle", "cx": 180.0, "cy": 180.0,
        "rx": 142.0, "ry": 142.0, "angle_deg": 45.0,
    }

    assert _shape_extents(circle) == pytest.approx((142.0, 142.0))


def test_rotated_ellipse_uses_exact_axis_aligned_extents():
    ellipse = {
        "shape": "ellipse", "cx": 0.0, "cy": 0.0,
        "rx": 20.0, "ry": 10.0, "angle_deg": 45.0,
    }

    expected_extent = np.sqrt(250.0)
    assert _shape_extents(ellipse) == pytest.approx((expected_extent, expected_extent))


def test_circle_candidate_score_ignores_fit_angle():
    expected = {
        "shape": "circle", "cx": 180.0, "cy": 180.0,
        "rx": 142.0, "ry": 142.0, "angle_deg": -7.7,
    }
    candidate = {**expected, "angle_deg": 89.0}

    assert _shape_error(candidate, expected, (360, 360), 0.16) == pytest.approx(0.0)


def test_near_circular_ellipse_candidate_has_stable_zero_angle():
    expected = {
        "shape": "ellipse", "cx": 180.0, "cy": 180.0,
        "rx": 150.0, "ry": 149.0, "angle_deg": 0.0,
    }

    ranked = GeometryCalibrator._rank_candidates(
        [_ellipse_contour(150.0, 149.0, angle_deg=77.0)],
        expected,
        (360, 360),
        0.16,
        prefer_larger=False,
    )

    assert ranked[0]["angle_deg"] == pytest.approx(0.0)


def test_rule_topology_can_select_candidate_beyond_similarity_top_three():
    expected = _circle_anchor(radius=150.0)
    contours = [_circle_contour(radius) for radius in (150.0, 149.0, 151.0, 140.0)]

    ranked = GeometryCalibrator._rank_candidates(
        contours, expected, (360, 360), 0.16, prefer_larger=False,
    )
    valid = _filter_rule_candidates(ranked, expected, "inside")

    assert any(candidate["rx"] == pytest.approx(140.0, abs=0.5) for candidate in valid)


def test_fit_reference_reports_when_quality_candidates_fail_topology():
    image = np.full((360, 360, 3), 245, dtype=np.uint8)
    cv2.circle(image, (180, 180), 150, (35, 35, 35), -1)
    seed = {"shape": "circle", "cx": 180.0, "cy": 180.0, "r": 150.0}

    result = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=0.0)

    assert result["status"] == "low_confidence"
    assert result["rule_fit"]["reason_code"] == "topology_constraint_failed"
    assert result["rule_fit"]["quality_candidate_count"] > 0
    assert result["rule_fit"]["topology_valid_candidate_count"] == 0


def test_inner_circle_tracks_shift_scale_and_rotation_in_anchor_coordinates():
    image = _synthetic_ellipse(center=(155, 92), axes=(72, 54), angle=24.0)

    result = GeometryCalibrator().fit(image, _ellipse_profile(mode="inside"))

    assert result["status"] == "active"
    assert abs(result["anchor"]["cx"] - 155) <= 4
    assert abs(result["anchor"]["cy"] - 92) <= 4
    assert result["rules"][0]["edge_support"] >= 0.35
    assert result["ignore_mask"][92, 155] == 255
    assert result["ignore_mask"][20, 20] == 0


def test_rule_center_rotates_with_object_anchor():
    image = np.full((220, 320, 3), 35, dtype=np.uint8)
    anchor_center = np.array([155.0, 110.0], dtype=np.float32)
    anchor_axes = (88, 58)
    anchor_angle = 28.0
    cv2.ellipse(image, tuple(anchor_center.astype(int)), anchor_axes, anchor_angle,
                0, 360, (210, 210, 210), 3)
    local = np.array([[0.20 * anchor_axes[0], -0.10 * anchor_axes[1]]], dtype=np.float32)
    theta = np.deg2rad(anchor_angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    rule_center = (anchor_center + local @ rotation.T)[0]
    cv2.ellipse(image, tuple(np.rint(rule_center).astype(int)), (22, 15), anchor_angle,
                0, 360, (135, 135, 135), 3)
    profile = {
        "anchor": {"shape": "ellipse", "coarse": {
            "cx": 155 / 320, "cy": 110 / 220, "rx": 88 / 320,
            "ry": 58 / 220, "angle_deg": anchor_angle}},
        "rules": [{
            "rule_id": "inner", "name": "inner", "shape": "ellipse",
            "geometry": {"cx": 0.20, "cy": -0.10, "rx": 0.25, "ry": 0.26,
                         "angle_deg": 0.0},
            "mode": "inside", "margin_ratio": 0.08, "enabled": True,
        }],
    }

    result = GeometryCalibrator().fit(image, profile)

    assert result["status"] == "active"
    fitted = result["rules"][0]
    assert abs(fitted["cx"] - rule_center[0]) <= 5
    assert abs(fitted["cy"] - rule_center[1]) <= 5


def test_outer_rule_ignores_intrusion_but_keeps_shrunken_object_edge():
    image = _synthetic_circle_with_corner_intrusions()

    result = GeometryCalibrator().fit(image, _circle_profile(mode="outside"))

    assert result["status"] == "active"
    assert result["ignore_mask"][5, 5] == 255
    assert result["ignore_mask"][128, 128] == 0
    assert result["ignore_mask"][128, 42] == 0


def test_inside_rule_rejects_outer_contour_candidate():
    image = _synthetic_nested_circles()
    rule = {
        "rule_id": "inner",
        "shape": "circle",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 1.0},
        "mode": "inside",
        "margin_ratio": 0.0,
        "enabled": True,
    }

    fitted = GeometryCalibrator()._fit_rule(
        image, _circle_anchor(), rule, image.shape[:2]
    )

    assert fitted["status"] == "active"
    assert fitted["relation_to_anchor"] == "contained"
    assert fitted["topology_valid"] is True
    assert fitted["fitted_shape"]["rx"] < 150.0 * 0.98
    assert np.count_nonzero(fitted["ignore_mask"]) / fitted["ignore_mask"].size < 0.55


def test_inside_rule_without_nested_boundary_returns_low_confidence_without_mask():
    image = np.full((360, 360, 3), 245, dtype=np.uint8)
    cv2.circle(image, (180, 180), 150, (35, 35, 35), 4)
    rule = {
        "rule_id": "inner",
        "shape": "circle",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 1.0},
        "mode": "inside",
        "margin_ratio": 0.0,
        "enabled": True,
    }

    fitted = GeometryCalibrator()._fit_rule(
        image, _circle_anchor(), rule, image.shape[:2]
    )

    assert fitted["status"] == "low_confidence"
    assert fitted["reason_code"] == "topology_constraint_failed"
    assert "ignore_mask" not in fitted


def test_outside_rule_candidate_encloses_anchor():
    image = np.full((360, 360, 3), 245, dtype=np.uint8)
    cv2.circle(image, (180, 180), 150, (35, 35, 35), 4)
    rule = {
        "rule_id": "outside",
        "shape": "circle",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 1.5},
        "mode": "outside",
        "margin_ratio": 0.0,
        "enabled": True,
    }

    fitted = GeometryCalibrator()._fit_rule(
        image, _circle_anchor(radius=100.0), rule, image.shape[:2]
    )

    assert fitted["status"] == "active"
    assert fitted["relation_to_anchor"] == "encloses"
    assert fitted["topology_valid"] is True


def test_ellipse_rule_and_rotated_rectangle_keep_shape_diagnostics():
    ellipse = GeometryCalibrator().fit(_synthetic_ellipse(), _ellipse_profile(rule_shape="ellipse"))
    rectangle_profile = _circle_profile(shape="rotated_rectangle")
    rectangle_profile["anchor"] = {
        "shape": "rotated_rectangle",
        "coarse": {"cx": 0.50, "cy": 0.50, "half_width": 0.30,
                   "half_height": 0.20, "angle_deg": 28.0},
    }
    rectangle = GeometryCalibrator().fit(_synthetic_rotated_rectangle(), rectangle_profile)

    assert ellipse["status"] == "active"
    assert ellipse["rules"][0]["shape"] == "ellipse"
    assert rectangle["status"] == "active"
    assert rectangle["anchor"]["shape"] == "rotated_rectangle"
    assert abs(rectangle["anchor"]["angle_deg"] - 28.0) <= 3.0


def test_multiple_rules_are_combined_and_margin_direction_is_explicit():
    profile = _ellipse_profile(mode="inside")
    profile["rules"].append({
        "rule_id": "outer",
        "name": "outer",
        "shape": "circle",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.95},
        "mode": "outside",
        "margin_ratio": 0.02,
        "enabled": True,
    })
    result = GeometryCalibrator().fit(_synthetic_ellipse(), profile)

    assert result["status"] == "active"
    assert result["ignored_ratio"] > 0.05
    assert result["ignore_mask"][92, 155] == 255
    assert result["ignore_mask"][0, 0] == 255


def test_anchor_failure_returns_no_partial_mask():
    result = GeometryCalibrator().fit(
        np.full((256, 256, 3), 127, dtype=np.uint8),
        _circle_profile(),
    )

    assert result["status"] == "low_confidence"
    assert "ignore_mask" not in result


def test_apply_ignore_mask_fills_only_ignored_pixels():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[1, 1] = (1, 2, 3)
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[1, 1] = 255

    filled = apply_ignore_mask(image, mask, (9, 8, 7))

    assert tuple(filled[1, 1]) == (9, 8, 7)
    assert tuple(filled[0, 0]) == (0, 0, 0)
    assert tuple(image[1, 1]) == (1, 2, 3)


def test_apply_geometry_fit_repairs_inside_and_fills_outside():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    for x in range(image.shape[1]):
        image[:, x] = (x * 7, x * 5, x * 3)
    inside = np.zeros(image.shape[:2], dtype=np.uint8)
    outside = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(inside, (16, 16), 6, 255, -1)
    outside[0, 0] = 255
    fit = {
        "status": "active",
        "rules": [
            {"mode": "inside", "ignore_mask": inside},
            {"mode": "outside", "ignore_mask": outside},
        ],
    }

    repaired = apply_geometry_fit(image, fit, (9, 8, 7))

    assert tuple(repaired[0, 0]) == (9, 8, 7)
    assert np.any(repaired[inside != 0] != np.asarray((9, 8, 7), dtype=np.uint8))
    assert np.std(repaired[inside != 0].astype(np.float32)) > 0.0
    assert tuple(repaired[31, 31]) == tuple(image[31, 31])


def test_geometry_feature_mask_filters_inside_and_outside_rules():
    image_shape = (32, 32)
    inside = np.zeros(image_shape, dtype=np.uint8)
    outside = np.zeros(image_shape, dtype=np.uint8)
    inside[8:16, 8:16] = 255
    outside[:4, :4] = 255
    fit = {
        "status": "active",
        "rules": [
            {"mode": "inside", "ignore_mask": inside},
            {"mode": "outside", "ignore_mask": outside},
        ],
    }

    feature_mask = geometry_feature_mask(fit)

    assert feature_mask.shape == image_shape
    assert feature_mask[10, 10] == 255
    assert feature_mask[1, 1] == 255


def test_filter_features_by_mask_removes_aligned_tensors():
    features = {
        "keypoints": np.array([[[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]]], dtype=np.float32),
        "scores": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "descriptors": np.arange(12, dtype=np.float32).reshape(1, 4, 3),
    }
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[:4, :4] = 255

    filtered = filter_features_by_mask(features, mask)

    assert filtered["keypoints"].shape == (1, 2, 2)
    assert filtered["keypoints"][0].tolist() == [[5.0, 5.0], [9.0, 9.0]]
    assert filtered["scores"].shape == (1, 2)
    assert filtered["descriptors"].shape == (1, 4, 2)


def test_fit_reference_returns_object_geometry_and_ranked_candidates():
    image = _synthetic_ellipse(center=(168, 104), axes=(78, 57), angle=21.0)
    seed = {"shape": "circle", "cx": 168.0, "cy": 104.0, "r": 32.0,
            "angle_deg": 0.0}

    result = GeometryCalibrator().fit_reference(
        image, seed, mode="inside", margin_ratio=0.02
    )

    assert result["status"] == "active"
    assert result["anchor_fit"]["selected_candidate_index"] == 0
    assert 1 <= len(result["anchor_fit"]["candidates"]) <= 3
    assert 1 <= len(result["rule_fit"]["candidates"]) <= 3
    assert result["profile_patch"]["anchor"]["mode"] == "auto"
    assert result["profile_patch"]["geometry"]["r"] > 0
    assert result["fit_duration_ms"] >= 0


def test_fit_reference_honors_selected_candidate_without_mutating_seed():
    image = _synthetic_ellipse()
    seed = {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0}
    first = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=0.02)
    if len(first["rule_fit"]["candidates"]) < 2:
        pytest.skip("synthetic image exposes one valid rule candidate")
    second = GeometryCalibrator().fit_reference(
        image, seed, mode="inside", margin_ratio=0.02, rule_candidate_index=1
    )
    assert second["rule_fit"]["selected_candidate_index"] == 1
    assert seed == {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0}


def test_fit_reference_rejects_candidate_index_when_no_candidate_exists():
    image = np.full((128, 128, 3), 127, dtype=np.uint8)
    seed = {"shape": "circle", "cx": 64.0, "cy": 64.0, "r": 24.0}

    with pytest.raises(ValueError, match="anchor_candidate_index is out of range"):
        GeometryCalibrator().fit_reference(
            image, seed, mode="inside", anchor_candidate_index=0
        )


def test_fit_reference_uses_pixel_seed_to_select_inner_circle():
    image = np.full((256, 256, 3), 30, dtype=np.uint8)
    cv2.circle(image, (128, 128), 92, (215, 215, 215), 3)
    cv2.circle(image, (128, 128), 58, (150, 150, 150), 3)
    seed = {"shape": "circle", "cx": 128.0, "cy": 128.0, "r": 58.0}

    result = GeometryCalibrator().fit_reference(image, seed, mode="inside")

    fitted = result["rule_fit"]["fitted_shape"]
    assert abs(fitted["cx"] - 128.0) <= 4
    assert abs(fitted["cy"] - 128.0) <= 4
    assert abs((fitted["rx"] + fitted["ry"]) / 2.0 - 58.0) <= 8


def test_fit_reference_persists_selected_fitted_geometry_not_seed():
    image = np.full((256, 256, 3), 30, dtype=np.uint8)
    cv2.circle(image, (128, 128), 92, (215, 215, 215), 3)
    cv2.circle(image, (128, 128), 58, (150, 150, 150), 3)
    seed = {"shape": "circle", "cx": 128.0, "cy": 128.0, "r": 70.0}

    result = GeometryCalibrator().fit_reference(image, seed, mode="inside")

    assert result["status"] == "active"
    anchor = result["anchor_fit"]["fitted_shape"]
    fitted = result["rule_fit"]["fitted_shape"]
    expected_fitted = GeometryCalibrator._relative_geometry(fitted, anchor)
    expected_seed = GeometryCalibrator._relative_geometry(seed, anchor)
    assert result["profile_patch"]["geometry"]["r"] == pytest.approx(expected_fitted["r"])
    assert result["profile_patch"]["seed_geometry"]["r"] == pytest.approx(expected_seed["r"])
    assert result["profile_patch"]["geometry"]["r"] != pytest.approx(
        result["profile_patch"]["seed_geometry"]["r"]
    )


def test_two_direction_fits_reuse_one_contour_context(monkeypatch):
    calls = 0
    original = geometry_calibration._contours

    def counted(image):
        nonlocal calls
        calls += 1
        return original(image)

    monkeypatch.setattr(geometry_calibration, "_contours", counted)
    calibrator = GeometryCalibrator()
    image = _synthetic_ellipse()

    context = calibrator.prepare_context(image)
    calibrator.fit(image, _ellipse_profile(mode="inside"), context=context)
    calibrator.fit(image, _ellipse_profile(mode="inside"), context=context)

    assert calls == 1


def test_signed_inside_and_outside_masks_use_one_boundary_formula():
    shape = {"shape": "ellipse", "cx": 80.0, "cy": 60.0,
             "rx": 40.0, "ry": 30.0, "angle_deg": 17.0}
    inside = [_shape_mask(shape, (140, 180), "inside", value)
              for value in (-0.20, 0.0, 0.20)]
    outside = [_shape_mask(shape, (140, 180), "outside", value)
               for value in (-0.20, 0.0, 0.20)]

    assert np.count_nonzero(inside[0]) < np.count_nonzero(inside[1]) < np.count_nonzero(inside[2])
    assert np.count_nonzero(outside[0]) > np.count_nonzero(outside[1]) > np.count_nonzero(outside[2])


def test_fit_keeps_raw_shape_and_scales_only_effective_dimensions():
    profile = _ellipse_profile(mode="inside")
    profile["rules"][0]["margin_ratio"] = -0.10
    rule = GeometryCalibrator().fit(_synthetic_ellipse(), profile)["rules"][0]

    assert rule["effective_shape"]["rx"] < rule["fitted_shape"]["rx"]
    assert rule["effective_shape"]["ry"] < rule["fitted_shape"]["ry"]
    assert rule["effective_shape"]["cx"] == pytest.approx(rule["fitted_shape"]["cx"])
    assert rule["effective_shape"]["cy"] == pytest.approx(rule["fitted_shape"]["cy"])
    assert rule["effective_shape"]["angle_deg"] == pytest.approx(rule["fitted_shape"]["angle_deg"])


def test_fit_reference_candidates_do_not_change_with_margin_sign():
    image = _synthetic_ellipse(center=(168, 104), axes=(78, 57), angle=21.0)
    seed = {"shape": "circle", "cx": 168.0, "cy": 104.0, "r": 32.0}
    negative = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=-0.02)
    positive = GeometryCalibrator().fit_reference(image, seed, mode="inside", margin_ratio=0.02)

    assert negative["anchor_fit"]["candidates"] == positive["anchor_fit"]["candidates"]
    assert negative["rule_fit"]["candidates"] == positive["rule_fit"]["candidates"]
    assert negative["rule_fit"]["selected_candidate_index"] == positive["rule_fit"]["selected_candidate_index"]


@pytest.mark.parametrize("margin", [-0.94, 0.0, 0.94])
def test_signed_margin_boundary_values_are_accepted(margin):
    result = GeometryCalibrator().fit_reference(
        _synthetic_ellipse(),
        {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0},
        mode="inside",
        margin_ratio=margin,
    )
    assert result["status"] == "active"
    assert result["margin_ratio"] == pytest.approx(margin)
    assert "effective_shape" in result["rule_fit"]


@pytest.mark.parametrize("margin", [-0.95, 0.95, 1.0])
def test_signed_margin_out_of_range_is_rejected(margin):
    with pytest.raises(ValueError, match=r"margin_ratio must be in \[-0.94, 0.94\]"):
        GeometryCalibrator().fit_reference(
            _synthetic_ellipse(),
            {"shape": "circle", "cx": 155.0, "cy": 92.0, "r": 31.0},
            mode="inside",
            margin_ratio=margin,
        )
