from __future__ import annotations

import cv2
import numpy as np

from src.geometry_calibration import (
    GeometryCalibrator,
    apply_ignore_mask,
    filter_features_by_mask,
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


def test_inner_circle_tracks_shift_scale_and_rotation_in_anchor_coordinates():
    image = _synthetic_ellipse(center=(155, 92), axes=(72, 54), angle=24.0)

    result = GeometryCalibrator().fit(image, _ellipse_profile(mode="inside"))

    assert result["status"] == "active"
    assert abs(result["anchor"]["cx"] - 155) <= 4
    assert abs(result["anchor"]["cy"] - 92) <= 4
    assert result["rules"][0]["edge_support"] >= 0.35
    assert result["ignore_mask"][92, 155] == 255
    assert result["ignore_mask"][20, 20] == 0


def test_outer_rule_ignores_intrusion_but_keeps_shrunken_object_edge():
    image = _synthetic_circle_with_corner_intrusions()

    result = GeometryCalibrator().fit(image, _circle_profile(mode="outside"))

    assert result["status"] == "active"
    assert result["ignore_mask"][5, 5] == 255
    assert result["ignore_mask"][128, 128] == 0
    assert result["ignore_mask"][128, 42] == 0


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
