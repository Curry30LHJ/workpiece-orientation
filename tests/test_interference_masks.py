import numpy as np

from src.interference_masks import (
    build_active_mask_map,
    filter_template_features,
    project_region,
    resolve_propagated_region,
)


def test_active_mask_map_ignores_disabled_and_unresolved_annotation_groups():
    region = {"x": 1, "y": 2, "width": 3, "height": 4}
    masks = build_active_mask_map(
        1,
        1,
        [
            {"group_id": "active", "enabled": True,
             "propagation": {"state": "active"},
             "annotations": [{"orientation": "front", "index": 0,
                              "status": "active", "regions": [region]}]},
            {"group_id": "disabled", "enabled": False,
             "propagation": {"state": "active"},
             "annotations": [{"orientation": "front", "index": 0,
                              "status": "active", "regions": [{"x": 5, "y": 5, "width": 1, "height": 1}]}]},
            {"group_id": "review", "enabled": True,
             "propagation": {"state": "needs_review"},
             "annotations": [{"orientation": "back", "index": 0,
                              "status": "needs_review", "regions": [{"x": 6, "y": 6, "width": 1, "height": 1}]}]},
        ],
    )

    assert masks == {"front": [[region]], "back": [[]]}


def test_filtered_features_remove_only_template_keypoints_inside_region():
    features = {
        "keypoints": np.array([[[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]]], dtype=np.float32),
        "scores": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        "descriptors": np.arange(12, dtype=np.float32).reshape(1, 4, 3),
    }

    filtered = filter_template_features(features, [{"x": 0, "y": 0, "width": 4, "height": 4}])

    assert filtered["keypoints"].shape == (1, 2, 2)
    assert filtered["keypoints"][0].tolist() == [[5.0, 5.0], [9.0, 9.0]]
    assert filtered["scores"].shape == (1, 2)
    assert filtered["descriptors"].shape == (1, 4, 2)


def test_two_consistent_correspondences_activate_projected_region():
    source = {"x": 1, "y": 1, "width": 2, "height": 2}
    source_points = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    target_points = np.array([[5, 2], [15, 2], [15, 12], [5, 12]], dtype=np.float32)

    projected = project_region(source, source_points, target_points)
    result = resolve_propagated_region([projected, projected], image_shape=(20, 20))

    assert result["status"] == "active"
    assert result["confidence"] >= 0.9
    assert result["reason_code"] == "sources_agree"
    assert result["max_spread_px"] == 0.0
    assert result["area_ratio"] > 0.0


def test_disagreement_stays_in_review():
    first = {"x": 1, "y": 1, "width": 2, "height": 2}
    second = {"x": 12, "y": 12, "width": 2, "height": 2}

    result = resolve_propagated_region([first, second], image_shape=(20, 20))

    assert result["status"] == "needs_review"
    assert "disagreement" in result["reason"]
    assert result["reason_code"] == "source_disagreement"
    assert result["max_spread_px"] > 3.0
