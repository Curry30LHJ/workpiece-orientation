import numpy as np
import pytest

from src.fast_ridge import fit_ridge_head, pack_embeddings, predict_ridge


def test_pack_embeddings_l2_normalizes_each_slot_in_fixed_order():
    packed = pack_embeddings([
        np.array([3.0, 4.0], np.float32),
        np.array([0.0, 2.0], np.float32),
        np.array([5.0, 0.0], np.float32),
    ])
    np.testing.assert_allclose(
        packed,
        np.array([0.6, 0.8, 0.0, 1.0, 1.0, 0.0], np.float32),
        atol=1e-6,
    )


def test_ridge_accepts_unequal_counts_and_separates_both_classes():
    front = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], np.float32)
    back = np.array([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]], np.float32)
    head = fit_ridge_head(front, back)
    assert all(predict_ridge(head, row).label == "front" for row in front)
    assert all(predict_ridge(head, row).label == "back" for row in back)
    assert head.training_summary["class_counts"] == {"front": 3, "back": 5}


@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15), (35, 35)])
def test_ridge_supports_required_template_count_matrix(front_count, back_count):
    front = np.tile(np.array([[1.0, 0.0]], np.float32), (front_count, 1))
    back = np.tile(np.array([[0.0, 1.0]], np.float32), (back_count, 1))
    head = fit_ridge_head(front, back)
    assert head.training_summary["class_counts"] == {
        "front": front_count,
        "back": back_count,
    }


def test_one_plus_one_builds_but_marks_small_margin_for_review():
    head = fit_ridge_head(
        np.array([[1.0, 0.0]], np.float32),
        np.array([[0.0, 1.0]], np.float32),
    )
    decision = predict_ridge(head, np.array([0.51, 0.49], np.float32))
    assert decision.needs_review is True
    assert head.training_summary["validation_status"] == "few_shot_unverified"
    assert head.regularization == 1.0


def test_fit_is_bitwise_deterministic():
    front = np.eye(4, dtype=np.float32)[:2]
    back = -np.eye(4, dtype=np.float32)[:3]
    first = fit_ridge_head(front, back)
    second = fit_ridge_head(front, back)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.bias == second.bias
    assert first.review_threshold == second.review_threshold


def test_non_finite_features_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        fit_ridge_head(
            np.array([[np.nan, 0.0]], np.float32),
            np.array([[0.0, 1.0]], np.float32),
        )


def test_leave_one_source_out_excludes_rotations_from_the_held_out_source():
    front = np.array([[1.0, 0.0], [0.99, 0.01], [0.8, 0.2]], np.float32)
    back = np.array([[0.0, 1.0], [0.01, 0.99], [0.2, 0.8]], np.float32)
    head = fit_ridge_head(
        front,
        back,
        front_source_ids=[0, 0, 1],
        back_source_ids=[0, 0, 1],
        front_original_rows=[True, False, True],
        back_original_rows=[True, False, True],
    )
    assert head.training_summary["validation_folds"] == 4
    assert head.training_summary["validation_grouping"] == "leave_one_source_out"


def test_leave_one_source_out_validation_does_not_leak_held_out_rotation():
    front = np.array([[-1.0, -1.0], [-1.0, 2.0], [-2.0, 1.0], [2.0, -1.0]], np.float32)
    back = np.array([[2.0, 0.0], [1.0, 2.0], [1.0, 0.0], [-2.0, -1.0]], np.float32)
    head = fit_ridge_head(
        front,
        back,
        front_source_ids=[0, 0, 1, 1],
        back_source_ids=[0, 0, 1, 1],
        front_original_rows=[True, False, True, False],
        back_original_rows=[True, False, True, False],
        regularization_grid=[1.0],
    )
    assert head.training_summary["validation_status"] == "cross_validation_failed"


def test_validation_rebalances_class_weights_after_each_source_is_held_out():
    front = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 2.0], [0.0, 1.0]], np.float32)
    back = np.array([[-1.0, 2.0], [-1.0, 1.0], [-2.0, -1.0], [0.0, 2.0]], np.float32)
    head = fit_ridge_head(
        front,
        back,
        front_source_ids=[0, 0, 1, 1],
        back_source_ids=[0, 0, 1, 1],
        front_original_rows=[True, False, True, False],
        back_original_rows=[True, False, True, False],
    )
    assert head.regularization == 10.0
