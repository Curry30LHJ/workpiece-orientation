import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).parents[1]


def load_comparison_module():
    path = REPO_ROOT / "scripts" / "compare_cpp_python_inference.py"
    spec = importlib.util.spec_from_file_location(
        "compare_cpp_python_inference_under_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_f32_matrix_preserves_rows_and_rejects_wrong_size(tmp_path):
    module = load_comparison_module()
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    path = tmp_path / "embeddings.f32"
    matrix.tofile(path)

    actual = module.read_f32_matrix(path, rows=2, columns=2)

    np.testing.assert_array_equal(actual, matrix)
    with pytest.raises(ValueError, match="expected 6 float32 values, found 4"):
        module.read_f32_matrix(path, rows=2, columns=3)


def test_compare_embeddings_rejects_a_rotated_vector_even_when_shapes_match():
    module = load_comparison_module()
    reference = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidate = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    result = module.compare_embeddings(reference, candidate)

    assert result["passed"] is False
    assert result["minimum_cosine"] == pytest.approx(0.0)
    assert result["maximum_absolute_error"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "reference,candidate,message",
    [
        (
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            "non-zero norm",
        ),
        (
            np.array([[np.nan, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
            "finite values",
        ),
        (
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "identical 2-D shapes",
        ),
    ],
)
def test_compare_embeddings_rejects_invalid_matrices(reference, candidate, message):
    module = load_comparison_module()

    with pytest.raises(ValueError, match=message):
        module.compare_embeddings(reference, candidate)


def test_project_pipeline_p95_replaces_python_feature_time_once():
    module = load_comparison_module()

    projected = module.project_pipeline_p95(
        python_pipeline_p95_ms=132.0,
        python_feature_p95_ms=100.0,
        native_feature_p95_ms=80.0,
    )

    assert projected == pytest.approx(112.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        module.project_pipeline_p95(
            python_pipeline_p95_ms=99.0,
            python_feature_p95_ms=100.0,
            native_feature_p95_ms=80.0,
        )


def test_integration_gate_requires_both_measured_and_projected_fifteen_percent_gain():
    module = load_comparison_module()
    result = module.evaluate_integration_gate(
        embedding_metrics={"passed": True},
        labels_match=True,
        review_decisions_match=True,
        python_feature_p95_ms=100.0,
        native_feature_p95_ms=80.0,
        python_pipeline_p95_ms=132.0,
    )

    assert result["native_feature_improvement_ratio"] == pytest.approx(0.20)
    assert result["projected_pipeline_p95_ms"] == pytest.approx(112.0)
    assert result["projected_pipeline_improvement_ratio"] == pytest.approx(20.0 / 132.0)
    assert result["feature_gate_passed"] is True
    assert result["projected_pipeline_gate_passed"] is True
    assert result["passed"] is True


def test_integration_gate_rejects_when_projected_gain_is_below_fifteen_percent():
    module = load_comparison_module()

    result = module.evaluate_integration_gate(
        embedding_metrics={"passed": True},
        labels_match=True,
        review_decisions_match=True,
        python_feature_p95_ms=100.0,
        native_feature_p95_ms=80.0,
        python_pipeline_p95_ms=200.0,
    )

    assert result["feature_gate_passed"] is True
    assert result["projected_pipeline_gate_passed"] is False
    assert result["passed"] is False
