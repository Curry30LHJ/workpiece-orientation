import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

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


def _timing_series(value=1.0):
    return {
        "samples": [value],
        "p50": value,
        "p95": value,
        "p99": value,
        "max": value,
    }


def _valid_native_report(images, dimension=2):
    return {
        "schema_version": 1,
        "ok": True,
        "mode": "feature_extraction",
        "input_count": len(images),
        "feature_dimension": dimension,
        "worker_count": 1,
        "batch_size": len(images),
        "threads": 1,
        "warmup": 0,
        "iterations": 1,
        "ordered_images": [str(path) for path in images],
        "timings_ms": {
            "decode": _timing_series(),
            "preprocess": _timing_series(),
            "inference": _timing_series(),
            "normalize": _timing_series(),
            "total": _timing_series(5.0),
        },
    }


def test_validate_native_report_rejects_reordered_images(tmp_path):
    module = load_comparison_module()
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    report = _valid_native_report(images)
    report["ordered_images"] = [str(images[1]), str(images[0])]
    report_path = tmp_path / "native.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ordered_images"):
        module.validate_native_report(report_path, expected_images=images)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda report: report.update(feature_dimension=0), "feature_dimension"),
        (lambda report: report["timings_ms"].pop("normalize"), "normalize"),
    ],
)
def test_validate_native_report_rejects_incomplete_report(tmp_path, mutation, message):
    module = load_comparison_module()
    images = [tmp_path / "sample.png"]
    report = _valid_native_report(images)
    mutation(report)
    report_path = tmp_path / "native.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        module.validate_native_report(report_path, expected_images=images)


def test_run_native_embeddings_rejects_nonzero_exit_and_preserves_error_context(
    tmp_path, monkeypatch
):
    module = load_comparison_module()
    image = tmp_path / "sample.png"
    image.write_bytes(b"image")
    executable = tmp_path / "native.exe"
    executable.write_bytes(b"stub")
    model = tmp_path / "model"
    model.mkdir()
    (model / "inference.pdmodel").write_bytes(b"model")
    (model / "inference.pdiparams").write_bytes(b"params")
    config = tmp_path / "config.yaml"
    config.write_text(
        "RecPreProcess:\n"
        "  transform_ops:\n"
        "    - ResizeImage: {size: [224, 224]}\n"
        "    - NormalizeImage: {scale: 1.0/255.0, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}\n"
        "    - ToCHWImage: {}\n",
        encoding="utf-8",
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 9, "stdout", "native boom")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="native boom"):
        module.run_native_embeddings(
            executable,
            model,
            config,
            [image],
            tmp_path / "out",
            warmup=0,
            iterations=1,
            workers=1,
            batch_size=1,
            threads=1,
        )


def test_run_native_embeddings_validates_and_returns_ordered_matrix(
    tmp_path, monkeypatch
):
    module = load_comparison_module()
    images = [tmp_path / "first.png", tmp_path / "second.png"]
    for image in images:
        image.write_bytes(image.name.encode("ascii"))
    executable = tmp_path / "native.exe"
    executable.write_bytes(b"stub")
    model = tmp_path / "model"
    model.mkdir()
    (model / "inference.pdmodel").write_bytes(b"model")
    (model / "inference.pdiparams").write_bytes(b"params")
    config = tmp_path / "config.yaml"
    config.write_text(
        "RecPreProcess:\n"
        "  transform_ops:\n"
        "    - ResizeImage: {size: [224, 224]}\n"
        "    - NormalizeImage: {scale: 1.0/255.0, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}\n"
        "    - ToCHWImage: {}\n",
        encoding="utf-8",
    )

    def fake_run(command, **kwargs):
        image_list = Path(command[command.index("--image-list") + 1])
        report_path = Path(command[command.index("--report") + 1])
        embedding_path = Path(command[command.index("--dump-embeddings") + 1])
        ordered = json.loads(image_list.read_text(encoding="utf-8"))
        report = _valid_native_report(ordered, dimension=2)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32).tofile(embedding_path)
        assert command[command.index("--input-width") + 1] == "224"
        assert command[command.index("--input-height") + 1] == "224"
        return subprocess.CompletedProcess(command, 0, "native stdout", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = module.run_native_embeddings(
        executable,
        model,
        config,
        images,
        tmp_path / "out",
        warmup=1,
        iterations=2,
        workers=1,
        batch_size=2,
        threads=1,
    )

    assert isinstance(result, module.EmbeddingRun)
    assert result.ordered_images == tuple(str(image) for image in images)
    np.testing.assert_array_equal(
        result.embeddings, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    )
    assert result.timings_ms["native_stdout"] == "native stdout"


def test_compare_decisions_reports_one_changed_label_and_score(tmp_path):
    module = load_comparison_module()
    reference = [
        {
            "label": "front",
            "needs_review": False,
            "global_prediction": "front",
            "geometry_status": "active",
            "global_scores": {"front": 0.8, "back": 0.2},
            "decision_margin": 0.6,
        }
    ]
    candidate = [
        {
            "label": "back",
            "needs_review": False,
            "global_prediction": "front",
            "geometry_status": "active",
            "global_scores": {"front": 0.8, "back": 0.2},
            "decision_margin": 0.6001,
        }
    ]

    result = module.compare_decisions(reference, candidate)

    assert result["passed"] is False
    assert result["mismatches"][0]["field"] == "label"
    assert any(item["field"] == "margin" for item in result["mismatches"])


def test_build_comparison_report_records_hashes_and_gate():
    module = load_comparison_module()
    images = ("first.png", "second.png")
    python = module.EmbeddingRun(
        np.eye(2, dtype=np.float32),
        images,
        {"total": _timing_series(100.0), "input_sha256": ["a", "b"]},
        "python-fingerprint",
    )
    native = module.EmbeddingRun(
        np.eye(2, dtype=np.float32),
        images,
        {"total": _timing_series(80.0)},
        "native-fingerprint",
    )

    report = module.build_comparison_report(
        python,
        native,
        python_pipeline_p95_ms=132.0,
        labels_match=True,
        review_decisions_match=True,
    )

    assert report["input_sha256"] == ["a", "b"]
    assert report["embedding_metrics"]["passed"] is True
    assert report["gate"]["projected_pipeline_p95_ms"] == pytest.approx(112.0)


def test_replay_fast_decisions_uses_slot_digest_lookup_and_reports_mismatch(tmp_path):
    module = load_comparison_module()
    image_path = tmp_path / "query.png"
    image_path.write_bytes(b"query")
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

    class Geometry:
        def build_variants(self, value, _compiled):
            return SimpleNamespace(images=(value, value + 1, value + 2))

    class Engine:
        def __init__(self):
            self.geometry = Geometry()
            self.image_reader = lambda _path: image.copy()
            self.embed_batch = lambda _images: [np.array([1.0], dtype=np.float32)]

    class Classifier:
        def __init__(self):
            self.fast_engine = Engine()

        def predict_fast_with_cache(self, _cache, _path, *, library_revision):
            vector = self.fast_engine.embed_batch([image.copy()])[0]
            label = "front" if float(vector[0]) >= 0.0 else "back"
            return {
                "label": label,
                "needs_review": False,
                "global_prediction": label,
                "geometry_status": "active",
                "decision_margin": float(vector[0]),
                "library_revision": library_revision,
            }

    classifier = Classifier()
    cache = SimpleNamespace(fast_runtime=object())
    slots = (image, image + 1, image + 2)
    # Native lookup deliberately flips the sign to force an observable mismatch.
    native = {
        module._slot_digest(slot): np.array([-1.0], dtype=np.float32)
        for slot in slots
    }

    result = module.replay_fast_decisions(
        classifier, cache, [image_path], native, library_revision=1
    )

    assert result["passed"] is False
    assert result["mismatches"][0]["field"] == "label"
