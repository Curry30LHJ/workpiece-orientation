from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import sys

import cv2
import numpy as np
import pytest

from src.orientation_classifier import (
    ImageUnreadableError,
    OrientationClassifier,
    OrientationClassifierError,
    PropagationModelError,
    WorkpieceNotFoundError,
)


class FakeGlobalPredictor:
    def __init__(self):
        self.calls = 0
        self.markers = []

    def predict(self, images):
        self.calls += 1
        marker = int(images[0][0, 0, 0])
        self.markers.append(marker)
        vectors = {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
            3: np.array([0.90, 0.91], dtype=np.float32),
            4: np.array([0.90, 0.70], dtype=np.float32),
            7: np.array([0.95, 0.05], dtype=np.float32),
        }
        return [vectors[marker]]


class FakeExtractor:
    def __init__(self):
        self.calls = 0
        self.markers = []


class FakeMatcher:
    pass


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __getitem__(self, key):
        return FakeTensor(self.value[key])


def fake_extract_features(image, extractor, device, roi_ratio=1.0):
    extractor.calls += 1
    extractor.markers.append(int(image[0, 0, 0]))
    return {"marker": int(image[0, 0, 0])}


def fake_score_feature_pair(query_features, template_features, image_shape, matcher):
    scores = {
        3: {1: 12.4, 2: 5.1},
        4: {1: 4.0, 2: 9.0},
        7: {1: 12.4, 2: 5.1, 7: 12.4},
    }
    return {"score": scores[query_features["marker"]][template_features["marker"]]}


def write_marker(path: Path, marker: int) -> Path:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def classifier():
    return OrientationClassifier(
        global_predictor=FakeGlobalPredictor(),
        extractor=FakeExtractor(),
        matcher=FakeMatcher(),
        device="cpu",
        extract_features_fn=fake_extract_features,
        score_feature_pair_fn=fake_score_feature_pair,
    )


@pytest.fixture
def registered_classifier(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(5)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(5)]
    cache = classifier.build_template_cache(front, back)
    classifier.set_template_cache("m7", cache)
    return classifier


class FakeGeometryCalibrator:
    def fit(self, image, direction_profile):
        if direction_profile.get("fail_marker") == int(image[0, 0, 0]):
            return {"status": "low_confidence", "reason_code": "boundary_not_found", "rules": []}
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[0, 0] = 255
        return {
            "status": "active",
            "ignore_mask": mask,
            "ignored_ratio": 1.0 / mask.size,
            "rules": [{"status": "active", "edge_support": 1.0, "visible_ratio": 1.0}],
        }


def geometry_profile(*, fail_marker=None):
    directions = {}
    for side in ("front", "back"):
        directions[side] = {
            "side": side,
            "anchor": {"shape": "circle", "coarse": {"cx": 0.5, "cy": 0.5, "r": 0.4}},
            "rules": [{"rule_id": f"{side}-inner", "shape": "circle",
                       "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5},
                       "mode": "inside", "margin_ratio": 0.02, "enabled": True}],
            "fill_bgr": [7, 7, 7],
        }
    if fail_marker is not None:
        directions["back"]["fail_marker"] = fail_marker
    return {"profile_revision": 3, "directions": directions}


def test_active_geometry_profile_builds_directional_template_cache_and_query_features(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))

    classifier.geometry_calibrator = FakeGeometryCalibrator()
    candidate, report = classifier.prepare_geometry_cache(
        "m7", record, geometry_profile(), FakeGeometryCalibrator()
    )
    classifier.set_template_cache("m7", candidate)
    result = classifier.predict("m7", write_marker(tmp_path / "query.png", 3))

    assert candidate.geometry_profile_revision == 3
    assert report["front"][0]["status"] == "active"
    assert report["back"][0]["status"] == "active"
    assert classifier.global_predictor.markers[-2:] == [7, 7]
    assert classifier.extractor.markers[-2:] == [7, 7]
    assert result["geometry_mask"]["status"] == "active"
    assert result["geometry_mask"]["profile_revision"] == 3


def test_leave_one_out_scores_cached_candidate_without_feature_reextraction(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(2)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(2)]
    cache = classifier.build_template_cache(front, back)
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier._score_feature_pair = lambda *args: {"score": 1.0}
    before_global = classifier.global_predictor.calls
    before_local = classifier.extractor.calls

    report = classifier.leave_one_out_report(record, cache)

    assert report["status"] == "completed"
    assert report["evaluated"] == 4
    assert report["correct_to_wrong"] == 0
    assert classifier.global_predictor.calls == before_global
    assert classifier.extractor.calls == before_local


def test_geometry_fit_failure_uses_raw_cache_and_sets_review(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    candidate, _ = classifier.prepare_geometry_cache(
        "m7", SimpleNamespace(front_images=tuple(front), back_images=tuple(back)),
        geometry_profile(fail_marker=3), FakeGeometryCalibrator()
    )
    classifier.set_template_cache("m7", candidate)

    result = classifier.predict("m7", write_marker(tmp_path / "query.png", 3))

    assert result["geometry_mask"]["status"] == "low_confidence"
    assert result["geometry_mask"]["needs_review"] is True
    assert result["needs_review"] is True
    assert classifier.global_predictor.markers[-1] == 3
    assert classifier.extractor.markers[-1] == 3


def test_low_global_margin_is_overridden_by_decisive_local_evidence(registered_classifier, tmp_path):
    query = write_marker(tmp_path / "conflict.png", 3)

    result = registered_classifier.predict("m7", query)

    assert result["global_prediction"] == "back"
    assert result["local_prediction"] == "front"
    assert result["label"] == "front"
    assert result["decision_source"] == "local_override"
    assert result["needs_review"] is True


def test_confident_global_result_is_kept_even_when_local_differs(registered_classifier, tmp_path):
    query = write_marker(tmp_path / "global-confident.png", 4)

    result = registered_classifier.predict("m7", query)

    assert result["global_prediction"] == "front"
    assert result["local_prediction"] == "back"
    assert result["label"] == "front"
    assert result["decision_source"] == "global"
    assert result["needs_review"] is True


def test_unknown_workpiece_raises_workpiece_not_found(registered_classifier, tmp_path):
    query = write_marker(tmp_path / "query.png", 1)

    with pytest.raises(WorkpieceNotFoundError):
        registered_classifier.predict("missing", query)


def test_unreadable_query_raises_image_unreadable(registered_classifier, tmp_path):
    with pytest.raises(ImageUnreadableError):
        registered_classifier.predict("m7", tmp_path / "missing.png")


def test_projection_treats_invalid_match_indices_as_no_correspondence(classifier, tmp_path):
    source = write_marker(tmp_path / "source.png", 1)
    target = write_marker(tmp_path / "target.png", 2)
    classifier._extract_features = lambda image, extractor, device, roi_ratio=1.0: {
        "keypoints": FakeTensor(np.zeros((1, 4, 2), dtype=np.float32)),
    }
    classifier.matcher = lambda inputs: {
        "matches": [FakeTensor([[0, 99], [1, 99], [2, 99], [3, 99]])],
    }

    assert classifier.project_region_between_templates(
        source, target, {"x": 1, "y": 1, "width": 2, "height": 2}
    ) is None


def test_projection_converts_cuda_failure_to_propagation_model_error(classifier, tmp_path):
    source = write_marker(tmp_path / "source.png", 1)
    target = write_marker(tmp_path / "target.png", 2)
    classifier._extract_features = lambda image, extractor, device, roi_ratio=1.0: {}

    def fail(_inputs):
        raise RuntimeError("CUDA out of memory")

    classifier.matcher = fail
    with pytest.raises(PropagationModelError, match="局部特征模型不可用"):
        classifier.project_region_between_templates(
            source, target, {"x": 1, "y": 1, "width": 2, "height": 2}
        )


def test_projection_runs_lightglue_inside_torch_inference_context(classifier, tmp_path, monkeypatch):
    source = write_marker(tmp_path / "source.png", 1)
    target = write_marker(tmp_path / "target.png", 2)
    state = {"active": False}

    class FakeTorch:
        @staticmethod
        @contextmanager
        def inference_mode():
            state["active"] = True
            try:
                yield
            finally:
                state["active"] = False

    points = FakeTensor(np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32))
    classifier._extract_features = lambda image, extractor, device, roi_ratio=1.0: {"keypoints": points}

    def matcher(_inputs):
        assert state["active"] is True
        return {"matches": [FakeTensor([[0, 0], [1, 1], [2, 2], [3, 3]])]}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    classifier.matcher = matcher

    assert classifier.project_region_between_templates(
        source, target, {"x": 0, "y": 0, "width": 1, "height": 1}
    ) is not None


def test_build_template_cache_extracts_each_of_ten_templates_once(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(5)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(5)]

    cache = classifier.build_template_cache(front, back)

    assert classifier.global_predictor.calls == 10
    assert classifier.extractor.calls == 10
    assert cache.global_vectors["front"].shape == (5, 2)
    assert len(cache.local_features["back"]) == 5


def test_build_template_cache_accepts_unequal_counts_and_reports_all_templates(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(12)]
    progress = []

    cache = classifier.build_template_cache(
        front,
        back,
        progress_callback=lambda label, done, total: progress.append((label, done, total)),
    )

    assert cache.global_vectors["front"].shape == (1, 2)
    assert cache.global_vectors["back"].shape == (12, 2)
    assert len(cache.local_features["front"]) == 1
    assert len(cache.local_features["back"]) == 12
    assert progress[-1] == ("back", 12, 12)
    assert classifier.global_predictor.calls == 13
    assert classifier.extractor.calls == 13


def test_build_template_cache_rejects_an_empty_orientation(classifier, tmp_path):
    back = [write_marker(tmp_path / "back-0.png", 2)]

    with pytest.raises(ValueError, match="at least one"):
        classifier.build_template_cache([], back)


def test_prediction_scores_every_local_template(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(10)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(15)]
    scored = []

    def score(query_features, template_features, image_shape, matcher):
        scored.append(template_features["marker"])
        return {"score": 1.0}

    classifier._score_feature_pair = score
    classifier.set_template_cache("m", classifier.build_template_cache(front, back))
    classifier.predict("m", write_marker(tmp_path / "query.png", 3))

    assert len(scored) == 25


def test_template_cache_round_trip_preserves_unequal_template_sets(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2 + index) for index in range(3)]
    cache = classifier.build_template_cache(front, back)
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
    )
    record.root.mkdir()

    classifier.save_template_cache(record, cache)
    loaded = classifier.load_template_cache(record)

    assert loaded is not None
    assert loaded.global_vectors["front"].shape == (1, 2)
    assert loaded.global_vectors["back"].shape == (3, 2)
    assert len(loaded.local_features["front"]) == 1
    assert len(loaded.local_features["back"]) == 3


def test_template_cache_is_invalidated_when_template_content_changes(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    cache = classifier.build_template_cache(front, back)
    record = SimpleNamespace(root=tmp_path / "record", front_images=tuple(front), back_images=tuple(back))
    record.root.mkdir()
    classifier.save_template_cache(record, cache)

    write_marker(front[0], 99)

    assert classifier.load_template_cache(record) is None


def test_corrupt_template_cache_is_ignored(classifier, tmp_path):
    front = [write_marker(tmp_path / "front-0.png", 1)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    record = SimpleNamespace(root=tmp_path / "record", front_images=tuple(front), back_images=tuple(back))
    record.root.mkdir()
    (record.root / ".template_cache.pkl").write_bytes(b"not a pickle")

    assert classifier.load_template_cache(record) is None

def make_search_inputs(front_count, back_count, score_for):
    global_vectors = {}
    local_features = {}
    for label, count in (("front", front_count), ("back", back_count)):
        similarities = np.linspace(1.0, 0.1, count, dtype=np.float32)
        global_vectors[label] = similarities.reshape(-1, 1)
        local_features[label] = [
            {"label": label, "index": index, "score": float(score_for(label, index))}
            for index in range(count)
        ]
    return global_vectors, local_features


def run_local_search(classifier, global_vectors, local_features, global_scores):
    return classifier._search_local(
        query_features_by_label={
            "front": {"label": "front"},
            "back": {"label": "back"},
        },
        global_vectors=global_vectors,
        query_embeddings={
            "front": np.asarray([1.0], dtype=np.float32),
            "back": np.asarray([1.0], dtype=np.float32),
        },
        local_features=local_features,
        image_shape=(8, 8),
        global_scores=global_scores,
    )


def test_adaptive_local_search_stops_at_top5_without_reordering_features(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )

    def score_pair(query, template, image_shape, matcher):
        calls.append((template["label"], template["index"]))
        return {"score": template["score"]}

    classifier._score_feature_pair = score_pair
    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.scores == {"front": 8.0, "back": 5.0}
    assert result.diagnostics["stage"] == "top5"
    assert result.diagnostics["matched_counts"] == {"front": 5, "back": 5}
    assert result.diagnostics["available_counts"] == {"front": 12, "back": 13}
    assert result.diagnostics["expanded_because"] is None
    assert result.diagnostics["exhaustive"] is False
    assert calls == [
        (label, index)
        for label in ("front", "back")
        for index in range(5)
    ]
    assert result.trace["ranked_indices"]["front"] == list(range(12))


def test_adaptive_local_search_treats_exact_global_margin_threshold_as_low_margin(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.55, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "top5"
    assert result.diagnostics["global_margin_gate"] == "adaptive_low_margin"
    assert len(calls) == 10


def test_local_search_stable_ranking_preserves_cache_order_for_equal_similarity(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )
    for vectors in global_vectors.values():
        vectors[:] = 1.0
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.trace["ranked_indices"] == {
        "front": list(range(12)),
        "back": list(range(13)),
    }
    assert calls == [
        (label, index)
        for label in ("front", "back")
        for index in range(5)
    ]


def test_adaptive_local_search_expands_to_top10_without_rematching(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13,
        lambda label, index: 8.0 if label == "front" and index == 5 else 5.0,
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "top10"
    assert result.diagnostics["matched_counts"] == {"front": 10, "back": 10}
    assert result.diagnostics["expanded_because"] == "local_margin_low"
    assert len(calls) == 20
    assert len(set(calls)) == 20
    assert [item["stage"] for item in result.trace["stages"]] == ["top5", "top10"]


def test_adaptive_local_search_expands_after_local_conflict(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12,
        13,
        lambda label, index: (
            9.0 if label == "front" and index == 5
            else 8.0 if label == "back"
            else 5.0
        ),
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "top10"
    assert result.diagnostics["expanded_because"] == "local_conflict"
    assert len(calls) == 20
    assert len(set(calls)) == 20


def test_adaptive_local_search_reaches_full_once_when_still_uncertain(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 5.0
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == "full"
    assert result.diagnostics["matched_counts"] == {"front": 12, "back": 13}
    assert result.diagnostics["exhaustive"] is True
    assert result.diagnostics["expanded_because"] == "local_margin_low"
    assert len(calls) == 25
    assert len(set(calls)) == 25


@pytest.mark.parametrize(
    ("front_count", "back_count", "expected_stage", "expected_calls"),
    [(1, 1, "top5", 2), (5, 10, "top5", 10), (10, 15, "top5", 10)],
)
def test_adaptive_local_search_supports_unequal_counts(
    classifier, front_count, back_count, expected_stage, expected_calls
):
    calls = []
    global_vectors, local_features = make_search_inputs(
        front_count, back_count,
        lambda label, index: 8.0 if label == "front" else 5.0,
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.diagnostics["stage"] == expected_stage
    assert len(calls) == expected_calls


def test_local_search_rejects_misaligned_cache(classifier):
    global_vectors, local_features = make_search_inputs(
        5, 5, lambda label, index: 5.0
    )
    local_features["back"].pop()

    with pytest.raises(OrientationClassifierError, match="alignment error for back"):
        run_local_search(
            classifier, global_vectors, local_features,
            {"front": 0.51, "back": 0.50},
        )


@pytest.mark.parametrize("missing_label", ["front", "back"])
@pytest.mark.parametrize(
    ("missing_input", "message"),
    [
        ("query_embeddings", "missing query embedding"),
        ("query_features_by_label", "missing query local features"),
        ("global_scores", "missing global score"),
    ],
)
def test_local_search_rejects_missing_required_direction(
    classifier, missing_label, missing_input, message
):
    global_vectors, local_features = make_search_inputs(
        5, 5, lambda label, index: 8.0 if label == "front" else 5.0
    )
    inputs = {
        "query_features_by_label": {
            "front": {"label": "front"},
            "back": {"label": "back"},
        },
        "global_vectors": global_vectors,
        "query_embeddings": {
            "front": np.asarray([1.0], dtype=np.float32),
            "back": np.asarray([1.0], dtype=np.float32),
        },
        "local_features": local_features,
        "image_shape": (8, 8),
        "global_scores": {"front": 0.51, "back": 0.50},
    }
    inputs[missing_input].pop(missing_label)

    with pytest.raises(
        OrientationClassifierError,
        match=f"{message} for {missing_label}",
    ):
        classifier._search_local(**inputs)


def test_local_search_matching_time_excludes_template_tensor_movement(
    classifier, monkeypatch
):
    import src.orientation_classifier as classifier_module

    clock = {"value": 0.0}
    global_vectors, local_features = make_search_inputs(
        5, 5, lambda label, index: 8.0 if label == "front" else 5.0
    )

    def move_tensors(value, device):
        if isinstance(value, dict) and "index" in value:
            clock["value"] += 0.010
        return value

    def score_pair(query, template, image_shape, matcher):
        clock["value"] += 0.001
        return {"score": template["score"]}

    monkeypatch.setattr(classifier_module, "_move_tensors", move_tensors)
    monkeypatch.setattr(
        classifier_module.time, "perf_counter", lambda: clock["value"]
    )
    classifier._score_feature_pair = score_pair

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.51, "back": 0.50},
    )

    assert result.matching_ms == pytest.approx(10.0)


def test_adaptive_high_global_margin_searches_every_template_once(classifier):
    calls = []
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = run_local_search(
        classifier, global_vectors, local_features,
        {"front": 0.90, "back": 0.20},
    )

    assert result.diagnostics["stage"] == "full"
    assert result.diagnostics["global_margin_gate"] == "preserve_review_semantics"
    assert len(calls) == 25
    assert len(set(calls)) == 25


def test_exhaustive_local_search_matches_all_template_scores_and_fusion():
    classifier = OrientationClassifier(
        global_predictor=FakeGlobalPredictor(),
        extractor=FakeExtractor(),
        matcher=FakeMatcher(),
        device="cpu",
        extract_features_fn=fake_extract_features,
        score_feature_pair_fn=fake_score_feature_pair,
        local_search_mode="exhaustive",
    )
    global_vectors, local_features = make_search_inputs(
        12, 13, lambda label, index: 8.0 if label == "front" else 5.0
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: {
        "score": template["score"]
    }
    global_scores = {"front": 0.51, "back": 0.50}

    result = run_local_search(classifier, global_vectors, local_features, global_scores)
    expected_fusion = classifier._fuse_scores(
        global_scores, {"front": 8.0, "back": 5.0}, 0.0
    )

    assert result.scores == {"front": 8.0, "back": 5.0}
    assert result.diagnostics["stage"] == "full"
    assert result.diagnostics["mode"] == "exhaustive"
    assert result.diagnostics["matched_counts"] == {"front": 12, "back": 13}
    assert result.diagnostics["expanded_because"] == "exhaustive_mode"
    fusion = result.trace["stages"][-1]["fusion"]
    for key in (
        "label", "global_prediction", "global_scores", "global_margin",
        "local_prediction", "local_scores", "local_margin", "decision_source",
        "needs_review",
    ):
        assert fusion[key] == expected_fusion[key]


def test_local_search_mode_rejects_unsupported_values():
    with pytest.raises(ValueError, match="local_search_mode"):
        OrientationClassifier(
            global_predictor=FakeGlobalPredictor(),
            extractor=FakeExtractor(),
            matcher=FakeMatcher(),
            device="cpu",
            local_search_mode="fast",
        )
