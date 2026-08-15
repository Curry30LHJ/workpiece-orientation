from pathlib import Path

import cv2
import numpy as np
import pytest

from src.orientation_classifier import (
    ImageUnreadableError,
    OrientationClassifier,
    WorkpieceNotFoundError,
)


class FakeGlobalPredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        marker = int(images[0][0, 0, 0])
        vectors = {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
            3: np.array([0.90, 0.91], dtype=np.float32),
            4: np.array([0.90, 0.70], dtype=np.float32),
        }
        return [vectors[marker]]


class FakeExtractor:
    def __init__(self):
        self.calls = 0


class FakeMatcher:
    pass


def fake_extract_features(image, extractor, device, roi_ratio=1.0):
    extractor.calls += 1
    return {"marker": int(image[0, 0, 0])}


def fake_score_feature_pair(query_features, template_features, image_shape, matcher):
    scores = {
        3: {1: 12.4, 2: 5.1},
        4: {1: 4.0, 2: 9.0},
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


def test_build_template_cache_extracts_each_of_ten_templates_once(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(5)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(5)]

    cache = classifier.build_template_cache(front, back)

    assert classifier.global_predictor.calls == 10
    assert classifier.extractor.calls == 10
    assert cache.global_vectors["front"].shape == (5, 2)
    assert len(cache.local_features["back"]) == 5
