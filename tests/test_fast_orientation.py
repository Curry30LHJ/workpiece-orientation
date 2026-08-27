from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.fast_geometry import FastGeometryProcessor, compile_fast_geometry
from src.fast_orientation import FAST_ROTATION_DEGREES, FastOrientationEngine


def marker_image(marker: int, *, size: int = 64) -> np.ndarray:
    image = np.full((size, size, 3), marker, dtype=np.uint8)
    cv2.circle(image, (size // 3, size // 2), size // 7, (marker // 2, marker, 255 - marker), -1)
    return image


def write_marker(path: Path, marker: int) -> Path:
    assert cv2.imwrite(str(path), marker_image(marker))
    return path


def write_markers(tmp_path: Path, prefix: str, count: int, *, base: int) -> list[Path]:
    return [write_marker(tmp_path / f"{prefix}-{index}.png", base + index) for index in range(count)]


def read_marker(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert image is not None
    return image


class FakeCalibrator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def prepare_context(self, image: np.ndarray):
        return type("Context", (), {"image_shape": image.shape[:2], "contours": ()})()

    def fit(self, image: np.ndarray, direction: dict, *, context: object) -> dict:
        if self.fail:
            return {"status": "low_confidence", "rules": []}
        return {"status": "low_confidence", "rules": []}


class FakeEmbedder:
    def __init__(self, *, record_batches: bool = False) -> None:
        self.batch_sizes: list[int] = []
        self.batches: list[list[np.ndarray]] = []
        self.record_batches = record_batches
        self.seen_markers: set[int] = set()

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        self.batch_sizes.append(len(images))
        if self.record_batches:
            self.batches.append([image.copy() for image in images])
        result = []
        for image in images:
            marker = int(image[0, 0, 0])
            self.seen_markers.add(marker)
            result.append(np.array([float(marker), 255.0 - marker], dtype=np.float32))
        return result


def fake_engine(*, fail_geometry: bool = False, record_batches: bool = False):
    embedder = FakeEmbedder(record_batches=record_batches)
    geometry = FastGeometryProcessor(FakeCalibrator(fail=fail_geometry))
    return FastOrientationEngine(embedder, geometry, image_reader=read_marker), embedder


def configured_profile() -> dict:
    return {
        "schema_version": 1,
        "profile_revision": 8,
        "directions": {
            label: {
                "side": label,
                "anchor": {"shape": "circle", "coarse": {"cx": 0.5, "cy": 0.5, "r": 0.4, "angle_deg": 0.0}},
                "rules": [{"rule_id": label, "shape": "circle", "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5, "angle_deg": 0.0}, "mode": "inside", "enabled": True}],
                "fill_bgr": [7, 7, 7],
            }
            for label in ("front", "back")
        },
    }


def test_build_uses_every_unequal_template_and_keeps_actual_counts(tmp_path):
    front = write_markers(tmp_path, "front", 5, base=10)
    back = write_markers(tmp_path, "back", 12, base=100)
    engine, embedder = fake_engine()

    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=4, model_fingerprint="model-a")

    assert cache.template_counts == {"front": 5, "back": 12}
    assert cache.training_summary["original_samples"] == 17
    assert embedder.seen_markers >= {10, 11, 12, 13, 14, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111}


def test_few_shot_rotation_augments_only_small_side(tmp_path):
    front = write_markers(tmp_path, "front", 5, base=10)
    back = write_markers(tmp_path, "back", 20, base=100)
    engine, _ = fake_engine()

    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m")

    assert cache.training_summary["augmented_samples"] == {"front": 55, "back": 0}
    assert cache.training_summary["rotation_degrees"] == list(FAST_ROTATION_DEGREES)


def test_build_emits_template_and_augmentation_progress_units(tmp_path):
    progress: list[dict] = []
    front = write_markers(tmp_path, "front", 1, base=10)
    back = write_markers(tmp_path, "back", 1, base=200)
    engine, _ = fake_engine()

    engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m", progress_callback=progress.append)

    assert {item["phase"] for item in progress} == {"fast_originals", "fast_augmentation", "fast_ridge"}
    assert {item["unit"] for item in progress} == {"templates", "augmented_samples", "ridge_head"}


def test_predict_calls_one_three_image_batch_and_returns_fast_ridge_result(tmp_path):
    engine, embedder = fake_engine()
    front = write_markers(tmp_path, "front", 20, base=10)
    back = write_markers(tmp_path, "back", 20, base=180)
    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m")
    embedder.batch_sizes.clear()

    result = engine.predict(marker_image(10), cache)

    assert embedder.batch_sizes == [3]
    assert result["inference_engine"] == "fast_geometry"
    assert result["decision_source"] == "fast_ridge"
    assert result["label"] in {"front", "back"}
    assert "local_prediction" not in result
    assert result["timings_ms"].keys() >= {"geometry_context", "geometry_fit", "mask_build", "global_batch", "linear_head", "total"}


def test_geometry_failure_returns_direction_and_review_reason(tmp_path):
    engine, _ = fake_engine(fail_geometry=True)
    front = write_markers(tmp_path, "front", 2, base=10)
    back = write_markers(tmp_path, "back", 2, base=200)
    cache = engine.build_cache(front, back, geometry_profile=configured_profile(), library_revision=1, model_fingerprint="m")

    result = engine.predict(marker_image(10), cache)

    assert result["label"] in {"front", "back"}
    assert result["needs_review"] is True
    assert "FAST_GEOMETRY_LOW_CONFIDENCE" in result["review_reason_codes"]
    assert isinstance(result["review_reason"], str) and result["review_reason"]
    assert set(result["directions"]) == {"front", "back"}


def test_template_and_query_use_the_same_three_slot_preprocessing(tmp_path):
    front = write_marker(tmp_path / "front.png", 10)
    back = write_marker(tmp_path / "back.png", 200)
    engine, embedder = fake_engine(record_batches=True)
    profile = configured_profile()

    cache = engine.build_cache([front], [back], geometry_profile=profile, library_revision=1, model_fingerprint="m")
    template_slots = tuple(image.copy() for image in embedder.batches[0][:3])
    query_slots = engine.geometry.build_variants(read_marker(front), cache.compiled_geometry).images

    assert all(np.array_equal(left, right) for left, right in zip(template_slots, query_slots))


def test_predict_rejects_stale_cache_before_embedder_is_called(tmp_path):
    engine, embedder = fake_engine()
    front = write_markers(tmp_path, "front", 2, base=10)
    back = write_markers(tmp_path, "back", 2, base=200)
    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m")
    embedder.batch_sizes.clear()

    with pytest.raises(ValueError, match="FAST_CACHE_REVISION_MISMATCH"):
        engine.predict(marker_image(10), replace(cache, format_version=999))

    assert embedder.batch_sizes == []
