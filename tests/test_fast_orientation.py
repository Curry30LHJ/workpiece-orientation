from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

import src.fast_orientation as fast_orientation
from src.fast_geometry import FastGeometryProcessor, FastGeometryVariants, compile_fast_geometry
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


class SlotGeometry:
    def __init__(self, markers):
        self.markers = tuple(markers)

    def build_variants(self, image, compiled):
        images = tuple(np.full_like(image, marker) for marker in self.markers)
        return FastGeometryVariants(
            images, "not_configured", False, (), {},
            {"geometry_context": 0.0, "geometry_fit": 0.0, "mask_build": 0.0},
        )


class ComparisonErrorImage:
    def __init__(self, image: np.ndarray) -> None:
        self._image = image

    @property
    def shape(self):
        raise RuntimeError("comparison failed")

    @property
    def dtype(self):
        return self._image.dtype

    def __getitem__(self, key):
        return self._image[key]


class RaisingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, images):
        self.calls += 1
        raise RuntimeError("embedding failed")


def make_small_cache(engine, tmp_path):
    front = write_markers(tmp_path, "front", 20, base=10)
    back = write_markers(tmp_path, "back", 20, base=180)
    return engine.build_cache(
        front, back, geometry_profile=None, library_revision=1,
        model_fingerprint="m",
    )


def fake_engine(*, fail_geometry: bool = False, record_batches: bool = False):
    embedder = FakeEmbedder(record_batches=record_batches)
    geometry = FastGeometryProcessor(FakeCalibrator(fail=fail_geometry))
    return FastOrientationEngine(embedder, geometry, image_reader=read_marker), embedder


class RuleCenterGeometry:
    """Geometry boundary that reports rule centers in its resized context."""

    max_side = 256

    def __init__(self, centers: dict[str, tuple[float, float]], *, active_labels: set[str] | None = None) -> None:
        self.centers = centers
        self.active_labels = set(centers) if active_labels is None else active_labels

    def build_variants(self, image: np.ndarray, compiled: object) -> FastGeometryVariants:
        directions = {}
        for label, center in self.centers.items():
            rule_status = "active" if label in self.active_labels else "low_confidence"
            directions[label] = {
                "status": rule_status,
                "anchor": {"cx": center[0] + 3.0, "cy": center[1] + 3.0},
                "rules": [{
                    "status": rule_status,
                    "fitted_shape": {"cx": center[0], "cy": center[1]},
                }],
            }
        return FastGeometryVariants(
            (image.copy(), image.copy(), image.copy()),
            "active",
            False,
            (),
            directions,
            {"geometry_context": 0.0, "geometry_fit": 0.0, "mask_build": 0.0},
        )


class RepeatingEmbedder:
    def __init__(self, slots: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        self.slots = slots
        self.position = 0
        self.batch_sizes: list[int] = []

    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        self.batch_sizes.append(len(images))
        result = []
        for _ in images:
            result.append(self.slots[self.position % 3].copy())
            self.position += 1
        return result


def map_reader(images: dict[str, np.ndarray]):
    return lambda path: images[str(path)].copy()


def bright_marker_image(marker: int, *, size: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    height, width = size
    image = np.full((height, width, 3), marker, dtype=np.uint8)
    cv2.circle(image, center, 8, (255, 255, 255), -1)
    return image


def recorded_images(embedder: FakeEmbedder) -> list[np.ndarray]:
    return [image for batch in embedder.batches for image in batch]


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


def test_cpu_dedup_reuses_identical_query_slots(tmp_path):
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 10, 10)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    cache = make_small_cache(engine, tmp_path)
    embedder.batch_sizes.clear()

    result = engine.predict(marker_image(10), cache)

    assert embedder.batch_sizes == [1]
    assert result["timings_ms"]["global_input_slots"] == 3
    assert result["timings_ms"]["global_unique_slots"] == 1


def test_dedup_keeps_three_calls_when_geometry_slots_differ(tmp_path):
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 30)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    cache = make_small_cache(engine, tmp_path)
    embedder.batch_sizes.clear()

    engine.predict(marker_image(10), cache)

    assert embedder.batch_sizes == [3]


def test_dedup_maps_two_equal_slots_back_to_original_order():
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 20)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    slots = (marker_image(10), marker_image(20), marker_image(20))

    embeddings, unique_count = engine._embed_query_slots(slots)

    assert embedder.batch_sizes == [2]
    assert unique_count == 2
    assert np.array_equal(embeddings[1], embeddings[2])


def test_dedup_comparison_error_falls_back_to_one_three_slot_call():
    embedder = FakeEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 30)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )
    slots = (marker_image(10), ComparisonErrorImage(marker_image(20)), marker_image(30))

    embeddings, unique_count = engine._embed_query_slots(slots)

    assert embedder.batch_sizes == [3]
    assert len(embeddings) == 3
    assert unique_count == 3


def test_dedup_does_not_retry_when_embedder_raises():
    embedder = RaisingEmbedder()
    engine = FastOrientationEngine(
        embedder, SlotGeometry((10, 20, 30)), image_reader=read_marker,
        deduplicate_identical_slots=True,
    )

    with pytest.raises(RuntimeError, match="embedding failed"):
        engine._embed_query_slots((marker_image(10), marker_image(20), marker_image(30)))

    assert embedder.calls == 1


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


def test_augmentation_scales_rule_center_from_geometry_context_to_source_pixels():
    front = bright_marker_image(10, size=(512, 1024), center=(800, 400))
    back = bright_marker_image(200, size=(512, 1024), center=(800, 400))
    embedder = FakeEmbedder(record_batches=True)
    engine = FastOrientationEngine(
        embedder,
        RuleCenterGeometry({"front": (200.0, 100.0), "back": (200.0, 100.0)}),
        image_reader=map_reader({"front.png": front, "back.png": back}),
    )

    engine.build_cache([Path("front.png")], [Path("back.png")], geometry_profile=None, library_revision=1, model_fingerprint="m")

    first_rotated_raw_slot = recorded_images(embedder)[3]
    assert first_rotated_raw_slot[400, 800].tolist() == [255, 255, 255]


def test_augmentation_uses_the_active_rule_center_for_its_own_label():
    front = bright_marker_image(10, size=(64, 64), center=(12, 20))
    back = bright_marker_image(200, size=(64, 64), center=(44, 36))
    embedder = FakeEmbedder(record_batches=True)
    engine = FastOrientationEngine(
        embedder,
        RuleCenterGeometry({"front": (12.0, 20.0), "back": (44.0, 36.0)}),
        image_reader=map_reader({"front.png": front, "back.png": back}),
    )

    engine.build_cache([Path("front.png")], [Path("back.png")], geometry_profile=None, library_revision=1, model_fingerprint="m")

    images = recorded_images(embedder)
    assert images[3][20, 12].tolist() == [255, 255, 255]
    assert images[39][36, 44].tolist() == [255, 255, 255]


def test_augmentation_without_an_active_rule_center_uses_decoded_image_center():
    front = bright_marker_image(10, size=(64, 64), center=(32, 32))
    back = bright_marker_image(200, size=(64, 64), center=(32, 32))
    embedder = FakeEmbedder(record_batches=True)
    engine = FastOrientationEngine(
        embedder,
        RuleCenterGeometry({"front": (10.0, 10.0), "back": (54.0, 54.0)}, active_labels=set()),
        image_reader=map_reader({"front.png": front, "back.png": back}),
    )

    engine.build_cache([Path("front.png")], [Path("back.png")], geometry_profile=None, library_revision=1, model_fingerprint="m")

    assert recorded_images(embedder)[3][32, 32].tolist() == [255, 255, 255]


@pytest.mark.parametrize("slots", [
    (np.ones(2, np.float32), np.ones(3, np.float32), np.ones(2, np.float32)),
    (np.ones(2, np.float32), np.empty(0, np.float32), np.ones(2, np.float32)),
    (np.ones(2, np.float32), np.array([np.nan, 1.0], np.float32), np.ones(2, np.float32)),
])
def test_build_rejects_invalid_embedding_slots_before_ridge(monkeypatch, tmp_path, slots):
    front = write_marker(tmp_path / "front.png", 10)
    back = write_marker(tmp_path / "back.png", 200)
    engine = FastOrientationEngine(
        RepeatingEmbedder(slots),
        FastGeometryProcessor(FakeCalibrator()),
        image_reader=read_marker,
    )
    monkeypatch.setattr(fast_orientation, "fit_ridge_head", lambda *args, **kwargs: pytest.fail("Ridge must not run"))

    with pytest.raises(ValueError, match="FAST_FEATURE_INVALID"):
        engine.build_cache([front], [back], geometry_profile=None, library_revision=1, model_fingerprint="m")


@pytest.mark.parametrize("slots", [
    (np.ones(2, np.float32), np.ones(3, np.float32), np.ones(2, np.float32)),
    (np.ones(2, np.float32), np.empty(0, np.float32), np.ones(2, np.float32)),
    (np.ones(2, np.float32), np.array([np.inf, 1.0], np.float32), np.ones(2, np.float32)),
])
def test_predict_rejects_invalid_embedding_slots_before_ridge(monkeypatch, tmp_path, slots):
    engine, _ = fake_engine()
    front = write_markers(tmp_path, "front", 2, base=10)
    back = write_markers(tmp_path, "back", 2, base=200)
    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m")
    invalid = RepeatingEmbedder(slots)
    engine.embed_batch = invalid
    monkeypatch.setattr(fast_orientation, "predict_ridge", lambda *args, **kwargs: pytest.fail("Ridge must not run"))

    with pytest.raises(ValueError, match="FAST_FEATURE_INVALID"):
        engine.predict(marker_image(10), cache)

    assert invalid.batch_sizes == [3]


def test_predict_rejects_mutated_cache_invariants_and_accepts_valid_cache(tmp_path):
    engine, embedder = fake_engine()
    front = write_markers(tmp_path, "front", 2, base=10)
    back = write_markers(tmp_path, "back", 2, base=200)
    cache = engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m")

    assert engine.predict(marker_image(10), cache)["label"] in {"front", "back"}
    embedder.batch_sizes.clear()
    invalid_digest = deepcopy(cache.template_signature)
    invalid_digest["front"][0]["sha256"] = "not-a-sha256"
    invalid_combined = deepcopy(cache.template_signature)
    invalid_combined["combined_sha256"] = "0" * 64
    invalid_head = replace(cache.ridge_head, weights=np.ones(5, np.float32), feature_dim=5)
    invalid_caches = [
        replace(cache, cache_revision="0" * 64),
        replace(cache, template_counts={"front": 0, "back": 2}),
        replace(cache, template_signature=invalid_digest),
        replace(cache, template_signature=invalid_combined),
        replace(cache, ridge_head=invalid_head),
    ]

    for invalid_cache in invalid_caches:
        with pytest.raises(ValueError, match="FAST_CACHE_REVISION_MISMATCH"):
            engine.predict(marker_image(10), invalid_cache)

    assert embedder.batch_sizes == []


def test_augmentation_progress_is_emitted_as_samples_are_generated(tmp_path):
    events: list[dict] = []
    front = write_markers(tmp_path, "front", 2, base=10)
    back = write_markers(tmp_path, "back", 20, base=100)
    engine, _ = fake_engine()

    engine.build_cache(front, back, geometry_profile=None, library_revision=1, model_fingerprint="m", progress_callback=events.append)

    first_augmentation = next(index for index, event in enumerate(events) if event["phase"] == "fast_augmentation")
    second_original = next(index for index, event in enumerate(events) if event["phase"] == "fast_originals" and event["completed"] == 2)
    augmentation = [event for event in events if event["phase"] == "fast_augmentation"]
    assert first_augmentation < second_original
    assert [event["completed"] for event in augmentation] == list(range(1, 23))
    assert all(event["total"] == 22 and event["unit"] == "augmented_samples" for event in augmentation)
