from pathlib import Path

import cv2
import numpy as np
import pytest

from src.orientation_classifier import OrientationClassifier, TemplateCache
from src.workpiece_catalog import RestoreConflictError, WorkpieceCatalog
from src.workpiece_library import WorkpieceLibrary


def image(path: Path, marker: int) -> Path:
    assert cv2.imwrite(str(path), np.full((8, 8, 3), marker, dtype=np.uint8))
    return path


def builder(front, back, progress_callback=None):
    return TemplateCache(
        global_vectors={
            "front": np.zeros((len(front), 2), dtype=np.float32),
            "back": np.ones((len(back), 2), dtype=np.float32),
        },
        local_features={"front": [{} for _ in front], "back": [{} for _ in back]},
    )


class FakeClassifier:
    def __init__(self):
        self.caches = {}

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def remove_template_cache(self, workpiece_id):
        self.caches.pop(workpiece_id, None)

    def get_template_cache(self, workpiece_id):
        return self.caches.get(workpiece_id)

    def build_template_cache(self, front, back, progress_callback=None):
        return builder(front, back, progress_callback)


class CachingFakeClassifier(FakeClassifier):
    def __init__(self, preloaded=None):
        super().__init__()
        self.preloaded = preloaded
        self.load_calls = []
        self.save_calls = []
        self.build_calls = 0

    def build_template_cache(self, front, back, progress_callback=None):
        self.build_calls += 1
        return super().build_template_cache(front, back, progress_callback)

    def load_template_cache(self, record):
        self.load_calls.append(record.id)
        return self.preloaded

    def save_template_cache(self, record, cache):
        self.save_calls.append((record.id, cache))


def create_catalog(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    front = [image(tmp_path / "front.png", 10)]
    back = [image(tmp_path / "back.png", 20)]
    record, _ = catalog.register("M7", front, back, False)
    return catalog, classifier, record


def test_recycle_removes_prediction_and_restore_republishes_same_workpiece(tmp_path):
    catalog, classifier, record = create_catalog(tmp_path)

    recycled = catalog.recycle(record.id, operation_id="delete-1")

    assert recycled["id"] == record.id
    assert catalog.list_workpieces() == []
    assert record.id not in classifier.caches
    assert catalog.list_recycled()[0]["id"] == record.id

    restored = catalog.restore(record.id, operation_id="restore-1")

    assert restored.id == record.id
    assert catalog.list_workpieces() == [{"id": record.id, "name": "M7"}]
    assert record.id in classifier.caches


def test_restore_rejects_case_insensitive_name_conflict_without_mutating_recycle(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    catalog.recycle(record.id, operation_id="delete-1")
    front = [image(tmp_path / "other-front.png", 30)]
    back = [image(tmp_path / "other-back.png", 40)]
    catalog.register("M7", front, back, False)

    with pytest.raises(RestoreConflictError):
        catalog.restore(record.id, operation_id="restore-1")

    assert catalog.list_recycled()[0]["id"] == record.id


def test_purge_removes_only_recycled_workpiece(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    catalog.recycle(record.id, operation_id="delete-1")

    catalog.purge(record.id, operation_id="purge-1")

    assert catalog.list_recycled() == []


def test_annotation_snapshot_contains_template_metadata_and_revision(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    document = catalog.get_annotation_snapshot(record.id)

    assert document["workpiece_id"] == record.id
    assert document["revision"] == record.revision
    assert document["annotation_revision"] == 1
    assert document["active_annotation_revision"] == 1
    assert document["templates"] == [
        {
            "template_id": "front:00.png",
            "orientation": "front",
            "index": 0,
            "preview_path": str(record.front_images[0]),
            "width": 8,
            "height": 8,
            "readable": True,
            "mask_effect": {"keypoints_before": 0, "keypoints_after": 0, "remaining_ratio": 1.0},
        },
        {
            "template_id": "back:00.png",
            "orientation": "back",
            "index": 0,
            "preview_path": str(record.back_images[0]),
            "width": 8,
            "height": 8,
            "readable": True,
            "mask_effect": {"keypoints_before": 0, "keypoints_after": 0, "remaining_ratio": 1.0},
        },
    ]
    assert document["groups"] == []


def test_prepare_template_masks_returns_unpublished_candidate_with_snapshot_statistics():
    classifier = OrientationClassifier.__new__(OrientationClassifier)
    original = TemplateCache(
        global_vectors={"front": np.zeros((1, 2), dtype=np.float32), "back": np.ones((1, 2), dtype=np.float32)},
        local_features={
            "front": [{"keypoints": np.array([[[1.0, 1.0], [8.0, 8.0]]], dtype=np.float32)}],
            "back": [{}],
        },
        raw_local_features={
            "front": [{"keypoints": np.array([[[1.0, 1.0], [8.0, 8.0]]], dtype=np.float32)}],
            "back": [{}],
        },
    )
    classifier._template_caches = {"m7": original}
    ignored_regions = {"front": [[{"x": 0, "y": 0, "width": 4, "height": 4}]], "back": [[]]}

    candidate, statistics = classifier.prepare_template_masks("m7", ignored_regions)
    ignored_regions["front"][0][0]["x"] = 99

    assert classifier.get_template_cache("m7") is original
    assert candidate is not original
    assert candidate.local_features["front"][0]["keypoints"].shape == (1, 1, 2)
    assert statistics["front"][0] == {
        "keypoints_before": 2,
        "keypoints_after": 1,
        "remaining_ratio": pytest.approx(0.5),
    }
    assert candidate.ignored_regions["front"][0][0]["x"] == 0


def test_catalog_recover_uses_classifier_cache_before_rebuilding(tmp_path):
    initial_library = WorkpieceLibrary(tmp_path / "library")
    front = [image(tmp_path / "front.png", 10)]
    back = [image(tmp_path / "back.png", 20)]
    record, expected_cache = initial_library.register("M7", front, back, False, builder)
    classifier = CachingFakeClassifier(preloaded=expected_cache)
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    recovered = catalog.recover()

    assert [item.id for item, _ in recovered] == [record.id]
    assert classifier.load_calls == [record.id]
    assert classifier.build_calls == 0


def test_catalog_persists_cache_after_register_append_and_restore(tmp_path):
    classifier = CachingFakeClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    front = [image(tmp_path / "front.png", 10)]
    back = [image(tmp_path / "back.png", 20)]

    record, _ = catalog.register("M7", front, back, False)
    catalog.append_templates(record.id, [image(tmp_path / "front-extra.png", 11)], [])
    catalog.recycle(record.id, operation_id="recycle-1")
    catalog.restore(record.id, operation_id="restore-1")

    assert [item[0] for item in classifier.save_calls] == [record.id, record.id, record.id]
