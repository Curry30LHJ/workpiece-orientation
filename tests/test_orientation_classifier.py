from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from dataclasses import replace
import json
import os
import pickle
import shutil
import sys
import threading
import time
import types

import cv2
import numpy as np
import pytest

from src.orientation_classifier import (
    ComputeDeviceError,
    ImageUnreadableError,
    LocalSearchResult,
    ModelFingerprintError,
    OrientationClassifier,
    OrientationClassifierError,
    PropagationModelError,
    TemplateCache,
    WorkpieceNotFoundError,
)
from src.fast_geometry import FastGeometryProcessor
from src.fast_orientation import FastOrientationEngine


class FakeGlobalPredictor:
    def __init__(self):
        self.calls = 0
        self.markers = []
        self.batch_sizes = []

    def predict(self, images):
        self.calls += 1
        markers = [int(image[0, 0, 0]) for image in images]
        self.markers.append(markers)
        self.batch_sizes.append(len(images))
        vectors = {
            1: np.array([1.0, 0.0], dtype=np.float32),
            2: np.array([0.0, 1.0], dtype=np.float32),
            3: np.array([0.90, 0.91], dtype=np.float32),
            4: np.array([0.90, 0.70], dtype=np.float32),
            7: np.array([0.95, 0.05], dtype=np.float32),
        }
        return [vectors[marker] for marker in markers]


class FakeExtractor:
    def __init__(self):
        self.calls = 0
        self.markers = []


class FakeMatcher:
    def __init__(self):
        self.calls = 0


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
        3: {1: 12.4, 2: 5.1, 3: 12.4, 4: 5.1},
        4: {1: 4.0, 2: 9.0, 3: 4.0, 4: 9.0},
        7: {1: 12.4, 2: 5.1, 7: 12.4},
    }
    return {"score": scores[query_features["marker"]][template_features["marker"]]}


def write_marker(path: Path, marker: int) -> Path:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def build_fast_runtime(
    front,
    back,
    *,
    library_revision,
    model_fingerprint,
    geometry_profile=None,
):
    engine = make_fast_engine()
    return engine.build_cache(
        front,
        back,
        geometry_profile=geometry_profile,
        library_revision=library_revision,
        model_fingerprint=model_fingerprint,
    )


def make_fast_engine(embed_batch=None):
    def default_embed_batch(images):
        return [
            np.asarray(
                [float(np.mean(image)) / 255.0, 1.0 - float(np.mean(image)) / 255.0],
                dtype=np.float32,
            )
            for image in images
        ]

    return FastOrientationEngine(
        embed_batch or default_embed_batch,
        FastGeometryProcessor(FakeGeometryCalibrator()),
        image_reader=lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR),
    )


class ReturningFastEngine:
    def __init__(self):
        self.calls = 0

    def predict(self, image, cache):
        self.calls += 1
        return {
            "label": "front",
            "needs_review": False,
            "inference_engine": "fast_geometry",
            "elapsed_ms": 2.0,
            "timings_ms": {
                "geometry_context": 0.1,
                "geometry_fit": 0.2,
                "mask_build": 0.1,
                "global_batch": 1.0,
                "linear_head": 0.1,
                "total": 2.0,
            },
        }


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


def test_fast_prediction_uses_fast_runtime_without_extracting_local(classifier, tmp_path):
    front = [write_marker(tmp_path / "fast-front.png", 1)]
    back = [write_marker(tmp_path / "fast-back.png", 2)]
    cache = classifier.build_template_cache(front, back)
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=9,
        model_fingerprint="model-a",
    )
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = ReturningFastEngine()
    classifier.extractor.calls = 0
    classifier.matcher.calls = 0

    result = classifier.predict_with_cache(
        replace(cache, fast_runtime=runtime),
        write_marker(tmp_path / "fast-query.png", 3),
        library_revision=9,
    )

    assert result["label"] == "front"
    assert result["library_revision"] == 9
    assert classifier.extractor.calls == 0
    assert classifier.matcher.calls == 0
    assert result["inference_engine"] == "fast_geometry"


def test_fast_mode_rejects_missing_runtime_cache(classifier, tmp_path):
    cache = TemplateCache(
        global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        local_features={"front": [{}], "back": [{}]},
    )
    classifier.inference_mode = "fast_geometry"
    classifier.fast_engine = ReturningFastEngine()

    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_NOT_READY"):
        classifier.predict_with_cache(
            cache,
            write_marker(tmp_path / "missing-fast-query.png", 3),
            library_revision=9,
        )

    assert classifier.global_predictor.calls == 0
    assert classifier.extractor.calls == 0
    assert classifier.matcher.calls == 0
    assert classifier.fast_engine.calls == 0


def test_fast_cache_round_trip_binds_library_geometry_and_model_revisions(classifier, tmp_path):
    front = [write_marker(tmp_path / "round-trip-front.png", 1)]
    back = [write_marker(tmp_path / "round-trip-back.png", 2)]
    cache = classifier.build_template_cache(front, back)
    runtime = build_fast_runtime(
        front,
        back,
        geometry_profile=geometry_profile(),
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=3,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"

    classifier.save_template_cache(record, replace(cache, fast_runtime=runtime))
    with (record.root / ".template_cache.pkl").open("rb") as stream:
        base_payload = pickle.load(stream)
    loaded = classifier.load_template_cache(record)

    assert getattr(base_payload["cache"], "fast_runtime", None) is None
    assert loaded is not None
    assert loaded.fast_runtime is not None
    assert loaded.fast_runtime.library_revision == 7
    assert loaded.fast_runtime.geometry_profile_revision == 3
    assert loaded.fast_runtime.model_fingerprint == "model-a"
    assert loaded.fast_runtime.template_counts == {"front": 1, "back": 1}
    np.testing.assert_array_equal(
        loaded.fast_runtime.ridge_head.weights,
        runtime.ridge_head.weights,
    )
    assert loaded.fast_runtime.ridge_head.bias == runtime.ridge_head.bias
    assert loaded.fast_runtime.ridge_head.regularization == runtime.ridge_head.regularization
    assert loaded.fast_runtime.ridge_head.review_threshold == runtime.ridge_head.review_threshold
    assert loaded.fast_runtime.ridge_head.feature_dim == runtime.ridge_head.feature_dim


def test_staged_fast_cache_commit_and_rollback_preserve_previous_sidecar(classifier, tmp_path):
    front = [write_marker(tmp_path / "staged-front.png", 1)]
    back = [write_marker(tmp_path / "staged-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    old_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "old"})
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    target = record.root / ".fast_runtime_cache.pkl"
    previous = target.read_bytes()

    staged = classifier.stage_fast_runtime_cache(
        record,
        new_runtime,
        geometry_profile_revision=None,
    )

    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "old"
    committed = classifier.commit_staged_fast_runtime_cache(record, staged)
    committed_again = classifier.commit_staged_fast_runtime_cache(record, staged)
    assert committed_again == committed
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "new"
    classifier.rollback_committed_fast_runtime_cache(committed)
    classifier.rollback_committed_fast_runtime_cache(committed_again)
    classifier.discard_staged_fast_runtime_cache(staged)
    classifier.discard_staged_fast_runtime_cache(staged)
    assert target.read_bytes() == previous
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "old"


def test_staged_fast_cache_replacement_keeps_live_sidecar_readable(
    classifier,
    tmp_path,
    monkeypatch,
):
    front = [write_marker(tmp_path / "atomic-front.png", 1)]
    back = [write_marker(tmp_path / "atomic-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    old_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "old"})
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    staged = classifier.stage_fast_runtime_cache(
        record,
        new_runtime,
        geometry_profile_revision=None,
    )
    target = record.root / ".fast_runtime_cache.pkl"
    previous = target.read_bytes()
    replacing = threading.Event()
    release = threading.Event()
    errors = []
    committed = []
    real_replace = os.replace

    def block_atomic_replace(source, destination):
        if Path(source) == staged.temporary_path and Path(destination) == target:
            replacing.set()
            assert release.wait(2.0)
        return real_replace(source, destination)

    def commit():
        try:
            committed.append(classifier.commit_staged_fast_runtime_cache(record, staged))
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr("src.orientation_classifier.os.replace", block_atomic_replace)
    worker = threading.Thread(target=commit)
    worker.start()
    try:
        assert replacing.wait(1.0)
        assert target.is_file()
        assert target.read_bytes() == previous
        assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "old"
    finally:
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert errors == []
    assert len(committed) == 1
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "new"


def test_discard_after_interrupted_commit_restores_previous_sidecar(
    classifier,
    tmp_path,
    monkeypatch,
):
    front = [write_marker(tmp_path / "discard-front.png", 1)]
    back = [write_marker(tmp_path / "discard-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    old_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "old"})
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    target = record.root / ".fast_runtime_cache.pkl"
    previous = target.read_bytes()
    staged = classifier.stage_fast_runtime_cache(record, new_runtime)
    real_replace = os.replace

    def replace_then_raise(source, destination):
        real_replace(source, destination)
        if Path(source) == staged.temporary_path and Path(destination) == target:
            raise OSError("interrupted after atomic replacement")

    monkeypatch.setattr("src.orientation_classifier.os.replace", replace_then_raise)
    with pytest.raises(OSError, match="interrupted after atomic replacement"):
        classifier.commit_staged_fast_runtime_cache(record, staged)

    classifier.discard_staged_fast_runtime_cache(staged)
    classifier.discard_staged_fast_runtime_cache(staged)

    assert target.read_bytes() == previous
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "old"


def test_fast_cache_stage_recovery_discards_uncommitted_orphan_without_touching_live_sidecar(
    classifier,
    tmp_path,
):
    front = [write_marker(tmp_path / "orphan-front.png", 1)]
    back = [write_marker(tmp_path / "orphan-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    old_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "old"})
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    target = record.root / ".fast_runtime_cache.pkl"
    previous = target.read_bytes()
    staged = classifier.stage_fast_runtime_cache(record, new_runtime)

    classifier.recover_fast_runtime_cache_staging(record)

    assert target.read_bytes() == previous
    assert not staged.temporary_root.exists()
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "old"


def test_fast_cache_stage_recovery_restores_old_sidecar_after_interrupted_revision_commit(
    classifier,
    tmp_path,
):
    front = [write_marker(tmp_path / "restart-front.png", 1)]
    back = [write_marker(tmp_path / "restart-back.png", 2)]
    old_runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    new_runtime = build_fast_runtime(
        front,
        back,
        library_revision=8,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    target_record = SimpleNamespace(**{**record.__dict__, "revision": 8})
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    target = record.root / ".fast_runtime_cache.pkl"
    previous = target.read_bytes()
    staged = classifier.stage_fast_runtime_cache(target_record, new_runtime)
    os.replace(staged.temporary_path, target)

    classifier.recover_fast_runtime_cache_staging(record)

    assert target.read_bytes() == previous
    assert not staged.temporary_root.exists()
    assert classifier.load_fast_runtime_cache(record).library_revision == 7


def test_fast_cache_stage_recovery_keeps_valid_live_sidecar_when_cleaning_orphan(
    classifier,
    tmp_path,
):
    front = [write_marker(tmp_path / "valid-orphan-front.png", 1)]
    back = [write_marker(tmp_path / "valid-orphan-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    staged = classifier.stage_fast_runtime_cache(record, new_runtime)
    target = record.root / ".fast_runtime_cache.pkl"
    os.replace(staged.temporary_path, target)
    published = target.read_bytes()

    classifier.recover_fast_runtime_cache_staging(record)

    assert target.read_bytes() == published
    assert not staged.temporary_root.exists()
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == "new"


def test_fast_cache_stage_recovery_uses_committed_manifest_pointer_during_legacy_rollback(
    classifier,
    tmp_path,
):
    front = [write_marker(tmp_path / "manifest-front.png", 1)]
    back = [write_marker(tmp_path / "manifest-back.png", 2)]
    old_runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        geometry_profile={"profile_revision": 4},
        model_fingerprint="model-a",
    )
    new_runtime = build_fast_runtime(
        front,
        back,
        library_revision=8,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
    )
    target_record = SimpleNamespace(**{**record.__dict__, "revision": 8})
    record.root.mkdir()
    profile_root = record.root / "geometry_masks"
    profile_root.mkdir()
    (profile_root / "profile.json").write_text(
        json.dumps({"active_revision": 4}),
        encoding="utf-8",
    )
    manifest_path = record.root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"revision": 7, "geometry_mask_active_revision": 4}),
        encoding="utf-8",
    )
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, old_runtime)
    staged = classifier.stage_fast_runtime_cache(
        target_record,
        new_runtime,
        geometry_profile_revision=None,
    )
    target = record.root / ".fast_runtime_cache.pkl"
    os.replace(staged.temporary_path, target)
    published = target.read_bytes()
    manifest_path.write_text(
        json.dumps({"revision": 8, "geometry_mask_active_revision": None}),
        encoding="utf-8",
    )

    classifier.recover_fast_runtime_cache_staging(target_record)

    assert target.read_bytes() == published
    assert not staged.temporary_root.exists()
    assert classifier.load_fast_runtime_cache(target_record).library_revision == 8
    assert classifier.load_fast_runtime_cache(target_record).geometry_profile_revision is None


def test_staged_fast_cache_partial_serialization_failure_removes_sibling_stage(
    classifier,
    tmp_path,
    monkeypatch,
):
    front = [write_marker(tmp_path / "partial-front.png", 1)]
    back = [write_marker(tmp_path / "partial-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, runtime)
    previous = (record.root / ".fast_runtime_cache.pkl").read_bytes()

    def fail_after_partial_write(_payload, stream, **_kwargs):
        stream.write(b"partial")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr("src.orientation_classifier.pickle.dump", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="serialization failed"):
        classifier.stage_fast_runtime_cache(record, runtime)

    assert (record.root / ".fast_runtime_cache.pkl").read_bytes() == previous
    assert list(tmp_path.glob(".fast-runtime-stage-record-*")) == []


def test_staging_cleanup_failure_preserves_primary_error_and_releases_record_lock(
    classifier,
    tmp_path,
    monkeypatch,
):
    front = [write_marker(tmp_path / "cleanup-front.png", 1)]
    back = [write_marker(tmp_path / "cleanup-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, runtime)
    real_dump = pickle.dump
    real_rmtree = shutil.rmtree

    def fail_serialization(_payload, stream, **_kwargs):
        stream.write(b"partial")
        raise RuntimeError("primary serialization failure")

    def fail_stage_cleanup(path, *args, **kwargs):
        if Path(path).name.startswith(".fast-runtime-stage-record-"):
            raise OSError("stage cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("src.orientation_classifier.pickle.dump", fail_serialization)
    monkeypatch.setattr("src.orientation_classifier.shutil.rmtree", fail_stage_cleanup)
    try:
        classifier.stage_fast_runtime_cache(record, runtime)
    except Exception as exc:
        first_error = exc
    else:
        pytest.fail("fault-injected staging unexpectedly succeeded")
    monkeypatch.setattr("src.orientation_classifier.pickle.dump", real_dump)
    monkeypatch.setattr("src.orientation_classifier.shutil.rmtree", real_rmtree)

    completed = threading.Event()
    staged = []
    errors = []

    def stage_again():
        try:
            staged.append(classifier.stage_fast_runtime_cache(record, runtime))
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(target=stage_again, daemon=True)
    worker.start()
    assert completed.wait(1.0), "failed staging cleanup permanently held the record lock"
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert type(first_error) is RuntimeError
    assert str(first_error) == "primary serialization failure"
    assert errors == []
    assert len(staged) == 1
    classifier.discard_staged_fast_runtime_cache(staged[0])
    classifier.recover_fast_runtime_cache_staging(record)
    classifier.recover_fast_runtime_cache_staging(record)
    assert list(tmp_path.glob(".fast-runtime-stage-record-*")) == []


@pytest.mark.parametrize("failure_point", ["replace", "unlink", "cleanup"])
def test_failed_committed_rollback_releases_record_lock_and_remains_recoverable(
    classifier,
    tmp_path,
    monkeypatch,
    failure_point,
):
    front = [write_marker(tmp_path / "rollback-failure-front.png", 1)]
    back = [write_marker(tmp_path / "rollback-failure-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    old_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "old"})
    new_runtime = replace(runtime, training_summary={**runtime.training_summary, "marker": "new"})
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    if failure_point != "unlink":
        classifier.save_fast_runtime_cache(record, old_runtime)
    staged = classifier.stage_fast_runtime_cache(record, new_runtime)
    committed = classifier.commit_staged_fast_runtime_cache(record, staged)
    target = record.root / ".fast_runtime_cache.pkl"
    published = target.read_bytes()
    real_replace = os.replace
    real_unlink = Path.unlink
    real_rmtree = shutil.rmtree

    def fail_rollback(source, destination):
        if Path(source) == committed.backup_path and Path(destination) == target:
            raise OSError("rollback replace failure")
        return real_replace(source, destination)

    def fail_unlink(path, *args, **kwargs):
        if path == target:
            raise OSError("rollback unlink failure")
        return real_unlink(path, *args, **kwargs)

    def fail_cleanup(path, *args, **kwargs):
        if Path(path) == committed.temporary_root:
            raise OSError("rollback cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    if failure_point == "replace":
        monkeypatch.setattr("src.orientation_classifier.os.replace", fail_rollback)
    elif failure_point == "unlink":
        monkeypatch.setattr(Path, "unlink", fail_unlink)
    else:
        monkeypatch.setattr("src.orientation_classifier.shutil.rmtree", fail_cleanup)
    with pytest.raises(OSError, match=f"rollback {failure_point} failure"):
        classifier.rollback_committed_fast_runtime_cache(committed)
    monkeypatch.setattr("src.orientation_classifier.os.replace", real_replace)
    monkeypatch.setattr(Path, "unlink", real_unlink)
    monkeypatch.setattr("src.orientation_classifier.shutil.rmtree", real_rmtree)
    assert committed.temporary_root.is_dir()
    if failure_point == "replace":
        assert committed.backup_path is not None and committed.backup_path.is_file()
    expected_marker = "old" if failure_point == "cleanup" else "new"
    expected_sidecar = target.read_bytes()
    if failure_point != "cleanup":
        assert expected_sidecar == published

    completed = threading.Event()
    second_stage = []
    errors = []

    def stage_again():
        try:
            second_stage.append(classifier.stage_fast_runtime_cache(record, new_runtime))
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(target=stage_again, daemon=True)
    worker.start()
    assert completed.wait(1.0), "failed rollback permanently held the record lock"
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert errors == []
    assert len(second_stage) == 1
    classifier.discard_staged_fast_runtime_cache(second_stage[0])

    classifier.rollback_committed_fast_runtime_cache(committed)
    classifier.discard_staged_fast_runtime_cache(staged)
    assert target.read_bytes() == expected_sidecar
    classifier.recover_fast_runtime_cache_staging(record)
    assert not committed.temporary_root.exists()
    assert classifier.load_fast_runtime_cache(record).training_summary["marker"] == expected_marker


def test_recycled_stage_recovery_rejects_an_unrelated_parent(classifier, tmp_path):
    library_root = tmp_path / "library"
    record_root = library_root / ".recycled" / "m7"
    record_root.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    record = SimpleNamespace(id="m7", root=record_root)

    with pytest.raises(ValueError, match="staging parent is unrelated"):
        classifier.recover_recycled_fast_runtime_cache_staging(record, foreign)


def test_stage_recovery_does_not_follow_a_symlink_candidate(
    classifier,
    tmp_path,
    monkeypatch,
):
    record_root = tmp_path / "record"
    record_root.mkdir()
    record = SimpleNamespace(id="m7", root=record_root)
    candidate = tmp_path / f".fast-runtime-stage-record-{'a' * 32}"
    candidate.mkdir()
    marker = candidate / "outside-marker"
    marker.write_bytes(b"keep")
    real_is_symlink = Path.is_symlink

    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path == candidate or real_is_symlink(path),
    )

    classifier.recover_fast_runtime_cache_staging(record)

    assert candidate.is_dir()
    assert marker.read_bytes() == b"keep"


def test_same_record_fast_staging_is_serialized_until_first_transaction_finishes(
    classifier,
    tmp_path,
):
    front = [write_marker(tmp_path / "serialized-front.png", 1)]
    back = [write_marker(tmp_path / "serialized-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        id="m7",
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    first = classifier.stage_fast_runtime_cache(record, runtime)
    attempted = threading.Event()
    completed = threading.Event()
    second = []
    errors = []

    def stage_second():
        attempted.set()
        try:
            second.append(classifier.stage_fast_runtime_cache(record, runtime))
        except Exception as exc:
            errors.append(exc)
        finally:
            completed.set()

    worker = threading.Thread(target=stage_second)
    worker.start()
    assert attempted.wait(1.0)
    assert not completed.wait(0.1)
    classifier.discard_staged_fast_runtime_cache(first)
    assert completed.wait(1.0)
    worker.join(timeout=1.0)
    assert not worker.is_alive()

    assert errors == []
    assert len(second) == 1
    classifier.discard_staged_fast_runtime_cache(second[0])


def test_fast_cache_revision_mismatch_is_ignored_without_deleting_base_cache(classifier, tmp_path):
    front = [write_marker(tmp_path / "stale-front.png", 1)]
    back = [write_marker(tmp_path / "stale-back.png", 2)]
    cache = classifier.build_template_cache(front, back)
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    root = tmp_path / "record"
    root.mkdir()
    matching = SimpleNamespace(
        root=root,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
    )
    classifier.model_fingerprint = "model-a"
    classifier.save_template_cache(matching, replace(cache, fast_runtime=runtime))
    stale = SimpleNamespace(**{**vars(matching), "revision": 8})

    loaded = classifier.load_template_cache(stale)

    assert loaded is not None
    assert loaded.fast_runtime is None
    assert (root / ".template_cache.pkl").is_file()
    assert (root / ".fast_runtime_cache.pkl").is_file()


def test_attached_fast_cache_revision_mismatch_is_never_used(classifier, tmp_path):
    front = [write_marker(tmp_path / "attached-front.png", 1)]
    back = [write_marker(tmp_path / "attached-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = ReturningFastEngine()
    cache = TemplateCache(
        global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        local_features={"front": [{}], "back": [{}]},
        fast_runtime=runtime,
    )

    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            cache,
            write_marker(tmp_path / "attached-query.png", 3),
            library_revision=8,
        )

    assert classifier.fast_engine.calls == 0
    assert classifier.global_predictor.calls == 0
    assert classifier.extractor.calls == 0
    assert classifier.matcher.calls == 0


def test_attached_fast_cache_model_fingerprint_mismatch_is_never_used(classifier, tmp_path):
    front = [write_marker(tmp_path / "model-front.png", 1)]
    back = [write_marker(tmp_path / "model-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-b"
    classifier.fast_engine = ReturningFastEngine()
    cache = TemplateCache(
        global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        local_features={"front": [{}], "back": [{}]},
        fast_runtime=runtime,
    )

    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            cache,
            write_marker(tmp_path / "model-query.png", 3),
            library_revision=7,
        )

    assert classifier.fast_engine.calls == 0
    assert classifier.global_predictor.calls == 0
    assert classifier.extractor.calls == 0


def test_old_v2_template_cache_loads_with_fast_runtime_none(classifier, tmp_path):
    front = [write_marker(tmp_path / "v2-front.png", 1)]
    back = [write_marker(tmp_path / "v2-back.png", 2)]
    cache = classifier.build_template_cache(front, back)
    if "fast_runtime" in cache.__dict__:
        object.__delattr__(cache, "fast_runtime")
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=1,
    )
    record.root.mkdir()
    payload = {
        "signature": classifier._template_cache_signature(front, back),
        "cache": cache,
    }
    with (record.root / ".template_cache.pkl").open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = classifier.load_template_cache(record)

    assert loaded is not None
    np.testing.assert_array_equal(loaded.global_vectors["front"], cache.global_vectors["front"])
    assert loaded.local_features == cache.local_features
    assert loaded.fast_runtime is None


def test_legacy_load_rebuilds_fast_built_base_missing_local_features(classifier, tmp_path):
    front = [write_marker(tmp_path / "legacy-rebuild-front.png", 1)]
    back = [write_marker(tmp_path / "legacy-rebuild-back.png", 2)]
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=1,
    )
    record.root.mkdir()
    base = TemplateCache(
        global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        local_features={"front": [], "back": []},
        raw_global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        raw_local_features={"front": [], "back": []},
    )
    with (record.root / ".template_cache.pkl").open("wb") as stream:
        pickle.dump(
            {
                "signature": classifier._template_cache_signature(front, back),
                "cache": base,
            },
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    loaded = classifier.load_template_cache(record)

    assert loaded is not None
    assert loaded.local_features == {"front": [{"marker": 1}], "back": [{"marker": 2}]}
    assert classifier.extractor.calls == 2


def _install_fake_paddle_runtime(monkeypatch, *, compiled=True, gpu_count=1):
    selected = []
    paddle = types.ModuleType("paddle")
    paddle.is_compiled_with_cuda = lambda: compiled
    paddle.set_device = lambda value: selected.append(value) or value
    paddle.device = SimpleNamespace(
        cuda=SimpleNamespace(device_count=lambda: gpu_count),
        get_device=lambda: selected[-1] if selected else "cpu",
    )
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    return selected


def _install_fake_paddleclas(monkeypatch):
    captured = {}
    paddleclas = types.ModuleType("paddleclas")
    deploy = types.ModuleType("paddleclas.deploy")
    python_module = types.ModuleType("paddleclas.deploy.python")
    predict_rec = types.ModuleType("paddleclas.deploy.python.predict_rec")
    utils = types.ModuleType("paddleclas.deploy.utils")
    config_module = types.ModuleType("paddleclas.deploy.utils.config")

    class FakeRecPredictor:
        def __init__(self, config):
            captured["config"] = config

        def predict(self, images):
            captured.setdefault("predict_calls", []).append(len(images))
            return [np.asarray([1.0, 0.0], np.float32) for _ in images]

    def get_config(path, show=False):
        captured["config_path"] = path
        config = SimpleNamespace(Global=SimpleNamespace())
        captured["config"] = config
        return config

    predict_rec.RecPredictor = FakeRecPredictor
    config_module.get_config = get_config
    for name, module in {
        "paddleclas": paddleclas,
        "paddleclas.deploy": deploy,
        "paddleclas.deploy.python": python_module,
        "paddleclas.deploy.python.predict_rec": predict_rec,
        "paddleclas.deploy.utils": utils,
        "paddleclas.deploy.utils.config": config_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return captured


@pytest.mark.parametrize(
    "device,use_gpu,mkldnn,selected_device",
    [("gpu", True, False, "gpu:0"), ("cpu", False, True, "cpu")],
)
def test_load_configures_requested_paddle_device(
    tmp_path, monkeypatch, device, use_gpu, mkldnn, selected_device
):
    captured = _install_fake_paddleclas(monkeypatch)
    selected = _install_fake_paddle_runtime(monkeypatch)
    yaml_path = tmp_path / "部署 配置.yaml"
    yaml_path.write_text("Global: {}\n", encoding="utf-8")
    loaded = OrientationClassifier.load(
        tmp_path,
        tmp_path / "模型",
        paddle_config_path=yaml_path,
        compute_device=device,
        inference_mode="fast_geometry",
    )
    assert captured["config_path"] == str(yaml_path)
    assert captured["config"].Global.use_gpu is use_gpu
    assert captured["config"].Global.enable_mkldnn is mkldnn
    assert selected == [selected_device]
    assert captured["predict_calls"] == [1]
    assert loaded.compute_device == device


def test_gpu_load_refuses_missing_cuda_without_cpu_fallback(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch, compiled=False, gpu_count=0)
    with pytest.raises(ComputeDeviceError) as error:
        OrientationClassifier.load(
            tmp_path,
            tmp_path / "model",
            compute_device="gpu",
            inference_mode="fast_geometry",
        )
    assert error.value.code == "GPU_UNAVAILABLE"


def test_load_rejects_unknown_compute_device(tmp_path):
    with pytest.raises(ValueError, match="compute_device"):
        OrientationClassifier.load(
            tmp_path,
            tmp_path / "model",
            compute_device="automatic",
            inference_mode="fast_geometry",
        )


def test_cpu_load_never_probes_cuda(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    paddle = sys.modules["paddle"]

    def unexpected_cuda_probe():
        raise AssertionError("CPU edition probed CUDA")

    paddle.is_compiled_with_cuda = unexpected_cuda_probe
    paddle.device.cuda.device_count = unexpected_cuda_probe
    loaded = OrientationClassifier.load(
        tmp_path,
        tmp_path / "model",
        compute_device="cpu",
        inference_mode="fast_geometry",
    )
    assert loaded.compute_device == "cpu"


def test_load_rejects_wrong_expected_model_fingerprint(tmp_path, monkeypatch):
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "inference.pdmodel").write_bytes(b"model")
    with pytest.raises(ModelFingerprintError) as error:
        OrientationClassifier.load(
            tmp_path,
            model_dir,
            compute_device="gpu",
            expected_model_fingerprint="0" * 64,
            inference_mode="fast_geometry",
        )
    assert error.value.code == "MODEL_FINGERPRINT_MISMATCH"


def test_fast_load_does_not_construct_aliked_or_lightglue(tmp_path, monkeypatch):
    import src.aliked_lightglue_matcher as local_stack

    monkeypatch.setattr(
        local_stack,
        "build_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local stack loaded")),
    )
    _install_fake_paddleclas(monkeypatch)
    _install_fake_paddle_runtime(monkeypatch)

    loaded = OrientationClassifier.load(
        tmp_path,
        tmp_path / "model",
        inference_mode="fast_geometry",
    )

    assert loaded.extractor is None
    assert loaded.matcher is None
    assert loaded.fast_engine is not None


def test_fast_path_timing_includes_decode_and_full_classifier_call(classifier, tmp_path, monkeypatch):
    import src.orientation_classifier as classifier_module

    front = [write_marker(tmp_path / "timing-fast-front.png", 1)]
    back = [write_marker(tmp_path / "timing-fast-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=4,
        model_fingerprint="model-a",
    )
    base = classifier.build_template_cache(front, back)

    class DelayedFastEngine(ReturningFastEngine):
        def predict(self, image, cache):
            time.sleep(0.006)
            result = super().predict(image, cache)
            result["timings_ms"].update({"global_batch": 2.0, "linear_head": 1.0, "total": 3.0})
            return result

    original_reader = classifier_module.read_color_image

    def delayed_reader(path):
        time.sleep(0.006)
        return original_reader(path)

    monkeypatch.setattr(classifier_module, "read_color_image", delayed_reader)
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = DelayedFastEngine()
    cache = replace(base, fast_runtime=runtime)

    result = classifier.predict_with_cache(
        cache,
        write_marker(tmp_path / "timing-fast-query.png", 3),
        library_revision=4,
    )

    assert result["timings_ms"]["decode"] > 0.0
    assert result["timings_ms"]["total"] >= result["timings_ms"]["decode"] + 3.0
    assert result["elapsed_ms"] == result["timings_ms"]["total"]


def test_fast_array_input_reports_zero_decode_time(classifier, tmp_path):
    front = [write_marker(tmp_path / "array-fast-front.png", 1)]
    back = [write_marker(tmp_path / "array-fast-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=4,
        model_fingerprint="model-a",
    )
    base = classifier.build_template_cache(front, back)
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = ReturningFastEngine()
    cache = replace(base, fast_runtime=runtime)

    result = classifier.predict_with_cache(
        cache,
        np.full((8, 8, 3), 3, dtype=np.uint8),
        library_revision=4,
    )

    assert result["timings_ms"]["decode"] == 0.0


def test_profiled_fast_cache_is_rejected_against_no_profile_record(classifier, tmp_path):
    front = [write_marker(tmp_path / "profiled-front.png", 1)]
    back = [write_marker(tmp_path / "profiled-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        geometry_profile=geometry_profile(),
        library_revision=7,
        model_fingerprint="model-a",
    )
    root = tmp_path / "record"
    root.mkdir()
    profiled_record = SimpleNamespace(
        root=root,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=3,
    )
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(profiled_record, runtime)
    no_profile_record = SimpleNamespace(
        root=root,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )

    assert classifier.load_fast_runtime_cache(no_profile_record) is None

    base = classifier.build_template_cache(front, back)
    classifier.inference_mode = "fast_geometry"
    classifier.fast_engine = ReturningFastEngine()
    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            replace(base, fast_runtime=runtime, geometry_profile_revision=None),
            np.full((8, 8, 3), 3, dtype=np.uint8),
            library_revision=7,
        )
    assert classifier.fast_engine.calls == 0


def test_no_profile_fast_cache_is_accepted_only_for_no_profile(classifier, tmp_path):
    front = [write_marker(tmp_path / "no-profile-front.png", 1)]
    back = [write_marker(tmp_path / "no-profile-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=5,
        model_fingerprint="model-a",
    )
    root = tmp_path / "record"
    root.mkdir()
    no_profile_record = SimpleNamespace(
        root=root,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=5,
        geometry_profile_revision=None,
    )
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(no_profile_record, runtime)

    assert classifier.load_fast_runtime_cache(no_profile_record) is not None

    profiled_record = SimpleNamespace(
        root=root,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=5,
        geometry_profile_revision=3,
    )
    assert classifier.load_fast_runtime_cache(profiled_record) is None

    base = classifier.build_template_cache(front, back)
    classifier.inference_mode = "fast_geometry"
    classifier.fast_engine = ReturningFastEngine()
    accepted = classifier.predict_with_cache(
        replace(base, fast_runtime=runtime, geometry_profile_revision=None),
        np.full((8, 8, 3), 3, dtype=np.uint8),
        library_revision=5,
    )
    assert accepted["label"] == "front"
    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            replace(base, fast_runtime=runtime, geometry_profile_revision=3),
            np.full((8, 8, 3), 3, dtype=np.uint8),
            library_revision=5,
        )


def test_fast_cache_save_rejects_same_count_different_template_content(classifier, tmp_path):
    source_front = [write_marker(tmp_path / "source-front.png", 1)]
    source_back = [write_marker(tmp_path / "source-back.png", 2)]
    other_front = [write_marker(tmp_path / "other-front.png", 3)]
    other_back = [write_marker(tmp_path / "other-back.png", 4)]
    runtime = build_fast_runtime(
        source_front,
        source_back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(other_front),
        back_images=tuple(other_back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"

    with pytest.raises(ValueError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.save_fast_runtime_cache(record, runtime)

    assert not (record.root / ".fast_runtime_cache.pkl").exists()


def test_fast_cache_load_rejects_same_count_different_template_content(classifier, tmp_path):
    source_front = [write_marker(tmp_path / "load-source-front.png", 1)]
    source_back = [write_marker(tmp_path / "load-source-back.png", 2)]
    other_front = [write_marker(tmp_path / "load-other-front.png", 3)]
    other_back = [write_marker(tmp_path / "load-other-back.png", 4)]
    runtime = build_fast_runtime(
        source_front,
        source_back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(other_front),
        back_images=tuple(other_back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    persistence_signature = {
        "format_version": runtime.format_version,
        "library_revision": 7,
        "geometry_profile_revision": None,
        "model_fingerprint": "model-a",
        "template_content": classifier._template_cache_signature(other_front, other_back),
        "template_counts": {"front": 1, "back": 1},
        "feature_layout": ("raw", "front_masked", "back_masked"),
        "feature_dim": runtime.ridge_head.feature_dim,
    }
    with (record.root / ".fast_runtime_cache.pkl").open("wb") as stream:
        pickle.dump(
            {"signature": persistence_signature, "cache": runtime},
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    assert classifier.load_fast_runtime_cache(record) is None
    assert (record.root / ".fast_runtime_cache.pkl").is_file()


def test_fast_prediction_rejects_runtime_from_same_count_different_templates(classifier, tmp_path):
    source_front = [write_marker(tmp_path / "predict-source-front.png", 1)]
    source_back = [write_marker(tmp_path / "predict-source-back.png", 2)]
    other_front = [write_marker(tmp_path / "predict-other-front.png", 3)]
    other_back = [write_marker(tmp_path / "predict-other-back.png", 4)]
    runtime = build_fast_runtime(
        source_front,
        source_back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    base = classifier.build_template_cache(other_front, other_back)
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = ReturningFastEngine()

    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            replace(base, fast_runtime=runtime),
            np.full((8, 8, 3), 3, dtype=np.uint8),
            library_revision=7,
        )

    assert classifier.fast_engine.calls == 0


def test_fast_template_build_binds_explicit_library_revision(classifier, tmp_path):
    front = [write_marker(tmp_path / "revision-front.png", 1)]
    back = [write_marker(tmp_path / "revision-back.png", 2)]
    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = make_fast_engine()

    cache = classifier.build_template_cache(front, back, library_revision=4)
    result = classifier.predict_with_cache(
        cache,
        np.full((8, 8, 3), 1, dtype=np.uint8),
        library_revision=4,
    )

    assert cache.fast_runtime is not None
    assert cache.fast_runtime.library_revision == 4
    assert result["library_revision"] == 4
    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_REVISION_MISMATCH"):
        classifier.predict_with_cache(
            cache,
            np.full((8, 8, 3), 1, dtype=np.uint8),
            library_revision=1,
        )


def test_fast_cache_build_batches_yield_to_waiting_online_prediction(classifier, tmp_path):
    front = [write_marker(tmp_path / "gate-front.png", 1)]
    back = [write_marker(tmp_path / "gate-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=2,
        model_fingerprint="model-a",
    )
    base = classifier.build_template_cache(front, back)
    first_build_entered = threading.Event()
    release_first_build = threading.Event()
    online_submitted = threading.Event()
    online_entered = threading.Event()
    entries = []
    build_batches = 0

    def ordered_embed(images):
        nonlocal build_batches
        if len(images) == 3:
            entries.append("online")
            online_entered.set()
        else:
            build_batches += 1
            entries.append(f"build-{build_batches}")
            if build_batches == 1:
                first_build_entered.set()
                assert release_first_build.wait(2)
        return [
            np.asarray([float(np.mean(image)), 255.0 - float(np.mean(image))], np.float32)
            for image in images
        ]

    classifier.inference_mode = "fast_geometry"
    classifier.model_fingerprint = "model-a"
    classifier.fast_engine = make_fast_engine(ordered_embed)
    record = SimpleNamespace(
        root=tmp_path,
        front_images=tuple(front),
        back_images=tuple(back),
        revision=2,
        geometry_profile_revision=None,
    )
    errors = []

    def build():
        try:
            classifier.build_fast_runtime_cache(record, None)
        except BaseException as exc:
            errors.append(exc)

    def predict():
        online_submitted.set()
        try:
            classifier.predict_fast_with_cache(
                replace(base, fast_runtime=runtime),
                np.full((8, 8, 3), 1, dtype=np.uint8),
                library_revision=2,
            )
        except BaseException as exc:
            errors.append(exc)

    builder = threading.Thread(target=build)
    builder.start()
    assert first_build_entered.wait(2)
    online = threading.Thread(target=predict)
    online.start()
    assert online_submitted.wait(1)
    time.sleep(0.03)
    online_overlapped_build = online_entered.is_set()
    release_first_build.set()
    for thread in (builder, online):
        thread.join(timeout=3)
        assert not thread.is_alive()

    assert errors == []
    assert online_overlapped_build is False
    assert entries[:3] == ["build-1", "online", "build-2"]


def test_fast_cache_prewrite_failure_cleans_temp_and_preserves_sidecar(
    classifier, tmp_path, monkeypatch
):
    front = [write_marker(tmp_path / "cleanup-front.png", 1)]
    back = [write_marker(tmp_path / "cleanup-back.png", 2)]
    runtime = build_fast_runtime(
        front,
        back,
        library_revision=7,
        model_fingerprint="model-a",
    )
    record = SimpleNamespace(
        root=tmp_path / "record",
        front_images=tuple(front),
        back_images=tuple(back),
        revision=7,
        geometry_profile_revision=None,
    )
    record.root.mkdir()
    classifier.model_fingerprint = "model-a"
    classifier.save_fast_runtime_cache(record, runtime)
    sidecar = record.root / ".fast_runtime_cache.pkl"
    previous = sidecar.read_bytes()

    def fail_signature(*_args, **_kwargs):
        raise RuntimeError("signature failed")

    monkeypatch.setattr(classifier, "_fast_persistence_signature", fail_signature)
    with pytest.raises(RuntimeError, match="signature failed"):
        classifier.save_fast_runtime_cache(record, runtime)

    assert sidecar.read_bytes() == previous
    assert not any(path.name.startswith(".fast-runtime-") for path in record.root.iterdir())


def test_predict_with_cache_uses_supplied_snapshot_not_mutable_map(classifier, tmp_path):
    requested = TemplateCache(
        global_vectors={
            "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
            "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        local_features={"front": [{"marker": 1}], "back": [{"marker": 2}]},
    )
    mapped = TemplateCache(
        global_vectors={
            "front": np.asarray([[0.0, 1.0]], dtype=np.float32),
            "back": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        local_features={"front": [{"marker": 2}], "back": [{"marker": 1}]},
    )
    classifier.set_template_cache("m7", mapped)

    result = classifier.predict_with_cache(
        requested,
        write_marker(tmp_path / "snapshot-query.png", 3),
        library_revision=7,
    )

    assert result["label"] == "front"
    assert result["library_revision"] == 7


class FakeGeometryCalibrator:
    def prepare_context(self, image):
        return {"image_shape": image.shape[:2]}

    def fit(self, image, direction_profile, *, context=None):
        if direction_profile.get("fail_marker") == int(image[0, 0, 0]):
            return {"status": "low_confidence", "reason_code": "boundary_not_found", "rules": []}
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[0, 0] = 255
        rules = direction_profile.get("rules", [])
        return {
            "status": "active",
            "ignore_mask": mask,
            "ignored_ratio": 1.0 / mask.size,
            "rules": [
                {
                    "rule_id": rule.get("rule_id"),
                    "status": "active", "edge_support": 1.0, "visible_ratio": 1.0,
                }
                for rule in rules
            ],
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


def v2_geometry_profile(*, front_r=0.70, back_r=0.64):
    rule = {
        "rule_id": "glare", "name": "中心反光", "shape": "circle",
        "mode": "inside", "margin_ratio": -0.04,
        "margin_semantics": "signed_boundary_v2", "enabled": True,
    }
    anchor = {
        "shape": "circle", "mode": "auto",
        "coarse": {"cx": 0.5, "cy": 0.5, "r": 0.45, "angle_deg": 0.0},
    }

    def direction(side, radius):
        return {
            "anchor": anchor,
            "calibrations": {
                "glare": {
                    "state": "ready",
                    "geometry": {"cx": 0.0, "cy": 0.0, "r": radius, "angle_deg": 0.0},
                    "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": radius + 0.1, "angle_deg": 0.0},
                    "reference_template": {
                        "template_id": f"{side}:00.png", "direction": side,
                        "width": 360, "height": 360,
                    },
                    "diagnostics": {},
                }
            },
            "template_reviews": {},
            "fill_bgr": [3, 3, 3] if side == "front" else [4, 4, 4],
        }

    return {
        "schema_version": 2,
        "profile_revision": 7,
        "rules": [rule],
        "directions": {
            "front": direction("front", front_r),
            "back": direction("back", back_r),
        },
        "migration": {"source_schema_version": None, "conflicts": [], "resolutions": []},
    }


def test_v2_cache_materializes_same_rule_for_unequal_direction_counts(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(5)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(12)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))

    cache, report = classifier.prepare_geometry_cache(
        "m7", record, v2_geometry_profile(), FakeGeometryCalibrator()
    )

    assert cache.global_vectors["front"].shape[0] == 5
    assert cache.global_vectors["back"].shape[0] == 12
    assert {item["rules"][0]["rule_id"] for item in report["front"]} == {"glare"}
    assert {item["rules"][0]["rule_id"] for item in report["back"]} == {"glare"}
    assert cache.geometry_profile["schema_version"] == 2
    assert cache.geometry_profile_revision == 7


def test_geometry_prediction_batches_front_and_back_global_embeddings(classifier, tmp_path, monkeypatch):
    import src.orientation_classifier as classifier_module

    def apply_marker_fill(image, fit, fill_bgr):
        result = image.copy()
        result[0, 0] = np.asarray(fill_bgr, dtype=np.uint8)
        return result

    monkeypatch.setattr(classifier_module, "apply_geometry_fit", apply_marker_fill)
    front = [write_marker(tmp_path / "front.png", 1)]
    back = [write_marker(tmp_path / "back.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    cache, _ = classifier.prepare_geometry_cache(
        "m7", record, v2_geometry_profile(), FakeGeometryCalibrator()
    )
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    classifier.global_predictor.calls = 0
    classifier.global_predictor.markers.clear()
    classifier.global_predictor.batch_sizes.clear()

    result = classifier.predict_with_cache(cache, write_marker(tmp_path / "query.png", 3))

    assert result["geometry_mask"]["status"] == "active"
    assert classifier.global_predictor.calls == 1
    assert classifier.global_predictor.batch_sizes == [2]
    assert classifier.global_predictor.markers == [[3, 4]]
    assert set(result["geometry_mask"]["timings_ms"]) == {
        "fit_context", "fit_directions", "mask_build",
        "global_batch", "local_features", "local_matching", "fusion",
    }
    assert all(value >= 0.0 for value in result["geometry_mask"]["timings_ms"].values())
    assert result["local_search"]["mode"] == "adaptive"


def test_one_direction_fit_failure_uses_single_raw_baseline(classifier, tmp_path):
    class BackFailureCalibrator(FakeGeometryCalibrator):
        def fit(self, image, direction_profile, *, context=None):
            if direction_profile.get("fill_bgr") == [4, 4, 4]:
                return {
                    "status": "low_confidence",
                    "reason_code": "boundary_not_found",
                    "rules": [],
                }
            return super().fit(image, direction_profile, context=context)

    front = [write_marker(tmp_path / "front.png", 1)]
    back = [write_marker(tmp_path / "back.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    cache, _ = classifier.prepare_geometry_cache(
        "m7", record, v2_geometry_profile(), FakeGeometryCalibrator()
    )
    classifier.geometry_calibrator = BackFailureCalibrator()
    classifier.extractor.markers.clear()

    result = classifier.predict_with_cache(cache, write_marker(tmp_path / "query.png", 3))

    assert result["geometry_mask"]["status"] == "low_confidence"
    assert result["geometry_mask"]["needs_review"] is True
    assert result["geometry_mask"]["fallback"] == "raw_baseline"
    assert result["geometry_mask"]["reason_code"] == "GEOMETRY_FALLBACK_TO_BASELINE"
    assert result["geometry_mask"]["fallback_reason"] == "boundary_not_found"
    assert result["local_search"]["available_counts"] == {
        label: len(cache.raw_local_features[label]) for label in ("front", "back")
    }
    assert classifier.extractor.markers == [3]


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
    before_query_local_calls = classifier.extractor.calls
    result = classifier.predict("m7", write_marker(tmp_path / "query.png", 3))

    assert candidate.geometry_profile_revision == 3
    assert report["front"][0]["status"] == "active"
    assert report["back"][0]["status"] == "active"
    assert classifier.global_predictor.markers[-1] == [3, 3]
    assert classifier.extractor.calls == before_query_local_calls + 1
    assert classifier.extractor.markers[-1] == 3
    assert result["geometry_mask"]["status"] == "active"
    assert result["geometry_mask"]["profile_revision"] == 3


def test_geometry_cache_excludes_template_only_from_candidate_side(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(2)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    profile = geometry_profile()
    profile["directions"]["front"]["template_reviews"] = {
        "front:front-1.png": {"state": "excluded", "reason": "轮廓被遮挡"}
    }

    candidate, report = classifier.prepare_geometry_cache(
        "m7", record, profile, FakeGeometryCalibrator(), None
    )

    assert candidate.global_vectors["front"].shape[0] == len(record.front_images) - 1
    assert candidate.raw_global_vectors["front"].shape[0] == len(record.front_images)
    assert report["front"][1]["review_state"] == "excluded"


def test_geometry_cache_rejects_a_side_with_no_included_templates(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(2)]
    back = [write_marker(tmp_path / "back-0.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    profile = geometry_profile()
    profile["directions"]["front"]["template_reviews"] = {
        f"front:front-{index}.png": {"state": "excluded", "reason": "不可见"}
        for index in range(2)
    }

    with pytest.raises(ValueError, match="front has no included templates"):
        classifier.prepare_geometry_cache("m7", record, profile, FakeGeometryCalibrator(), None)


def test_unsafe_geometry_cache_falls_back_to_raw_baseline(classifier, tmp_path):
    class UnsafeGeometryCalibrator(FakeGeometryCalibrator):
        def fit(self, image, direction_profile):
            result = super().fit(image, direction_profile)
            result["ignored_ratio"] = 0.60
            return result

    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(2)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))

    candidate, _ = classifier.prepare_geometry_cache(
        "m7", record, geometry_profile(), UnsafeGeometryCalibrator(), None
    )
    assert candidate.geometry_unsafe is True
    classifier.geometry_calibrator = UnsafeGeometryCalibrator()
    classifier.set_template_cache("m7", candidate)

    result = classifier.predict("m7", write_marker(tmp_path / "query.png", 3))

    assert result["geometry_mask"]["status"] == "unsafe_template_geometry"
    assert result["needs_review"] is True


def test_leave_one_out_reextracts_query_features(classifier, tmp_path):
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
    assert classifier.global_predictor.calls == before_global + 4
    assert classifier.extractor.calls == before_local + 4


def test_paired_leave_one_out_counts_only_baseline_correct_to_candidate_wrong(
    classifier, tmp_path, monkeypatch
):
    front = [
        write_marker(tmp_path / "front-preexisting-error.png", 4),
        write_marker(tmp_path / "front-regression.png", 7),
        write_marker(tmp_path / "front-correct.png", 1),
    ]
    back = [
        write_marker(tmp_path / "back-0.png", 2),
        write_marker(tmp_path / "back-1.png", 2),
    ]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    cache = replace(classifier.build_template_cache(front, back), geometry_profile={"directions": {}})

    def prediction(label, *, global_label=None, local_label=None, source="global", geometry=False):
        result = {
            "label": label,
            "global_prediction": global_label or label,
            "local_prediction": local_label or label,
            "global_margin": 0.02,
            "local_margin": 3.9,
            "decision_source": source,
            "needs_review": source != "global",
        }
        if geometry:
            result["geometry_mask"] = {"status": "active"}
        return result

    def baseline(image, *_args, **_kwargs):
        marker = int(image[0, 0, 0])
        return prediction({4: "back", 7: "front", 1: "front", 2: "back"}[marker])

    def candidate(image, *_args, **_kwargs):
        marker = int(image[0, 0, 0])
        if marker == 7:
            return prediction(
                "back", global_label="front", local_label="back",
                source="local_override", geometry=True,
            )
        return prediction(
            {4: "back", 1: "front", 2: "back"}[marker], geometry=True
        )

    monkeypatch.setattr(classifier, "_predict_baseline", baseline)
    monkeypatch.setattr(classifier, "_predict_geometry", candidate)

    report = classifier.leave_one_out_report(record, cache)

    assert report["correct_to_wrong"] == 1
    assert report["wrong_to_wrong"] == 1
    assert report["correct_to_correct"] == 3
    assert report["wrong_to_correct"] == 0
    assert report["changed_predictions"] == [{
        "template_id": "front:front-regression.png",
        "expected": "front",
        "predicted": "back",
        "baseline_predicted": "front",
        "candidate_predicted": "back",
        "baseline_global_prediction": "front",
        "baseline_local_prediction": "front",
        "candidate_global_prediction": "front",
        "candidate_local_prediction": "back",
        "candidate_decision_source": "local_override",
        "candidate_global_margin": 0.02,
        "candidate_local_margin": 3.9,
        "cause": "geometry_local_override",
    }]


def test_geometry_local_features_use_raw_image_and_filter_complete_ignore_mask(
    classifier, tmp_path
):
    def features_with_keypoints(image, extractor, _device, roi_ratio=1.0):
        extractor.calls += 1
        marker = int(image[0, 0, 0])
        extractor.markers.append(marker)
        return {
            "marker": marker,
            "keypoints": np.asarray([[[0.0, 0.0], [4.0, 4.0]]], dtype=np.float32),
            "descriptors": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        }

    classifier._extract_features = features_with_keypoints
    front = [write_marker(tmp_path / "front.png", 1)]
    back = [write_marker(tmp_path / "back.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))

    candidate, report = classifier.prepare_geometry_cache(
        "m7", record, geometry_profile(), FakeGeometryCalibrator()
    )

    assert candidate.local_features["front"][0]["marker"] == 1
    assert candidate.local_features["back"][0]["marker"] == 2
    assert classifier._feature_keypoint_count(candidate.local_features["front"][0]) == 1
    assert classifier._feature_keypoint_count(candidate.local_features["back"][0]) == 1
    assert report["front"][0]["feature_mask_mode"] == "all_ignored_regions"
    assert report["front"][0]["keypoints_before"] == 2
    assert report["front"][0]["keypoints_after"] == 1


def test_geometry_query_extracts_raw_local_features_once_then_filters_each_direction(
    classifier, tmp_path
):
    def features_with_keypoints(image, extractor, _device, roi_ratio=1.0):
        extractor.calls += 1
        marker = int(image[0, 0, 0])
        extractor.markers.append(marker)
        return {
            "marker": marker,
            "keypoints": np.asarray([[[0.0, 0.0], [4.0, 4.0]]], dtype=np.float32),
            "descriptors": np.asarray([[[1.0], [2.0]]], dtype=np.float32),
        }

    classifier._extract_features = features_with_keypoints
    front = [write_marker(tmp_path / "front.png", 1)]
    back = [write_marker(tmp_path / "back.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    candidate, _ = classifier.prepare_geometry_cache(
        "m7", record, geometry_profile(), FakeGeometryCalibrator()
    )
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    classifier.extractor.markers.clear()

    classifier.predict_with_cache(candidate, write_marker(tmp_path / "query.png", 3))

    assert classifier.extractor.markers == [3]


def test_leave_one_out_does_not_block_on_explicitly_excluded_templates(classifier, tmp_path):
    front = [write_marker(tmp_path / f"front-{index}.png", 1) for index in range(3)]
    back = [write_marker(tmp_path / f"back-{index}.png", 2) for index in range(3)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    cache = classifier.build_template_cache(front, back)
    cache = replace(
        cache,
        geometry_template_indices={"front": [1, 2], "back": [0, 1, 2]},
        geometry_template_report={
            "front": [
                {"index": 0, "template_id": "front:front-0.png", "status": "excluded",
                 "review_state": "excluded"},
                {"index": 1, "template_id": "front:front-1.png", "status": "active"},
                {"index": 2, "template_id": "front:front-2.png", "status": "active"},
            ],
            "back": [],
        },
    )
    classifier._score_feature_pair = lambda *args: {"score": 1.0}

    report = classifier.leave_one_out_report(record, cache)

    assert report["status"] == "completed"
    assert report["skipped"] == 0
    assert report["excluded"] == 1
    assert report["excluded_templates"] == ["front:front-0.png"]


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
    assert classifier.global_predictor.markers[-1] == [3]
    assert classifier.extractor.markers[-1] == 3


def test_needs_reseed_direction_uses_raw_cache_until_preview_succeeds(classifier, monkeypatch):
    import src.orientation_classifier as classifier_module

    monkeypatch.setattr(
        classifier_module,
        "_read_image",
        lambda path: np.full(
            (8, 8, 3),
            1 if "front" in str(path) else (2 if "back" in str(path) else 3),
            dtype=np.uint8,
        ),
    )
    front = [Path(f"front-{index}.png") for index in range(2)]
    back = [Path(f"back-{index}.png") for index in range(2)]
    base = classifier.build_template_cache(front, back)
    classifier.set_template_cache("m7", base)
    profile = geometry_profile()
    profile["directions"]["front"]["rules"][0].update({
        "enabled": False,
        "editor_state": "needs_reseed",
    })

    candidate, report = classifier.prepare_geometry_cache(
        "m7", SimpleNamespace(front_images=tuple(front), back_images=tuple(back)),
        profile, FakeGeometryCalibrator(), None
    )

    np.testing.assert_array_equal(candidate.global_vectors["front"], base.global_vectors["front"])
    assert candidate.local_features["front"] is base.local_features["front"]
    assert all(item["status"] == "needs_reseed" for item in report["front"])
    assert report["back"][0]["status"] == "active"
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    classifier.set_template_cache("m7", candidate)
    result = classifier.predict("m7", Path("query.png"))
    assert result["geometry_mask"]["status"] == "needs_reseed"
    assert result["geometry_mask"]["needs_review"] is True
    assert result["needs_review"] is True


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
    classifier = OrientationClassifier(
        global_predictor=FakeGlobalPredictor(),
        extractor=FakeExtractor(),
        matcher=FakeMatcher(),
        device="cpu",
        extract_features_fn=fake_extract_features,
        score_feature_pair_fn=fake_score_feature_pair,
        local_search_mode="exhaustive",
    )
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


def test_baseline_prediction_exposes_adaptive_search_diagnostics(classifier, tmp_path):
    calls = []
    cache = TemplateCache(
        global_vectors={
            "front": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (12, 1)),
            "back": np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (13, 1)),
        },
        local_features={
            "front": [
                {"label": "front", "index": index, "score": 5.0}
                for index in range(12)
            ],
            "back": [
                {"label": "back", "index": index, "score": 8.0}
                for index in range(13)
            ],
        },
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )

    result = classifier.predict_with_cache(
        cache, write_marker(tmp_path / "adaptive-query.png", 3)
    )

    assert result["label"] == "back"
    assert result["needs_review"] is False
    assert result["local_search"]["stage"] == "top5"
    assert result["local_search"]["matched_counts"] == {"front": 5, "back": 5}
    assert len(calls) == 10


def test_baseline_exhaustive_search_matches_explicit_full_fusion(tmp_path):
    classifier = OrientationClassifier(
        global_predictor=FakeGlobalPredictor(),
        extractor=FakeExtractor(),
        matcher=FakeMatcher(),
        device="cpu",
        extract_features_fn=fake_extract_features,
        score_feature_pair_fn=fake_score_feature_pair,
        local_search_mode="exhaustive",
    )
    calls = []
    front_scores = [5.0] * 11 + [9.5]
    back_scores = [8.0] * 12 + [9.0]
    cache = TemplateCache(
        global_vectors={
            "front": np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (12, 1)),
            "back": np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (13, 1)),
        },
        local_features={
            "front": [
                {"label": "front", "index": index, "score": score}
                for index, score in enumerate(front_scores)
            ],
            "back": [
                {"label": "back", "index": index, "score": score}
                for index, score in enumerate(back_scores)
            ],
        },
    )
    classifier._score_feature_pair = lambda query, template, shape, matcher: (
        calls.append((template["label"], template["index"]))
        or {"score": template["score"]}
    )
    global_scores = {
        "front": float(np.float32(0.9)),
        "back": float(np.float32(0.91)),
    }
    expected = classifier._fuse_scores(
        global_scores,
        {"front": max(front_scores), "back": max(back_scores)},
        0.0,
    )

    result = classifier.predict_with_cache(
        cache, write_marker(tmp_path / "exhaustive-query.png", 3)
    )

    assert len(calls) == 25
    assert result["local_search"]["stage"] == "full"
    for key in (
        "label", "needs_review", "global_scores", "local_scores",
        "global_margin", "local_margin", "decision_source",
    ):
        assert result[key] == expected[key]


@pytest.mark.parametrize("geometry", [False, True])
@pytest.mark.parametrize("label", ["front", "back"])
@pytest.mark.parametrize(
    "corruption",
    ["missing_globals", "missing_locals", "empty_globals", "empty_locals", "mismatch"],
)
def test_predict_with_cache_rejects_corrupt_direction_before_scoring(
    classifier, tmp_path, geometry, label, corruption
):
    global_vectors = {
        "front": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "back": np.asarray([[0.0, 1.0]], dtype=np.float32),
    }
    local_features = {"front": [{"marker": 1}], "back": [{"marker": 2}]}
    if corruption == "missing_globals":
        global_vectors.pop(label)
    elif corruption == "missing_locals":
        local_features.pop(label)
    elif corruption == "empty_globals":
        global_vectors[label] = np.empty((0, 2), dtype=np.float32)
    elif corruption == "empty_locals":
        local_features[label] = []
    else:
        global_vectors[label] = np.repeat(global_vectors[label], 2, axis=0)
    cache = TemplateCache(
        global_vectors=global_vectors,
        local_features=local_features,
        geometry_profile={"directions": {}} if geometry else None,
    )
    classifier.geometry_calibrator = FakeGeometryCalibrator()

    with pytest.raises(
        OrientationClassifierError,
        match=f"template cache alignment error for {label}",
    ):
        classifier.predict_with_cache(
            cache, write_marker(tmp_path / f"corrupt-{geometry}-{label}-{corruption}.png", 3)
        )


def test_geometry_timings_exclude_matching_from_local_features(
    classifier, tmp_path, monkeypatch
):
    import src.orientation_classifier as classifier_module

    front = [write_marker(tmp_path / "timing-front.png", 1)]
    back = [write_marker(tmp_path / "timing-back.png", 2)]
    record = SimpleNamespace(front_images=tuple(front), back_images=tuple(back))
    classifier.set_template_cache("m7", classifier.build_template_cache(front, back))
    cache, _ = classifier.prepare_geometry_cache(
        "m7", record, v2_geometry_profile(), FakeGeometryCalibrator()
    )
    classifier.geometry_calibrator = FakeGeometryCalibrator()
    now = [100.0]
    extraction_calls = []
    original_extract_local = classifier._extract_local

    def timed_extract_local(image):
        extraction_calls.append(int(image[0, 0, 0]))
        result = original_extract_local(image)
        now[0] += 0.010
        return result

    def timed_search(**_kwargs):
        now[0] += 0.100
        return LocalSearchResult(
            scores={"front": 5.0, "back": 8.0},
            diagnostics={"mode": "adaptive"},
            matching_ms=37.0,
            trace={},
        )

    monkeypatch.setattr(classifier_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setattr(classifier, "_extract_local", timed_extract_local)
    monkeypatch.setattr(classifier, "_search_local", timed_search)

    result = classifier.predict_with_cache(
        cache, write_marker(tmp_path / "timing-query.png", 3)
    )
    timings = result["geometry_mask"]["timings_ms"]

    assert extraction_calls == [3]
    assert timings["local_features"] == pytest.approx(10.0)
    assert timings["local_matching"] == pytest.approx(37.0)


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
