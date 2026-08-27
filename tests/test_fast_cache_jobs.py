from dataclasses import replace
import gc
import json
from pathlib import Path
import threading
import weakref

import cv2
import numpy as np
import pytest

from src.fast_cache_jobs import FastCacheJobManager
from src.fast_geometry import FastGeometryProcessor
from src.fast_orientation import FastOrientationEngine
from src.geometry_mask_profiles import GeometryMaskProfiles
from src.orientation_classifier import OrientationClassifier, TemplateCache, _file_sha256
from src.workpiece_catalog import WorkpieceCatalog
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


def test_job_transitions_queued_running_ready_and_publishes_once():
    manager = FastCacheJobManager()
    started = threading.Event()
    release = threading.Event()
    published = []

    def build(progress):
        started.set()
        assert release.wait(2.0)
        progress({"phase": "fast_ridge", "completed": 3, "total": 3})
        return "cache"

    queued = manager.schedule(
        workpiece_id="m7",
        library_revision=4,
        geometry_profile_revision=2,
        build=build,
        publish=lambda cache: published.append(cache) is None,
    )

    assert queued.state == "queued"
    assert started.wait(1.0)
    assert manager.snapshot("m7").state == "running"
    release.set()
    manager.shutdown()

    ready = manager.snapshot("m7")
    assert ready.state == "ready"
    assert (ready.completed, ready.total) == (3, 3)
    assert published == ["cache"]


def test_duplicate_schedule_for_same_revision_returns_same_job():
    manager = FastCacheJobManager()
    started = threading.Event()
    release = threading.Event()
    calls = {"build": 0, "publish": 0}

    def build(_progress):
        calls["build"] += 1
        started.set()
        assert release.wait(2.0)
        return "cache"

    first = manager.schedule(
        workpiece_id="m7", library_revision=7, geometry_profile_revision=None,
        build=build, publish=lambda _cache: calls.__setitem__("publish", calls["publish"] + 1) or True,
    )
    second = manager.schedule(
        workpiece_id="m7", library_revision=7, geometry_profile_revision=None,
        build=build, publish=lambda _cache: pytest.fail("duplicate publish callback used"),
    )

    assert (
        first.workpiece_id,
        first.library_revision,
        first.geometry_profile_revision,
    ) == (
        second.workpiece_id,
        second.library_revision,
        second.geometry_profile_revision,
    )
    assert second.state in {"queued", "running"}
    assert started.wait(1.0)
    release.set()
    manager.shutdown()
    assert calls == {"build": 1, "publish": 1}


def test_stale_result_is_discarded_when_library_revision_changes(tmp_path):
    manager = FastCacheJobManager()
    started = threading.Event()
    release = threading.Event()
    active_revision = {"value": 1}
    sidecar = tmp_path / ".fast_runtime_cache.pkl"

    def build(_progress):
        started.set()
        assert release.wait(2.0)
        return b"cache"

    def publish(cache):
        if active_revision["value"] != 1:
            return False
        sidecar.write_bytes(cache)
        return True

    manager.schedule(
        workpiece_id="m7", library_revision=1, geometry_profile_revision=None,
        build=build, publish=publish,
    )
    assert started.wait(1.0)
    active_revision["value"] = 2
    release.set()
    manager.shutdown()

    assert manager.snapshot("m7").state == "stale"
    assert not sidecar.exists()


def test_failure_preserves_previous_ready_cache_and_exposes_error():
    manager = FastCacheJobManager()
    active = {"cache": "old-ready"}
    started = threading.Event()
    release = threading.Event()

    def fail(_progress):
        started.set()
        assert release.wait(2.0)
        raise RuntimeError("build failed")

    manager.schedule(
        workpiece_id="m7", library_revision=2, geometry_profile_revision=None,
        build=fail,
        publish=lambda cache: active.__setitem__("cache", cache) or True,
    )
    assert started.wait(1.0)
    release.set()
    manager.shutdown()

    snapshot = manager.snapshot("m7")
    assert active["cache"] == "old-ready"
    assert snapshot.state == "failed"
    assert snapshot.error == "build failed"


def test_shutdown_waits_for_current_step_without_starting_more_jobs():
    manager = FastCacheJobManager()
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    stopping_observed = threading.Event()
    shutdown_done = threading.Event()

    def first(_progress):
        first_started.set()
        assert first_release.wait(2.0)
        return "first"

    manager.schedule(
        workpiece_id="first", library_revision=1, geometry_profile_revision=None,
        build=first, publish=lambda _cache: True,
    )
    manager.schedule(
        workpiece_id="second", library_revision=1, geometry_profile_revision=None,
        build=lambda _progress: second_started.set() or "second",
        publish=lambda _cache: True,
    )
    assert first_started.wait(1.0)

    def observe_stopping():
        with manager._condition:
            manager._condition.wait_for(lambda: manager._stopping)
            stopping_observed.set()

    observer = threading.Thread(target=observe_stopping)
    observer.start()
    stopper = threading.Thread(target=lambda: (manager.shutdown(), shutdown_done.set()))
    stopper.start()
    try:
        assert stopping_observed.wait(1.0)
        assert not shutdown_done.is_set()
    finally:
        first_release.set()
        with manager._condition:
            manager._condition.notify_all()
        stopper.join(timeout=2.0)
        observer.join(timeout=2.0)

    assert shutdown_done.is_set()
    assert not second_started.is_set()
    assert manager.snapshot("first").state == "ready"
    assert manager.snapshot("second").state == "queued"


def test_superseded_terminal_jobs_release_callback_captures():
    manager = FastCacheJobManager()
    releases = [threading.Event() for _ in range(3)]
    starts = [threading.Event() for _ in range(3)]
    references = []

    class Capture:
        pass

    def make_build(index):
        capture = Capture()
        references.append(weakref.ref(capture))

        def build(_progress, retained=capture):
            assert retained is not None
            starts[index].set()
            assert releases[index].wait(2.0)
            return index

        return build

    builds = [make_build(index) for index in range(3)]
    for revision, build in enumerate(builds, start=1):
        manager.schedule(
            workpiece_id="m7",
            library_revision=revision,
            geometry_profile_revision=None,
            build=build,
            publish=lambda _cache: True,
        )
    del builds
    del build

    try:
        assert starts[0].wait(1.0)
        releases[0].set()
        assert starts[1].wait(1.0)
        gc.collect()
        assert references[0]() is None
        releases[1].set()
        assert starts[2].wait(1.0)
        gc.collect()
        assert references[1]() is None
    finally:
        for release in releases:
            release.set()
        manager.shutdown()
    gc.collect()
    assert references[2]() is None


def test_terminal_job_retention_is_bounded_across_repeated_revisions():
    manager = FastCacheJobManager()
    published = []
    all_published = threading.Event()

    def publish(value):
        published.append(value)
        if value == 25:
            all_published.set()
        return True

    for revision in range(1, 26):
        manager.schedule(
            workpiece_id="m7",
            library_revision=revision,
            geometry_profile_revision=None,
            build=lambda _progress, value=revision: value,
            publish=publish,
        )

    assert all_published.wait(2.0)
    manager.shutdown()

    assert published == list(range(1, 26))
    assert len(manager._jobs) == 1
    latest = manager.snapshot("m7")
    assert latest is not None
    assert latest.library_revision == 25
    assert latest.state == "ready"
    duplicate = manager.schedule(
        workpiece_id="m7",
        library_revision=25,
        geometry_profile_revision=None,
        build=lambda _progress: pytest.fail("latest terminal revision rebuilt"),
        publish=lambda _cache: pytest.fail("latest terminal revision republished"),
    )
    assert duplicate == latest


class BlockingFastClassifier:
    inference_mode = "fast_geometry"

    def __init__(self):
        self.caches = {}
        self.started = threading.Event()
        self.release = threading.Event()
        self.saved = []

    def build_template_cache(self, front, back, progress_callback=None, *, library_revision=1):
        return builder(front, back, progress_callback)

    def build_fast_runtime_cache(self, record, geometry_profile, progress_callback=None):
        self.started.set()
        assert self.release.wait(2.0)
        if progress_callback is not None:
            progress_callback({"phase": "fast_ridge", "completed": 2, "total": 2})
        return {"library_revision": record.revision, "geometry": geometry_profile}

    def save_fast_runtime_cache(self, record, runtime):
        (record.root / ".fast_runtime_cache.pkl").write_bytes(b"ready")
        self.saved.append((record.id, runtime))

    @staticmethod
    def stage_fast_runtime_cache(record, runtime, *, geometry_profile_revision=None):
        return record, runtime, geometry_profile_revision

    def commit_staged_fast_runtime_cache(self, record, staged):
        _captured_record, runtime, _geometry_revision = staged
        self.save_fast_runtime_cache(record, runtime)
        return staged

    @staticmethod
    def finalize_staged_fast_runtime_cache(_committed):
        return None

    @staticmethod
    def discard_staged_fast_runtime_cache(_staged):
        return None

    @staticmethod
    def rollback_committed_fast_runtime_cache(_committed):
        return None

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def remove_template_cache(self, workpiece_id):
        self.caches.pop(workpiece_id, None)

    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        if cache.fast_runtime is None:
            raise RuntimeError("FAST_CACHE_NOT_READY: fast runtime cache is unavailable")
        return {"label": "front", "library_revision": library_revision}


class BlockingRestoreClassifier(BlockingFastClassifier):
    def __init__(self):
        super().__init__()
        self.base_build_started = threading.Event()
        self.base_build_release = threading.Event()

    def build_template_cache(self, front, back, progress_callback=None, *, library_revision=1):
        self.base_build_started.set()
        assert self.base_build_release.wait(2.0)
        return builder(front, back, progress_callback)


class BlockingPersistenceClassifier(BlockingFastClassifier):
    def __init__(self):
        super().__init__()
        self.persistence_started = threading.Event()
        self.persistence_release = threading.Event()

    def _block_persistence(self):
        self.persistence_started.set()
        assert self.persistence_release.wait(2.0)

    def save_fast_runtime_cache(self, record, runtime):
        self._block_persistence()
        super().save_fast_runtime_cache(record, runtime)

    def stage_fast_runtime_cache(self, record, runtime, *, geometry_profile_revision=None):
        self._block_persistence()
        return record, runtime, geometry_profile_revision

    def commit_staged_fast_runtime_cache(self, record, staged):
        _captured_record, runtime, _geometry_revision = staged
        super().save_fast_runtime_cache(record, runtime)
        return staged


class ProductionFastPredictor:
    @staticmethod
    def predict(images):
        return [
            np.asarray(
                [float(np.mean(item)) / 255.0, 1.0 - float(np.mean(item)) / 255.0],
                dtype=np.float32,
            )
            for item in images
        ]


def production_fast_classifier():
    classifier = OrientationClassifier(
        global_predictor=ProductionFastPredictor(),
        extractor=None,
        matcher=None,
        device="cpu",
        inference_mode="fast_geometry",
        model_fingerprint="catalog-model-a",
    )
    classifier.fast_engine = FastOrientationEngine(
        classifier._global_embeddings,
        FastGeometryProcessor(object()),
        image_reader=lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR),
    )
    return classifier

def _persist_base_library(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, cache = library.register(
        "M7",
        [image(tmp_path / "front.png", 10)],
        [image(tmp_path / "back.png", 20)],
        False,
        builder,
    )

    return record, cache


def _write_geometry_recovery_state(
    record,
    *,
    manifest_active_revision,
    document_active_revision,
    manifest_pointer_present=True,
):
    profile = {
        "schema_version": 1,
        "directions": {
            "front": {"anchor": None, "rules": []},
            "back": {"anchor": None, "rules": []},
        },
    }
    profile_root = record.root / "geometry_masks"
    revisions_root = profile_root / "revisions"
    revisions_root.mkdir(parents=True, exist_ok=True)
    for revision in {value for value in (manifest_active_revision, document_active_revision) if value}:
        (revisions_root / f"{revision}.json").write_text(
            json.dumps({
                "profile": profile,
                "previous_active_revision": None,
            }),
            encoding="utf-8",
        )
    (profile_root / "profile.json").write_text(
        json.dumps({
            "schema_version": 1,
            "library_revision": record.revision,
            "draft_revision": max(manifest_active_revision or 0, document_active_revision or 0),
            "active_revision": document_active_revision,
            "previous_active_revision": None,
            "draft": profile,
            "active": profile if document_active_revision is not None else None,
        }),
        encoding="utf-8",
    )
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_pointer_present:
        manifest["geometry_mask_active_revision"] = manifest_active_revision
        manifest["geometry_mask_previous_active_revision"] = None
    else:
        manifest.pop("geometry_mask_active_revision", None)
        manifest.pop("geometry_mask_previous_active_revision", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_recover_does_not_resurrect_profile_when_manifest_pointer_is_explicit_null(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    _write_geometry_recovery_state(
        record,
        manifest_active_revision=None,
        document_active_revision=1,
    )
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "geometry-recovery-jobs",
    )
    catalog.set_geometry_profiles(profiles)

    catalog.recover()
    try:
        assert classifier.started.wait(1.0)
        recovering = catalog.capture_snapshot(record.id)
        assert recovering.cache.geometry_profile is None
        assert recovering.cache.geometry_profile_revision is None
        assert catalog.fast_cache_status(record.id)["state"] == "running"
        summary = catalog.list_workpiece_summaries()[0]
        assert summary["geometry_rule_count"] == 0
        assert summary["fast_cache"]["state"] == "running"
    finally:
        classifier.release.set()
        catalog.shutdown()
        profiles.shutdown()

    ready = catalog.capture_snapshot(record.id)
    assert ready.cache.geometry_profile_revision is None
    assert ready.cache.fast_runtime["geometry"] is None
    assert classifier.saved[-1][1]["geometry"] is None
    assert (record.root / ".fast_runtime_cache.pkl").read_bytes() == b"ready"
    assert profiles.snapshot(record.id)["active_revision"] is None
    assert catalog.list_workpiece_summaries()[0]["fast_cache"]["state"] == "ready"


def test_recover_uses_manifest_immutable_revision_when_profile_document_is_ahead(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    _write_geometry_recovery_state(
        record,
        manifest_active_revision=1,
        document_active_revision=2,
    )
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "geometry-recovery-jobs",
    )
    catalog.set_geometry_profiles(profiles)

    catalog.recover()
    try:
        assert classifier.started.wait(1.0)
        recovering = catalog.capture_snapshot(record.id)
        assert recovering.cache.geometry_profile_revision == 1
        assert recovering.cache.geometry_profile["profile_revision"] == 1
        job = catalog.fast_jobs.snapshot(record.id)
        assert job is not None
        assert job.geometry_profile_revision == 1
        assert catalog.list_workpiece_summaries()[0]["fast_cache"]["state"] == "running"
    finally:
        classifier.release.set()
        catalog.shutdown()
        profiles.shutdown()

    ready = catalog.capture_snapshot(record.id)
    assert ready.cache.geometry_profile_revision == 1
    assert ready.cache.fast_runtime["geometry"]["profile_revision"] == 1
    assert classifier.saved[-1][1]["geometry"]["profile_revision"] == 1
    assert (record.root / ".fast_runtime_cache.pkl").read_bytes() == b"ready"
    assert profiles.snapshot(record.id)["active_revision"] == 1
    assert catalog.list_workpiece_summaries()[0]["fast_cache"]["state"] == "ready"


def test_recover_preserves_legacy_profile_identity_when_manifest_pointer_is_missing(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    _write_geometry_recovery_state(
        record,
        manifest_active_revision=1,
        document_active_revision=1,
        manifest_pointer_present=False,
    )
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "legacy-geometry-recovery-jobs",
    )
    catalog.set_geometry_profiles(profiles)

    catalog.recover()
    try:
        assert classifier.started.wait(1.0)
        snapshot = catalog.capture_snapshot(record.id)
        assert snapshot.cache.geometry_profile_revision == 1
        assert catalog.fast_jobs.snapshot(record.id).geometry_profile_revision == 1
    finally:
        classifier.release.set()
        catalog.shutdown()
        profiles.shutdown()

    assert catalog.capture_snapshot(record.id).cache.fast_runtime["geometry"]["profile_revision"] == 1
    assert profiles.snapshot(record.id)["active_revision"] == 1


def test_legacy_recover_does_not_resurrect_explicitly_cleared_manifest_profile(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    _write_geometry_recovery_state(
        record,
        manifest_active_revision=None,
        document_active_revision=1,
    )

    class LegacyRecoveryClassifier(BlockingFastClassifier):
        inference_mode = "legacy"

        def __init__(self):
            super().__init__()
            self.geometry_prepares = []

        def prepare_geometry_cache(
            self,
            workpiece_id,
            _record,
            profile,
            _calibrator=None,
            _progress_callback=None,
        ):
            self.geometry_prepares.append(profile)
            current = self.caches[workpiece_id]
            return replace(
                current,
                geometry_profile=profile,
                geometry_profile_revision=profile.get("profile_revision"),
                ignored_regions={"front": [[{"x": 0.1}]], "back": []},
            ), {"front": [], "back": []}

    classifier = LegacyRecoveryClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "legacy-null-recovery-jobs",
    )
    catalog.set_geometry_profiles(profiles)

    catalog.recover()

    snapshot = catalog.capture_snapshot(record.id)
    manifest = json.loads((snapshot.record.root / "manifest.json").read_text(encoding="utf-8"))
    profile_snapshot = profiles.snapshot(record.id)
    summary = catalog.list_workpiece_summaries()[0]
    assert classifier.geometry_prepares == []
    assert snapshot.cache is base_cache
    assert snapshot.cache.geometry_profile is None
    assert snapshot.cache.geometry_profile_revision is None
    assert snapshot.cache.ignored_regions is None
    assert catalog.get(record.id) == snapshot.record
    assert manifest["geometry_mask_active_revision"] is None
    assert profile_snapshot["active_revision"] is None
    assert profile_snapshot["active"] is None
    assert profile_snapshot["library_revision"] == snapshot.record.revision
    assert summary["revision"] == snapshot.record.revision
    assert summary["geometry_status"] == profile_snapshot["profile_status"]
    assert summary["geometry_rule_count"] == 0
    profiles.shutdown()
    catalog.shutdown()


def test_restore_uses_valid_base_without_waiting_for_base_or_fast_builder(tmp_path):
    classifier = BlockingRestoreClassifier()
    classifier.inference_mode = "legacy"
    classifier.base_build_release.set()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base_cache = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    classifier.base_build_started.clear()
    classifier.base_build_release.clear()
    classifier.inference_mode = "fast_geometry"
    catalog.recycle(record.id, operation_id="restore-async-recycle")
    classifier.load_template_cache = lambda _record: base_cache
    restored = []
    restore_done = threading.Event()

    def run_restore():
        restored.append(catalog.restore(record.id, operation_id="restore-async"))
        restore_done.set()

    worker = threading.Thread(target=run_restore)
    worker.start()
    try:
        assert restore_done.wait(1.0)
        assert not classifier.base_build_started.is_set()
        assert classifier.started.wait(1.0)
        snapshot = catalog.capture_snapshot(record.id)
        assert snapshot.record == restored[0]
        assert snapshot.cache.global_vectors is base_cache.global_vectors
        assert snapshot.cache.local_features is base_cache.local_features
        assert snapshot.cache.fast_runtime is None
    finally:
        classifier.base_build_release.set()
        classifier.release.set()
        worker.join(timeout=2.0)
        catalog.shutdown()


@pytest.mark.parametrize("inference_mode", ["fast_geometry", "compare"])
def test_restore_materializes_published_profile_without_sync_geometry_rebuild(
    tmp_path,
    inference_mode,
):
    classifier = BlockingFastClassifier()
    classifier.inference_mode = "legacy"
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base_cache = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    catalog.recycle(record.id, operation_id=f"profile-recycle-{inference_mode}")
    classifier.inference_mode = inference_mode
    classifier.load_template_cache = lambda _record: base_cache

    class PublishedProfiles:
        def __init__(self):
            self.rebuild_started = threading.Event()
            self.rebuild_release = threading.Event()

        def sync_library_revision(self, _record):
            return None

        def snapshot(self, _workpiece_id):
            return {
                "active_revision": 8,
                "active": {"profile_revision": 8, "rules": []},
            }

        def rebuild_active_cache(self, *_args, **_kwargs):
            self.rebuild_started.set()
            assert self.rebuild_release.wait(2.0)
            return None

    profiles = PublishedProfiles()
    catalog.set_geometry_profiles(profiles)
    restored = []
    restore_done = threading.Event()
    worker = threading.Thread(
        target=lambda: (restored.append(catalog.restore(
            record.id,
            operation_id=f"profile-restore-{inference_mode}",
        )), restore_done.set())
    )
    worker.start()
    try:
        assert restore_done.wait(1.0)
        assert not profiles.rebuild_started.is_set()
        assert classifier.started.wait(1.0)
        snapshot = catalog.capture_snapshot(record.id)
        assert snapshot.record == restored[0]
        assert snapshot.cache.geometry_profile_revision == 8
        assert snapshot.cache.fast_runtime is None
    finally:
        profiles.rebuild_release.set()
        classifier.release.set()
        worker.join(timeout=2.0)
        catalog.shutdown()


def test_blocked_fast_persistence_does_not_hold_catalog_lock(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingPersistenceClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    catalog.recover()
    assert classifier.started.wait(1.0)
    classifier.release.set()
    assert classifier.persistence_started.wait(1.0)
    captured = []
    prediction_errors = []
    reader_done = threading.Event()

    def read_while_persistence_is_blocked():
        captured.append(catalog.capture_snapshot(record.id))
        try:
            catalog.predict(record.id, tmp_path / "query.png")
        except Exception as exc:
            prediction_errors.append(exc)
        reader_done.set()

    reader = threading.Thread(target=read_while_persistence_is_blocked)
    reader.start()
    try:
        assert reader_done.wait(1.0)
        assert captured[0].cache.fast_runtime is None
        assert len(prediction_errors) == 1
        assert str(prediction_errors[0]) == "FAST_CACHE_NOT_READY: fast runtime cache is unavailable"
    finally:
        classifier.persistence_release.set()
        reader.join(timeout=2.0)
        catalog.shutdown()

    assert catalog.capture_snapshot(record.id).cache.fast_runtime is not None


def test_fast_sidecar_hashing_never_blocks_catalog_snapshot_or_prediction(
    tmp_path,
    monkeypatch,
):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = production_fast_classifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    hash_calls = []
    hash_started = threading.Event()
    read_snapshots = []
    prediction_errors = []
    real_hash = _file_sha256

    def audited_hash(path):
        hash_calls.append(Path(path))
        hash_started.set()
        reader_done = threading.Event()

        def read_catalog():
            read_snapshots.append(catalog.capture_snapshot(record.id))
            try:
                catalog.predict(record.id, tmp_path / "query.png")
            except Exception as exc:
                prediction_errors.append(str(exc))
            reader_done.set()

        reader = threading.Thread(target=read_catalog)
        reader.start()
        assert reader_done.wait(1.0), "file hash ran while catalog readers were blocked"
        reader.join(timeout=1.0)
        return real_hash(path)

    monkeypatch.setattr("src.orientation_classifier._file_sha256", audited_hash)

    catalog.recover()
    assert hash_started.wait(1.0)
    catalog.shutdown()

    assert hash_calls
    assert read_snapshots
    assert set(prediction_errors) == {"FAST_CACHE_NOT_READY: fast runtime cache is unavailable"}
    assert catalog.fast_cache_status(record.id)["state"] == "ready"


def test_append_publishes_new_revision_before_blocked_fast_worker_finishes(tmp_path):
    classifier = BlockingFastClassifier()
    classifier.inference_mode = "legacy"
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    classifier.inference_mode = "fast_geometry"

    appended, candidate = catalog.append_templates(
        record.id,
        [image(tmp_path / "front-extra.png", 11)],
        [],
        operation_id="append-fast-background",
    )

    try:
        assert classifier.started.wait(1.0)
        snapshot = catalog.capture_snapshot(record.id)
        assert snapshot.record == appended
        assert snapshot.cache is candidate
        assert snapshot.cache.fast_runtime is None
    finally:
        classifier.release.set()
        catalog.shutdown()

    assert catalog.capture_snapshot(record.id).cache.fast_runtime == {
        "library_revision": appended.revision,
        "geometry": None,
    }


def test_recycle_keeps_blocked_fast_worker_from_recreating_sidecar(tmp_path):
    classifier = BlockingFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    assert classifier.started.wait(1.0)

    catalog.recycle(record.id, operation_id="recycle-with-fast-worker")
    recycled = catalog.library.get_recycled(record.id)
    assert catalog.fast_cache_status(record.id)["state"] == "running"

    classifier.release.set()
    catalog.shutdown()

    assert catalog.fast_cache_status(record.id)["state"] == "stale"
    assert classifier.saved == []
    assert not (recycled.root / ".fast_runtime_cache.pkl").exists()


def test_restart_cleans_root_fast_stage_after_recycle_without_harming_live_sidecar(
    tmp_path,
):
    library = WorkpieceLibrary(tmp_path / "library")
    record, base_cache = library.register(
        "M7",
        [image(tmp_path / "front.png", 10)],
        [image(tmp_path / "back.png", 20)],
        False,
        builder,
    )
    classifier = production_fast_classifier()
    old_runtime = classifier.build_fast_runtime_cache(record, None)
    classifier.save_fast_runtime_cache(record, old_runtime)
    staged = classifier.stage_fast_runtime_cache(
        record,
        replace(
            old_runtime,
            training_summary={**old_runtime.training_summary, "marker": "uncommitted"},
        ),
    )
    catalog = WorkpieceCatalog(library, classifier, fast_jobs=RecordingFastJobs())

    catalog.recycle(record.id, operation_id="orphan-stage-recycle")
    recycled = library.get_recycled(record.id)
    live_runtime = classifier.build_fast_runtime_cache(recycled, None)
    classifier.save_fast_runtime_cache(recycled, live_runtime)
    live_sidecar = recycled.root / ".fast_runtime_cache.pkl"
    live_bytes = live_sidecar.read_bytes()
    assert staged.temporary_root.parent == library.library_dir
    assert staged.temporary_root.is_dir()
    assert (staged.temporary_root / ".previous-fast-runtime-cache.pkl").is_file()

    restarted_classifier = production_fast_classifier()
    restarted_classifier.load_template_cache = lambda _record: base_cache
    restarted = WorkpieceCatalog(
        WorkpieceLibrary(library.library_dir),
        restarted_classifier,
        fast_jobs=RecordingFastJobs(),
    )
    restarted.recover()

    recovered_recycled = restarted.library.get_recycled(record.id)
    recovered_sidecar = recovered_recycled.root / ".fast_runtime_cache.pkl"
    assert not staged.temporary_root.exists()
    assert recovered_sidecar.read_bytes() == live_bytes
    assert restarted_classifier.load_fast_runtime_cache(recovered_recycled) is not None

    restarted.purge(record.id, operation_id="orphan-stage-purge")

    assert not recovered_recycled.root.exists()
    assert list(library.library_dir.glob(f".fast-runtime-stage-{record.id}-*")) == []


def test_purge_after_restart_removes_only_the_recycled_records_root_fast_stage(
    tmp_path,
):
    library = WorkpieceLibrary(tmp_path / "library")
    first, _ = library.register(
        "M7",
        [image(tmp_path / "first-front.png", 10)],
        [image(tmp_path / "first-back.png", 20)],
        False,
        builder,
    )
    second, _ = library.register(
        "M8",
        [image(tmp_path / "second-front.png", 30)],
        [image(tmp_path / "second-back.png", 40)],
        False,
        builder,
    )
    classifier = production_fast_classifier()
    first_runtime = classifier.build_fast_runtime_cache(first, None)
    second_runtime = classifier.build_fast_runtime_cache(second, None)
    classifier.save_fast_runtime_cache(first, first_runtime)
    classifier.save_fast_runtime_cache(second, second_runtime)
    first_stage = classifier.stage_fast_runtime_cache(first, first_runtime)
    second_stage = classifier.stage_fast_runtime_cache(second, second_runtime)
    second_sidecar = second.root / ".fast_runtime_cache.pkl"
    second_bytes = second_sidecar.read_bytes()
    WorkpieceCatalog(library, classifier, fast_jobs=RecordingFastJobs()).recycle(
        first.id,
        operation_id="purge-orphan-recycle",
    )

    restarted_library = WorkpieceLibrary(library.library_dir)
    restarted_library.recover(builder)
    restarted = WorkpieceCatalog(
        restarted_library,
        production_fast_classifier(),
        fast_jobs=RecordingFastJobs(),
    )
    assert first_stage.temporary_root.is_dir()
    assert second_stage.temporary_root.is_dir()

    restarted.purge(first.id, operation_id="purge-root-orphan")

    assert not first_stage.temporary_root.exists()
    assert second_stage.temporary_root.is_dir()
    assert second_sidecar.read_bytes() == second_bytes
    classifier.discard_staged_fast_runtime_cache(second_stage)


def test_restore_cleans_recycled_records_root_fast_stage_before_moving_record(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, base_cache = library.register(
        "M7",
        [image(tmp_path / "restore-front.png", 10)],
        [image(tmp_path / "restore-back.png", 20)],
        False,
        builder,
    )
    classifier = production_fast_classifier()
    runtime = classifier.build_fast_runtime_cache(record, None)
    classifier.save_fast_runtime_cache(record, runtime)
    staged = classifier.stage_fast_runtime_cache(record, runtime)
    catalog = WorkpieceCatalog(library, classifier, fast_jobs=RecordingFastJobs())
    catalog.recycle(record.id, operation_id="restore-orphan-recycle")
    recycled = library.get_recycled(record.id)
    recycled_sidecar = recycled.root / ".fast_runtime_cache.pkl"
    sidecar_bytes = recycled_sidecar.read_bytes()
    restarted_library = WorkpieceLibrary(library.library_dir)
    restarted_library.recover(builder)
    restarted_classifier = production_fast_classifier()
    restarted_classifier.load_template_cache = lambda _record: base_cache
    restarted = WorkpieceCatalog(
        restarted_library,
        restarted_classifier,
        fast_jobs=RecordingFastJobs(),
    )

    restored = restarted.restore(record.id, operation_id="restore-root-orphan")

    assert not staged.temporary_root.exists()
    assert (restored.root / ".fast_runtime_cache.pkl").read_bytes() == sidecar_bytes


def test_recovery_activates_base_and_rebuilds_fast_cache_in_background(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    recovered = catalog.recover()

    assert [item.id for item, _cache in recovered] == [record.id]
    assert classifier.started.wait(1.0)
    before = catalog.capture_snapshot(record.id)
    assert before.cache is base_cache
    with pytest.raises(RuntimeError, match="FAST_CACHE_NOT_READY"):
        catalog.predict(record.id, tmp_path / "query.png")

    classifier.release.set()
    catalog.shutdown()
    after = catalog.capture_snapshot(record.id)
    assert after.record == before.record
    assert after.cache.global_vectors is before.cache.global_vectors
    assert after.cache.local_features is before.cache.local_features
    assert after.cache.fast_runtime == {"library_revision": record.revision, "geometry": None}
    assert classifier.saved == [(record.id, after.cache.fast_runtime)]
    assert catalog.fast_cache_status(record.id)["state"] == "ready"


def test_recovery_of_published_geometry_queues_matching_fast_profile(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache

    class PublishedProfiles:
        def sync_library_revision(self, _record):
            return None

        def snapshot(self, _workpiece_id):
            return {
                "active_revision": 5,
                "active": {"profile_revision": 5, "rules": []},
                "profile_status": "ok",
            }

        def rebuild_active_cache(self, *_args):
            pytest.fail("startup recovery synchronously rebuilt geometry")

    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "library"),
        classifier,
        PublishedProfiles(),
    )

    catalog.recover()

    assert classifier.started.wait(1.0)
    recovering = catalog.capture_snapshot(record.id)
    assert recovering.cache.geometry_profile_revision == 5
    assert recovering.cache.fast_runtime is None
    classifier.release.set()
    catalog.shutdown()
    ready = catalog.capture_snapshot(record.id).cache.fast_runtime
    assert ready == {
        "library_revision": record.revision,
        "geometry": {"profile_revision": 5, "rules": []},
    }


def test_compare_recovery_materializes_profile_without_sync_geometry_rebuild(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingFastClassifier()
    classifier.inference_mode = "compare"
    classifier.load_template_cache = lambda _record: base_cache

    class PublishedProfiles:
        def __init__(self):
            self.rebuild_started = threading.Event()

        def sync_library_revision(self, _record):
            return None

        def snapshot(self, _workpiece_id):
            return {
                "active_revision": 6,
                "active": {"profile_revision": 6, "rules": []},
            }

        def rebuild_active_cache(self, *_args, **_kwargs):
            self.rebuild_started.set()
            pytest.fail("compare recovery synchronously rebuilt geometry")

    profiles = PublishedProfiles()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier, profiles)

    catalog.recover()

    assert not profiles.rebuild_started.is_set()
    assert classifier.started.wait(1.0)
    assert catalog.capture_snapshot(record.id).cache.geometry_profile_revision == 6
    classifier.release.set()
    catalog.shutdown()
    assert catalog.capture_snapshot(record.id).cache.fast_runtime["geometry"]["profile_revision"] == 6


def test_purge_cannot_be_undone_by_old_background_worker(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingFastClassifier()
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    catalog.recover()
    assert classifier.started.wait(1.0)

    catalog.recycle(record.id, operation_id="recycle")
    catalog.purge(record.id, operation_id="purge")
    classifier.release.set()
    catalog.shutdown()

    assert not record.root.exists()
    assert classifier.saved == []
    assert catalog.fast_cache_status(record.id)["state"] == "stale"


class GeometryFastClassifier(BlockingFastClassifier):
    inference_mode = "legacy"

    def __init__(self):
        super().__init__()
        self.block_build = False
        self.fail_build = False

    def build_fast_runtime_cache(self, record, geometry_profile, progress_callback=None):
        if self.block_build:
            self.started.set()
            assert self.release.wait(2.0)
        if self.fail_build:
            raise RuntimeError("geometry fast build failed")
        return {
            "library_revision": record.revision,
            "geometry_profile_revision": None if geometry_profile is None else geometry_profile["profile_revision"],
        }

    def save_template_cache(self, record, cache):
        if cache.fast_runtime is not None:
            self.save_fast_runtime_cache(record, cache.fast_runtime)

    def prepare_template_masks(self, workpiece_id, ignored_regions, *, base_cache=None):
        base = base_cache or self.caches[workpiece_id]
        return replace(
            base,
            ignored_regions=ignored_regions,
            geometry_profile=None,
            geometry_profile_revision=None,
            fast_runtime=None,
        ), {}


class FailureAtomicGeometryClassifier(GeometryFastClassifier):
    def __init__(self):
        super().__init__()
        self.fail_next_activation = False
        self.sidecar_committed = threading.Event()

    def set_template_cache(self, workpiece_id, cache):
        super().set_template_cache(workpiece_id, cache)
        if self.fail_next_activation:
            self.fail_next_activation = False
            raise RuntimeError("activation failed after sidecar commit")

    @staticmethod
    def stage_fast_runtime_cache(record, runtime, *, geometry_profile_revision=None):
        target = record.root / ".fast_runtime_cache.pkl"
        return {
            "target": target,
            "payload": json.dumps(runtime, sort_keys=True).encode("utf-8"),
            "previous": target.read_bytes() if target.is_file() else None,
            "runtime": runtime,
        }

    def commit_staged_fast_runtime_cache(self, record, staged):
        staged["target"].write_bytes(staged["payload"])
        self.saved.append((record.id, staged["runtime"]))
        self.sidecar_committed.set()
        return staged

    @staticmethod
    def rollback_committed_fast_runtime_cache(committed):
        if committed["previous"] is None:
            if committed["target"].is_file():
                committed["target"].unlink()
        else:
            committed["target"].write_bytes(committed["previous"])


class BuildSaveOnlyFastClassifier(GeometryFastClassifier):
    stage_fast_runtime_cache = None
    commit_staged_fast_runtime_cache = None
    finalize_staged_fast_runtime_cache = None
    discard_staged_fast_runtime_cache = None
    rollback_committed_fast_runtime_cache = None

    def __init__(self):
        super().__init__()
        self.fast_build_calls = 0

    def build_fast_runtime_cache(self, record, geometry_profile, progress_callback=None):
        self.fast_build_calls += 1
        return {
            "library_revision": record.revision,
            "geometry_profile_revision": (
                None if geometry_profile is None else geometry_profile["profile_revision"]
            ),
        }


class RecordingFastJobs:
    def __init__(self):
        self.scheduled = []

    def schedule(self, **kwargs):
        self.scheduled.append(kwargs)

    @staticmethod
    def snapshot(_workpiece_id):
        return None

    @staticmethod
    def shutdown():
        return None


@pytest.mark.parametrize("inference_mode", ["fast_geometry", "compare"])
def test_build_save_only_classifier_is_not_scheduled_and_reports_capability_error(
    tmp_path,
    inference_mode,
):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BuildSaveOnlyFastClassifier()
    classifier.inference_mode = inference_mode
    classifier.load_template_cache = lambda _record: base_cache
    jobs = RecordingFastJobs()

    class PublishedProfiles:
        def __init__(self):
            self.rebuild_calls = 0

        @staticmethod
        def sync_library_revision(_record):
            return None

        @staticmethod
        def snapshot(_workpiece_id):
            return {
                "active_revision": 6,
                "active": {"profile_revision": 6, "rules": []},
            }

        def rebuild_active_cache(self, *_args, **_kwargs):
            self.rebuild_calls += 1
            pytest.fail("unsupported fast classifier triggered synchronous geometry rebuild")

    profiles = PublishedProfiles()
    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "library"),
        classifier,
        profiles,
        fast_jobs=jobs,
    )

    catalog.recover()

    snapshot = catalog.capture_snapshot(record.id)
    assert snapshot.cache.geometry_profile_revision == 6
    assert snapshot.cache.fast_runtime is None
    assert profiles.rebuild_calls == 0
    assert classifier.fast_build_calls == 0
    assert jobs.scheduled == []
    assert catalog.fast_cache_status(record.id) == {
        "state": "not_ready",
        "completed": 0,
        "total": 0,
        "elapsed_ms": 0.0,
        "error": "FAST_CACHE_CAPABILITY_UNAVAILABLE: staged fast-cache persistence is unavailable",
    }


@pytest.mark.parametrize("inference_mode", ["fast_geometry", "compare"])
def test_geometry_publish_with_build_save_only_classifier_skips_unpublishable_build(
    tmp_path,
    inference_mode,
):
    classifier = BuildSaveOnlyFastClassifier()
    classifier.inference_mode = inference_mode
    jobs = RecordingFastJobs()
    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "library"),
        classifier,
        fast_jobs=jobs,
    )
    record, base = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    candidate = replace(
        base,
        geometry_profile={"profile_revision": 3},
        geometry_profile_revision=3,
    )

    published = catalog.publish_geometry_profile(
        record.id,
        candidate,
        profile_revision=3,
        previous_profile_revision=None,
        expected_revision=record.revision,
        operation_id=f"build-save-only-{inference_mode}",
    )

    assert published.revision == record.revision + 1
    assert catalog.capture_snapshot(record.id).cache is candidate
    assert classifier.fast_build_calls == 0
    assert jobs.scheduled == []
    assert catalog.fast_cache_status(record.id)["error"] == (
        "FAST_CACHE_CAPABILITY_UNAVAILABLE: staged fast-cache persistence is unavailable"
    )


def test_background_activation_failure_restores_snapshot_classifier_and_sidecar(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = FailureAtomicGeometryClassifier()
    classifier.inference_mode = "fast_geometry"
    classifier.block_build = True
    classifier.load_template_cache = lambda _record: base_cache
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    catalog.recover()
    assert classifier.started.wait(1.0)
    before = catalog.capture_snapshot(record.id)
    manifest_path = record.root / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    sidecar = record.root / ".fast_runtime_cache.pkl"
    sidecar.write_bytes(b"old-sidecar")
    sidecar_before = sidecar.read_bytes()
    classifier.fail_next_activation = True
    classifier.release.set()
    catalog.shutdown()

    assert classifier.sidecar_committed.is_set()
    status = catalog.fast_cache_status(record.id)
    assert status["state"] == "failed"
    assert status["error"] == "activation failed after sidecar commit"
    assert catalog.capture_snapshot(record.id) is before
    assert classifier.caches[record.id] is before.cache
    assert catalog.get(record.id) is before.record
    assert manifest_path.read_bytes() == manifest_before
    assert sidecar.read_bytes() == sidecar_before


def test_geometry_activation_failure_restores_snapshot_pointer_and_sidecar(tmp_path):
    classifier = FailureAtomicGeometryClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    before = catalog.capture_snapshot(record.id)
    manifest_path = record.root / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    sidecar = record.root / ".fast_runtime_cache.pkl"
    sidecar.write_bytes(b"old-sidecar")
    sidecar_before = sidecar.read_bytes()
    classifier.inference_mode = "fast_geometry"
    classifier.fail_next_activation = True
    candidate = replace(
        base,
        geometry_profile={"profile_revision": 3},
        geometry_profile_revision=3,
    )

    with pytest.raises(RuntimeError, match="activation failed after sidecar commit"):
        catalog.publish_geometry_profile(
            record.id,
            candidate,
            profile_revision=3,
            previous_profile_revision=None,
            expected_revision=record.revision,
            operation_id="geometry-activation-failure",
        )

    assert classifier.sidecar_committed.is_set()
    assert catalog.capture_snapshot(record.id) is before
    assert classifier.caches[record.id] is before.cache
    assert catalog.get(record.id) is before.record
    assert manifest_path.read_bytes() == manifest_before
    assert sidecar.read_bytes() == sidecar_before
    catalog.shutdown()


def test_legacy_rollback_activation_failure_restores_snapshot_pointer_and_sidecar(tmp_path):
    classifier = FailureAtomicGeometryClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    group = {
        "group_id": "legacy",
        "name": "legacy",
        "enabled": True,
        "propagation": {"state": "active"},
        "annotations": [{
            "orientation": "front",
            "index": 0,
            "status": "active",
            "regions": [{"x": 0, "y": 0, "width": 2, "height": 2}],
        }],
    }
    catalog.commit_annotation_document(
        record.id,
        [group],
        expected_revision=record.revision,
        operation_id="legacy-before-failure",
        active_groups=[group],
    )
    record = catalog.get(record.id)
    classifier.inference_mode = "fast_geometry"
    current = catalog.capture_snapshot(record.id)
    published = catalog.publish_geometry_profile(
        record.id,
        replace(
            current.cache,
            geometry_profile={"profile_revision": 1},
            geometry_profile_revision=1,
            fast_runtime=None,
        ),
        profile_revision=1,
        previous_profile_revision=None,
        expected_revision=record.revision,
        operation_id="geometry-before-legacy-failure",
    )
    before = catalog.capture_snapshot(record.id)
    library_before = catalog.get(record.id)
    manifest_path = published.root / "manifest.json"
    manifest_before = manifest_path.read_bytes()
    sidecar = published.root / ".fast_runtime_cache.pkl"
    sidecar_before = sidecar.read_bytes()
    classifier.sidecar_committed.clear()
    classifier.fail_next_activation = True

    with pytest.raises(RuntimeError, match="activation failed after sidecar commit"):
        catalog.restore_legacy_annotation_cache(
            record.id,
            expected_revision=published.revision,
            operation_id="legacy-activation-failure",
        )

    assert classifier.sidecar_committed.is_set()
    assert catalog.capture_snapshot(record.id) is before
    assert classifier.caches[record.id] is before.cache
    assert catalog.get(record.id) is library_before
    assert catalog.get(record.id) == before.record
    assert manifest_path.read_bytes() == manifest_before
    assert sidecar.read_bytes() == sidecar_before
    catalog.shutdown()


def test_geometry_publication_builds_target_revision_before_pointer_swap(tmp_path):
    classifier = GeometryFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    classifier.inference_mode = "fast_geometry"
    profile = {"profile_revision": 3}
    candidate = replace(base, geometry_profile=profile, geometry_profile_revision=3)

    published = catalog.publish_geometry_profile(
        record.id,
        candidate,
        profile_revision=3,
        previous_profile_revision=None,
        expected_revision=record.revision,
        operation_id="publish-fast-3",
    )

    runtime = catalog.capture_snapshot(record.id).cache.fast_runtime
    assert published.revision == record.revision + 1
    assert runtime == {"library_revision": published.revision, "geometry_profile_revision": 3}
    assert classifier.saved[-1] == (record.id, runtime)
    catalog.shutdown()


def test_failed_geometry_replacement_keeps_pointer_while_fast_build_is_blocked(tmp_path):
    classifier = GeometryFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    before = catalog.capture_snapshot(record.id)
    classifier.inference_mode = "fast_geometry"
    classifier.block_build = True
    classifier.fail_build = True
    candidate = replace(
        base,
        geometry_profile={"profile_revision": 4},
        geometry_profile_revision=4,
    )

    errors = []
    done = threading.Event()

    def publish():
        try:
            catalog.publish_geometry_profile(
                record.id,
                candidate,
                profile_revision=4,
                previous_profile_revision=None,
                expected_revision=record.revision,
                operation_id="publish-fast-failure",
            )
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=publish)
    worker.start()
    try:
        assert classifier.started.wait(1.0)
        assert catalog.capture_snapshot(record.id) is before
        assert catalog.get(record.id).revision == record.revision
        assert not done.is_set()
    finally:
        classifier.release.set()
        worker.join(timeout=2.0)
        catalog.shutdown()

    assert done.is_set()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "geometry fast build failed"
    assert catalog.capture_snapshot(record.id) is before
    assert catalog.get(record.id).revision == record.revision


def test_geometry_rollback_keeps_active_cache_while_fast_worker_is_blocked(tmp_path):
    classifier = GeometryFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    group = {
        "group_id": "legacy",
        "name": "legacy",
        "enabled": True,
        "propagation": {"state": "active"},
        "annotations": [{
            "orientation": "front",
            "index": 0,
            "status": "active",
            "regions": [{"x": 0, "y": 0, "width": 2, "height": 2}],
        }],
    }
    catalog.commit_annotation_document(
        record.id,
        [group],
        expected_revision=record.revision,
        operation_id="legacy-before-geometry",
        active_groups=[group],
    )
    record = catalog.get(record.id)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "geometry-jobs",
    )
    catalog.set_geometry_profiles(profiles)
    profile = profiles.snapshot(record.id)["draft"]
    profile_root = record.root / "geometry_masks"
    (profile_root / "revisions").mkdir(parents=True, exist_ok=True)
    profile_document = {
        "schema_version": profile["schema_version"],
        "library_revision": record.revision + 1,
        "draft_revision": 0,
        "active_revision": 1,
        "previous_active_revision": None,
        "draft": profile,
        "active": profile,
    }
    profile_path = profile_root / "profile.json"
    profile_path.write_text(json.dumps(profile_document), encoding="utf-8")
    (profile_root / "revisions" / "1.json").write_text(
        json.dumps({
            "profile": profile,
            "previous_active_revision": None,
            "previous_source": "legacy",
        }),
        encoding="utf-8",
    )
    classifier.inference_mode = "fast_geometry"
    current = catalog.capture_snapshot(record.id)
    published = catalog.publish_geometry_profile(
        record.id,
        replace(
            current.cache,
            geometry_profile={**profile, "profile_revision": 1},
            geometry_profile_revision=1,
            fast_runtime=None,
        ),
        profile_revision=1,
        previous_profile_revision=None,
        expected_revision=record.revision,
        operation_id="publish-before-rollback",
    )
    before = catalog.capture_snapshot(record.id)
    classifier.block_build = True
    classifier.started.clear()
    classifier.release.clear()
    rolled = []
    errors = []
    done = threading.Event()

    def rollback():
        try:
            rolled.append(profiles.rollback(
                record.id,
                expected_library_revision=published.revision,
                operation_id="rollback-to-legacy",
            ))
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    worker = threading.Thread(target=rollback)
    worker.start()
    try:
        assert classifier.started.wait(1.0)
        assert catalog.capture_snapshot(record.id) is before
        assert json.loads(profile_path.read_text(encoding="utf-8"))["active_revision"] == 1
        assert not done.is_set()
    finally:
        classifier.release.set()
        worker.join(timeout=2.0)
        catalog.shutdown()
        profiles.shutdown()

    assert done.is_set()
    assert errors == []
    assert rolled[0]["active_revision"] is None
    ready = catalog.capture_snapshot(record.id)
    assert ready.record.revision == published.revision + 1
    assert ready.cache.geometry_profile_revision is None
    assert ready.cache.fast_runtime["geometry_profile_revision"] is None
