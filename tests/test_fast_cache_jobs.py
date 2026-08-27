from dataclasses import replace
from pathlib import Path
import threading

import cv2
import numpy as np
import pytest

from src.fast_cache_jobs import FastCacheJobManager
from src.orientation_classifier import TemplateCache
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
    stopper = threading.Thread(target=lambda: (manager.shutdown(), shutdown_done.set()))
    stopper.start()
    assert not shutdown_done.wait(0.05)
    first_release.set()
    assert shutdown_done.wait(1.0)
    stopper.join(timeout=1.0)

    assert not second_started.is_set()
    assert manager.snapshot("first").state == "ready"
    assert manager.snapshot("second").state == "queued"


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

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def remove_template_cache(self, workpiece_id):
        self.caches.pop(workpiece_id, None)

    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        if cache.fast_runtime is None:
            raise RuntimeError("FAST_CACHE_NOT_READY: fast runtime cache is unavailable")
        return {"label": "front", "library_revision": library_revision}


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


def test_compare_recovery_builds_legacy_geometry_without_waiting_for_fast_runtime(tmp_path):
    record, base_cache = _persist_base_library(tmp_path)
    classifier = BlockingFastClassifier()
    classifier.inference_mode = "compare"
    classifier.load_template_cache = lambda _record: base_cache

    class PublishedProfiles:
        def __init__(self):
            self.fast_runtime_flags = []

        def sync_library_revision(self, _record):
            return None

        def rebuild_active_cache(self, workpiece_id, _record, *, build_fast_runtime=True):
            self.fast_runtime_flags.append(build_fast_runtime)
            base = classifier.caches[workpiece_id]
            return replace(
                base,
                geometry_profile={"profile_revision": 6, "rules": []},
                geometry_profile_revision=6,
                fast_runtime=None,
            )

    profiles = PublishedProfiles()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier, profiles)

    catalog.recover()

    assert profiles.fast_runtime_flags == [False]
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
        self.fail_build = False

    def build_fast_runtime_cache(self, record, geometry_profile, progress_callback=None):
        if self.fail_build:
            raise RuntimeError("geometry fast build failed")
        return {
            "library_revision": record.revision,
            "geometry_profile_revision": None if geometry_profile is None else geometry_profile["profile_revision"],
        }

    def save_template_cache(self, record, cache):
        if cache.fast_runtime is not None:
            self.save_fast_runtime_cache(record, cache.fast_runtime)


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


def test_geometry_publication_build_failure_leaves_pointer_and_cache_unchanged(tmp_path):
    classifier = GeometryFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, base = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False,
    )
    before = catalog.capture_snapshot(record.id)
    classifier.inference_mode = "fast_geometry"
    classifier.fail_build = True
    candidate = replace(
        base,
        geometry_profile={"profile_revision": 4},
        geometry_profile_revision=4,
    )

    with pytest.raises(RuntimeError, match="geometry fast build failed"):
        catalog.publish_geometry_profile(
            record.id,
            candidate,
            profile_revision=4,
            previous_profile_revision=None,
            expected_revision=record.revision,
            operation_id="publish-fast-failure",
        )

    assert catalog.capture_snapshot(record.id) is before
    assert catalog.get(record.id).revision == record.revision
    catalog.shutdown()
