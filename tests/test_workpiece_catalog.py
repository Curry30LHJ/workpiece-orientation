from dataclasses import replace
from pathlib import Path
import json
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.fast_geometry import FastGeometryProcessor
from src.fast_orientation import FastOrientationEngine
from src.orientation_classifier import OrientationClassifier, OrientationClassifierError, TemplateCache
from src.geometry_mask_profiles import GeometryMaskProfiles
from src.workpiece_catalog import RestoreConflictError, WorkpieceCatalog
from src.workpiece_library import StaleWorkpieceRevisionError, WorkpieceLibrary


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


class PersistedGlobalPredictor:
    def predict(self, images):
        embeddings = []
        for item in images:
            mean = float(np.mean(item)) / 255.0
            embeddings.append(np.asarray([mean, 1.0 - mean], dtype=np.float32))
        return embeddings


def persisted_fast_classifier(inference_mode):
    classifier = OrientationClassifier(
        global_predictor=PersistedGlobalPredictor(),
        extractor=object() if inference_mode == "compare" else None,
        matcher=object() if inference_mode == "compare" else None,
        device="cpu",
        extract_features_fn=lambda item, *_args, **_kwargs: {
            "marker": int(item[0, 0, 0])
        },
        score_feature_pair_fn=lambda *_args, **_kwargs: {"score": 1.0},
        inference_mode=inference_mode,
        model_fingerprint="catalog-model-a",
    )
    classifier.fast_engine = FastOrientationEngine(
        classifier._global_embeddings,
        FastGeometryProcessor(object()),
        image_reader=lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR),
    )
    return classifier


class FakeClassifier:
    def __init__(self):
        self.caches = {}

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def remove_template_cache(self, workpiece_id):
        self.caches.pop(workpiece_id, None)

    def get_template_cache(self, workpiece_id):
        return self.caches.get(workpiece_id)

    def prepare_geometry_cache(
        self, workpiece_id, record, profile, calibrator=None, progress_callback=None, *, base_cache=None
    ):
        base = base_cache or self.caches[workpiece_id]
        candidate = TemplateCache(
            global_vectors=base.global_vectors,
            local_features=base.local_features,
            raw_global_vectors=base.raw_global_vectors or base.global_vectors,
            raw_local_features=base.raw_local_features or base.local_features,
            geometry_profile=profile,
            geometry_profile_revision=profile.get("profile_revision"),
            geometry_template_report={"front": [], "back": []},
        )
        return candidate, {"front": [], "back": []}

    def build_template_cache(self, front, back, progress_callback=None):
        return builder(front, back, progress_callback)

    @staticmethod
    def leave_one_out_report(record, cache):
        return {
            "status": "completed",
            "correct_to_wrong": 0,
            "evaluated": len(record.front_images) + len(record.back_images),
            "skipped": 0,
            "fit_failures": [],
            "changed_predictions": [],
        }

    def prepare_template_masks(self, workpiece_id, ignored_regions, *, base_cache=None):
        base = base_cache or self.caches[workpiece_id]
        return TemplateCache(
            global_vectors=base.global_vectors,
            local_features=base.local_features,
            raw_global_vectors=base.raw_global_vectors or base.global_vectors,
            raw_local_features=base.raw_local_features or base.local_features,
            ignored_regions=ignored_regions,
        ), {}

    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        return {"label": "front", "library_revision": library_revision}


class BatchCatalogClassifier(FakeClassifier):
    def __init__(self):
        super().__init__()
        self.snapshot_calls = 0
        self.batch_paths = None

    def predict_many_with_cache(self, cache, image_paths, *, library_revision=None):
        self.snapshot_calls += 1
        self.batch_paths = list(image_paths)
        return [
            {"index": index, "image_path": str(path), "label": "front", "library_revision": library_revision}
            for index, path in enumerate(image_paths)
        ]

    def batch_capabilities(self):
        return {"batch_ready": True, "worker_count": 1, "threads_per_worker": 1}


class LegacyReadyBatchCatalogClassifier(BatchCatalogClassifier):
    batch_capabilities = None
    batch_ready = True


class PartialFailureCatalogClassifier(FakeClassifier):
    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        if Path(image_path).name == "bad.png":
            raise OSError("unable to decode image")
        return super().predict_with_cache(cache, image_path, library_revision=library_revision)


class FastCacheFailureCatalogClassifier(FakeClassifier):
    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        raise OrientationClassifierError("FAST_CACHE_NOT_READY: cache is unavailable")


class ReorderedBatchCatalogClassifier(BatchCatalogClassifier):
    def predict_many_with_cache(self, cache, image_paths, *, library_revision=None):
        return list(reversed([
            {"index": index, "image_path": str(path), "label": f"label-{index}"}
            for index, path in enumerate(image_paths)
        ]))


class MalformedBatchCatalogClassifier(BatchCatalogClassifier):
    def predict_many_with_cache(self, cache, image_paths, *, library_revision=None):
        return [
            {"index": 0, "image_path": str(image_paths[0]), "label": "front"},
            {"index": 0, "image_path": str(image_paths[1]), "label": "back"},
        ]


class RevisionAwareFastClassifier(FakeClassifier):
    inference_mode = "fast_geometry"

    def build_template_cache(
        self,
        front,
        back,
        progress_callback=None,
        *,
        library_revision=1,
    ):
        cache = builder(front, back, progress_callback)
        return TemplateCache(
            global_vectors=cache.global_vectors,
            local_features=cache.local_features,
            fast_runtime=SimpleNamespace(library_revision=library_revision),
        )


class BlockingPredictClassifier(FakeClassifier):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def _block(self, library_revision=None):
        self.started.set()
        assert self.release.wait(2.0)
        return {"label": "front", "library_revision": library_revision}

    def predict(self, workpiece_id, image_path):
        return self._block()

    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        return self._block(library_revision)


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


class SlowAppendClassifier(FakeClassifier):
    def __init__(self):
        super().__init__()
        self.block_appends = False
        self.build_started = threading.Event()
        self.release_build = threading.Event()

    def build_template_cache(self, front, back, progress_callback=None):
        if self.block_appends:
            self.build_started.set()
            assert self.release_build.wait(2.0)
        return super().build_template_cache(front, back, progress_callback)


class SummaryGeometryProfiles:
    def snapshot(self, workpiece_id):
        return {
            "profile_status": "ok",
            "active": {"rules": [{"rule_id": "glare"}, {"rule_id": "intrusion"}]},
        }


class BlockingCatalogGeometryProfiles:
    def __init__(self):
        self.catalog = None
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def snapshot(self, workpiece_id):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            assert self.release.wait(2.0)
        record = self.catalog.get(workpiece_id)
        return {
            "profile_status": "ok",
            "active": {"rules": []},
            "library_revision": record.revision,
        }


class AlwaysMutatingSummaryGeometryProfiles:
    def __init__(self):
        self.catalog = None
        self.calls = 0

    def snapshot(self, workpiece_id):
        self.calls += 1
        current = self.catalog.capture_snapshot(workpiece_id)
        self.catalog.publish_geometry_profile(
            workpiece_id,
            current.cache,
            profile_revision=self.calls,
            previous_profile_revision=None,
            expected_revision=current.record.revision,
            operation_id=f"summary-churn-{self.calls}",
        )
        return {
            "profile_status": "ok",
            "active": {"rules": []},
        }


def create_catalog(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    front = [image(tmp_path / "front.png", 10)]
    back = [image(tmp_path / "back.png", 20)]
    record, _ = catalog.register("M7", front, back, False)
    return catalog, classifier, record


def create_catalog_with_counts(tmp_path, front_count, back_count):
    classifier = FakeClassifier()
    library = WorkpieceLibrary(tmp_path / "summary-library")
    catalog = WorkpieceCatalog(library, classifier, SummaryGeometryProfiles())
    front = [
        image(tmp_path / f"summary-front-{i}.png", 10 + i)
        for i in range(front_count)
    ]
    back = [
        image(tmp_path / f"summary-back-{i}.png", 80 + i)
        for i in range(back_count)
    ]
    record, _ = catalog.register("M-summary", front, back, False)
    return catalog, classifier, record


def test_predict_batch_captures_one_snapshot_and_returns_input_order(tmp_path, monkeypatch):
    library = WorkpieceLibrary(tmp_path / "batch-library")
    classifier = BatchCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    front = [image(tmp_path / "batch-front.png", 10)]
    back = [image(tmp_path / "batch-back.png", 20)]
    record, _ = catalog.register("M7", front, back, False)
    captured = catalog.capture_snapshot
    calls = []

    def count_capture(workpiece_id):
        calls.append(workpiece_id)
        return captured(workpiece_id)

    monkeypatch.setattr(catalog, "capture_snapshot", count_capture)

    result = catalog.predict_many(record.id, [Path("a.png"), Path("b.png"), Path("c.png")])

    assert [item["index"] for item in result] == [0, 1, 2]
    assert calls == [record.id]


def test_predict_many_uses_legacy_batch_ready_attribute_without_capabilities(tmp_path):
    library = WorkpieceLibrary(tmp_path / "legacy-batch-library")
    classifier = LegacyReadyBatchCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register("M7", [image(tmp_path / "legacy-front.png", 10)], [image(tmp_path / "legacy-back.png", 20)], False)

    result = catalog.predict_many(record.id, [Path("a.png"), Path("b.png")])

    assert [item["index"] for item in result] == [0, 1]
    assert classifier.batch_paths == [Path("a.png"), Path("b.png")]


def test_predict_many_keeps_partial_scalar_errors_in_input_order(tmp_path):
    library = WorkpieceLibrary(tmp_path / "partial-batch-library")
    classifier = PartialFailureCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register("M7", [image(tmp_path / "partial-front.png", 10)], [image(tmp_path / "partial-back.png", 20)], False)

    result = catalog.predict_many(record.id, [Path("a.png"), Path("bad.png"), Path("c.png")])

    assert [item["index"] for item in result] == [0, 1, 2]
    assert [item["ok"] for item in result] == [True, False, True]
    assert result[1]["error"]["code"] == "MODEL_ERROR"


def test_predict_many_reraises_fast_cache_errors_from_scalar_fallback(tmp_path):
    library = WorkpieceLibrary(tmp_path / "fast-cache-batch-library")
    classifier = FastCacheFailureCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register("M7", [image(tmp_path / "fast-front.png", 10)], [image(tmp_path / "fast-back.png", 20)], False)

    with pytest.raises(OrientationClassifierError, match="FAST_CACHE_NOT_READY"):
        catalog.predict_many(record.id, [Path("a.png")])


def test_predict_many_reorders_batch_results_by_declared_index(tmp_path):
    library = WorkpieceLibrary(tmp_path / "reordered-batch-library")
    classifier = ReorderedBatchCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register("M7", [image(tmp_path / "reordered-front.png", 10)], [image(tmp_path / "reordered-back.png", 20)], False)

    result = catalog.predict_many(record.id, [Path("a.png"), Path("b.png")])

    assert [item["index"] for item in result] == [0, 1]
    assert [item["prediction"]["label"] for item in result] == ["label-0", "label-1"]


def test_predict_many_rejects_duplicate_batch_result_indices(tmp_path):
    library = WorkpieceLibrary(tmp_path / "malformed-batch-library")
    classifier = MalformedBatchCatalogClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register("M7", [image(tmp_path / "malformed-batch-front.png", 10)], [image(tmp_path / "malformed-batch-back.png", 20)], False)

    with pytest.raises(Exception, match="batch result"):
        catalog.predict_many(record.id, [Path("a.png"), Path("b.png")])


def test_workpiece_summary_reports_unequal_counts_and_rules(tmp_path):
    catalog, _, record = create_catalog_with_counts(tmp_path, 1, 12)

    summary = catalog.list_workpiece_summaries()[0]

    assert summary["id"] == record.id
    assert summary["template_counts"] == {"front": 1, "back": 12}
    assert summary["geometry_rule_count"] == 2
    assert summary["geometry_status"] == "ok"
    assert summary["detectable"] is True
    assert summary["fast_cache"] == {
        "state": "not_ready",
        "completed": 0,
        "total": 0,
        "elapsed_ms": 0.0,
        "error": None,
    }


def test_details_return_all_unequal_and_over_thirty_templates(tmp_path):
    catalog, _, record = create_catalog_with_counts(tmp_path, 31, 1)

    details = catalog.get_workpiece_details(record.id)
    template_ids = [item["template_id"] for item in details["templates"]]

    assert details["template_counts"] == {"front": 31, "back": 1}
    assert len(template_ids) == 32
    assert len(set(template_ids)) == 32
    assert template_ids[0] == "front:00.png"
    assert template_ids[-1] == "back:00.png"
    assert all(Path(item["preview_path"]).is_absolute() for item in details["templates"])
    assert all(item["readable"] is True for item in details["templates"])


def test_legacy_details_are_derived_without_manifest_or_cache_mutation(tmp_path):
    classifier = CachingFakeClassifier()
    library = WorkpieceLibrary(tmp_path / "legacy-library")
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register(
        "M-legacy",
        [image(tmp_path / f"legacy-front-{index}.png", 10 + index) for index in range(5)],
        [image(tmp_path / f"legacy-back-{index}.png", 30 + index) for index in range(5)],
        False,
    )
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("template_inventory")
    manifest.pop("created_at")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    before = manifest_path.read_bytes()
    before_mtime = manifest_path.stat().st_mtime_ns
    build_calls = classifier.build_calls
    save_calls = list(classifier.save_calls)

    details = catalog.get_workpiece_details(record.id)

    assert details["template_counts"] == {"front": 5, "back": 5}
    assert len(details["templates"]) == 10
    assert {item["source"] for item in details["templates"]} == {"initial_registration"}
    assert {item["added_at"] for item in details["templates"]} == {None}
    assert manifest_path.read_bytes() == before
    assert manifest_path.stat().st_mtime_ns == before_mtime
    assert classifier.build_calls == build_calls
    assert classifier.save_calls == save_calls


def test_summary_detectable_reflects_runtime_snapshot_presence(tmp_path):
    library = WorkpieceLibrary(tmp_path / "snapshot-library")
    record, _ = library.register(
        "M-snapshot",
        [image(tmp_path / "snapshot-front.png", 10)],
        [image(tmp_path / "snapshot-back.png", 20)],
        False,
        builder,
    )
    catalog = WorkpieceCatalog(library, FakeClassifier())

    assert catalog.list_workpiece_summaries()[0]["detectable"] is False

    catalog.recover()

    assert catalog.list_workpiece_summaries()[0]["id"] == record.id
    assert catalog.list_workpiece_summaries()[0]["detectable"] is True


def test_details_keep_all_templates_and_mark_an_unreadable_preview(tmp_path):
    catalog, _, record = create_catalog_with_counts(tmp_path, 2, 1)
    record.front_images[1].write_bytes(b"corrupt")

    details = catalog.get_workpiece_details(record.id)

    assert len(details["templates"]) == 3
    readable = {item["template_id"]: item["readable"] for item in details["templates"]}
    assert readable == {
        "front:00.png": True,
        "front:01.png": False,
        "back:00.png": True,
    }


def test_details_resolve_preview_paths_from_relative_library_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(Path("relative-library")), classifier)
    record, _ = catalog.register(
        "M-relative",
        [image(tmp_path / "relative-front.png", 10)],
        [image(tmp_path / "relative-back.png", 20)],
        False,
    )

    details = catalog.get_workpiece_details(record.id)

    assert all(Path(item["preview_path"]).is_absolute() for item in details["templates"])


def test_append_forwards_source_and_details_switch_after_commit(tmp_path):
    classifier = SlowAppendClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "append-details-library"), classifier)
    record, _ = catalog.register(
        "M-append-details",
        [image(tmp_path / "append-details-front.png", 10)],
        [image(tmp_path / "append-details-back.png", 20)],
        False,
    )
    classifier.block_appends = True
    result = []
    errors = []

    def append():
        try:
            result.append(catalog.append_templates(
                record.id,
                [image(tmp_path / "append-details-new.png", 30)],
                [],
                operation_id="append-details-1",
                source="confirmed_inspection",
            ))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=append)
    worker.start()
    assert classifier.build_started.wait(1.0)
    old_details = catalog.get_workpiece_details(record.id)
    try:
        assert len(old_details["templates"]) == 2
    finally:
        classifier.release_build.set()
        worker.join(timeout=2.0)

    assert not errors
    assert result
    new_details = catalog.get_workpiece_details(record.id)
    assert len(new_details["templates"]) == 3
    assert new_details["templates"][-1]["source"] == "confirmed_inspection"


def test_details_retry_after_append_during_geometry_snapshot(tmp_path):
    profiles = BlockingCatalogGeometryProfiles()
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "details-retry-library"),
        classifier,
        profiles,
    )
    profiles.catalog = catalog
    record, _ = catalog.register(
        "M-details-retry",
        [image(tmp_path / "details-retry-front.png", 10)],
        [image(tmp_path / "details-retry-back.png", 20)],
        False,
    )
    results = []
    errors = []

    def read_details():
        try:
            results.append(catalog.get_workpiece_details(record.id))
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=read_details)
    reader.start()
    assert profiles.started.wait(1.0)
    try:
        appended, _ = catalog.append_templates(
            record.id,
            [image(tmp_path / "details-retry-new.png", 30)],
            [],
            operation_id="details-retry-append",
        )
    finally:
        profiles.release.set()
        reader.join(timeout=2.0)

    assert not reader.is_alive()
    assert errors == []
    assert len(results) == 1
    assert results[0]["revision"] == appended.revision
    assert results[0]["template_counts"] == {"front": 2, "back": 1}
    assert len(results[0]["templates"]) == 3
    assert profiles.calls >= 2


def test_summary_list_skips_recycled_record_during_geometry_snapshot(tmp_path):
    profiles = BlockingCatalogGeometryProfiles()
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "summary-retry-library"),
        classifier,
        profiles,
    )
    profiles.catalog = catalog
    record, _ = catalog.register(
        "M-summary-retry",
        [image(tmp_path / "summary-retry-front.png", 10)],
        [image(tmp_path / "summary-retry-back.png", 20)],
        False,
    )
    results = []
    errors = []

    def read_summaries():
        try:
            results.append(catalog.list_workpiece_summaries())
        except Exception as exc:
            errors.append(exc)

    reader = threading.Thread(target=read_summaries)
    reader.start()
    assert profiles.started.wait(1.0)
    try:
        catalog.recycle(record.id, operation_id="summary-retry-recycle")
    finally:
        profiles.release.set()
        reader.join(timeout=2.0)

    assert not reader.is_alive()
    assert errors == []
    assert results == [[]]


def test_summary_list_reports_stale_after_all_snapshot_retries_change_revision(tmp_path):
    profiles = AlwaysMutatingSummaryGeometryProfiles()
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(
        WorkpieceLibrary(tmp_path / "summary-churn-library"),
        classifier,
        profiles,
    )
    profiles.catalog = catalog
    catalog.register(
        "M-summary-churn",
        [image(tmp_path / "summary-churn-front.png", 10)],
        [image(tmp_path / "summary-churn-back.png", 20)],
        False,
    )

    with pytest.raises(StaleWorkpieceRevisionError, match="changed repeatedly"):
        catalog.list_workpiece_summaries()

    assert profiles.calls == 3


def test_predict_releases_catalog_lock_before_classifier_runs(tmp_path):
    classifier = BlockingPredictClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7",
        [image(tmp_path / "front-lock.png", 10)],
        [image(tmp_path / "back-lock.png", 20)],
        False,
    )
    prediction = threading.Thread(
        target=catalog.predict,
        args=(record.id, tmp_path / "query-lock.png"),
    )
    captured = []
    reader = threading.Thread(target=lambda: captured.append(catalog.capture_snapshot(record.id)))

    prediction.start()
    assert classifier.started.wait(1.0)
    reader.start()
    reader.join(timeout=0.2)
    try:
        assert not reader.is_alive()
        assert captured[0].record.revision == record.revision
    finally:
        classifier.release.set()
        prediction.join(timeout=1.0)
        reader.join(timeout=1.0)


def test_append_build_does_not_block_prediction_and_swaps_revision_after_commit(tmp_path):
    classifier = SlowAppendClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7",
        [image(tmp_path / "front-slow.png", 10)],
        [image(tmp_path / "back-slow.png", 20)],
        False,
    )
    classifier.block_appends = True
    appended = []
    append_errors = []
    def append():
        try:
            appended.append(catalog.append_templates(
                record.id,
                [image(tmp_path / "front-confirmed.png", 11)],
                [],
                operation_id="append-nonblocking",
            ))
        except Exception as exc:
            append_errors.append(exc)

    append_thread = threading.Thread(target=append)
    prediction = []
    prediction_thread = threading.Thread(
        target=lambda: prediction.append(catalog.predict(record.id, tmp_path / "query.png"))
    )

    append_thread.start()
    assert classifier.build_started.wait(1.0)
    prediction_thread.start()
    prediction_thread.join(timeout=0.2)
    try:
        assert not prediction_thread.is_alive()
        assert prediction[0]["library_revision"] == record.revision
    finally:
        classifier.release_build.set()
        append_thread.join(timeout=2.0)
        prediction_thread.join(timeout=1.0)

    assert not append_errors
    assert appended
    assert catalog.predict(record.id, tmp_path / "query-after.png")["library_revision"] == record.revision + 1


def test_append_reports_committing_before_atomic_snapshot_swap(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    events = []

    def capture_progress(event):
        events.append({
            **event,
            "active_revision": catalog.capture_snapshot(record.id).record.revision,
        })

    appended, _ = catalog.append_templates(
        record.id,
        [image(tmp_path / "commit-progress.png", 11)],
        [],
        operation_id="commit-progress",
        progress_callback=capture_progress,
    )

    committing = [event for event in events if event["phase"] == "committing"]
    assert len(committing) == 1
    assert committing[0]["completed"] == 3
    assert committing[0]["total"] == 3
    assert committing[0]["active_revision"] == record.revision
    assert appended.revision == record.revision + 1


def test_committing_progress_observer_failure_does_not_abort_append(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    phases = []

    def failing_observer(event):
        phases.append(event["phase"])
        if event["phase"] == "committing":
            raise RuntimeError("observer failed")

    appended, _ = catalog.append_templates(
        record.id,
        [image(tmp_path / "observer-failure.png", 12)],
        [],
        operation_id="observer-failure",
        progress_callback=failing_observer,
    )

    assert "committing" in phases
    assert appended.revision == record.revision + 1


@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15), (35, 35)])
def test_prediction_stays_available_during_sized_background_append(
    tmp_path, front_count, back_count
):
    classifier = SlowAppendClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    front = [
        image(tmp_path / f"front-{index:02d}.png", index + 1)
        for index in range(front_count)
    ]
    back = [
        image(tmp_path / f"back-{index:02d}.png", index + 101)
        for index in range(back_count)
    ]
    record, _ = catalog.register(f"M-{front_count}-{back_count}", front, back, False)

    idle_started = time.perf_counter()
    idle_result = catalog.predict(record.id, tmp_path / "idle-query.png")
    idle_elapsed_ms = (time.perf_counter() - idle_started) * 1000.0

    classifier.block_appends = True
    append_errors = []
    append_results = []
    append_done = threading.Event()
    background_started = time.perf_counter()

    def append():
        try:
            append_results.append(
                catalog.append_templates(
                    record.id,
                    [image(tmp_path / "confirmed.png", 240)],
                    [],
                    operation_id=f"append-{front_count}-{back_count}",
                )
            )
        except Exception as exc:
            append_errors.append(exc)
        finally:
            append_done.set()

    append_thread = threading.Thread(target=append)
    append_thread.start()
    assert classifier.build_started.wait(10.0)

    online_arrived = time.perf_counter()
    concurrent_result = catalog.predict(record.id, tmp_path / "online-query.png")
    concurrent_elapsed_ms = (time.perf_counter() - online_arrived) * 1000.0
    assert append_thread.is_alive()
    assert idle_result["library_revision"] == record.revision
    assert concurrent_result["library_revision"] == record.revision

    classifier.release_build.set()
    assert append_done.wait(30.0)
    append_thread.join(timeout=1.0)
    background_elapsed_ms = (time.perf_counter() - background_started) * 1000.0

    assert not append_thread.is_alive()
    assert not append_errors
    assert append_results
    updated_revision = catalog.capture_snapshot(record.id).record.revision
    assert updated_revision == record.revision + 1
    print(
        "snapshot_timing "
        f"templates={front_count}+{back_count} "
        f"idle_prediction_ms={idle_elapsed_ms:.3f} "
        f"online_during_append_ms={concurrent_elapsed_ms:.3f} "
        f"service_arrival_to_result_ms={concurrent_elapsed_ms:.3f} "
        f"background_append_ms={background_elapsed_ms:.3f} "
        f"revision={record.revision}->{updated_revision}"
    )


def test_catalog_rejects_second_prepared_append_from_same_base_revision(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    first = catalog.library.prepare_append(
        record,
        [image(tmp_path / "first-confirmed.png", 11)],
        [],
        catalog.classifier.build_template_cache,
        operation_id="append-first",
    )
    second = catalog.library.prepare_append(
        record,
        [image(tmp_path / "second-confirmed.png", 12)],
        [],
        catalog.classifier.build_template_cache,
        operation_id="append-second",
    )
    try:
        committed = catalog.commit_prepared_append(first, first.candidate_cache)
        with pytest.raises(StaleWorkpieceRevisionError, match="revision changed"):
            catalog.commit_prepared_append(second, second.candidate_cache)
        snapshot = catalog.capture_snapshot(record.id)
        assert committed.revision == record.revision + 1
        assert len(snapshot.record.front_images) == len(record.front_images) + 1
    finally:
        catalog.library.abort_prepared(first)
        catalog.library.abort_prepared(second)


def test_recycle_removes_prediction_and_restore_republishes_same_workpiece(tmp_path):
    catalog, classifier, record = create_catalog(tmp_path)

    recycled = catalog.recycle(record.id, operation_id="delete-1")

    assert recycled["id"] == record.id
    assert catalog.list_workpieces() == []
    assert record.id not in classifier.caches
    assert catalog.list_recycled()[0]["id"] == record.id

    restored = catalog.restore(record.id, operation_id="restore-1")

    assert restored.id == record.id
    restored_summary = catalog.list_workpieces()[0]
    assert {"id": restored_summary["id"], "name": restored_summary["name"]} == {
        "id": record.id,
        "name": "M7",
    }
    assert record.id in classifier.caches


def test_restore_cache_miss_builds_fast_cache_for_recycled_library_revision(tmp_path):
    classifier = RevisionAwareFastClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7",
        [image(tmp_path / "front.png", 10)],
        [image(tmp_path / "back.png", 20)],
        False,
    )
    updated, _ = catalog.append_templates(
        record.id,
        [image(tmp_path / "front-extra.png", 11)],
        [],
        operation_id="append-revision-2",
    )
    catalog.recycle(record.id, operation_id="recycle-revision-2")

    restored = catalog.restore(record.id, operation_id="restore-revision-2")

    assert restored.revision == updated.revision + 2 == 4
    assert classifier.caches[record.id].fast_runtime.library_revision == restored.revision


@pytest.mark.parametrize("inference_mode", ["fast_geometry", "compare"])
def test_restore_queues_stale_fast_sidecar_rebuild_while_preserving_valid_base_cache(
    tmp_path,
    inference_mode,
):
    classifier = persisted_fast_classifier(inference_mode)
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7",
        [image(tmp_path / "front.png", 10)],
        [image(tmp_path / "back.png", 20)],
        False,
    )
    catalog.recycle(record.id, operation_id=f"recycle-{inference_mode}")
    recycled = catalog.library.get_recycled(record.id)
    base_path = recycled.root / ".template_cache.pkl"
    fast_path = recycled.root / ".fast_runtime_cache.pkl"

    loaded_base = classifier.load_template_cache(recycled)

    assert base_path.is_file()
    assert fast_path.is_file()
    assert loaded_base is not None
    assert loaded_base.fast_runtime is None

    restored = catalog.restore(record.id, operation_id=f"restore-{inference_mode}")
    catalog.shutdown()
    snapshot = catalog.capture_snapshot(record.id)
    persisted = classifier.load_template_cache(restored)

    assert restored.revision == 3
    assert snapshot.cache.fast_runtime is not None
    assert snapshot.cache.fast_runtime.library_revision == restored.revision
    assert persisted is not None
    assert persisted.fast_runtime is not None
    assert persisted.fast_runtime.library_revision == restored.revision
    result = classifier.predict_fast_with_cache(
        snapshot.cache,
        np.full((8, 8, 3), 10, dtype=np.uint8),
        library_revision=restored.revision,
    )
    assert result["inference_engine"] == "fast_geometry"


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


def test_annotation_commit_publishes_updated_runtime_snapshot_revision(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    original_cache = catalog.capture_snapshot(record.id).cache

    result = catalog.commit_annotation_document(
        record.id,
        [],
        expected_revision=record.revision,
        operation_id="annotation-revision-snapshot",
    )

    snapshot = catalog.capture_snapshot(record.id)
    assert snapshot.record.revision == result["revision"] == record.revision + 1
    assert snapshot.cache is original_cache
    assert catalog.predict(record.id, tmp_path / "query.png")["library_revision"] == result["revision"]


def test_legacy_annotation_save_publishes_masked_runtime_snapshot(tmp_path):
    catalog, classifier, record = create_catalog(tmp_path)
    group = _active_legacy_group()

    result = catalog.save_annotations(
        record.id,
        [group],
        operation_id="legacy-save-snapshot",
    )

    snapshot = catalog.capture_snapshot(record.id)
    assert snapshot.record.revision == result["revision"] == record.revision + 1
    assert snapshot.cache.ignored_regions["front"][0]
    assert classifier.get_template_cache(record.id) is snapshot.cache


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
    mapped = TemplateCache(
        global_vectors=original.global_vectors,
        local_features={"front": [{"keypoints": np.empty((1, 0, 2), dtype=np.float32)}], "back": [{}]},
    )
    classifier._template_caches = {"m7": mapped}
    ignored_regions = {"front": [[{"x": 0, "y": 0, "width": 4, "height": 4}]], "back": [[]]}

    candidate, statistics = classifier.prepare_template_masks(
        "m7", ignored_regions, base_cache=original
    )
    ignored_regions["front"][0][0]["x"] = 99

    assert classifier.get_template_cache("m7") is mapped
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
    front = [image(tmp_path / f"front-{index}.png", 10 + index) for index in range(5)]
    back = [image(tmp_path / f"back-{index}.png", 20 + index) for index in range(5)]
    record, expected_cache = initial_library.register("M7", front, back, False, builder)
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("template_inventory")
    manifest.pop("created_at")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    before = manifest_path.read_bytes()
    before_mtime = manifest_path.stat().st_mtime_ns
    classifier = CachingFakeClassifier(preloaded=expected_cache)
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    recovered = catalog.recover()
    details = catalog.get_workpiece_details(record.id)

    assert [item.id for item, _ in recovered] == [record.id]
    assert classifier.load_calls == [record.id]
    assert classifier.build_calls == 0
    assert classifier.save_calls == []
    assert len(details["templates"]) == 10
    assert {item["added_at"] for item in details["templates"]} == {None}
    assert manifest_path.read_bytes() == before
    assert manifest_path.stat().st_mtime_ns == before_mtime


def test_catalog_recover_resolves_fast_cache_staging_before_loading_cache(tmp_path):
    initial_library = WorkpieceLibrary(tmp_path / "library")
    record, expected_cache = initial_library.register(
        "M7",
        [image(tmp_path / "recovery-order-front.png", 10)],
        [image(tmp_path / "recovery-order-back.png", 20)],
        False,
        builder,
    )

    class RecoveryAwareClassifier(CachingFakeClassifier):
        def __init__(self):
            super().__init__(preloaded=expected_cache)
            self.recovery_order = []

        def recover_fast_runtime_cache_staging(self, candidate_record):
            self.recovery_order.append(("recover", candidate_record.id))

        def load_template_cache(self, candidate_record):
            self.recovery_order.append(("load", candidate_record.id))
            return super().load_template_cache(candidate_record)

    classifier = RecoveryAwareClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    catalog.recover()

    assert classifier.recovery_order == [("recover", record.id), ("load", record.id)]


def test_recover_does_not_hold_catalog_lock_during_geometry_rebuild(tmp_path):
    initial = WorkpieceLibrary(tmp_path / "library")
    record, _ = initial.register(
        "M7",
        [image(tmp_path / "recover-front.png", 10)],
        [image(tmp_path / "recover-back.png", 20)],
        False,
        builder,
    )
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)

    class BlockingProfiles:
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def rebuild_active_cache(self, workpiece_id, candidate_record):
            self.started.set()
            assert self.release.wait(2.0)
            return None

        def sync_library_revision(self, candidate_record):
            return None

    profiles = BlockingProfiles()
    catalog.set_geometry_profiles(profiles)
    recover_errors = []

    def recover():
        try:
            catalog.recover()
        except Exception as exc:
            recover_errors.append(exc)

    recovery = threading.Thread(target=recover)
    recovery.start()
    assert profiles.started.wait(1.0)
    captured = []
    reader = threading.Thread(target=lambda: captured.append(catalog.capture_snapshot(record.id)))
    reader.start()
    reader.join(timeout=0.2)
    try:
        assert not reader.is_alive()
        assert captured[0].record.revision == record.revision
    finally:
        profiles.release.set()
        recovery.join(timeout=2.0)
        reader.join(timeout=1.0)
    assert not recover_errors


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


def _geometry_profile():
    anchor = {
        "shape": "ellipse",
        "mode": "auto",
        "coarse": {
            "cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4, "angle_deg": 0.0,
        },
    }
    calibration = {
        "state": "ready",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5, "angle_deg": 0.0},
        "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5, "angle_deg": 0.0},
        "diagnostics": {},
    }
    return {
        "schema_version": 2,
        "rules": [{
            "rule_id": "glare", "name": "反光", "shape": "circle",
            "mode": "inside", "margin_ratio": 0.02,
            "margin_semantics": "signed_boundary_v2", "enabled": True,
        }],
        "directions": {
            "front": {
                "anchor": anchor,
                "calibrations": {"glare": calibration},
                "template_reviews": {},
            },
            "back": {
                "anchor": anchor,
                "calibrations": {"glare": calibration},
                "template_reviews": {},
            },
        },
        "migration": {"source_schema_version": None, "conflicts": [], "resolutions": []},
    }


def test_recover_rebuilds_active_geometry_cache_and_archives_legacy_masks(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    classifier = FakeClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register(
        "M7", [image(tmp_path / "front.png", 10)], [image(tmp_path / "back.png", 20)], False
    )
    profiles = GeometryMaskProfiles(catalog, start_worker=False, storage_dir=tmp_path / "geometry-jobs")
    catalog.set_geometry_profiles(profiles)
    draft = profiles.save_draft(
        record.id, _geometry_profile(), expected_library_revision=record.revision,
        expected_draft_revision=0, operation_id="draft-1"
    )
    job = profiles.start_validation(
        record.id, expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"], operation_id="validate-1"
    )
    profiles.run_next(force=True)
    profiles.publish(
        record.id, job["job_id"], expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"], operation_id="publish-1"
    )
    manifest_path = catalog.get(record.id).root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["interference_groups"] = [{"group_id": "legacy", "annotations": []}]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restarted_classifier = FakeClassifier()
    restarted_catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), restarted_classifier)
    restarted_profiles = GeometryMaskProfiles(
        restarted_catalog, start_worker=False, storage_dir=tmp_path / "geometry-jobs-restarted"
    )
    restarted_catalog.set_geometry_profiles(restarted_profiles)
    restarted_catalog.recover()

    cache = restarted_classifier.get_template_cache(record.id)
    assert cache.geometry_profile_revision == 1
    assert cache.ignored_regions in ({}, None)
    assert restarted_catalog.get_annotation_snapshot(record.id)["legacy_archived"] is True


def test_append_preserves_active_geometry_and_updates_staged_library_revision(tmp_path):
    catalog, classifier, record = create_catalog(tmp_path)
    profiles = GeometryMaskProfiles(
        catalog,
        calibrator=object(),
        start_worker=False,
        storage_dir=tmp_path / "geometry-append-jobs",
    )
    catalog.set_geometry_profiles(profiles)
    draft = profiles.save_draft(
        record.id,
        _geometry_profile(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="append-geometry-draft",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"],
        operation_id="append-geometry-validate",
    )
    profiles.run_next(force=True)
    profiles.publish(
        record.id,
        job["job_id"],
        expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"],
        operation_id="append-geometry-publish",
    )
    published_record = catalog.get(record.id)
    revision_path = published_record.root / "geometry_masks" / "revisions" / "1.json"
    immutable_revision = revision_path.read_bytes()

    appended, _ = catalog.append_templates(
        record.id,
        [image(tmp_path / "geometry-confirmed.png", 11)],
        [],
        operation_id="append-with-geometry",
    )

    snapshot = catalog.capture_snapshot(record.id)
    profile_document = json.loads(
        (appended.root / "geometry_masks" / "profile.json").read_text(encoding="utf-8")
    )
    assert snapshot.record.revision == published_record.revision + 1
    assert snapshot.cache.geometry_profile_revision == 1
    assert profile_document["library_revision"] == snapshot.record.revision
    assert revision_path.read_bytes() == immutable_revision
    assert classifier.get_template_cache(record.id) is snapshot.cache


def _active_legacy_group():
    return {
        "group_id": "legacy", "name": "旧标注", "enabled": True,
        "propagation": {"state": "active"},
        "annotations": [{
            "orientation": "front", "index": 0, "status": "active",
            "regions": [{"x": 0, "y": 0, "width": 2, "height": 2}],
        }],
    }


def test_recover_applies_active_legacy_groups_when_geometry_manager_exists(tmp_path):
    catalog, _, record = create_catalog(tmp_path)
    group = _active_legacy_group()
    catalog.commit_annotation_document(
        record.id, [group], expected_revision=record.revision,
        operation_id="legacy-active", active_groups=[group],
    )
    restarted_classifier = FakeClassifier()
    restarted_catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), restarted_classifier)
    profiles = GeometryMaskProfiles(restarted_catalog, start_worker=False,
                                    storage_dir=tmp_path / "geometry-jobs")
    restarted_catalog.set_geometry_profiles(profiles)

    restarted_catalog.recover()

    cache = restarted_classifier.get_template_cache(record.id)
    assert cache.ignored_regions
    assert cache.geometry_profile is None


def test_first_geometry_publish_can_rollback_to_legacy_cache(tmp_path):
    catalog, classifier, record = create_catalog(tmp_path)
    group = _active_legacy_group()
    catalog.commit_annotation_document(
        record.id, [group], expected_revision=record.revision,
        operation_id="legacy-active", active_groups=[group],
    )
    record = catalog.get(record.id)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")
    catalog.set_geometry_profiles(profiles)
    draft = profiles.save_draft(
        record.id, _geometry_profile(), expected_library_revision=record.revision,
        expected_draft_revision=0, operation_id="draft-geometry",
    )
    job = profiles.start_validation(
        record.id, expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"], operation_id="validate-geometry",
    )
    profiles.run_next(force=True)
    published = profiles.publish(
        record.id, job["job_id"], expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"], operation_id="publish-geometry",
    )
    assert published["active_revision"] == 1

    rolled = profiles.rollback(
        record.id, expected_library_revision=record.revision + 1,
        operation_id="rollback-legacy",
    )
    assert rolled["active_revision"] is None
    assert classifier.get_template_cache(record.id).ignored_regions


def test_legacy_rollback_persists_none_geometry_sidecar_with_production_validator(tmp_path):
    classifier = persisted_fast_classifier("fast_geometry")
    catalog = WorkpieceCatalog(WorkpieceLibrary(tmp_path / "library"), classifier)
    record, _ = catalog.register(
        "M7",
        [image(tmp_path / "front.png", 10)],
        [image(tmp_path / "back.png", 20)],
        False,
    )
    catalog.shutdown()
    group = _active_legacy_group()
    catalog.commit_annotation_document(
        record.id,
        [group],
        expected_revision=record.revision,
        operation_id="production-legacy-active",
        active_groups=[group],
    )
    record = catalog.get(record.id)
    profiles = GeometryMaskProfiles(
        catalog,
        start_worker=False,
        storage_dir=tmp_path / "production-geometry-jobs",
    )
    catalog.set_geometry_profiles(profiles)
    profile = profiles.snapshot(record.id)["draft"]
    profile_root = record.root / "geometry_masks"
    (profile_root / "revisions").mkdir(parents=True, exist_ok=True)
    profile_path = profile_root / "profile.json"
    profile_document = {
        "schema_version": profile["schema_version"],
        "library_revision": record.revision + 1,
        "draft_revision": 0,
        "active_revision": 1,
        "previous_active_revision": None,
        "draft": profile,
        "active": profile,
    }
    profile_path.write_text(json.dumps(profile_document), encoding="utf-8")
    (profile_root / "revisions" / "1.json").write_text(
        json.dumps({
            "profile": profile,
            "previous_active_revision": None,
            "previous_source": "legacy",
        }),
        encoding="utf-8",
    )
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
        operation_id="production-publish-geometry",
    )
    assert classifier.load_fast_runtime_cache(published) is not None

    before = catalog.capture_snapshot(record.id)
    original_stage = classifier.stage_fast_runtime_cache

    def fail_stage(*_args, **_kwargs):
        raise RuntimeError("staged legacy persistence failed")

    classifier.stage_fast_runtime_cache = fail_stage
    with pytest.raises(RuntimeError, match="staged legacy persistence failed"):
        catalog.restore_legacy_annotation_cache(
            record.id,
            expected_revision=published.revision,
            operation_id="production-rollback-stage-failure",
        )
    assert catalog.capture_snapshot(record.id) is before
    assert catalog.get(record.id).revision == published.revision
    classifier.stage_fast_runtime_cache = original_stage

    restored = catalog.restore_legacy_annotation_cache(
        record.id,
        expected_revision=published.revision,
        operation_id="production-rollback-legacy",
    )
    profile_document.update({
        "library_revision": restored.revision,
        "active_revision": None,
        "previous_active_revision": None,
        "active": None,
    })
    profile_path.write_text(json.dumps(profile_document), encoding="utf-8")

    runtime = classifier.load_fast_runtime_cache(restored)
    assert runtime is not None
    assert runtime.library_revision == restored.revision
    assert runtime.geometry_profile_revision is None
    profiles.shutdown()
