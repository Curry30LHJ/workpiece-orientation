import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.geometry_mask_profiles import (
    GeometryProfilePublishError,
    GeometryMaskProfiles,
    InvalidGeometryProfileError,
    StaleGeometryProfileError,
)
from src.orientation_classifier import TemplateCache
from src.workpiece_catalog import WorkpieceCatalog
from src.workpiece_library import WorkpieceLibrary


def write_image(path: Path, marker: int) -> Path:
    image = np.full((32, 32, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def fake_builder(front, back, progress_callback=None):
    return TemplateCache(
        global_vectors={
            "front": np.zeros((len(front), 2), dtype=np.float32),
            "back": np.ones((len(back), 2), dtype=np.float32),
        },
        local_features={"front": [{} for _ in front], "back": [{} for _ in back]},
    )


class GeometryFakeClassifier:
    def __init__(self):
        self.caches = {}

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def get_template_cache(self, workpiece_id):
        return self.caches.get(workpiece_id)

    def build_template_cache(self, front, back, progress_callback=None):
        return fake_builder(front, back, progress_callback)

    def prepare_geometry_cache(self, workpiece_id, record, profile, calibrator=None, progress_callback=None):
        base = self.caches[workpiece_id]
        total = len(record.front_images) + len(record.back_images)
        completed = 0
        report = {"front": [], "back": []}
        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            for index, _ in enumerate(paths):
                completed += 1
                report[label].append({"index": index, "status": "active", "ignored_ratio": 0.1,
                                      "remaining_ratio": 0.9})
                if progress_callback is not None:
                    progress_callback(label, index + 1, len(paths))
        candidate = TemplateCache(
            global_vectors=base.global_vectors,
            local_features=base.local_features,
            raw_global_vectors=base.raw_global_vectors or base.global_vectors,
            raw_local_features=base.raw_local_features or base.local_features,
            geometry_profile=profile,
            geometry_profile_revision=profile.get("profile_revision"),
            geometry_template_report=report,
        )
        return candidate, report


def make_geometry_catalog(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "library")
    classifier = GeometryFakeClassifier()
    catalog = WorkpieceCatalog(library, classifier)
    record, _ = catalog.register(
        "M7",
        [write_image(tmp_path / "front.png", 10)],
        [write_image(tmp_path / "back.png", 20)],
        False,
    )
    return catalog, classifier, record


def make_library(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7",
        [write_image(tmp_path / "front.png", 10)],
        [write_image(tmp_path / "back.png", 20)],
        False,
        fake_builder,
    )
    return library, record


def circle_profile():
    return {
        "directions": {
            "front": {
                "anchor": {
                    "shape": "ellipse",
                    "coarse": {"cx": 0.5, "cy": 0.5, "rx": 0.4, "ry": 0.4, "angle_deg": 0},
                },
                "rules": [
                    {
                        "rule_id": "inner-glare",
                        "name": "内腔反光",
                        "shape": "circle",
                        "geometry": {"cx": 0, "cy": 0, "r": 0.5},
                        "mode": "inside",
                        "margin_ratio": 0.02,
                        "enabled": True,
                    }
                ],
            },
            "back": {"anchor": None, "rules": []},
        }
    }


def test_old_library_has_empty_geometry_profile_without_manifest_migration(tmp_path: Path):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)

    snapshot = profiles.snapshot(record.id)

    assert snapshot["library_revision"] == record.revision
    assert snapshot["draft_revision"] == 0
    assert snapshot["active_revision"] is None
    assert snapshot["draft"]["directions"]["front"]["rules"] == []
    assert not (record.root / "geometry_masks" / "profile.json").exists()


def test_save_draft_requires_both_revisions_and_is_idempotent(tmp_path: Path):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)
    draft = circle_profile()

    first = profiles.save_draft(
        record.id,
        draft,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="op-1",
    )
    second = profiles.save_draft(
        record.id,
        draft,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="op-1",
    )

    assert first == second
    assert first["draft_revision"] == 1
    assert first["draft"]["directions"]["front"]["rules"][0]["rule_id"] == "inner-glare"
    document = json.loads((record.root / "geometry_masks" / "profile.json").read_text(encoding="utf-8"))
    assert document["draft_revision"] == 1
    assert (record.root / "geometry_masks" / "revisions").is_dir()
    assert (record.root / "geometry_masks" / "previews").is_dir()


def test_stale_draft_does_not_mutate_profile(tmp_path: Path):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)
    first = profiles.save_draft(
        record.id,
        circle_profile(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="op-1",
    )
    path = record.root / "geometry_masks" / "profile.json"
    before = path.read_bytes()

    with pytest.raises(StaleGeometryProfileError):
        profiles.save_draft(
            record.id,
            circle_profile(),
            expected_library_revision=record.revision,
            expected_draft_revision=0,
            operation_id="op-2",
        )

    assert path.read_bytes() == before
    assert profiles.snapshot(record.id)["draft_revision"] == first["draft_revision"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda draft: draft["directions"]["front"].update(
            {"anchor": {"shape": "triangle", "coarse": {"cx": 0.5, "cy": 0.5}}}
        ),
        lambda draft: draft["directions"]["front"]["rules"].append(
            {
                "rule_id": "inner-glare",
                "name": "重复",
                "shape": "circle",
                "geometry": {"r": 0.2},
                "mode": "inside",
                "margin_ratio": 0,
                "enabled": True,
            }
        ),
        lambda draft: draft["directions"]["front"]["rules"][0].update({"mode": "sideways"}),
    ],
)
def test_invalid_geometry_profile_is_rejected_without_file(tmp_path: Path, mutate):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)
    draft = circle_profile()
    mutate(draft)

    with pytest.raises(InvalidGeometryProfileError):
        profiles.save_draft(
            record.id,
            draft,
            expected_library_revision=record.revision,
            expected_draft_revision=0,
            operation_id="bad-op",
        )

    assert not (record.root / "geometry_masks" / "profile.json").exists()


def test_corrupt_profile_is_isolated_from_old_library(tmp_path: Path):
    library, record = make_library(tmp_path)
    profile_path = record.root / "geometry_masks" / "profile.json"
    profile_path.parent.mkdir()
    profile_path.write_text("{not-json", encoding="utf-8")
    profiles = GeometryMaskProfiles(library, start_worker=False)

    snapshot = profiles.snapshot(record.id)

    assert snapshot["profile_status"] == "corrupt"
    assert snapshot["draft_revision"] == 0


def test_validation_returns_job_immediately_and_exposes_template_progress(tmp_path: Path):
    catalog, classifier, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")

    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="validate-1",
    )

    assert job["state"] == "queued"
    assert job["job_id"]
    completed = profiles.run_next(force=True)
    assert completed is not None
    assert completed["state"] == "completed"
    assert completed["progress"]["completed"] == 2
    assert completed["report"]["front"][0]["status"] == "active"
    assert classifier.get_template_cache(record.id) is not None


def test_restart_marks_running_geometry_job_interrupted_but_keeps_draft(tmp_path: Path):
    catalog, _, record = make_geometry_catalog(tmp_path)
    storage = tmp_path / "jobs"
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False, storage_dir=storage)
    profiles.save_draft(
        record.id,
        circle_profile(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="draft-1",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=1,
        operation_id="validate-1",
    )
    payload = json.loads((storage / "jobs.json").read_text(encoding="utf-8"))
    payload["jobs"][0]["state"] = "running"
    (storage / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    restarted = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False, storage_dir=storage)

    assert restarted.get_job(job["job_id"])["state"] == "interrupted"
    assert restarted.snapshot(record.id)["draft_revision"] == 1


def test_publish_failure_keeps_old_cache_and_active_pointer(tmp_path: Path, monkeypatch):
    catalog, classifier, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")
    draft = profiles.save_draft(
        record.id,
        circle_profile(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="draft-1",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"],
        operation_id="validate-1",
    )
    profiles.run_next(force=True)
    old_cache = classifier.get_template_cache(record.id)

    def fail_publish(*args, **kwargs):
        raise RuntimeError("candidate swap failed")

    monkeypatch.setattr(catalog, "publish_geometry_profile", fail_publish, raising=False)
    with pytest.raises(GeometryProfilePublishError):
        profiles.publish(
            record.id,
            job["job_id"],
            expected_library_revision=record.revision,
            expected_draft_revision=draft["draft_revision"],
            operation_id="publish-1",
        )

    assert classifier.get_template_cache(record.id) is old_cache
    assert profiles.snapshot(record.id)["active_revision"] is None


def test_publish_and_rollback_swap_manifest_revision_and_runtime_cache(tmp_path: Path):
    catalog, classifier, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")
    draft = profiles.save_draft(
        record.id,
        circle_profile(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="draft-1",
    )
    first_job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"],
        operation_id="validate-1",
    )
    profiles.run_next(force=True)
    first = profiles.publish(
        record.id,
        first_job["job_id"],
        expected_library_revision=record.revision,
        expected_draft_revision=draft["draft_revision"],
        operation_id="publish-1",
    )

    assert first["active_revision"] == 1
    assert catalog.get(record.id).revision == 2
    first_cache = classifier.get_template_cache(record.id)

    second_draft = profiles.save_draft(
        record.id,
        circle_profile(),
        expected_library_revision=2,
        expected_draft_revision=1,
        operation_id="draft-2",
    )
    second_job = profiles.start_validation(
        record.id,
        expected_library_revision=2,
        expected_draft_revision=second_draft["draft_revision"],
        operation_id="validate-2",
    )
    profiles.run_next(force=True)
    second = profiles.publish(
        record.id,
        second_job["job_id"],
        expected_library_revision=2,
        expected_draft_revision=second_draft["draft_revision"],
        operation_id="publish-2",
    )

    assert second["active_revision"] == 2
    assert second["previous_active_revision"] == 1
    assert classifier.get_template_cache(record.id) is not first_cache

    rolled_back = profiles.rollback(record.id, expected_library_revision=3, operation_id="rollback-1")

    assert rolled_back["active_revision"] == 1
    assert rolled_back["previous_active_revision"] is None
    assert catalog.get(record.id).revision == 4
