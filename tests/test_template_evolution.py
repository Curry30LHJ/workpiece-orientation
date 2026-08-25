from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from src.orientation_classifier import PropagationModelError, TemplateCache
from src.template_evolution import DuplicateTemplateError, TemplateEvolution
from src.interference_masks import InvalidMaskError
from src.workpiece_catalog import WorkpieceCatalog
from src.workpiece_library import StaleWorkpieceRevisionError, WorkpieceLibrary


def image(path: Path, marker: int) -> Path:
    assert cv2.imwrite(str(path), np.full((8, 8, 3), marker, dtype=np.uint8))
    return path


class FakeClassifier:
    def __init__(self):
        self.caches = {}

    def build_template_cache(self, front, back, progress_callback=None):
        return TemplateCache(
            global_vectors={
                "front": np.zeros((len(front), 2), dtype=np.float32),
                "back": np.ones((len(back), 2), dtype=np.float32),
            },
            local_features={"front": [{} for _ in front], "back": [{} for _ in back]},
        )

    def set_template_cache(self, workpiece_id, cache):
        self.caches[workpiece_id] = cache

    def set_template_masks(self, workpiece_id, masks):
        self.masks = (workpiece_id, masks)

    def get_template_cache(self, workpiece_id):
        return None

    def project_region_between_templates(self, source_path, target_path, region):
        return dict(region)


def setup_catalog(tmp_path, front_count=1):
    classifier = FakeClassifier()
    library = WorkpieceLibrary(tmp_path / "library")
    catalog = WorkpieceCatalog(library, classifier)
    front = [image(tmp_path / f"front-{index}.png", 10 + index) for index in range(front_count)]
    back = image(tmp_path / "back.png", 20)
    record, _ = catalog.register("M7", front, [back], False)
    return catalog, record


def test_confirmations_for_same_workpiece_coalesce_and_append(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    first = image(tmp_path / "confirmed-front.png", 30)
    second = image(tmp_path / "confirmed-back.png", 40)
    now = [100.0]
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", clock=lambda: now[0], start_worker=False)

    job1 = evolution.submit_confirmation(record.id, "front", first, operation_id="confirm-1")
    now[0] += 2.0
    job2 = evolution.submit_confirmation(record.id, "back", second, operation_id="confirm-2")

    assert job2["job_id"] == job1["job_id"]
    assert len(evolution.get_job(job1["job_id"])["items"]) == 2
    evolution.run_next()

    updated = catalog.get(record.id)
    assert updated.revision == record.revision + 1
    assert len(updated.front_images) == 2
    assert len(updated.back_images) == 2
    assert evolution.get_job(job1["job_id"])["state"] == "completed"


def test_restart_reconciles_manifest_commit_without_appending_twice(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    storage = tmp_path / "restart-jobs"
    evolution = TemplateEvolution(catalog, storage, start_worker=False)
    job = evolution.submit_confirmation(
        record.id,
        "front",
        image(tmp_path / "restart-confirmed.png", 31),
        operation_id="restart-confirmation",
    )
    staged = Path(evolution.get_job(job["job_id"])["items"][0]["path"])
    committed, _ = catalog.append_templates(
        record.id,
        [staged],
        [],
        operation_id=job["job_id"],
    )
    with evolution._condition:
        evolution._jobs[job["job_id"]]["state"] = "building"
        evolution._persist()

    restarted = TemplateEvolution(catalog, storage, start_worker=False)
    result = restarted.run_next(force=True)

    assert result["state"] == "completed"
    assert result["revision"] == committed.revision
    assert len(catalog.get(record.id).front_images) == 2
    assert not staged.exists()


def test_restart_rejects_unrelated_revision_instead_of_reappending(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    storage = tmp_path / "unrelated-jobs"
    evolution = TemplateEvolution(catalog, storage, start_worker=False)
    job = evolution.submit_confirmation(
        record.id,
        "front",
        image(tmp_path / "queued-confirmed.png", 32),
        operation_id="queued-confirmation",
    )
    catalog.append_templates(
        record.id,
        [image(tmp_path / "unrelated-confirmed.png", 33)],
        [],
        operation_id="different-operation",
    )
    with evolution._condition:
        evolution._jobs[job["job_id"]]["state"] = "building"
        evolution._persist()

    restarted = TemplateEvolution(catalog, storage, start_worker=False)
    result = restarted.run_next(force=True)

    assert result["state"] == "failed"
    assert result["error"] == "workpiece revision changed without a matching template operation"
    assert len(catalog.get(record.id).front_images) == 2


def test_corrupt_jobs_document_is_quarantined_without_disabling_catalog(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    storage = tmp_path / "corrupt-jobs"
    storage.mkdir()
    jobs_path = storage / "jobs.json"
    jobs_path.write_text("{not-json", encoding="utf-8")

    evolution = TemplateEvolution(catalog, storage, start_worker=False)

    quarantined = list(storage.glob("jobs.corrupt-*.json"))
    assert evolution.list_jobs() == []
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == "{not-json"
    assert not jobs_path.exists()
    assert catalog.get(record.id).id == record.id


def test_completed_job_records_its_job_id_in_template_manifest(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "operation-jobs", start_worker=False)
    job = evolution.submit_confirmation(
        record.id,
        "front",
        image(tmp_path / "operation-confirmed.png", 34),
        operation_id="operation-confirmation",
    )

    result = evolution.run_next(force=True)

    manifest = json.loads(
        (catalog.get(record.id).root / "manifest.json").read_text(encoding="utf-8")
    )
    assert result["state"] == "completed"
    assert manifest["last_template_update"]["operation_id"] == job["job_id"]


class LowConfidenceGeometryProfiles:
    def validate_new_template(self, workpiece_id, orientation, image_path):
        return {
            "status": "low_confidence",
            "needs_review": True,
            "reason": "synthetic boundary fit failure",
        }


def test_confirmed_template_waits_for_geometry_review_before_append(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    new_image = image(tmp_path / "low-confidence-front.png", 31)
    evolution = TemplateEvolution(
        catalog,
        tmp_path / "jobs",
        start_worker=False,
        geometry_profiles=LowConfidenceGeometryProfiles(),
    )

    job = evolution.submit_confirmation(record.id, "front", new_image, operation_id="confirm-low")
    completed = evolution.run_next(force=True)

    assert completed["state"] == "needs_review"
    assert completed["review_reason"] == "geometry_mask_low_confidence"
    assert len(catalog.get(record.id).front_images) == 1


def test_exact_duplicate_is_rejected_before_queue(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    duplicate = catalog.get(record.id).front_images[0]
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    with pytest.raises(DuplicateTemplateError):
        evolution.submit_confirmation(record.id, "front", duplicate, operation_id="confirm-duplicate")


def test_confirmation_during_build_creates_successor_after_predecessor(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    first = image(tmp_path / "during-front.png", 50)
    second = image(tmp_path / "after-front.png", 60)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first_job = evolution.submit_confirmation(record.id, "front", first, operation_id="during-1")
    with evolution._condition:
        evolution._jobs[first_job["job_id"]]["state"] = "building"
    successor = evolution.submit_confirmation(record.id, "front", second, operation_id="during-2")

    assert successor["job_id"] != first_job["job_id"]
    assert successor["predecessor_job_id"] == first_job["job_id"]
    evolution._jobs[first_job["job_id"]]["state"] = "completed"
    evolution.run_next()
    assert evolution.get_job(successor["job_id"])["state"] == "completed"


def test_saving_two_seed_annotations_persists_group_and_activates_seed_masks(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    result = evolution.save_annotations(
        record.id,
        [{
            "group_id": "glare",
            "name": "边缘反光",
            "annotations": [
                {"orientation": "front", "index": 0, "regions": [{"x": 0, "y": 0, "width": 1, "height": 1}], "trusted": True},
                {"orientation": "back", "index": 0, "regions": [{"x": 0, "y": 0, "width": 1, "height": 1}], "trusted": True},
            ],
        }],
        expected_revision=record.revision,
        operation_id="annotation-1",
    )

    assert result["groups"][0]["group_id"] == "glare"
    manifest = (catalog.get(record.id).root / "manifest.json").read_text(encoding="utf-8")
    assert "边缘反光" in manifest


def test_two_trusted_seeds_propagate_to_unmarked_template(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    result = evolution.save_annotations(
        record.id,
        [{
            "group_id": "edge",
            "name": "边缘",
            "annotations": [
                {"orientation": "front", "index": 0, "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
                {"orientation": "front", "index": 1, "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
            ],
        }],
        expected_revision=record.revision,
        operation_id="annotation-propagate",
    )

    group = result["groups"][0]
    assert group["propagation"]["state"] == "active"
    assert any(item.get("index") == 2 and item.get("provenance") == "automatic"
               for item in group["annotations"])


def test_two_trusted_seeds_report_propagation_progress(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    progress = []

    evolution.save_annotations(
        record.id,
        [{
            "group_id": "edge",
            "name": "边缘",
            "annotations": [
                {"orientation": "front", "index": 0,
                 "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
                {"orientation": "front", "index": 1,
                 "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
            ],
        }],
        expected_revision=record.revision,
        operation_id="annotation-propagate-progress",
        progress_callback=progress.append,
    )

    assert progress[0] == {"phase": "propagating_annotations", "completed": 0, "total": 4}
    assert progress[-1] == {"phase": "propagating_annotations", "completed": 4, "total": 4}


def test_correction_by_template_id_becomes_a_propagation_seed(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "orientation": "front", "index": 0,
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True,
        }]}],
        expected_revision=record.revision,
        operation_id="annotation-first-seed",
    )

    result = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "front:01.png", "review_action": "correct",
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}],
        }]}],
        expected_revision=first["revision"],
        operation_id="annotation-second-seed",
    )

    annotations = result["groups"][0]["annotations"]
    second = next(item for item in annotations if item.get("template_id") == "front:01.png")
    assert (second["orientation"], second["index"]) == ("front", 1)
    assert any(item.get("template_id") == "front:02.png"
               and item.get("provenance") == "automatic" for item in annotations)


def test_recoverable_projection_exception_marks_only_target_and_continues(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=4)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "orientation": "front", "index": 0,
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True,
        }]}],
        expected_revision=record.revision,
        operation_id="projection-failure-seed-0",
    )

    def project(source_path, target_path, region):
        if Path(target_path).name == "02.png":
            raise RuntimeError("synthetic pair failure")
        return dict(region)

    catalog.classifier.project_region_between_templates = project
    result = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "front:01.png", "review_action": "correct",
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}],
        }]}],
        expected_revision=first["revision"],
        operation_id="projection-failure-seed-1",
    )

    group = result["groups"][0]
    failed = next(item for item in group["targets"] if item["template_id"] == "front:02.png")
    succeeded = next(item for item in group["targets"] if item["template_id"] == "front:03.png")
    assert failed["state"] == "unresolved"
    assert failed["diagnostics"]["reason_code"] == "projection_failed"
    assert failed["diagnostics"]["projection_failure_count"] == 2
    assert succeeded["provenance"] == "automatic"
    assert result["active_annotation_revision"] == first["active_annotation_revision"]


def test_repeated_projection_exceptions_abort_without_committing(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=4)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "orientation": "front", "index": 0,
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True,
        }]}],
        expected_revision=record.revision,
        operation_id="projection-budget-seed-0",
    )

    def project(source_path, target_path, region):
        raise RuntimeError("synthetic repeated pair failure")

    catalog.classifier.project_region_between_templates = project
    with pytest.raises(PropagationModelError, match="递推失败次数过多"):
        evolution.save_annotations(
            record.id,
            [{"group_id": "edge", "annotations": [{
                "template_id": "front:01.png", "review_action": "correct",
                "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}],
            }]}],
            expected_revision=first["revision"],
            operation_id="projection-budget-seed-1",
        )

    snapshot = evolution.get_annotations(record.id)
    assert snapshot["revision"] == first["revision"]
    assert snapshot["active_annotation_revision"] == first["active_annotation_revision"]


def test_target_geometry_exception_isolated_from_later_targets(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=4)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "orientation": "front", "index": 0,
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True,
        }]}],
        expected_revision=record.revision,
        operation_id="target-geometry-seed-0",
    )

    def project(source_path, target_path, region):
        if Path(target_path).name == "02.png":
            return {"x": 1}
        return dict(region)

    catalog.classifier.project_region_between_templates = project
    result = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "front:01.png", "review_action": "correct",
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}],
        }]}],
        expected_revision=first["revision"],
        operation_id="target-geometry-seed-1",
    )

    group = result["groups"][0]
    failed = next(item for item in group["targets"] if item["template_id"] == "front:02.png")
    succeeded = next(item for item in group["targets"] if item["template_id"] == "front:03.png")
    assert failed["state"] == "unresolved"
    assert failed["diagnostics"]["reason_code"] == "projection_failed"
    assert succeeded["provenance"] == "automatic"


def test_invalid_annotation_rectangle_is_rejected(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    with pytest.raises(InvalidMaskError):
        evolution.save_annotations(
            record.id,
            [{"group_id": "bad", "name": "bad", "annotations": [
                {"orientation": "front", "index": 0, "trusted": True,
                 "regions": [{"x": 0, "y": 0, "width": 0, "height": 2}]},
                {"orientation": "back", "index": 0, "trusted": True,
                 "regions": [{"x": 0, "y": 0, "width": 1, "height": 1}]},
            ]}],
            expected_revision=record.revision,
            operation_id="annotation-invalid",
        )


def test_propagation_diagnostics_include_source_lineage_and_safety_metrics(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    result = evolution.save_annotations(
        record.id,
        [{
            "group_id": "edge",
            "name": "边缘",
            "annotations": [
                {"orientation": "front", "index": 0,
                 "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
                {"orientation": "front", "index": 1,
                 "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
            ],
        }],
        expected_revision=record.revision,
        operation_id="annotation-diagnostics",
    )

    automatic = next(item for item in result["groups"][0]["annotations"] if item.get("provenance") == "automatic")
    diagnostics = automatic["diagnostics"]
    assert diagnostics["reason_code"] == "sources_agree"
    assert diagnostics["source_template_ids"] == ["front:00.png", "front:01.png"]
    assert diagnostics["attempted_source_count"] == 2
    assert diagnostics["successful_projection_count"] == 2
    assert diagnostics["area_ratio"] > 0.0


def test_delete_group_preserves_other_group_and_removes_draft_group(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    first = evolution.save_annotations(
        record.id,
        [{"group_id": "glare", "name": "反光", "annotations": []}],
        expected_revision=record.revision,
        operation_id="annotation-glare",
    )
    second = evolution.save_annotations(
        record.id,
        [{"group_id": "shadow", "name": "阴影", "annotations": []}],
        expected_revision=first["revision"],
        operation_id="annotation-shadow",
    )

    result = evolution.delete_group(
        record.id,
        "glare",
        expected_revision=second["revision"],
        operation_id="annotation-delete-glare",
    )

    assert [group["group_id"] for group in result["groups"]] == ["shadow"]


def test_stale_annotation_save_does_not_change_snapshot(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)

    first = evolution.save_annotations(
        record.id,
        [{"group_id": "glare", "name": "反光", "annotations": []}],
        expected_revision=record.revision,
        operation_id="annotation-first",
    )
    with pytest.raises(StaleWorkpieceRevisionError, match="revision"):
        evolution.save_annotations(
            record.id,
            [{"group_id": "shadow", "name": "阴影", "annotations": []}],
            expected_revision=record.revision,
            operation_id="annotation-stale",
        )
    assert [group["group_id"] for group in evolution.get_annotations(record.id)["groups"]] == ["glare"]


def test_review_accept_and_absent_actions_update_target_state(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    seeded = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "name": "边缘", "annotations": [
            {"orientation": "front", "index": 0,
             "template_id": "front:00.png",
             "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
            {"orientation": "front", "index": 1,
             "template_id": "front:01.png",
             "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
        ]}],
        expected_revision=record.revision,
        operation_id="review-seed",
    )
    accepted = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "front:02.png", "review_action": "accept", "regions": [],
        }]}],
        expected_revision=seeded["revision"],
        operation_id="review-accept",
    )
    target = next(item for item in accepted["groups"][0]["annotations"]
                  if item.get("template_id") == "front:02.png")
    assert target["trusted"] is True
    assert target["status"] == "active"
    assert target["provenance"] == "accepted"

    absent = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "back:00.png", "review_action": "absent", "regions": [],
        }]}],
        expected_revision=accepted["revision"],
        operation_id="review-absent",
    )
    target = next(item for item in absent["groups"][0]["annotations"]
                  if item.get("template_id") == "back:00.png")
    assert target["status"] == "confirmed_absent"
    assert target["regions"] == []


def test_repropagate_keeps_manual_seed_annotation(tmp_path):
    catalog, record = setup_catalog(tmp_path, front_count=3)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    seeded = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "name": "边缘", "annotations": [
            {"orientation": "front", "index": 0,
             "template_id": "front:00.png",
             "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
            {"orientation": "front", "index": 1,
             "template_id": "front:01.png",
             "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}], "trusted": True},
        ]}],
        expected_revision=record.revision,
        operation_id="repropagate-seed",
    )

    result = evolution.save_annotations(
        record.id,
        [{"group_id": "edge", "annotations": [{
            "template_id": "front:00.png", "review_action": "repropagate",
            "regions": [{"x": 1, "y": 1, "width": 2, "height": 2}],
        }]}],
        expected_revision=seeded["revision"],
        operation_id="repropagate-manual-seed",
    )

    target = next(item for item in result["groups"][0]["annotations"]
                  if item.get("template_id") == "front:00.png")
    assert target["trusted"] is True
    assert target["regions"] == [{"x": 1, "y": 1, "width": 2, "height": 2}]
    propagated = next(item for item in result["groups"][0]["annotations"]
                      if item.get("template_id") == "front:02.png")
    assert propagated["provenance"] == "automatic"


def test_disabling_group_removes_only_that_group_from_active_snapshot(tmp_path):
    catalog, record = setup_catalog(tmp_path)
    evolution = TemplateEvolution(catalog, tmp_path / "jobs", start_worker=False)
    saved = evolution.save_annotations(
        record.id,
        [{"group_id": "glare", "name": "反光", "annotations": []}],
        expected_revision=record.revision,
        operation_id="disable-seed",
    )

    disabled = evolution.set_group_enabled(
        record.id, "glare", False,
        expected_revision=saved["revision"], operation_id="disable-glare",
    )

    group = disabled["groups"][0]
    assert group["enabled"] is False
    assert group["active_state"] == "disabled"
