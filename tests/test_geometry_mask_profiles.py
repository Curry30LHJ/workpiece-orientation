import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from src import geometry_mask_profiles as geometry_profiles_module
from src.geometry_mask_profiles import (
    FittedGeometryMissingError,
    GeometryCacheRevisionMismatchError,
    GeometryProfilePublishError,
    GeometryMaskProfiles,
    InvalidGeometryProfileError,
    MissingDirectionCalibrationError,
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
            direction_profile = profile.get("directions", {}).get(label, {})
            reviews = direction_profile.get("template_reviews", {}) if isinstance(direction_profile, dict) else {}
            for index, _ in enumerate(paths):
                completed += 1
                template_id = f"{label}:{paths[index].name}"
                review = reviews.get(template_id, {}) if isinstance(reviews, dict) else {}
                report[label].append({"index": index, "template_id": template_id,
                                      "status": "active", "review_state": review.get("state", "included"),
                                      "review_reason": review.get("reason", ""),
                                      "ignored_ratio": 0.1, "remaining_ratio": 0.9})
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


def v2_profile_fixture():
    logical_rule = {
        "rule_id": "glare",
        "name": "中心反光",
        "shape": "circle",
        "mode": "inside",
        "margin_ratio": -0.04,
        "margin_semantics": "signed_boundary_v2",
        "enabled": True,
    }
    anchor = {
        "shape": "circle",
        "mode": "auto",
        "coarse": {"cx": 0.5, "cy": 0.5, "r": 0.45, "angle_deg": 0.0},
    }
    return {
        "schema_version": 2,
        "rules": [logical_rule],
        "directions": {
            "front": {
                "anchor": anchor,
                "calibrations": {
                    "glare": {
                        "state": "ready",
                        "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.70, "angle_deg": 0.0},
                        "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": 0.80, "angle_deg": 0.0},
                        "reference_template": {
                            "template_id": "front:00.png", "direction": "front",
                            "width": 360, "height": 360,
                        },
                        "diagnostics": {},
                    }
                },
                "template_reviews": {},
            },
            "back": {
                "anchor": anchor,
                "calibrations": {
                    "glare": {
                        "state": "ready",
                        "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.64, "angle_deg": 0.0},
                        "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": 0.76, "angle_deg": 0.0},
                        "reference_template": {
                            "template_id": "back:00.png", "direction": "back",
                            "width": 360, "height": 360,
                        },
                        "diagnostics": {},
                    }
                },
                "template_reviews": {},
            },
        },
        "migration": {"source_schema_version": None, "conflicts": [], "resolutions": []},
    }


def v2_profile_missing_back():
    profile = v2_profile_fixture()
    profile["directions"]["back"]["calibrations"] = {}
    return profile


def v2_profile_missing_fitted_geometry():
    profile = v2_profile_fixture()
    calibration = profile["directions"]["back"]["calibrations"]["glare"]
    calibration["state"] = "needs_review"
    calibration.pop("geometry")
    return profile


@pytest.mark.parametrize(
    ("profile_factory", "code", "error_type"),
    [
        (v2_profile_missing_back, "MISSING_DIRECTION_CALIBRATION", MissingDirectionCalibrationError),
        (v2_profile_missing_fitted_geometry, "FITTED_GEOMETRY_MISSING", FittedGeometryMissingError),
    ],
)
def test_enabled_v2_rule_validation_requires_both_ready_calibrations(
    tmp_path: Path, profile_factory, code: str, error_type,
):
    catalog, _, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(
        catalog, calibrator=object(), start_worker=False, storage_dir=tmp_path / "jobs"
    )
    saved = profiles.save_draft(
        record.id,
        profile_factory(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id=f"save-{code}",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"],
        operation_id=f"validate-{code}",
    )

    completed = profiles.run_next(force=True)

    assert completed["job_id"] == job["job_id"]
    assert any(item["code"] == code for item in completed["blocking_issues"])
    with pytest.raises(error_type, match=code):
        profiles.publish(
            record.id,
            job["job_id"],
            expected_library_revision=record.revision,
            expected_draft_revision=saved["draft_revision"],
            operation_id=f"publish-{code}",
        )


def test_v2_validation_blocks_cache_revision_mismatch(tmp_path: Path):
    catalog, classifier, record = make_geometry_catalog(tmp_path)
    original_prepare = classifier.prepare_geometry_cache

    def mismatched_prepare(*args, **kwargs):
        candidate, report = original_prepare(*args, **kwargs)
        return replace(candidate, geometry_profile_revision=999), report

    classifier.prepare_geometry_cache = mismatched_prepare
    profiles = GeometryMaskProfiles(
        catalog, calibrator=object(), start_worker=False, storage_dir=tmp_path / "jobs"
    )
    saved = profiles.save_draft(
        record.id,
        v2_profile_fixture(),
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="save-cache-mismatch",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"],
        operation_id="validate-cache-mismatch",
    )

    completed = profiles.run_next(force=True)

    assert any(
        item["code"] == "PROFILE_CACHE_REVISION_MISMATCH"
        for item in completed["blocking_issues"]
    )
    with pytest.raises(GeometryCacheRevisionMismatchError, match="PROFILE_CACHE_REVISION_MISMATCH"):
        profiles.publish(
            record.id,
            job["job_id"],
            expected_library_revision=record.revision,
            expected_draft_revision=saved["draft_revision"],
            operation_id="publish-cache-mismatch",
        )


def legacy_pair_fixture():
    profile = circle_profile()
    profile["schema_version"] = 1
    front_rule = profile["directions"]["front"]["rules"][0]
    front_rule.update({"rule_id": "front-glare", "name": "中心反光",
                       "margin_ratio": -0.04, "margin_semantics": "signed_boundary_v2"})
    back_rule = dict(front_rule)
    back_rule.update({
        "rule_id": "back-glare",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.64, "angle_deg": 0.0},
    })
    profile["directions"]["back"] = {
        "anchor": profile["directions"]["front"]["anchor"],
        "rules": [back_rule],
    }
    return profile


def legacy_front_zero_back_four_fixture():
    profile = legacy_pair_fixture()
    anchor = profile["directions"]["back"]["anchor"]
    base = profile["directions"]["back"]["rules"][0]
    glare_a = {**base, "rule_id": "back-glare-a"}
    glare_b = {**base, "rule_id": "back-glare-b"}
    intrusion_a = {
        **base,
        "rule_id": "back-intrusion-a",
        "name": "四周侵入",
        "mode": "outside",
        "geometry": {"cx": 0.0, "cy": 0.0, "r": 1.34, "angle_deg": 0.0},
    }
    intrusion_b = {**intrusion_a, "rule_id": "back-intrusion-b"}
    profile["directions"] = {
        "front": {"anchor": None, "rules": []},
        "back": {"anchor": anchor, "rules": [glare_a, glare_b, intrusion_a, intrusion_b]},
    }
    return profile


def test_v2_uses_one_logical_rule_with_two_direction_calibrations():
    normalize_profile_v2 = getattr(geometry_profiles_module, "normalize_profile_v2")

    profile = normalize_profile_v2(v2_profile_fixture())

    assert [rule["rule_id"] for rule in profile["rules"]] == ["glare"]
    assert profile["directions"]["front"]["calibrations"]["glare"]["geometry"]["r"] == 0.70
    assert profile["directions"]["back"]["calibrations"]["glare"]["geometry"]["r"] == 0.64


def test_unique_legacy_front_back_rules_are_paired_without_copying_geometry():
    migrate_profile_v1 = getattr(geometry_profiles_module, "migrate_profile_v1")

    migrated = migrate_profile_v1(legacy_pair_fixture())

    assert len(migrated["rules"]) == 1
    rule_id = migrated["rules"][0]["rule_id"]
    assert migrated["directions"]["front"]["calibrations"][rule_id]["geometry"]["r"] == 0.5
    assert migrated["directions"]["back"]["calibrations"][rule_id]["geometry"]["r"] == 0.64
    assert migrated["migration"]["conflicts"] == []


def test_legacy_duplicates_are_preserved_and_reported_as_conflicts():
    migrate_profile_v1 = getattr(geometry_profiles_module, "migrate_profile_v1")

    migrated = migrate_profile_v1(legacy_front_zero_back_four_fixture())

    assert len(migrated["rules"]) == 4
    source_ids = {
        item["source_rule_id"]
        for conflict in migrated["migration"]["conflicts"]
        for item in conflict["sources"]
    }
    assert source_ids == {
        "back-glare-a", "back-glare-b", "back-intrusion-a", "back-intrusion-b",
    }
    assert all(
        rule["rule_id"] in migrated["directions"]["back"]["calibrations"]
        for rule in migrated["rules"]
    )


def test_saving_v2_draft_keeps_legacy_active_unchanged(tmp_path: Path):
    library, record = make_library(tmp_path)
    profile_path = record.root / "geometry_masks" / "profile.json"
    legacy_active = legacy_pair_fixture()
    legacy_document = {
        "schema_version": 1,
        "library_revision": record.revision,
        "draft_revision": 2,
        "active_revision": 1,
        "previous_active_revision": None,
        "draft": legacy_active,
        "active": legacy_active,
    }
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(legacy_document, ensure_ascii=False), encoding="utf-8")
    profiles = GeometryMaskProfiles(library, start_worker=False)

    saved = profiles.save_draft(
        record.id,
        v2_profile_fixture(),
        expected_library_revision=record.revision,
        expected_draft_revision=2,
        operation_id="save-v2-draft",
    )

    stored = json.loads(profile_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 2
    assert stored["draft"]["schema_version"] == 2
    assert stored["active"] == legacy_active
    assert saved["active"] == legacy_active


def test_keep_only_resolution_removes_explicit_duplicate_and_records_audit():
    resolve_migration_conflict = getattr(
        geometry_profiles_module, "resolve_migration_conflict"
    )
    migrate_profile_v1 = getattr(geometry_profiles_module, "migrate_profile_v1")
    profile = migrate_profile_v1(legacy_front_zero_back_four_fixture())
    conflict = next(
        item for item in profile["migration"]["conflicts"]
        if {source["source_rule_id"] for source in item["sources"]}
        == {"back-glare-a", "back-glare-b"}
    )

    resolved = resolve_migration_conflict(
        profile,
        conflict["conflict_id"],
        {"action": "keep_only", "survivor_rule_id": "back-glare-a"},
    )

    assert {rule["rule_id"] for rule in resolved["rules"]} == {
        "back-glare-a", "back-intrusion-a", "back-intrusion-b",
    }
    assert "back-glare-b" not in resolved["directions"]["back"]["calibrations"]
    assert len(resolved["migration"]["conflicts"]) == 1
    assert len(resolved["migration"]["resolutions"]) == 1
    audit = resolved["migration"]["resolutions"][0]
    assert audit["conflict_id"] == conflict["conflict_id"]
    assert audit["resolution"] == {
        "action": "keep_only", "survivor_rule_id": "back-glare-a",
    }
    assert audit["resolved_at"] > 0


def test_unresolved_migration_conflict_blocks_publish(tmp_path: Path):
    catalog, _, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(
        catalog, calibrator=object(), start_worker=False, storage_dir=tmp_path / "jobs"
    )
    migrate_profile_v1 = getattr(geometry_profiles_module, "migrate_profile_v1")
    conflicted = migrate_profile_v1(legacy_front_zero_back_four_fixture())
    saved = profiles.save_draft(
        record.id,
        conflicted,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="save-conflicted-v2",
    )
    job = profiles.start_validation(
        record.id,
        expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"],
        operation_id="validate-conflicted-v2",
    )
    completed = profiles.run_next(force=True)
    assert completed["state"] == "completed"

    with pytest.raises(GeometryProfilePublishError, match="MIGRATION_CONFLICT"):
        profiles.publish(
            record.id,
            job["job_id"],
            expected_library_revision=record.revision,
            expected_draft_revision=saved["draft_revision"],
            operation_id="publish-conflicted-v2",
        )


def test_resolve_migration_is_revisioned_idempotent_and_keeps_active(tmp_path: Path):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)
    migrate_profile_v1 = getattr(geometry_profiles_module, "migrate_profile_v1")
    conflicted = migrate_profile_v1(legacy_front_zero_back_four_fixture())
    saved = profiles.save_draft(
        record.id,
        conflicted,
        expected_library_revision=record.revision,
        expected_draft_revision=0,
        operation_id="save-before-resolve",
    )
    conflict = next(
        item for item in saved["draft"]["migration"]["conflicts"]
        if {source["source_rule_id"] for source in item["sources"]}
        == {"back-glare-a", "back-glare-b"}
    )
    resolution = {"action": "keep_only", "survivor_rule_id": "back-glare-a"}

    first = profiles.resolve_migration(
        record.id,
        conflict["conflict_id"],
        resolution,
        expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"],
        operation_id="resolve-glare",
    )
    second = profiles.resolve_migration(
        record.id,
        conflict["conflict_id"],
        resolution,
        expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"],
        operation_id="resolve-glare",
    )

    assert first == second
    assert first["draft_revision"] == saved["draft_revision"] + 1
    assert first["active"] is None
    assert len(first["draft"]["migration"]["conflicts"]) == 1


def test_old_library_has_empty_geometry_profile_without_manifest_migration(tmp_path: Path):
    library, record = make_library(tmp_path)
    profiles = GeometryMaskProfiles(library, start_worker=False)

    snapshot = profiles.snapshot(record.id)

    assert snapshot["library_revision"] == record.revision
    assert snapshot["draft_revision"] == 0
    assert snapshot["active_revision"] is None
    assert snapshot["draft"]["schema_version"] == 2
    assert snapshot["draft"]["rules"] == []
    assert snapshot["draft"]["directions"]["front"]["calibrations"] == {}
    assert snapshot["draft"]["directions"]["back"]["calibrations"] == {}
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
    assert first["draft"]["rules"][0]["rule_id"] == "inner-glare"
    assert first["draft"]["rules"][0]["margin_semantics"] == "signed_boundary_v2"
    assert "inner-glare" in first["draft"]["directions"]["front"]["calibrations"]
    assert first["draft"]["directions"]["back"]["calibrations"] == {}
    document = json.loads((record.root / "geometry_masks" / "profile.json").read_text(encoding="utf-8"))
    assert document["draft_revision"] == 1
    assert document["draft"]["directions"]["front"]["rules"][0]["margin_semantics"] == "signed_boundary_v2"
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


def test_profile_preserves_reference_seed_and_template_reviews():
    profile = circle_profile()
    front = profile["directions"]["front"]
    front["reference_template"] = {
        "template_id": "front:00.png", "direction": "front",
        "width": 512, "height": 512,
    }
    front["anchor"]["mode"] = "auto"
    front["rules"][0]["seed_geometry"] = dict(front["rules"][0]["geometry"])
    front["rules"][0]["editor_state"] = "ready"
    front["template_reviews"] = {
        "front:00.png": {"state": "excluded", "reason": "边缘遮挡严重"}
    }

    from src.geometry_mask_profiles import normalize_geometry_profile
    normalized = normalize_geometry_profile(profile)

    assert normalized["directions"]["front"]["reference_template"]["width"] == 512
    assert normalized["directions"]["front"]["anchor"]["mode"] == "auto"
    assert normalized["directions"]["front"]["rules"][0]["seed_geometry"]
    assert normalized["directions"]["front"]["rules"][0]["editor_state"] == "ready"
    assert normalized["directions"]["front"]["template_reviews"]["front:00.png"]["state"] == "excluded"


def test_signed_rule_keeps_negative_margin_and_marker():
    from src.geometry_mask_profiles import normalize_geometry_profile

    profile = circle_profile()
    rule = profile["directions"]["front"]["rules"][0]
    rule.update({"margin_ratio": -0.02, "margin_semantics": "signed_boundary_v2"})

    normalized = normalize_geometry_profile(profile)
    result = normalized["directions"]["front"]["rules"][0]

    assert result["margin_ratio"] == pytest.approx(-0.02)
    assert result["margin_semantics"] == "signed_boundary_v2"


def test_front_and_back_signed_offsets_are_normalized_independently():
    from src.geometry_mask_profiles import normalize_geometry_profile

    profile = circle_profile()
    profile["directions"]["back"] = {
        "anchor": profile["directions"]["front"]["anchor"],
        "rules": [{
            "rule_id": "back-edge",
            "name": "反面边界",
            "shape": "circle",
            "geometry": {"cx": 0, "cy": 0, "r": 0.5},
            "mode": "outside",
            "margin_ratio": 0.02,
            "margin_semantics": "signed_boundary_v2",
            "enabled": True,
        }],
    }
    profile["directions"]["front"]["rules"][0].update({
        "margin_ratio": 0.02,
        "margin_semantics": "signed_boundary_v2",
    })

    normalized = normalize_geometry_profile(profile)

    assert normalized["directions"]["front"]["rules"][0]["margin_ratio"] == pytest.approx(0.02)
    assert normalized["directions"]["back"]["rules"][0]["margin_ratio"] == pytest.approx(0.02)
    assert normalized["directions"]["front"]["rules"][0]["mode"] == "inside"
    assert normalized["directions"]["back"]["rules"][0]["mode"] == "outside"


@pytest.mark.parametrize(
    ("mode", "legacy_margin", "expected"),
    [("inside", 0.02, 0.02), ("outside", 0.02, -0.02), ("outside", 0.0, 0.0)],
)
def test_unmarked_legacy_margin_is_migrated_by_mode(mode, legacy_margin, expected):
    from src.geometry_mask_profiles import normalize_geometry_profile

    profile = circle_profile()
    rule = profile["directions"]["front"]["rules"][0]
    rule["mode"] = mode
    rule["margin_ratio"] = legacy_margin
    rule.pop("margin_semantics", None)

    normalized = normalize_geometry_profile(profile)
    migrated = normalized["directions"]["front"]["rules"][0]

    assert migrated["margin_ratio"] == pytest.approx(expected)
    assert migrated["margin_semantics"] == "signed_boundary_v2"


def test_missing_legacy_margin_uses_old_default_but_new_marker_defaults_to_zero():
    from src.geometry_mask_profiles import normalize_geometry_profile

    legacy = circle_profile()
    legacy["directions"]["front"]["rules"][0].pop("margin_ratio", None)
    old_result = normalize_geometry_profile(legacy)
    assert old_result["directions"]["front"]["rules"][0]["margin_ratio"] == pytest.approx(0.02)

    new = circle_profile()
    new_rule = new["directions"]["front"]["rules"][0]
    new_rule.pop("margin_ratio", None)
    new_rule["margin_semantics"] = "signed_boundary_v2"
    new_result = normalize_geometry_profile(new)
    assert new_result["directions"]["front"]["rules"][0]["margin_ratio"] == pytest.approx(0.0)


def test_validation_blocking_issues_reject_critical_geometry_and_regression():
    report = {
        "front": [{"index": 0, "ignored_ratio": 0.55, "remaining_ratio": 0.20}],
        "back": [],
    }
    regression = {"status": "completed", "skipped": 0, "correct_to_wrong": 1}

    issues = GeometryMaskProfiles._validation_blocking_issues(report, regression)
    codes = {item["code"] for item in issues}

    assert {"geometry_mask_too_large", "keypoint_retention_critical", "geometry_regression"} <= codes


def test_validation_blocking_issues_expose_paired_fusion_regression_templates():
    report = {"front": [], "back": []}
    change = {
        "template_id": "front:22.png",
        "expected": "front",
        "baseline_predicted": "front",
        "candidate_predicted": "back",
        "candidate_global_prediction": "front",
        "candidate_local_prediction": "back",
        "candidate_decision_source": "local_override",
        "cause": "geometry_local_override",
    }
    regression = {
        "status": "completed",
        "skipped": 0,
        "correct_to_wrong": 1,
        "changed_predictions": [change],
    }

    issues = GeometryMaskProfiles._validation_blocking_issues(report, regression)

    issue = next(item for item in issues if item["code"] == "geometry_fusion_regression")
    assert issue["templates"] == ["front:22.png"]
    assert issue["changed_predictions"] == [change]


def test_validation_blocking_issues_do_not_block_absolute_candidate_error():
    regression = {
        "status": "completed",
        "skipped": 0,
        "correct_to_wrong": 0,
        "wrong_to_wrong": 1,
        "changed_predictions": [],
    }

    issues = GeometryMaskProfiles._validation_blocking_issues(
        {"front": [], "back": []}, regression
    )

    assert not any(item["code"] in {"geometry_regression", "geometry_fusion_regression"}
                   for item in issues)


def test_validation_blocking_issues_reject_incomplete_leave_one_out():
    report = {"front": [], "back": []}
    regression = {
        "status": "incomplete",
        "skipped": 1,
        "correct_to_wrong": 0,
        "fit_failures": [{"template_id": "front:09.png", "reason": "unsafe_template_geometry"}],
    }

    issues = GeometryMaskProfiles._validation_blocking_issues(report, regression)

    issue = next(item for item in issues if item["code"] == "leave_one_out_incomplete")
    assert issue["templates"] == ["front:09.png"]


def test_validation_warning_thresholds_remain_nonblocking():
    report = {
        "front": [{"index": 0, "status": "active", "ignored_ratio": 0.41, "remaining_ratio": 0.29}],
        "back": [],
    }

    warnings = GeometryMaskProfiles._validation_warnings(report)
    issues = GeometryMaskProfiles._validation_blocking_issues(
        report, {"status": "completed", "skipped": 0, "correct_to_wrong": 0}
    )

    assert {item["code"] for item in warnings} == {
        "effective_area_low", "keypoint_retention_low"
    }
    assert issues == []


def test_old_profile_snapshot_migrates_without_rewriting_source(tmp_path: Path):
    library, record = make_library(tmp_path)
    profile_path = record.root / "geometry_masks" / "profile.json"
    legacy = {
        "schema_version": 1,
        "library_revision": record.revision,
        "draft_revision": 1,
        "active_revision": None,
        "previous_active_revision": None,
        "draft": circle_profile(),
        "active": None,
    }
    legacy["draft"]["directions"]["front"]["rules"][0]["mode"] = "outside"
    legacy["draft"]["directions"]["front"]["rules"][0].pop("margin_ratio", None)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    before = profile_path.read_bytes()

    profiles = GeometryMaskProfiles(library, start_worker=False)
    snapshot = profiles.snapshot(record.id)

    assert snapshot["draft"]["schema_version"] == 2
    assert [rule["rule_id"] for rule in snapshot["draft"]["rules"]] == ["inner-glare"]
    rule = snapshot["draft"]["rules"][0]
    assert rule["margin_ratio"] == pytest.approx(-0.02)
    assert rule["margin_semantics"] == "signed_boundary_v2"
    assert "inner-glare" in snapshot["draft"]["directions"]["front"]["calibrations"]
    assert snapshot["draft"]["directions"]["back"]["calibrations"] == {}
    assert profile_path.read_bytes() == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda profile: profile["directions"]["front"]["rules"][0].update(
            {"margin_semantics": "unknown_v9"}
        ),
        lambda profile: profile["directions"]["front"]["rules"][0].update(
            {"margin_ratio": -0.02}
        ),
    ],
)
def test_invalid_margin_semantics_are_rejected(mutate):
    from src.geometry_mask_profiles import normalize_geometry_profile

    profile = circle_profile()
    mutate(profile)
    with pytest.raises(InvalidGeometryProfileError):
        normalize_geometry_profile(profile)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda profile: profile["directions"]["front"].update(
            {"template_reviews": {"front:00.png": {"state": "excluded", "reason": ""}}}
        ),
        lambda profile: profile["directions"]["front"].update(
            {"template_reviews": {"back:back.png": {"state": "review", "reason": "遮挡"}}}
        ),
        lambda profile: profile["directions"]["front"].update(
            {"template_reviews": {"front:00.png": {"state": "unknown", "reason": "原因"}}}
        ),
        lambda profile: profile["directions"]["front"]["rules"][0].update({"editor_state": "unknown"}),
        lambda profile: profile["directions"]["front"]["rules"][0].update(
            {"editor_state": "needs_reseed", "enabled": True}
        ),
    ],
)
def test_invalid_editor_metadata_is_rejected(mutate):
    from src.geometry_mask_profiles import normalize_geometry_profile
    profile = circle_profile()
    mutate(profile)
    with pytest.raises(InvalidGeometryProfileError):
        normalize_geometry_profile(profile)


def test_preview_rule_checks_revision_and_does_not_mutate_profile(tmp_path: Path):
    catalog, _, record = make_geometry_catalog(tmp_path)
    calibrator = Mock()
    calibrator.fit_reference.return_value = {
        "status": "active",
        "anchor_fit": {"candidates": [], "selected_candidate_index": 0},
        "rule_fit": {"candidates": [], "selected_candidate_index": 0},
        "profile_patch": {
            "anchor": {"shape": "ellipse", "mode": "auto",
                       "coarse": {"cx": 0.5, "cy": 0.5, "rx": 0.4,
                                  "ry": 0.4, "angle_deg": 0.0}},
            "geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5},
            "seed_geometry": {"cx": 0.0, "cy": 0.0, "r": 0.5},
        },
        "fit_duration_ms": 1.0,
    }
    profiles = GeometryMaskProfiles(
        catalog, calibrator=calibrator, start_worker=False,
        storage_dir=tmp_path / "jobs",
    )
    before = profiles.snapshot(record.id)

    preview = profiles.preview_rule(
        record.id,
        expected_library_revision=record.revision,
        rule_id="glare",
        direction="front",
        template_id="front:00.png",
        seed_shape={"shape": "circle", "cx": 16.0, "cy": 16.0, "r": 10.0},
        mode="inside",
        margin_ratio=0.02,
    )

    assert preview["rule_id"] == "glare"
    assert preview["direction"] == "front"
    assert preview["template_id"] == "front:00.png"
    assert preview["base_library_revision"] == record.revision
    assert preview["profile_patch"]["reference_template"]["direction"] == "front"
    assert profiles.snapshot(record.id) == before
    calibrator.fit_reference.assert_called_once()


def test_preview_rule_returns_low_confidence_when_no_boundary_was_fitted(tmp_path: Path):
    catalog, _, record = make_geometry_catalog(tmp_path)
    calibrator = Mock()
    calibrator.fit_reference.return_value = {
        "status": "low_confidence",
        "anchor_fit": {"candidates": [], "selected_candidate_index": None},
        "rule_fit": {"candidates": [], "selected_candidate_index": None},
        "fit_duration_ms": 1.0,
    }
    profiles = GeometryMaskProfiles(
        catalog, calibrator=calibrator, start_worker=False,
        storage_dir=tmp_path / "jobs",
    )

    preview = profiles.preview_rule(
        record.id,
        expected_library_revision=record.revision,
        rule_id="glare",
        direction="back",
        template_id="back:00.png",
        seed_shape={"shape": "circle", "cx": 16.0, "cy": 16.0, "r": 10.0},
        mode="inside",
        margin_ratio=-0.04,
    )

    assert preview["status"] == "low_confidence"
    assert preview["rule_id"] == "glare"
    assert preview["direction"] == "back"
    assert preview["template_id"] == "back:00.png"
    assert preview["base_library_revision"] == record.revision
    assert "profile_patch" not in preview


@pytest.mark.parametrize("review_state", ["review", "excluded"])
def test_template_review_state_reaches_validation_gate(tmp_path: Path, review_state: str):
    catalog, _, record = make_geometry_catalog(tmp_path)
    profiles = GeometryMaskProfiles(catalog, calibrator=object(), start_worker=False,
                                    storage_dir=tmp_path / "jobs")
    draft = v2_profile_fixture()
    draft["directions"]["front"]["template_reviews"] = {
        "front:00.png": {"state": review_state, "reason": "边缘遮挡"}
    }
    saved = profiles.save_draft(
        record.id, draft, expected_library_revision=record.revision,
        expected_draft_revision=0, operation_id=f"draft-{review_state}",
    )
    job = profiles.start_validation(
        record.id, expected_library_revision=record.revision,
        expected_draft_revision=saved["draft_revision"], operation_id=f"validate-{review_state}",
    )
    completed = profiles.run_next(force=True)
    assert completed["state"] == "completed"
    if review_state == "review":
        assert completed["blocking_issues"][0]["code"] == "template_needs_review"
        with pytest.raises(GeometryProfilePublishError, match="blocking issues"):
            profiles.publish(
                record.id, job["job_id"], expected_library_revision=record.revision,
                expected_draft_revision=saved["draft_revision"], operation_id="publish-review",
                override_reason="已检查",
            )
    else:
        assert completed["blocking_issues"] == []
        assert any(item["code"] == "template_excluded" for item in completed["warnings"])


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
        v2_profile_fixture(),
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
        v2_profile_fixture(),
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
