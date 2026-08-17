import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.geometry_mask_profiles import (
    GeometryMaskProfiles,
    InvalidGeometryProfileError,
    StaleGeometryProfileError,
)
from src.orientation_classifier import TemplateCache
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

