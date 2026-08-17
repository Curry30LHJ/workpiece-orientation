import json
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from src.orientation_classifier import TemplateCache
from src.workpiece_library import (
    FeatureBuildError,
    InvalidTemplateSetError,
    InvalidWorkpieceNameError,
    StaleWorkpieceRevisionError,
    WorkpieceExistsError,
    WorkpieceLibrary,
)


def write_image(path: Path, marker: int) -> Path:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def image_set(tmp_path: Path, prefix: str, marker: int, count: int) -> list[Path]:
    return [write_image(tmp_path / f"{prefix}-{index}.png", marker + index) for index in range(count)]


def fake_builder(front: list[Path], back: list[Path], progress_callback=None) -> TemplateCache:
    return TemplateCache(
        global_vectors={
            "front": np.zeros((len(front), 2), dtype=np.float32),
            "back": np.ones((len(back), 2), dtype=np.float32),
        },
        local_features={"front": [{} for _ in front], "back": [{} for _ in back]},
    )


def test_register_creates_uuid_manifest_and_label_folders(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "library")
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 20, 5)

    record, cache = library.register("M7", front, back, False, fake_builder)

    assert len(record.id) == 32
    assert cache.global_vectors["front"].shape == (5, 2)
    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "M7"
    assert manifest["labels"] == {"0": "front", "1": "back"}
    assert manifest["template_counts"] == {"front": 5, "back": 5}
    assert len(list((record.root / "0").glob("*"))) == 5
    assert len(list((record.root / "1").glob("*"))) == 5


def test_register_accepts_unicode_source_paths(tmp_path: Path):
    source_front = image_set(tmp_path, "front", 10, 5)
    source_back = image_set(tmp_path, "back", 20, 5)
    unicode_root = tmp_path / "中文工件" / "待建库"
    unicode_root.mkdir(parents=True)
    front = []
    back = []
    for index, source in enumerate(source_front):
        target = unicode_root / f"正面-{index}.png"
        shutil.copy2(source, target)
        front.append(target)
    for index, source in enumerate(source_back):
        target = unicode_root / f"反面-{index}.png"
        shutil.copy2(source, target)
        back.append(target)

    record, _ = WorkpieceLibrary(tmp_path / "library").register("M7", front, back, False, fake_builder)

    assert record.name == "M7"


def test_register_rejects_existing_name_without_replace(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 20, 5)
    old, _ = library.register("M7", front, back, False, fake_builder)

    with pytest.raises(WorkpieceExistsError):
        library.register("m7", front, back, False, fake_builder)

    assert library.get(old.id).root == old.root
    assert library.list_workpieces() == [{"id": old.id, "name": "M7"}]


def test_failed_replace_preserves_old_directory(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "lib")
    old_front = image_set(tmp_path, "old-front", 10, 5)
    old_back = image_set(tmp_path, "old-back", 20, 5)
    old, _ = library.register("M7", old_front, old_back, False, fake_builder)
    new_front = image_set(tmp_path, "new-front", 30, 5)
    new_back = image_set(tmp_path, "new-back", 40, 5)

    def raising_builder(front, back, progress_callback=None):
        raise FeatureBuildError("feature extraction failed")

    with pytest.raises(FeatureBuildError):
        library.register("M7", new_front, new_back, True, raising_builder)

    current = library.get(old.id)
    assert current.root == old.root
    assert cv2.imread(str(current.front_images[0]))[0, 0, 0] == 10
    assert not list((tmp_path / "lib").glob(".staging-*"))


@pytest.mark.parametrize("name", ["", "  ", "a/b", "a\\b", "a\x00b", "a\n b"])
def test_invalid_names_are_rejected(name, tmp_path):
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 20, 5)

    with pytest.raises(InvalidWorkpieceNameError):
        WorkpieceLibrary(tmp_path / "lib").register(name, front, back, False, fake_builder)


def test_duplicate_template_path_is_rejected(tmp_path):
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 20, 5)
    front[1] = front[0]

    with pytest.raises(InvalidTemplateSetError):
        WorkpieceLibrary(tmp_path / "lib").register("M7", front, back, False, fake_builder)


def test_empty_template_set_is_rejected(tmp_path):
    front = []
    back = image_set(tmp_path, "back", 20, 5)

    with pytest.raises(InvalidTemplateSetError):
        WorkpieceLibrary(tmp_path / "lib").register("M7", front, back, False, fake_builder)


def test_recover_skips_corrupt_record_but_loads_valid_record(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    valid_front = image_set(tmp_path, "valid-front", 10, 5)
    valid_back = image_set(tmp_path, "valid-back", 20, 5)
    valid, _ = library.register("M7", valid_front, valid_back, False, fake_builder)
    corrupt = tmp_path / "lib" / "corrupt"
    (corrupt / "0").mkdir(parents=True)
    (corrupt / "1").mkdir()
    (corrupt / "manifest.json").write_text("not-json", encoding="utf-8")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(record.id, record.name) for record, _ in recovered] == [(valid.id, "M7")]


def test_recover_rebuilds_each_valid_cache_once(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    library.register("M7", image_set(tmp_path, "m7-front", 10, 5), image_set(tmp_path, "m7-back", 20, 5), False, fake_builder)
    library.register("M8", image_set(tmp_path, "m8-front", 30, 5), image_set(tmp_path, "m8-back", 40, 5), False, fake_builder)
    calls = []

    def counting_builder(front, back, progress_callback=None):
        calls.append((tuple(front), tuple(back)))
        return fake_builder(front, back, progress_callback)

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(counting_builder)

    assert len(recovered) == 2
    assert len(calls) == 2


def test_recover_uses_valid_cache_loader_without_rebuilding(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, expected_cache = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )
    loaded = []

    def cache_loader(candidate):
        loaded.append(candidate.id)
        return expected_cache

    def unexpected_builder(front, back, progress_callback=None):
        raise AssertionError("cache hit must not rebuild template features")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(
        unexpected_builder,
        cache_loader=cache_loader,
    )

    assert [(item.id, item.name) for item, _ in recovered] == [(record.id, "M7")]
    assert loaded == [record.id]
    assert recovered[0][1] is expected_cache


def test_recover_rebuilds_and_saves_when_cache_loader_misses(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )
    rebuilt = []
    saved = []

    def counting_builder(front, back, progress_callback=None):
        rebuilt.append((tuple(front), tuple(back)))
        return fake_builder(front, back, progress_callback)

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(
        counting_builder,
        cache_loader=lambda candidate: None,
        cache_saver=lambda candidate, cache: saved.append((candidate.id, cache)),
    )

    assert len(recovered) == 1
    assert len(rebuilt) == 1
    assert saved == [(record.id, recovered[0][1])]


def test_recover_restores_valid_backup_when_formal_directory_is_missing(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 5), image_set(tmp_path, "back", 20, 5), False, fake_builder
    )
    backup = record.root.parent / ".backup-crash"
    os.replace(record.root, backup)

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(item.id, item.name) for item, _ in recovered] == [(record.id, "M7")]
    assert (tmp_path / "lib" / record.id).is_dir()
    assert not backup.exists()


def test_register_persists_unequal_counts_and_manifest_counts(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    front = image_set(tmp_path, "front", 10, 5)
    back = image_set(tmp_path, "back", 100, 12)

    record, cache = library.register("M7", front, back, False, fake_builder)

    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_counts"] == {"front": 5, "back": 12}
    assert len(record.front_images) == len(cache.local_features["front"]) == 5
    assert len(record.back_images) == len(cache.local_features["back"]) == 12


@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15)])
def test_register_accepts_requested_template_count_pairs(tmp_path, front_count, back_count):
    library = WorkpieceLibrary(tmp_path / "library")
    record, cache = library.register(
        "M7",
        image_set(tmp_path, "front", 10, front_count),
        image_set(tmp_path, "back", 200, back_count),
        False,
        fake_builder,
    )

    assert (len(record.front_images), len(record.back_images)) == (front_count, back_count)
    assert (len(cache.local_features["front"]), len(cache.local_features["back"])) == (front_count, back_count)


def test_register_allows_more_than_thirty_templates_without_truncation(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 31)
    back = image_set(tmp_path, "back", 200, 1)

    record, _ = library.register("M7", front, back, False, fake_builder)

    assert len(list((record.root / "0").iterdir())) == 31
    assert len(list((record.root / "1").iterdir())) == 1


def test_register_supports_more_than_two_digit_template_indices(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 101)
    back = image_set(tmp_path, "back", 200, 1)

    record, _ = library.register("M7", front, back, False, fake_builder)

    assert record.front_images[0].name == "000.png"
    assert record.front_images[-1].name == "100.png"


def test_register_rejects_same_decoded_image_under_different_paths(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10, 1)
    duplicate = tmp_path / "copy.png"
    shutil.copy2(front[0], duplicate)

    with pytest.raises(InvalidTemplateSetError):
        library.register("M7", front, [duplicate], False, fake_builder)


def test_recover_accepts_old_manifest_without_template_counts(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 5), image_set(tmp_path, "back", 20, 5), False, fake_builder
    )
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("template_counts")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(item.name, len(item.front_images), len(item.back_images)) for item, _ in recovered] == [("M7", 5, 5)]


def test_recover_skips_manifest_with_mismatched_template_counts(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 5), image_set(tmp_path, "back", 20, 5), False, fake_builder
    )
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_counts"]["back"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert recovered == []


def test_register_reports_progress_without_affecting_cache_result(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    events = []

    record, cache = library.register(
        "M7",
        image_set(tmp_path, "front", 10, 1),
        image_set(tmp_path, "back", 20, 2),
        False,
        fake_builder,
        progress_callback=events.append,
    )

    assert record.name == "M7"
    assert len(cache.local_features["front"]) == 1
    assert any(event["phase"] == "committing" for event in events)


def test_annotation_document_reads_legacy_groups_and_only_safe_groups_as_active(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["interference_groups"] = [
        {"group_id": "active", "enabled": True,
         "propagation": {"state": "active"}, "annotations": []},
        {"group_id": "review", "enabled": True,
         "propagation": {"state": "needs_review"}, "annotations": []},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    document = library.get_annotation_document(record.id)

    assert [item["group_id"] for item in document["draft_groups"]] == ["active", "review"]
    assert [item["group_id"] for item in document["active_groups"]] == ["active"]
    assert document["active_annotation_revision"] == document["annotation_revision"]


def test_annotation_document_rejects_stale_replace_without_mutating_manifest(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )
    manifest_path = record.root / "manifest.json"
    before = manifest_path.read_bytes()

    with pytest.raises(StaleWorkpieceRevisionError):
        library.replace_annotation_document(
            record.id,
            [{"group_id": "glare", "name": "反光", "annotations": []}],
            expected_revision=record.revision - 1,
            active_groups=[],
        )

    assert manifest_path.read_bytes() == before
    assert library.get(record.id).revision == record.revision


def test_replace_geometry_profile_pointers_updates_manifest_and_record_revision(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )

    updated = library.replace_geometry_profile_pointers(
        record.id,
        expected_revision=record.revision,
        active_revision=3,
        previous_active_revision=2,
    )

    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    assert updated.revision == record.revision + 1
    assert library.get(record.id).revision == updated.revision
    assert manifest["geometry_mask_active_revision"] == 3
    assert manifest["geometry_mask_previous_active_revision"] == 2
    assert manifest["revision"] == updated.revision


def test_annotation_document_replaces_draft_and_explicit_active_groups(tmp_path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10, 1), image_set(tmp_path, "back", 20, 1), False, fake_builder
    )

    updated = library.replace_annotation_document(
        record.id,
        [{"group_id": "draft", "name": "待复核", "propagation": {"state": "needs_review"}, "annotations": []}],
        expected_revision=record.revision,
        active_groups=[],
    )
    document = library.get_annotation_document(record.id)

    assert updated.revision == record.revision + 1
    assert document["draft_groups"][0]["group_id"] == "draft"
    assert document["active_groups"] == []
    assert document["active_annotation_revision"] == document["annotation_revision"]
