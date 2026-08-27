import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.orientation_classifier import OrientationClassifier, TemplateCache
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


def fast_revision_builder(
    front: list[Path],
    back: list[Path],
    progress_callback=None,
    *,
    library_revision: int = 1,
) -> TemplateCache:
    cache = fake_builder(front, back, progress_callback)
    return TemplateCache(
        global_vectors=cache.global_vectors,
        local_features=cache.local_features,
        fast_runtime=SimpleNamespace(library_revision=library_revision),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _registered_library(tmp_path: Path) -> tuple[WorkpieceLibrary, object]:
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7",
        image_set(tmp_path, "base-front", 10, 2),
        image_set(tmp_path, "base-back", 20, 2),
        False,
        fake_builder,
    )
    return library, record


def _write_geometry_sidecars(root: Path) -> dict[str, str]:
    revisions = root / "geometry_masks" / "revisions"
    previews = root / "geometry_masks" / "previews"
    revisions.mkdir(parents=True)
    previews.mkdir(parents=True)
    (root / "geometry_masks" / "profile.json").write_text(
        json.dumps({"library_revision": 1, "active_revision": 1}), encoding="utf-8"
    )
    (revisions / "1.json").write_text('{"revision": 1}', encoding="utf-8")
    (previews / "front-00.png").write_bytes(b"immutable-preview")
    (root / ".template_cache.pkl").write_bytes(b"stale-cache")
    return _tree_hashes(root)


def test_prepared_append_preserves_geometry_sidecars_until_commit(tmp_path):
    library, record = _registered_library(tmp_path)
    before = _write_geometry_sidecars(record.root)

    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "confirmed.png", 31)],
        [],
        fake_builder,
        operation_id="job-append-1",
    )

    assert _tree_hashes(record.root) == before
    assert library.get(record.id) == record
    assert not (prepared.staging_root / ".template_cache.pkl").exists()
    committed, backup = library.commit_prepared(prepared)
    assert (committed.root / "geometry_masks" / "revisions" / "1.json").read_bytes() == b'{"revision": 1}'
    assert (committed.root / "geometry_masks" / "previews" / "front-00.png").read_bytes() == b"immutable-preview"
    assert len(committed.front_images) == len(record.front_images) + 1
    manifest = json.loads((committed.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["last_template_update"]["operation_id"] == "job-append-1"
    assert manifest["last_template_update"]["base_revision"] == record.revision
    assert manifest["last_template_update"]["target_revision"] == committed.revision
    assert len(manifest["last_template_update"]["item_digests"]) == 1
    library.remove_retired(backup)


def test_prepared_append_builder_failure_leaves_active_tree_unchanged(tmp_path):
    library, record = _registered_library(tmp_path)
    before = _write_geometry_sidecars(record.root)

    def raising_builder(front, back, progress_callback=None):
        raise FeatureBuildError("feature extraction failed")

    with pytest.raises(FeatureBuildError):
        library.prepare_append(
            record,
            [write_image(tmp_path / "failed-confirmed.png", 32)],
            [],
            raising_builder,
            operation_id="job-append-failed",
        )

    assert _tree_hashes(record.root) == before
    assert not list(record.root.parent.glob(".staging-*"))


def test_abort_prepared_is_idempotent_and_stale_owner_cannot_commit(tmp_path):
    library, record = _registered_library(tmp_path)
    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "stale-confirmed.png", 33)],
        [],
        fake_builder,
        operation_id="job-append-stale",
    )
    library.recycle(record.id)

    with pytest.raises(StaleWorkpieceRevisionError):
        library.commit_prepared(prepared)

    library.abort_prepared(prepared)
    library.abort_prepared(prepared)
    assert not prepared.staging_root.exists()


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


def test_register_writes_inventory_for_every_copied_template(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "library")
    record, _ = library.register(
        "M7",
        image_set(tmp_path, "front-inventory", 10, 2),
        image_set(tmp_path, "back-inventory", 20, 3),
        False,
        fake_builder,
    )

    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    inventory = manifest["template_inventory"]

    assert [(item["template_id"], item["direction"], item["filename"]) for item in inventory] == [
        ("front:00.png", "front", "00.png"),
        ("front:01.png", "front", "01.png"),
        ("back:00.png", "back", "00.png"),
        ("back:01.png", "back", "01.png"),
        ("back:02.png", "back", "02.png"),
    ]
    assert {item["source"] for item in inventory} == {"initial_registration"}
    assert {item["added_at"] for item in inventory} == {manifest["created_at"]}


def test_prepare_append_preserves_inventory_and_adds_confirmed_templates(tmp_path: Path):
    library, record = _registered_library(tmp_path)
    manifest_path = record.root / "manifest.json"
    original_inventory = json.loads(manifest_path.read_text(encoding="utf-8"))["template_inventory"]
    original_manifest = manifest_path.read_bytes()

    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "confirmed-front.png", 31)],
        [write_image(tmp_path / "confirmed-back.png", 41)],
        fake_builder,
        operation_id="confirmed-1",
        source="confirmed_inspection",
    )
    try:
        staged_manifest = json.loads(
            (prepared.staging_root / "manifest.json").read_text(encoding="utf-8")
        )
        staged_inventory = staged_manifest["template_inventory"]

        assert manifest_path.read_bytes() == original_manifest
        assert staged_inventory[:4] == original_inventory
        assert [(item["template_id"], item["source"]) for item in staged_inventory[4:]] == [
            ("front:02.png", "confirmed_inspection"),
            ("back:02.png", "confirmed_inspection"),
        ]
        for item in staged_inventory[4:]:
            added_at = datetime.fromisoformat(item["added_at"])
            assert added_at.utcoffset() is not None
            assert added_at.utcoffset().total_seconds() == 0

        assert library.get_template_inventory(record.id) == original_inventory
        committed, retired = library.commit_prepared(prepared)
        assert library.get_template_inventory(committed.id) == staged_inventory
        library.remove_retired(retired)
    finally:
        library.abort_prepared(prepared)


def test_prepare_append_builds_fast_cache_for_staged_library_revision(tmp_path: Path):
    library, record = _registered_library(tmp_path)

    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "fast-front.png", 31)],
        [],
        fast_revision_builder,
        operation_id="fast-revision-append",
    )
    try:
        assert prepared.staged_record.revision == 2
        assert prepared.candidate_cache.fast_runtime.library_revision == 2
    finally:
        library.abort_prepared(prepared)


def test_append_across_filename_width_keeps_existing_inventory_stable(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "wide-library")
    record, _ = library.register(
        "M-wide",
        image_set(tmp_path, "wide-front", 1, 100),
        [write_image(tmp_path / "wide-back.png", 200)],
        False,
        fake_builder,
    )
    original_inventory = library.get_template_inventory(record.id)

    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "wide-front-new.png", 201)],
        [],
        fake_builder,
        operation_id="wide-append-1",
    )
    committed, retired = library.commit_prepared(prepared)
    library.remove_retired(retired)
    inventory = library.get_template_inventory(committed.id)

    assert inventory[:101] == original_inventory
    assert inventory[-1]["template_id"] == "front:100.png"
    assert len({item["template_id"] for item in inventory}) == 102
    for item in inventory:
        label_dir = "0" if item["direction"] == "front" else "1"
        template_path = committed.root / label_dir / item["filename"]
        assert template_path.is_file()
        assert cv2.imread(str(template_path)) is not None

    loaded = []
    build_calls = []
    cache_io = OrientationClassifier.__new__(OrientationClassifier)
    cache_io.save_template_cache(committed, prepared.candidate_cache)

    def cache_loader(candidate):
        loaded.append(candidate)
        return cache_io.load_template_cache(candidate)

    def unexpected_builder(front, back, progress_callback=None):
        build_calls.append((tuple(front), tuple(back)))
        return fake_builder(front, back, progress_callback)

    restarted = WorkpieceLibrary(library.library_dir)
    recovered = restarted.recover(
        unexpected_builder,
        cache_loader=cache_loader,
    )
    recovered_record = recovered[0][0]

    assert [path.name for path in recovered_record.front_images] == [
        *(f"{index:02d}.png" for index in range(100)),
        "100.png",
    ]
    expected_template_ids = [
        *(f"front:{index:02d}.png" for index in range(100)),
        "back:00.png",
        "front:100.png",
    ]
    assert [item["template_id"] for item in restarted.get_template_inventory(record.id)] == expected_template_ids
    assert [item.id for item in loaded] == [record.id]
    assert build_calls == []
    assert recovered[0][1].global_vectors["front"].shape == (101, 2)


@pytest.mark.parametrize(
    "corruption",
    ["missing_source", "duplicate_entry", "wrong_template_id", "missing_template"],
)
def test_present_malformed_template_inventory_is_rejected_without_rewrite(tmp_path, corruption):
    library, record = _registered_library(tmp_path)
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "missing_source":
        manifest["template_inventory"][0].pop("source")
    elif corruption == "duplicate_entry":
        manifest["template_inventory"][1] = dict(manifest["template_inventory"][0])
    elif corruption == "wrong_template_id":
        manifest["template_inventory"][0]["template_id"] = "front:not-the-file.png"
    else:
        manifest["template_inventory"].pop()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    before = manifest_path.read_bytes()

    with pytest.raises(InvalidTemplateSetError, match="template_inventory"):
        library.get_template_inventory(record.id)

    assert manifest_path.read_bytes() == before


def test_aborted_prepared_append_leaves_active_manifest_unchanged(tmp_path: Path):
    library, record = _registered_library(tmp_path)
    manifest_path = record.root / "manifest.json"
    before = manifest_path.read_bytes()
    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "aborted-front.png", 34)],
        [],
        fake_builder,
        operation_id="aborted-1",
    )

    library.abort_prepared(prepared)

    assert manifest_path.read_bytes() == before


def test_failed_prepared_append_leaves_active_manifest_unchanged(tmp_path: Path):
    library, record = _registered_library(tmp_path)
    manifest_path = record.root / "manifest.json"
    before = manifest_path.read_bytes()

    def raising_builder(front, back, progress_callback=None):
        raise FeatureBuildError("feature extraction failed")

    with pytest.raises(FeatureBuildError):
        library.prepare_append(
            record,
            [write_image(tmp_path / "failed-inventory-front.png", 35)],
            [],
            raising_builder,
            operation_id="failed-inventory-1",
        )

    assert manifest_path.read_bytes() == before


def test_replace_registration_builds_fresh_inventory_for_replacement_only(tmp_path: Path):
    library, old = _registered_library(tmp_path)

    replacement, _ = library.register(
        "M7",
        [write_image(tmp_path / "replacement-front.png", 50)],
        image_set(tmp_path, "replacement-back", 60, 3),
        True,
        fake_builder,
    )

    inventory = library.get_template_inventory(replacement.id)
    assert replacement.id != old.id
    assert [item["template_id"] for item in inventory] == [
        "front:00.png",
        "back:00.png",
        "back:01.png",
        "back:02.png",
    ]
    assert {item["source"] for item in inventory} == {"initial_registration"}
    assert old.id not in {item["id"] for item in library.list_workpieces()}


def test_workpiece_metadata_uses_created_append_and_legacy_timestamps(tmp_path: Path):
    library, record = _registered_library(tmp_path)
    manifest_path = record.root / "manifest.json"
    created_at = json.loads(manifest_path.read_text(encoding="utf-8"))["created_at"]

    assert library.get_workpiece_metadata(record.id)["updated_at"] == created_at

    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "metadata-front.png", 36)],
        [],
        fake_builder,
        operation_id="metadata-append-1",
    )
    committed, retired = library.commit_prepared(prepared)
    library.remove_retired(retired)
    appended_at = json.loads(manifest_path.read_text(encoding="utf-8"))["updated_at"]
    assert appended_at != created_at
    assert library.get_workpiece_metadata(committed.id)["updated_at"] == appended_at

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("created_at")
    manifest.pop("updated_at")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert library.get_workpiece_metadata(committed.id)["updated_at"] is None


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


def test_recover_rebuilds_fast_cache_for_manifest_library_revision(tmp_path):
    library, record = _registered_library(tmp_path)
    prepared = library.prepare_append(
        record,
        [write_image(tmp_path / "recovery-front.png", 31)],
        [],
        fake_builder,
        operation_id="recovery-revision-append",
    )
    committed, retired = library.commit_prepared(prepared)
    library.remove_retired(retired)

    recovered = WorkpieceLibrary(library.library_dir).recover(
        fast_revision_builder,
        cache_loader=lambda candidate: None,
    )

    assert recovered[0][0].revision == committed.revision == 2
    assert recovered[0][1].fast_runtime.library_revision == committed.revision


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
