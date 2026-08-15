import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.orientation_classifier import TemplateCache
from src.workpiece_library import (
    FeatureBuildError,
    InvalidTemplateSetError,
    InvalidWorkpieceNameError,
    WorkpieceExistsError,
    WorkpieceLibrary,
)


def write_image(path: Path, marker: int) -> Path:
    image = np.full((8, 8, 3), marker, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)
    return path


def image_set(tmp_path: Path, prefix: str, marker: int) -> list[Path]:
    return [write_image(tmp_path / f"{prefix}-{index}.png", marker) for index in range(5)]


def fake_builder(front: list[Path], back: list[Path]) -> TemplateCache:
    return TemplateCache(
        global_vectors={
            "front": np.zeros((len(front), 2), dtype=np.float32),
            "back": np.ones((len(back), 2), dtype=np.float32),
        },
        local_features={"front": [{} for _ in front], "back": [{} for _ in back]},
    )


def test_register_creates_uuid_manifest_and_fixed_label_folders(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "库")
    front = image_set(tmp_path, "front", 10)
    back = image_set(tmp_path, "back", 20)

    record, cache = library.register("M7", front, back, False, fake_builder)

    assert len(record.id) == 32
    assert cache.global_vectors["front"].shape == (5, 2)
    manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["name"] == "M7"
    assert manifest["labels"] == {"0": "front", "1": "back"}
    assert len(list((record.root / "0").glob("*"))) == 5
    assert len(list((record.root / "1").glob("*"))) == 5


def test_register_rejects_existing_name_without_replace(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "lib")
    front = image_set(tmp_path, "front", 10)
    back = image_set(tmp_path, "back", 20)
    old, _ = library.register("M7", front, back, False, fake_builder)

    with pytest.raises(WorkpieceExistsError):
        library.register("m7", front, back, False, fake_builder)

    assert library.get(old.id).root == old.root
    assert library.list_workpieces() == [{"id": old.id, "name": "M7"}]


def test_failed_replace_preserves_old_directory(tmp_path: Path):
    library = WorkpieceLibrary(tmp_path / "lib")
    old_front = image_set(tmp_path, "old-front", 10)
    old_back = image_set(tmp_path, "old-back", 20)
    old, _ = library.register("M7", old_front, old_back, False, fake_builder)
    new_front = image_set(tmp_path, "new-front", 30)
    new_back = image_set(tmp_path, "new-back", 40)

    def raising_builder(front, back):
        raise FeatureBuildError("feature extraction failed")

    with pytest.raises(FeatureBuildError):
        library.register("M7", new_front, new_back, True, raising_builder)

    current = library.get(old.id)
    assert current.root == old.root
    assert cv2.imread(str(current.front_images[0]))[0, 0, 0] == 10
    assert not list((tmp_path / "lib").glob(".staging-*"))


@pytest.mark.parametrize("name", ["", "  ", "a/b", "a\\b", "a\x00b", "a\n b"])
def test_invalid_names_are_rejected(name, tmp_path):
    front = image_set(tmp_path, "front", 10)
    back = image_set(tmp_path, "back", 20)

    with pytest.raises(InvalidWorkpieceNameError):
        WorkpieceLibrary(tmp_path / "lib").register(name, front, back, False, fake_builder)


def test_duplicate_template_path_is_rejected(tmp_path):
    front = image_set(tmp_path, "front", 10)
    back = image_set(tmp_path, "back", 20)
    front[1] = front[0]

    with pytest.raises(InvalidTemplateSetError):
        WorkpieceLibrary(tmp_path / "lib").register("M7", front, back, False, fake_builder)


def test_invalid_template_count_or_image_is_rejected(tmp_path):
    front = image_set(tmp_path, "front", 10)[:4]
    back = image_set(tmp_path, "back", 20)

    with pytest.raises(InvalidTemplateSetError):
        WorkpieceLibrary(tmp_path / "lib").register("M7", front, back, False, fake_builder)


def test_recover_skips_corrupt_record_but_loads_valid_record(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    valid_front = image_set(tmp_path, "valid-front", 10)
    valid_back = image_set(tmp_path, "valid-back", 20)
    valid, _ = library.register("M7", valid_front, valid_back, False, fake_builder)
    corrupt = tmp_path / "lib" / "corrupt"
    (corrupt / "0").mkdir(parents=True)
    (corrupt / "1").mkdir()
    (corrupt / "manifest.json").write_text("not-json", encoding="utf-8")

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(record.id, record.name) for record, _ in recovered] == [(valid.id, "M7")]


def test_recover_rebuilds_each_valid_cache_once(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    library.register("M7", image_set(tmp_path, "m7-front", 10), image_set(tmp_path, "m7-back", 20), False, fake_builder)
    library.register("M8", image_set(tmp_path, "m8-front", 30), image_set(tmp_path, "m8-back", 40), False, fake_builder)
    calls = []

    def counting_builder(front, back):
        calls.append((tuple(front), tuple(back)))
        return fake_builder(front, back)

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(counting_builder)

    assert len(recovered) == 2
    assert len(calls) == 2


def test_recover_restores_valid_backup_when_formal_directory_is_missing(tmp_path):
    library = WorkpieceLibrary(tmp_path / "lib")
    record, _ = library.register(
        "M7", image_set(tmp_path, "front", 10), image_set(tmp_path, "back", 20), False, fake_builder
    )
    backup = record.root.parent / ".backup-crash"
    os.replace(record.root, backup)

    recovered = WorkpieceLibrary(tmp_path / "lib").recover(fake_builder)

    assert [(item.id, item.name) for item, _ in recovered] == [(record.id, "M7")]
    assert (tmp_path / "lib" / record.id).is_dir()
    assert not backup.exists()
