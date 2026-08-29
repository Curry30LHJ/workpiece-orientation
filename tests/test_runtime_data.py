import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.runtime_data import (
    DATA_LAYOUT_VERSION,
    RuntimeDataError,
    legacy_runtime_data,
    prepare_runtime_data,
)


def fixed_clock():
    return datetime(2026, 8, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_first_run_creates_empty_portable_layout(tmp_path: Path):
    paths = prepare_runtime_data(tmp_path / "甲方 数据", clock=fixed_clock)
    assert paths.workpieces.is_dir()
    assert paths.rules.is_dir()
    assert paths.cache.is_dir()
    assert paths.logs.is_dir()
    assert paths.temp.is_dir()
    assert list(paths.workpieces.iterdir()) == []
    assert json.loads(paths.metadata.read_text(encoding="utf-8")) == {
        "layout_version": DATA_LAYOUT_VERSION
    }


def test_unversioned_legacy_library_is_backed_up_and_migrated(tmp_path: Path):
    root = tmp_path / "data"
    (root / "piece-a").mkdir(parents=True)
    (root / "piece-a" / "manifest.json").write_text(
        json.dumps({"id": "piece-a", "name": "M1"}), encoding="utf-8"
    )
    (root / ".recycled").mkdir()
    (root / ".geometry-mask-jobs").mkdir()
    (root / ".evolution").mkdir()
    (root / "diagnostics").mkdir()
    (root / "diagnostics" / "old.log").write_text("old", encoding="utf-8")
    paths = prepare_runtime_data(root, clock=fixed_clock)
    assert (paths.workpieces / "piece-a" / "manifest.json").is_file()
    assert (paths.workpieces / ".recycled").is_dir()
    assert (paths.workpieces / ".geometry-mask-jobs").is_dir()
    assert (paths.workpieces / ".evolution").is_dir()
    assert (paths.logs / "old.log").read_text(encoding="utf-8") == "old"
    assert paths.backup == tmp_path / "data.backup-20260828-120000"
    assert (paths.backup / "piece-a" / "manifest.json").is_file()


def test_unknown_legacy_entry_is_rejected_without_mutation(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "unknown.bin").write_bytes(b"keep")
    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root, clock=fixed_clock)
    assert error.value.code == "DATA_LAYOUT_AMBIGUOUS"
    assert (root / "unknown.bin").read_bytes() == b"keep"
    assert not (tmp_path / "data.backup-20260828-120000").exists()


def test_newer_layout_is_rejected_without_rewrite(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    marker = root / "data_layout.json"
    marker.write_text(json.dumps({"layout_version": 99}), encoding="utf-8")
    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root)
    assert error.value.code == "DATA_VERSION_UNSUPPORTED"
    assert json.loads(marker.read_text(encoding="utf-8"))["layout_version"] == 99


def test_owned_temp_directory_is_cleared_on_start(tmp_path: Path):
    paths = prepare_runtime_data(tmp_path / "data")
    (paths.temp / "stale.tmp").write_text("stale", encoding="utf-8")
    prepare_runtime_data(paths.root)
    assert list(paths.temp.iterdir()) == []


def test_final_rename_failure_restores_original_and_removes_staging(tmp_path: Path, monkeypatch):
    import src.runtime_data as runtime_data

    root = tmp_path / "data"
    (root / "piece-a").mkdir(parents=True)
    (root / "piece-a" / "manifest.json").write_text("{}", encoding="utf-8")
    staging_root = tmp_path / ".data.staging-fault-injected"
    actual_replace = runtime_data.os.replace

    def fail_final_replace(source, destination):
        if Path(source) == staging_root and Path(destination) == root:
            raise OSError("injected final rename failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(runtime_data.uuid, "uuid4", lambda: type("Id", (), {"hex": "fault-injected"})())
    monkeypatch.setattr(runtime_data.os, "replace", fail_final_replace)

    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root, clock=fixed_clock)

    assert error.value.code == "DATA_MIGRATION_FAILED"
    assert (root / "piece-a" / "manifest.json").is_file()
    assert not staging_root.exists()
    assert not (tmp_path / "data.backup-20260828-120000").exists()


def test_versioned_temp_symlink_is_rejected_without_clearing_outside_root(tmp_path: Path):
    root = tmp_path / "data"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    paths = prepare_runtime_data(root)
    paths.temp.rmdir()
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(paths.temp), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    with pytest.raises(RuntimeDataError) as error:
        prepare_runtime_data(root)

    assert error.value.code == "DATA_LAYOUT_AMBIGUOUS"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_temp_cleanup_removes_nested_junction_without_traversing_it(tmp_path: Path):
    paths = prepare_runtime_data(tmp_path / "data")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    ordinary = paths.temp / "ordinary"
    ordinary.mkdir()
    (ordinary / "stale.tmp").write_text("stale", encoding="utf-8")
    junction = paths.temp / "outside-junction"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    prepare_runtime_data(paths.root)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not junction.exists()
    assert not ordinary.exists()


def test_legacy_runtime_data_preserves_configured_library_directory(tmp_path: Path):
    library_dir = tmp_path / "legacy library"
    paths = legacy_runtime_data(library_dir)
    assert paths.root == library_dir.resolve()
    assert paths.workpieces == library_dir.resolve()
    assert paths.logs == library_dir.resolve() / "diagnostics"
    assert paths.temp == library_dir.resolve() / "temp"
