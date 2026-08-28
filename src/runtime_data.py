"""Versioned writable data directories for portable runtime packages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import stat
import uuid


DATA_LAYOUT_VERSION = 1
_METADATA_NAME = "data_layout.json"
_LAYOUT_DIRECTORIES = ("workpieces", "rules", "cache", "logs", "temp")
_LEGACY_SYSTEM_DIRECTORIES = {".recycled", ".geometry-mask-jobs", ".evolution"}


class RuntimeDataError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RuntimeDataPaths:
    root: Path
    workpieces: Path
    rules: Path
    cache: Path
    logs: Path
    temp: Path
    metadata: Path
    backup: Path | None = None


def _paths(root: Path, *, backup: Path | None = None) -> RuntimeDataPaths:
    root = root.resolve()
    return RuntimeDataPaths(
        root=root,
        workpieces=root / "workpieces",
        rules=root / "rules",
        cache=root / "cache",
        logs=root / "logs",
        temp=root / "temp",
        metadata=root / _METADATA_NAME,
        backup=backup,
    )


def _raise_not_writable(root: Path, error: OSError) -> None:
    raise RuntimeDataError("DATA_DIRECTORY_NOT_WRITABLE", f"Data directory is not writable: {root}") from error


def _verify_writable(root: Path) -> None:
    probe = root / f".write-probe-{uuid.uuid4().hex}"
    try:
        probe.touch(exist_ok=False)
        probe.unlink()
    except OSError as error:
        if probe.exists():
            try:
                probe.unlink()
            except OSError:
                pass
        _raise_not_writable(root, error)


def _owned_directory(root: Path, directory: Path) -> None:
    if (
        directory.parent != root
        or directory.is_symlink()
        or directory.resolve().parent != root
        or (directory.exists() and not directory.is_dir())
    ):
        raise RuntimeDataError("DATA_LAYOUT_AMBIGUOUS", f"Unsafe data layout path: {directory}")


def _is_reparse_point(path: Path) -> bool:
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _write_metadata(metadata: Path) -> None:
    temporary = metadata.with_name(f".{metadata.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps({"layout_version": DATA_LAYOUT_VERSION}), encoding="utf-8")
        os.replace(temporary, metadata)
    except OSError as error:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
        _raise_not_writable(metadata.parent, error)


def _clear_owned_temp(paths: RuntimeDataPaths) -> None:
    _owned_directory(paths.root, paths.temp)
    try:
        for child in paths.temp.iterdir():
            if child.is_symlink():
                child.unlink()
            elif _is_reparse_point(child):
                if child.is_dir():
                    child.rmdir()
                else:
                    child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    except OSError as error:
        _raise_not_writable(paths.root, error)


def _ensure_layout(root: Path, *, write_metadata: bool, clear_temp: bool) -> RuntimeDataPaths:
    paths = _paths(root)
    try:
        paths.root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        _raise_not_writable(paths.root, error)
    _verify_writable(paths.root)
    for name in _LAYOUT_DIRECTORIES:
        directory = getattr(paths, name)
        _owned_directory(paths.root, directory)
        try:
            directory.mkdir(exist_ok=True)
        except OSError as error:
            _raise_not_writable(paths.root, error)
    if write_metadata:
        _write_metadata(paths.metadata)
    if clear_temp:
        _clear_owned_temp(paths)
    return paths


def _layout_version(metadata: Path) -> int:
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
        version = value["layout_version"]
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeDataError("DATA_VERSION_UNSUPPORTED", "Data layout metadata is invalid") from error
    if type(version) is not int:
        raise RuntimeDataError("DATA_VERSION_UNSUPPORTED", "Data layout version is invalid")
    return version


def _is_in_place_layout(entries: list[Path]) -> bool:
    return all(entry.name in _LAYOUT_DIRECTORIES for entry in entries)


def _is_legacy_library(entries: list[Path]) -> bool:
    if not entries:
        return False
    for entry in entries:
        if entry.name == "diagnostics" or entry.name in _LEGACY_SYSTEM_DIRECTORIES:
            if not entry.is_dir() or entry.is_symlink():
                return False
            continue
        if not entry.is_dir() or entry.is_symlink() or not (entry / "manifest.json").is_file():
            return False
    return True


def _migration_paths(root: Path, clock) -> tuple[Path, Path]:
    now = clock() if clock is not None else datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    return (
        root.parent / f".{root.name}.staging-{uuid.uuid4().hex}",
        root.parent / f"{root.name}.backup-{timestamp}",
    )


def _migrate_legacy_library(root: Path, entries: list[Path], *, clock) -> RuntimeDataPaths:
    staging, backup = _migration_paths(root, clock)
    if staging.exists() or backup.exists():
        raise RuntimeDataError("DATA_MIGRATION_FAILED", "Data migration staging or backup path already exists")
    try:
        staged_paths = _ensure_layout(staging, write_metadata=True, clear_temp=False)
        for entry in entries:
            if entry.name == "diagnostics":
                shutil.copytree(entry, staged_paths.logs, dirs_exist_ok=True)
            else:
                shutil.copytree(entry, staged_paths.workpieces / entry.name)
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except OSError as error:
            try:
                os.replace(backup, root)
            except OSError as restore_error:
                raise RuntimeDataError(
                    "DATA_MIGRATION_FAILED", "Data migration failed and could not restore the original library"
                ) from restore_error
            raise RuntimeDataError("DATA_MIGRATION_FAILED", "Data migration could not replace the original library") from error
    except RuntimeDataError:
        raise
    except OSError as error:
        raise RuntimeDataError("DATA_MIGRATION_FAILED", "Data migration failed") from error
    finally:
        if staging.exists():
            try:
                shutil.rmtree(staging)
            except OSError:
                pass
    return _paths(root, backup=backup)


def prepare_runtime_data(root: Path, *, clock=None) -> RuntimeDataPaths:
    """Create, validate, or transactionally migrate a portable data root."""
    root = Path(root).resolve()
    if root.exists() and not root.is_dir():
        raise RuntimeDataError("DATA_LAYOUT_AMBIGUOUS", f"Data root is not a directory: {root}")
    if not root.exists():
        return _ensure_layout(root, write_metadata=True, clear_temp=True)

    metadata = root / _METADATA_NAME
    if metadata.exists():
        version = _layout_version(metadata)
        if version != DATA_LAYOUT_VERSION:
            raise RuntimeDataError("DATA_VERSION_UNSUPPORTED", f"Unsupported data layout version: {version}")
        return _ensure_layout(root, write_metadata=False, clear_temp=True)

    entries = list(root.iterdir())
    if _is_in_place_layout(entries):
        return _ensure_layout(root, write_metadata=True, clear_temp=True)
    if not _is_legacy_library(entries):
        raise RuntimeDataError("DATA_LAYOUT_AMBIGUOUS", f"Unrecognized legacy data layout: {root}")
    return _migrate_legacy_library(root, entries, clock=clock)


def legacy_runtime_data(library_dir: Path) -> RuntimeDataPaths:
    """Expose legacy development paths without relocating the library."""
    root = Path(library_dir).resolve()
    return RuntimeDataPaths(
        root=root,
        workpieces=root,
        rules=root / "rules",
        cache=root / "cache",
        logs=root / "diagnostics",
        temp=root / "temp",
        metadata=root / _METADATA_NAME,
    )
