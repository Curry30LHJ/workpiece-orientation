"""Transactional persistent library for five front and five back templates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import os
import shutil
import tempfile
import uuid
from typing import Callable, Sequence

import cv2

from src.orientation_classifier import TemplateCache


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
TEMPLATE_COUNT = 5


class WorkpieceLibraryError(RuntimeError):
    """Base error for persistent workpiece operations."""


class InvalidWorkpieceNameError(WorkpieceLibraryError):
    """Raised when a display name cannot be safely persisted."""


class InvalidTemplateSetError(WorkpieceLibraryError):
    """Raised when a front/back template set is not exactly five valid images."""


class WorkpieceExistsError(WorkpieceLibraryError):
    """Raised when a display name already exists without explicit replacement."""


class FeatureBuildError(WorkpieceLibraryError):
    """Raised by a feature builder when a staging set cannot be indexed."""


@dataclass(frozen=True)
class WorkpieceRecord:
    id: str
    name: str
    root: Path
    front_images: tuple[Path, ...]
    back_images: tuple[Path, ...]


CacheBuilder = Callable[[Sequence[Path], Sequence[Path]], TemplateCache]


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise InvalidWorkpieceNameError("Workpiece name must be text")
    value = name.strip()
    if not value or value in {".", ".."}:
        raise InvalidWorkpieceNameError("Workpiece name cannot be empty")
    if any(character in value for character in ("/", "\\")):
        raise InvalidWorkpieceNameError("Workpiece name cannot contain path separators")
    if any(ord(character) < 32 for character in value):
        raise InvalidWorkpieceNameError("Workpiece name cannot contain control characters")
    return value


def _validate_images(paths: Sequence[Path], label: str) -> tuple[Path, ...]:
    if len(paths) != TEMPLATE_COUNT:
        raise InvalidTemplateSetError(f"{label} requires exactly five images")
    resolved: list[Path] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        key = str(path.resolve()).casefold()
        if key in seen:
            raise InvalidTemplateSetError(f"Duplicate {label} template: {path}")
        seen.add(key)
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            raise InvalidTemplateSetError(f"Invalid {label} image: {path}")
        if cv2.imread(str(path)) is None:
            raise InvalidTemplateSetError(f"Unreadable {label} image: {path}")
        resolved.append(path.resolve())
    return tuple(resolved)


class WorkpieceLibrary:
    """Store complete workpiece records and replace them transactionally."""

    def __init__(self, library_dir: Path):
        self.library_dir = Path(library_dir)
        self._records: dict[str, WorkpieceRecord] = {}

    def _find_by_name(self, name: str) -> WorkpieceRecord | None:
        key = name.casefold()
        return next((record for record in self._records.values() if record.name.casefold() == key), None)

    @staticmethod
    def _record_from_root(root: Path) -> WorkpieceRecord:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_id = manifest["id"]
        name = _validate_name(manifest["name"])
        if record_id != root.name or manifest.get("labels") != {"0": "front", "1": "back"}:
            raise InvalidTemplateSetError(f"Invalid manifest: {manifest_path}")
        front = _validate_images(sorted((root / "0").iterdir()), "front")
        back = _validate_images(sorted((root / "1").iterdir()), "back")
        return WorkpieceRecord(record_id, name, root, front, back)

    def _copy_templates(self, paths: Sequence[Path], target: Path) -> tuple[Path, ...]:
        target.mkdir(parents=True, exist_ok=False)
        copied = []
        for index, source in enumerate(paths):
            destination = target / f"{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied.append(destination)
        return tuple(copied)

    def register(
        self,
        name: str,
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        replace: bool,
        build_cache: CacheBuilder,
    ) -> tuple[WorkpieceRecord, TemplateCache]:
        display_name = _validate_name(name)
        front = _validate_images(front_images, "front")
        back = _validate_images(back_images, "back")
        all_sources = {str(path).casefold() for path in (*front, *back)}
        if len(all_sources) != TEMPLATE_COUNT * 2:
            raise InvalidTemplateSetError("A source image cannot be used twice")
        existing = self._find_by_name(display_name)
        if existing is not None and not replace:
            raise WorkpieceExistsError(f"Workpiece already exists: {display_name}")

        self.library_dir.mkdir(parents=True, exist_ok=True)
        new_id = uuid.uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.library_dir))
        final_root = self.library_dir / new_id
        backup_root: Path | None = None
        try:
            staged_front = self._copy_templates(front, staging / "0")
            staged_back = self._copy_templates(back, staging / "1")
            manifest = {
                "schema_version": 1,
                "id": new_id,
                "name": display_name,
                "labels": {"0": "front", "1": "back"},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            cache = build_cache(staged_front, staged_back)
            if existing is not None:
                backup_root = self.library_dir / f".backup-{uuid.uuid4().hex}"
                os.replace(existing.root, backup_root)
            os.replace(staging, final_root)
            record = WorkpieceRecord(
                new_id,
                display_name,
                final_root,
                tuple(final_root / "0" / path.name for path in staged_front),
                tuple(final_root / "1" / path.name for path in staged_back),
            )
            self._records[new_id] = record
            if existing is not None:
                self._records.pop(existing.id, None)
            if backup_root is not None:
                shutil.rmtree(backup_root)
            return record, cache
        except Exception:
            if final_root.exists():
                shutil.rmtree(final_root)
            if backup_root is not None and backup_root.exists() and existing is not None:
                os.replace(backup_root, existing.root)
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def recover(self, build_cache: CacheBuilder) -> list[tuple[WorkpieceRecord, TemplateCache]]:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._records.clear()
        for temporary in self.library_dir.iterdir():
            if temporary.name.startswith(".staging-"):
                shutil.rmtree(temporary, ignore_errors=True)
            elif temporary.name.startswith(".backup-"):
                try:
                    manifest = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
                    formal_root = self.library_dir / str(manifest["id"])
                    if formal_root.exists():
                        shutil.rmtree(temporary, ignore_errors=True)
                    else:
                        os.replace(temporary, formal_root)
                except Exception as exc:
                    LOGGER.warning("Removing invalid backup %s: %s", temporary, exc)
                    shutil.rmtree(temporary, ignore_errors=True)
        recovered: list[tuple[WorkpieceRecord, TemplateCache]] = []
        for root in sorted(self.library_dir.iterdir(), key=lambda path: path.name):
            if not root.is_dir() or root.name.startswith("."):
                continue
            try:
                record = self._record_from_root(root)
                if self._find_by_name(record.name) is not None:
                    raise InvalidTemplateSetError(f"Duplicate workpiece name: {record.name}")
                cache = build_cache(record.front_images, record.back_images)
            except Exception as exc:
                LOGGER.warning("Skipping invalid workpiece %s: %s", root, exc)
                continue
            self._records[record.id] = record
            recovered.append((record, cache))
        return recovered

    def list_workpieces(self) -> list[dict[str, str]]:
        return [
            {"id": record.id, "name": record.name}
            for record in sorted(self._records.values(), key=lambda item: item.name.casefold())
        ]

    def get(self, workpiece_id: str) -> WorkpieceRecord:
        return self._records[workpiece_id]
