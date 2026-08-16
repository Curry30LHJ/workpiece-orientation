"""Transactional persistent library for variable-sized template sets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import os
import shutil
import uuid
from typing import Any, Callable, Sequence

from src.image_io import read_color_image
from src.orientation_classifier import TemplateCache


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


class WorkpieceLibraryError(RuntimeError):
    """Base error for persistent workpiece operations."""


class InvalidWorkpieceNameError(WorkpieceLibraryError):
    """Raised when a display name cannot be safely persisted."""


class InvalidTemplateSetError(WorkpieceLibraryError):
    """Raised when a front/back template set is empty or contains invalid images."""


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


CacheProgressCallback = Callable[[str, int, int], None]
CacheBuilder = Callable[[Sequence[Path], Sequence[Path], CacheProgressCallback | None], TemplateCache]
ProgressCallback = Callable[[dict[str, Any]], None]


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


def _image_fingerprint(image: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def _validate_images(
    paths: Sequence[Path],
    label: str,
    seen_images: dict[str, Path] | None = None,
    on_image: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, ...]:
    if not paths:
        raise InvalidTemplateSetError(f"{label} requires at least one image")
    resolved: list[Path] = []
    local_seen: dict[str, Path] = {}
    for index, raw_path in enumerate(paths, start=1):
        path = Path(raw_path)
        path_key = f"path:{str(path.resolve()).casefold()}"
        if path_key in local_seen or (seen_images is not None and path_key in seen_images):
            raise InvalidTemplateSetError(f"Duplicate {label} template: {path}")
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            raise InvalidTemplateSetError(f"Invalid {label} image: {path}")
        image = read_color_image(path)
        if image is None:
            raise InvalidTemplateSetError(f"Unreadable {label} image: {path}")
        digest_key = f"content:{_image_fingerprint(image)}"
        if digest_key in local_seen or (seen_images is not None and digest_key in seen_images):
            raise InvalidTemplateSetError(f"Duplicate {label} image content: {path}")
        resolved_path = path.resolve()
        local_seen[path_key] = resolved_path
        local_seen[digest_key] = resolved_path
        if seen_images is not None:
            seen_images[path_key] = resolved_path
            seen_images[digest_key] = resolved_path
        resolved.append(resolved_path)
        if on_image is not None:
            on_image(label, index, len(paths))
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
        seen_images: dict[str, Path] = {}
        front = _validate_images(
            sorted((root / "0").iterdir(), key=lambda path: path.name.casefold()),
            "front",
            seen_images,
        )
        back = _validate_images(
            sorted((root / "1").iterdir(), key=lambda path: path.name.casefold()),
            "back",
            seen_images,
        )
        counts = manifest.get("template_counts")
        if counts is not None:
            if (
                not isinstance(counts, dict)
                or set(counts) != {"front", "back"}
                or any(type(value) is not int or value <= 0 for value in counts.values())
                or counts != {"front": len(front), "back": len(back)}
            ):
                raise InvalidTemplateSetError(f"Invalid template_counts: {manifest_path}")
        return WorkpieceRecord(record_id, name, root, front, back)

    def _copy_templates(
        self,
        paths: Sequence[Path],
        target: Path,
        on_image: Callable[[int, int], None] | None = None,
    ) -> tuple[Path, ...]:
        target.mkdir(parents=True, exist_ok=False)
        copied = []
        width = max(2, len(str(len(paths) - 1)))
        for index, source in enumerate(paths):
            destination = target / f"{index:0{width}d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            copied.append(destination)
            if on_image is not None:
                on_image(index + 1, len(paths))
        return tuple(copied)

    def register(
        self,
        name: str,
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        replace: bool,
        build_cache: CacheBuilder,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[WorkpieceRecord, TemplateCache]:
        display_name = _validate_name(name)
        total = len(front_images) + len(back_images)

        def report(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event)
            except Exception:
                LOGGER.debug("Ignoring registration progress callback failure", exc_info=True)

        report({"phase": "validating", "completed": 0, "total": total})
        seen_images: dict[str, Path] = {}
        front = _validate_images(
            front_images,
            "front",
            seen_images,
            lambda label, completed, side_total: report(
                {
                    "phase": "validating",
                    "label": label,
                    "phase_completed": completed,
                    "phase_total": side_total,
                    "completed": 0,
                    "total": total,
                }
            ),
        )
        back = _validate_images(
            back_images,
            "back",
            seen_images,
            lambda label, completed, side_total: report(
                {
                    "phase": "validating",
                    "label": label,
                    "phase_completed": completed,
                    "phase_total": side_total,
                    "completed": 0,
                    "total": total,
                }
            ),
        )
        existing = self._find_by_name(display_name)
        if existing is not None and not replace:
            raise WorkpieceExistsError(f"Workpiece already exists: {display_name}")

        self.library_dir.mkdir(parents=True, exist_ok=True)
        new_id = uuid.uuid4().hex
        staging = self.library_dir / f".staging-{uuid.uuid4().hex}"
        staging.mkdir()
        final_root = self.library_dir / new_id
        backup_root: Path | None = None
        try:
            report({"phase": "copying", "completed": 0, "total": total})
            staged_front = self._copy_templates(
                front,
                staging / "0",
                lambda completed, side_total: report(
                    {
                        "phase": "copying",
                        "label": "front",
                        "phase_completed": completed,
                        "phase_total": side_total,
                        "completed": 0,
                        "total": total,
                    }
                ),
            )
            staged_back = self._copy_templates(
                back,
                staging / "1",
                lambda completed, side_total: report(
                    {
                        "phase": "copying",
                        "label": "back",
                        "phase_completed": completed,
                        "phase_total": side_total,
                        "completed": 0,
                        "total": total,
                    }
                ),
            )
            manifest = {
                "schema_version": 1,
                "id": new_id,
                "name": display_name,
                "labels": {"0": "front", "1": "back"},
                "template_counts": {"front": len(front), "back": len(back)},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report({"phase": "features", "completed": 0, "total": total})

            def cache_progress(label: str, completed: int, side_total: int) -> None:
                offset = len(front) if label == "back" else 0
                report({"phase": "features", "completed": offset + completed, "total": total})

            cache = build_cache(staged_front, staged_back, cache_progress)
            report({"phase": "committing", "completed": total, "total": total})
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
                cache = build_cache(record.front_images, record.back_images, None)
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
