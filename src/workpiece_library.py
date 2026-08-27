"""Transactional persistent library for variable-sized template sets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import logging
from pathlib import Path
import os
import shutil
import uuid
from typing import Any, Callable, Sequence

from src.image_io import read_color_image
from src.orientation_classifier import TEMPLATE_CACHE_FILE_NAME, TemplateCache


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


class StaleWorkpieceRevisionError(WorkpieceLibraryError):
    """Raised when an annotation mutation targets an older workpiece revision."""


@dataclass(frozen=True)
class WorkpieceRecord:
    id: str
    name: str
    root: Path
    front_images: tuple[Path, ...]
    back_images: tuple[Path, ...]
    revision: int = 1
    state: str = "active"


@dataclass
class PreparedTemplateUpdate:
    """Own a fully built template update until commit or abort."""

    operation_id: str
    workpiece_id: str
    base_revision: int
    base_root: Path
    staging_root: Path
    staged_record: WorkpieceRecord
    candidate_cache: TemplateCache
    item_digests: tuple[str, ...]
    consumed: bool = False


CacheProgressCallback = Callable[[str, int, int], None]
CacheBuilder = Callable[[Sequence[Path], Sequence[Path], CacheProgressCallback | None], TemplateCache]
CacheLoader = Callable[[WorkpieceRecord], TemplateCache | None]
CacheSaver = Callable[[WorkpieceRecord, TemplateCache], None]
ProgressCallback = Callable[[dict[str, Any]], None]


def _call_cache_builder(
    build_cache: CacheBuilder,
    front_images: Sequence[Path],
    back_images: Sequence[Path],
    progress_callback: CacheProgressCallback | None,
    *,
    library_revision: int,
) -> TemplateCache:
    """Pass a known revision when supported without breaking legacy builders."""
    try:
        inspect.signature(build_cache).bind(
            front_images,
            back_images,
            progress_callback,
            library_revision=library_revision,
        )
    except (TypeError, ValueError):
        return build_cache(front_images, back_images, progress_callback)
    return build_cache(
        front_images,
        back_images,
        progress_callback,
        library_revision=library_revision,
    )


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
        self._recycled_records: dict[str, WorkpieceRecord] = {}

    def _find_by_name(self, name: str) -> WorkpieceRecord | None:
        key = name.casefold()
        return next((record for record in self._records.values() if record.name.casefold() == key), None)

    @staticmethod
    def _validated_template_inventory(
        manifest_path: Path,
        manifest: dict[str, Any],
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
    ) -> tuple[list[dict[str, Any]], tuple[Path, ...], tuple[Path, ...]] | None:
        if "template_inventory" not in manifest:
            return None
        inventory = manifest["template_inventory"]
        if not isinstance(inventory, list):
            raise InvalidTemplateSetError(f"Invalid template_inventory: {manifest_path}")
        paths_by_direction = {
            "front": {path.name: path for path in front_paths},
            "back": {path.name: path for path in back_paths},
        }
        expected = {
            (direction, filename)
            for direction, paths in paths_by_direction.items()
            for filename in paths
        }
        seen: set[tuple[str, str]] = set()
        ordered_paths: dict[str, list[Path]] = {"front": [], "back": []}
        validated: list[dict[str, Any]] = []
        required = {"template_id", "direction", "filename", "source", "added_at"}
        for item in inventory:
            if not isinstance(item, dict) or not required.issubset(item):
                raise InvalidTemplateSetError(f"Invalid template_inventory: {manifest_path}")
            template_id = item["template_id"]
            direction = item["direction"]
            filename = item["filename"]
            source = item["source"]
            added_at = item["added_at"]
            if (
                not isinstance(direction, str)
                or direction not in {"front", "back"}
                or not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or any(separator in filename for separator in ("/", "\\"))
                or not isinstance(template_id, str)
                or template_id != f"{direction}:{filename}"
                or not isinstance(source, str)
                or not source
                or (added_at is not None and (not isinstance(added_at, str) or not added_at))
            ):
                raise InvalidTemplateSetError(f"Invalid template_inventory: {manifest_path}")
            key = (direction, filename)
            if key in seen or key not in expected:
                raise InvalidTemplateSetError(f"Invalid template_inventory: {manifest_path}")
            seen.add(key)
            ordered_paths[direction].append(paths_by_direction[direction][filename])
            validated.append(deepcopy(item))
        if seen != expected:
            raise InvalidTemplateSetError(f"Invalid template_inventory: {manifest_path}")
        return validated, tuple(ordered_paths["front"]), tuple(ordered_paths["back"])

    @staticmethod
    def _record_from_root(root: Path) -> WorkpieceRecord:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_id = manifest["id"]
        name = _validate_name(manifest["name"])
        if record_id != root.name or manifest.get("labels") != {"0": "front", "1": "back"}:
            raise InvalidTemplateSetError(f"Invalid manifest: {manifest_path}")
        front_paths = tuple(sorted((root / "0").iterdir(), key=lambda path: path.name.casefold()))
        back_paths = tuple(sorted((root / "1").iterdir(), key=lambda path: path.name.casefold()))
        inventory = WorkpieceLibrary._validated_template_inventory(
            manifest_path,
            manifest,
            front_paths,
            back_paths,
        )
        if inventory is not None:
            _, front_paths, back_paths = inventory
        seen_images: dict[str, Path] = {}
        front = _validate_images(
            front_paths,
            "front",
            seen_images,
        )
        back = _validate_images(
            back_paths,
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
        revision = manifest.get("revision", 1)
        if type(revision) is not int or revision <= 0:
            raise InvalidTemplateSetError(f"Invalid revision: {manifest_path}")
        state = manifest.get("state", "active")
        if state not in {"active", "recycled"}:
            raise InvalidTemplateSetError(f"Invalid state: {manifest_path}")
        return WorkpieceRecord(record_id, name, root, front, back, revision, state)

    @staticmethod
    def _update_manifest(record: WorkpieceRecord, *, revision: int, state: str) -> None:
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["revision"] = revision
        manifest["state"] = state
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

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

    @staticmethod
    def _inventory_entries(
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        *,
        source: str,
        added_at: str | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "template_id": f"{direction}:{path.name}",
                "direction": direction,
                "filename": path.name,
                "source": source,
                "added_at": added_at,
            }
            for direction, paths in (("front", front_images), ("back", back_images))
            for path in paths
        ]

    @staticmethod
    def _copy_appended_templates(
        paths: Sequence[Path],
        target: Path,
        *,
        start_index: int,
    ) -> tuple[Path, ...]:
        copied: list[Path] = []
        candidate_index = start_index
        for source in paths:
            while True:
                width = max(2, len(str(candidate_index)))
                destination = target / f"{candidate_index:0{width}d}{source.suffix.lower()}"
                candidate_index += 1
                if not destination.exists():
                    break
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
            created_at = datetime.now(timezone.utc).isoformat()
            manifest = {
                "schema_version": 1,
                "id": new_id,
                "name": display_name,
                "labels": {"0": "front", "1": "back"},
                "template_counts": {"front": len(front), "back": len(back)},
                "created_at": created_at,
                "template_inventory": self._inventory_entries(
                    staged_front,
                    staged_back,
                    source="initial_registration",
                    added_at=created_at,
                ),
                "revision": 1,
                "state": "active",
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report({"phase": "features", "completed": 0, "total": total})

            def cache_progress(label: str, completed: int, side_total: int) -> None:
                offset = len(front) if label == "back" else 0
                report({"phase": "features", "completed": offset + completed, "total": total})

            cache = _call_cache_builder(
                build_cache,
                staged_front,
                staged_back,
                cache_progress,
                library_revision=1,
            )
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
                1,
                "active",
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

    def prepare_append(
        self,
        base_record: WorkpieceRecord,
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        build_cache: CacheBuilder,
        *,
        operation_id: str,
        progress_callback: ProgressCallback | None = None,
        source: str = "manual_append",
    ) -> PreparedTemplateUpdate:
        """Build a complete append candidate without mutating the active record."""
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a non-empty string")
        existing = self._records.get(base_record.id)
        if (
            existing is None
            or existing.revision != base_record.revision
            or existing.root != base_record.root
        ):
            raise StaleWorkpieceRevisionError(
                f"Workpiece revision changed: expected {base_record.revision}"
            )
        if not front_images and not back_images:
            raise InvalidTemplateSetError("At least one confirmed template is required")
        total = (
            len(base_record.front_images)
            + len(base_record.back_images)
            + len(front_images)
            + len(back_images)
        )

        def report(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event)
            except Exception:
                LOGGER.debug("Ignoring append progress callback failure", exc_info=True)

        seen_images: dict[str, Path] = {}
        for path in (*base_record.front_images, *base_record.back_images):
            image = read_color_image(path)
            if image is None:
                raise InvalidTemplateSetError(f"Unreadable existing template: {path}")
            resolved = Path(path).resolve()
            seen_images[f"path:{str(resolved).casefold()}"] = resolved
            seen_images[f"content:{_image_fingerprint(image)}"] = resolved
        new_front = _validate_images(front_images, "front", seen_images) if front_images else ()
        new_back = _validate_images(back_images, "back", seen_images) if back_images else ()
        all_front = tuple(base_record.front_images) + new_front
        all_back = tuple(base_record.back_images) + new_back
        item_digests = tuple(
            _image_fingerprint(read_color_image(path))
            for path in (*new_front, *new_back)
        )
        old_inventory = self.get_template_inventory(base_record.id)
        staging = self.library_dir / f".staging-{uuid.uuid4().hex}"
        try:
            report({"phase": "copying", "completed": 0, "total": total})
            shutil.copytree(base_record.root, staging)
            (staging / TEMPLATE_CACHE_FILE_NAME).unlink(missing_ok=True)
            staged_existing_front = tuple(staging / "0" / path.name for path in base_record.front_images)
            staged_existing_back = tuple(staging / "1" / path.name for path in base_record.back_images)
            staged_new_front = self._copy_appended_templates(
                new_front,
                staging / "0",
                start_index=len(base_record.front_images),
            )
            staged_new_back = self._copy_appended_templates(
                new_back,
                staging / "1",
                start_index=len(base_record.back_images),
            )
            staged_front = staged_existing_front + staged_new_front
            staged_back = staged_existing_back + staged_new_back
            manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
            updated_at = datetime.now(timezone.utc).isoformat()
            manifest.update(
                {
                    "schema_version": max(3, int(manifest.get("schema_version", 1))),
                    "id": base_record.id,
                    "name": base_record.name,
                    "labels": {"0": "front", "1": "back"},
                    "template_counts": {"front": len(all_front), "back": len(all_back)},
                    "revision": base_record.revision + 1,
                    "state": "active",
                    "updated_at": updated_at,
                    "template_inventory": old_inventory + self._inventory_entries(
                        staged_new_front,
                        staged_new_back,
                        source=source,
                        added_at=updated_at,
                    ),
                    "last_template_update": {
                        "operation_id": operation_id,
                        "base_revision": base_record.revision,
                        "target_revision": base_record.revision + 1,
                        "item_digests": list(item_digests),
                    },
                }
            )
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report({"phase": "features", "completed": 0, "total": total})

            def cache_progress(label: str, completed: int, side_total: int) -> None:
                offset = len(all_front) if label == "back" else 0
                report({"phase": "features", "completed": offset + completed, "total": total})

            cache = _call_cache_builder(
                build_cache,
                staged_front,
                staged_back,
                cache_progress,
                library_revision=base_record.revision + 1,
            )
            staged_record = WorkpieceRecord(
                base_record.id,
                base_record.name,
                staging,
                staged_front,
                staged_back,
                base_record.revision + 1,
                "active",
            )
            return PreparedTemplateUpdate(
                operation_id=operation_id,
                workpiece_id=base_record.id,
                base_revision=base_record.revision,
                base_root=base_record.root,
                staging_root=staging,
                staged_record=staged_record,
                candidate_cache=cache,
                item_digests=item_digests,
            )
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def commit_prepared(
        self,
        prepared: PreparedTemplateUpdate,
    ) -> tuple[WorkpieceRecord, Path | None]:
        """Commit a prepared directory using only checked atomic renames."""
        if prepared.consumed:
            raise StaleWorkpieceRevisionError("Prepared template update was already consumed")
        current = self._records.get(prepared.workpiece_id)
        if (
            current is None
            or current.revision != prepared.base_revision
            or current.root != prepared.base_root
            or not current.root.exists()
        ):
            raise StaleWorkpieceRevisionError(
                f"Workpiece revision changed: expected {prepared.base_revision}"
            )
        if not prepared.staging_root.exists():
            raise StaleWorkpieceRevisionError("Prepared template staging directory is missing")
        backup_root = self.library_dir / f".backup-{uuid.uuid4().hex}"
        final_root = current.root
        committed = WorkpieceRecord(
            current.id,
            current.name,
            final_root,
            tuple(final_root / "0" / path.name for path in prepared.staged_record.front_images),
            tuple(final_root / "1" / path.name for path in prepared.staged_record.back_images),
            prepared.staged_record.revision,
            "active",
        )
        os.replace(final_root, backup_root)
        try:
            os.replace(prepared.staging_root, final_root)
        except Exception:
            os.replace(backup_root, final_root)
            raise
        self._records[prepared.workpiece_id] = committed
        prepared.consumed = True
        return committed, backup_root

    def abort_prepared(self, prepared: PreparedTemplateUpdate) -> None:
        """Discard a prepared update; repeated calls are safe."""
        if prepared.consumed:
            return
        staging = prepared.staging_root.resolve()
        library_root = self.library_dir.resolve()
        if staging.parent != library_root or not staging.name.startswith(".staging-"):
            raise WorkpieceLibraryError(f"Invalid staging directory: {staging}")
        if staging.exists():
            shutil.rmtree(staging)
        prepared.consumed = True

    def remove_retired(self, retired_root: Path | None) -> None:
        """Remove the exact backup returned by commit after publication unlocks."""
        if retired_root is None:
            return
        retired = Path(retired_root).resolve()
        library_root = self.library_dir.resolve()
        if retired.parent != library_root or not retired.name.startswith(".backup-"):
            raise WorkpieceLibraryError(f"Invalid retired directory: {retired}")
        if retired.exists():
            shutil.rmtree(retired)

    def append_templates(
        self,
        workpiece_id: str,
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        build_cache: CacheBuilder,
        *,
        operation_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
        source: str = "manual_append",
    ) -> tuple[WorkpieceRecord, TemplateCache]:
        """Compatibility wrapper around prepare, commit, and retired cleanup."""
        prepared = self.prepare_append(
            self._records[workpiece_id],
            front_images,
            back_images,
            build_cache,
            operation_id=operation_id or uuid.uuid4().hex,
            progress_callback=progress_callback,
            source=source,
        )
        try:
            record, retired = self.commit_prepared(prepared)
        except Exception:
            self.abort_prepared(prepared)
            raise
        self.remove_retired(retired)
        return record, prepared.candidate_cache

    def recover(
        self,
        build_cache: CacheBuilder,
        *,
        cache_loader: CacheLoader | None = None,
        cache_saver: CacheSaver | None = None,
    ) -> list[tuple[WorkpieceRecord, TemplateCache]]:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self._records.clear()
        self._recycled_records.clear()
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
                if record.state != "active":
                    raise InvalidTemplateSetError(f"Inactive workpiece in active directory: {root}")
                if self._find_by_name(record.name) is not None:
                    raise InvalidTemplateSetError(f"Duplicate workpiece name: {record.name}")
                cache = None
                if cache_loader is not None:
                    try:
                        cache = cache_loader(record)
                    except Exception as exc:
                        LOGGER.warning("Ignoring invalid template cache for %s: %s", root, exc)
                if cache is None:
                    cache = _call_cache_builder(
                        build_cache,
                        record.front_images,
                        record.back_images,
                        None,
                        library_revision=record.revision,
                    )
                    if cache_saver is not None:
                        try:
                            cache_saver(record, cache)
                        except Exception as exc:
                            LOGGER.warning("Unable to persist template cache for %s: %s", root, exc)
            except Exception as exc:
                LOGGER.warning("Skipping invalid workpiece %s: %s", root, exc)
                continue
            self._records[record.id] = record
            recovered.append((record, cache))
        recycle_root = self.library_dir / ".recycled"
        if recycle_root.is_dir():
            for root in sorted(recycle_root.iterdir(), key=lambda path: path.name):
                if not root.is_dir():
                    continue
                try:
                    record = self._record_from_root(root)
                    if record.state != "recycled":
                        raise InvalidTemplateSetError(f"Invalid recycled state: {root}")
                    self._recycled_records[record.id] = record
                except Exception as exc:
                    LOGGER.warning("Skipping invalid recycled workpiece %s: %s", root, exc)
        return recovered

    def list_workpieces(self) -> list[dict[str, str]]:
        return [
            {"id": record.id, "name": record.name}
            for record in sorted(self._records.values(), key=lambda item: item.name.casefold())
        ]

    def get_workpiece_metadata(self, workpiece_id: str) -> dict[str, Any]:
        record = self._records[workpiece_id]
        manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
        return {"updated_at": manifest.get("updated_at") or manifest.get("created_at")}

    def get_template_inventory(self, workpiece_id: str) -> list[dict[str, Any]]:
        record = self._records[workpiece_id]
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inventory = self._validated_template_inventory(
            manifest_path,
            manifest,
            tuple((record.root / "0").iterdir()),
            tuple((record.root / "1").iterdir()),
        )
        if inventory is not None:
            return inventory[0]
        return self._inventory_entries(
            record.front_images,
            record.back_images,
            source="initial_registration",
            added_at=manifest.get("created_at"),
        )

    def get(self, workpiece_id: str) -> WorkpieceRecord:
        return self._records[workpiece_id]

    def replace_geometry_profile_pointers(
        self,
        workpiece_id: str,
        *,
        expected_revision: int,
        active_revision: int | None,
        previous_active_revision: int | None,
    ) -> WorkpieceRecord:
        """Atomically update geometry profile pointers and the record revision.

        This method intentionally does not inspect validation reports or build a
        cache.  The catalog owns those concerns and calls this small persistence
        boundary only after a candidate profile has been fully prepared.
        """
        record = self._records[workpiece_id]
        if type(expected_revision) is not int or expected_revision != record.revision:
            raise StaleWorkpieceRevisionError(
                f"Workpiece revision changed: expected {expected_revision}, current {record.revision}"
            )
        for value, field in (
            (active_revision, "active_revision"),
            (previous_active_revision, "previous_active_revision"),
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise WorkpieceLibraryError(f"{field} must be a positive integer or null")
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        next_revision = record.revision + 1
        manifest.update(
            {
                "schema_version": max(2, int(manifest.get("schema_version", 1))),
                "revision": next_revision,
                "state": "active",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "geometry_mask_active_revision": active_revision,
                "geometry_mask_previous_active_revision": previous_active_revision,
            }
        )
        temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        updated = WorkpieceRecord(
            record.id,
            record.name,
            record.root,
            record.front_images,
            record.back_images,
            next_revision,
            "active",
        )
        self._records[workpiece_id] = updated
        return updated

    def list_recycled(self) -> list[dict[str, str]]:
        return [
            {"id": record.id, "name": record.name, "revision": record.revision}
            for record in sorted(self._recycled_records.values(), key=lambda item: item.name.casefold())
        ]

    def get_recycled(self, workpiece_id: str) -> WorkpieceRecord:
        return self._recycled_records[workpiece_id]

    def recycle(self, workpiece_id: str) -> WorkpieceRecord:
        record = self._records[workpiece_id]
        recycle_root = self.library_dir / ".recycled"
        recycle_root.mkdir(parents=True, exist_ok=True)
        target = recycle_root / record.id
        if target.exists():
            raise WorkpieceLibraryError(f"Recycled workpiece already exists: {record.id}")
        next_revision = record.revision + 1
        self._update_manifest(record, revision=next_revision, state="recycled")
        os.replace(record.root, target)
        recycled = WorkpieceRecord(
            record.id,
            record.name,
            target,
            tuple(target / "0" / path.name for path in record.front_images),
            tuple(target / "1" / path.name for path in record.back_images),
            next_revision,
            "recycled",
        )
        self._records.pop(record.id, None)
        self._recycled_records[record.id] = recycled
        return recycled

    def restore(self, workpiece_id: str) -> WorkpieceRecord:
        record = self._recycled_records[workpiece_id]
        if self._find_by_name(record.name) is not None or workpiece_id in self._records:
            raise WorkpieceLibraryError(f"Restore conflict for workpiece: {record.name}")
        target = self.library_dir / record.id
        if target.exists():
            raise WorkpieceLibraryError(f"Restore target already exists: {record.id}")
        next_revision = record.revision + 1
        self._update_manifest(record, revision=next_revision, state="active")
        os.replace(record.root, target)
        restored = WorkpieceRecord(
            record.id,
            record.name,
            target,
            tuple(target / "0" / path.name for path in record.front_images),
            tuple(target / "1" / path.name for path in record.back_images),
            next_revision,
            "active",
        )
        self._recycled_records.pop(record.id, None)
        self._records[record.id] = restored
        return restored

    def purge(self, workpiece_id: str) -> None:
        record = self._recycled_records[workpiece_id]
        shutil.rmtree(record.root)
        self._recycled_records.pop(workpiece_id, None)

    def save_annotation_groups(self, workpiece_id: str, groups: list[dict]) -> WorkpieceRecord:
        record = self._records[workpiece_id]
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["schema_version"] = max(2, int(manifest.get("schema_version", 1)))
        next_revision = record.revision + 1
        manifest["revision"] = next_revision
        manifest["state"] = "active"
        existing_groups = manifest.get("interference_groups", [])
        merged: dict[str, dict] = {str(group.get("group_id", group.get("name", ""))): group for group in existing_groups}
        for group in groups:
            key = str(group.get("group_id", group.get("name", "")))
            if key not in merged:
                merged[key] = group
                continue
            current = dict(merged[key])
            annotations = list(current.get("annotations", []))
            for annotation in group.get("annotations", []):
                identity = (annotation.get("orientation"), annotation.get("index"))
                annotations = [item for item in annotations
                               if (item.get("orientation"), item.get("index")) != identity]
                annotations.append(annotation)
            current.update({key_name: value for key_name, value in group.items() if key_name != "annotations"})
            current["annotations"] = annotations
            merged[key] = current
        manifest["interference_groups"] = list(merged.values())
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
        updated = WorkpieceRecord(
            record.id,
            record.name,
            record.root,
            record.front_images,
            record.back_images,
            next_revision,
            "active",
        )
        self._records[workpiece_id] = updated
        return updated

    def get_annotation_document(self, workpiece_id: str) -> dict[str, Any]:
        """Return a normalized draft/active annotation document.

        Older manifests contain only ``interference_groups``.  They remain
        readable; only groups whose propagation is already active are treated
        as active until the first versioned mutation writes both collections.
        """
        record = self._records[workpiece_id]
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        draft_groups = deepcopy(manifest.get("interference_groups", []))
        stored_active = manifest.get("active_interference_groups")
        if stored_active is None:
            active_groups = [
                deepcopy(group)
                for group in draft_groups
                if group.get("enabled", True)
                and group.get("propagation", {}).get("state") == "active"
            ]
        else:
            active_groups = deepcopy(stored_active)
        annotation_revision = manifest.get("annotation_revision", record.revision)
        active_annotation_revision = manifest.get("active_annotation_revision", annotation_revision)
        if type(annotation_revision) is not int or annotation_revision < 0:
            annotation_revision = record.revision
        if type(active_annotation_revision) is not int or active_annotation_revision < 0:
            active_annotation_revision = annotation_revision
        return {
            "revision": record.revision,
            "annotation_revision": annotation_revision,
            "active_annotation_revision": active_annotation_revision,
            "draft_groups": draft_groups,
            "active_groups": active_groups,
        }

    def replace_annotation_document(
        self,
        workpiece_id: str,
        draft_groups: list[dict],
        *,
        expected_revision: int,
        active_groups: list[dict] | None = None,
    ) -> WorkpieceRecord:
        """Atomically replace draft annotations and optionally publish active groups."""
        record = self._records[workpiece_id]
        if type(expected_revision) is not int or expected_revision != record.revision:
            raise StaleWorkpieceRevisionError(
                f"Workpiece revision changed: expected {expected_revision}, current {record.revision}"
            )
        manifest_path = record.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_document = self.get_annotation_document(workpiece_id)
        next_revision = record.revision + 1
        next_annotation_revision = previous_document["annotation_revision"] + 1
        if active_groups is None:
            next_active_groups = previous_document["active_groups"]
            next_active_annotation_revision = previous_document["active_annotation_revision"]
        else:
            next_active_groups = deepcopy(active_groups)
            next_active_annotation_revision = next_annotation_revision
        manifest.update(
            {
                "schema_version": max(2, int(manifest.get("schema_version", 1))),
                "revision": next_revision,
                "state": "active",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "interference_groups": deepcopy(draft_groups),
                "active_interference_groups": next_active_groups,
                "annotation_revision": next_annotation_revision,
                "active_annotation_revision": next_active_annotation_revision,
            }
        )
        temp = manifest_path.with_suffix(".tmp")
        try:
            temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temp.replace(manifest_path)
        except Exception:
            if temp.exists():
                temp.unlink()
            raise
        updated = WorkpieceRecord(
            record.id,
            record.name,
            record.root,
            record.front_images,
            record.back_images,
            next_revision,
            "active",
        )
        self._records[workpiece_id] = updated
        return updated
