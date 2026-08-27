"""Coherent workpiece lifecycle and runtime-cache publication boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import logging
import threading
from typing import Any, Sequence
import uuid

from src.image_io import read_color_image
from src.fast_cache_jobs import FastCacheJobManager
from src.interference_masks import build_active_mask_map, validate_region
from src.orientation_classifier import TemplateCache
from src.workpiece_library import (
    PreparedTemplateUpdate,
    StaleWorkpieceRevisionError,
    WorkpieceLibrary,
    WorkpieceRecord,
    _call_cache_builder,
)


LOGGER = logging.getLogger(__name__)
SUMMARY_READ_ATTEMPTS = 3


class WorkpieceCatalogError(RuntimeError):
    """Base catalog error."""


class RestoreConflictError(WorkpieceCatalogError):
    """Raised when restoring would collide with an active workpiece."""


@dataclass(frozen=True)
class ActiveWorkpieceSnapshot:
    """One coherent runtime record/cache revision used by a prediction."""

    record: WorkpieceRecord
    cache: TemplateCache


class WorkpieceCatalog:
    """Own persistent lifecycle transitions and classifier cache publication."""

    def __init__(self, library: WorkpieceLibrary, classifier: Any, geometry_profiles: Any | None = None,
                 fast_jobs: FastCacheJobManager | None = None):
        self.library = library
        self.classifier = classifier
        self.geometry_profiles = geometry_profiles
        self._lock = threading.RLock()
        self._operations: dict[str, Any] = {}
        self._operation_events: dict[str, threading.Event] = {}
        self._snapshots: dict[str, ActiveWorkpieceSnapshot] = {}
        self.fast_jobs = fast_jobs or FastCacheJobManager()

    def _activate(self, record: WorkpieceRecord, cache: TemplateCache) -> None:
        self._snapshots[record.id] = ActiveWorkpieceSnapshot(record, cache)
        self.classifier.set_template_cache(record.id, cache)

    def _fast_cache_enabled(self) -> bool:
        return (
            getattr(self.classifier, "inference_mode", None) in {"fast_geometry", "compare"}
            and callable(getattr(self.classifier, "build_fast_runtime_cache", None))
            and callable(getattr(self.classifier, "save_fast_runtime_cache", None))
        )

    def _schedule_fast_cache(self, record: WorkpieceRecord, cache: TemplateCache) -> None:
        if not self._fast_cache_enabled() or getattr(cache, "fast_runtime", None) is not None:
            return
        geometry_revision = getattr(cache, "geometry_profile_revision", None)
        geometry_profile = getattr(cache, "geometry_profile", None)

        def build(progress):
            return self.classifier.build_fast_runtime_cache(
                record,
                geometry_profile,
                progress,
            )

        def publish(runtime) -> bool:
            def matches_current_revision() -> bool:
                current = self._snapshots.get(record.id)
                return (
                    current is not None
                    and current.record.revision == record.revision
                    and getattr(current.cache, "geometry_profile_revision", None) == geometry_revision
                )

            with self._lock:
                if not matches_current_revision():
                    return False
            try:
                staged = self._stage_fast_runtime(record, runtime, geometry_revision)
            except Exception:
                with self._lock:
                    if not matches_current_revision():
                        return False
                raise
            committed = None
            try:
                with self._lock:
                    if not matches_current_revision():
                        return False
                    current = self._snapshots[record.id]
                    updated_cache = replace(current.cache, fast_runtime=runtime)
                    committed = self.classifier.commit_staged_fast_runtime_cache(
                        current.record,
                        staged,
                    )
                    try:
                        self._activate(current.record, updated_cache)
                    except Exception:
                        if committed is not None:
                            self.classifier.rollback_committed_fast_runtime_cache(committed)
                            committed = None
                        raise
                if committed is not None:
                    try:
                        self.classifier.finalize_staged_fast_runtime_cache(committed)
                    except Exception as exc:
                        LOGGER.warning("Unable to clean committed fast cache staging: %s", exc)
                return True
            finally:
                if staged is not None and committed is None:
                    self.classifier.discard_staged_fast_runtime_cache(staged)

        self.fast_jobs.schedule(
            workpiece_id=record.id,
            library_revision=record.revision,
            geometry_profile_revision=geometry_revision,
            build=build,
            publish=publish,
        )

    def _stage_fast_runtime(self, record: WorkpieceRecord, runtime: Any, geometry_revision: int | None):
        methods = (
            "stage_fast_runtime_cache",
            "commit_staged_fast_runtime_cache",
            "finalize_staged_fast_runtime_cache",
            "discard_staged_fast_runtime_cache",
            "rollback_committed_fast_runtime_cache",
        )
        if not all(callable(getattr(self.classifier, name, None)) for name in methods):
            raise WorkpieceCatalogError("classifier does not support staged fast-cache persistence")
        staged = self.classifier.stage_fast_runtime_cache(
            record,
            runtime,
            geometry_profile_revision=geometry_revision,
        )
        if staged is None:
            raise WorkpieceCatalogError("classifier returned no staged fast-cache payload")
        return staged

    def _build_fast_candidate(
        self,
        record: WorkpieceRecord,
        cache: TemplateCache,
    ) -> TemplateCache:
        if not self._fast_cache_enabled():
            return cache
        runtime = self.classifier.build_fast_runtime_cache(
            record,
            getattr(cache, "geometry_profile", None),
            None,
        )
        return replace(cache, fast_runtime=runtime)

    def _materialize_fast_profile(
        self,
        record: WorkpieceRecord,
        base_cache: TemplateCache,
    ) -> TemplateCache:
        geometry_revision = None
        geometry_profile = None
        geometry_profiles = self.geometry_profiles
        if geometry_profiles is not None:
            geometry_profiles.sync_library_revision(record)
            profile_snapshot = geometry_profiles.snapshot(record.id)
            active_revision = profile_snapshot.get("active_revision")
            active_profile = profile_snapshot.get("active")
            if type(active_revision) is int and isinstance(active_profile, dict):
                geometry_revision = active_revision
                geometry_profile = dict(active_profile)
                geometry_profile["profile_revision"] = active_revision
        runtime = getattr(base_cache, "fast_runtime", None)
        runtime_library_revision = getattr(runtime, "library_revision", None)
        runtime_geometry_revision = getattr(runtime, "geometry_profile_revision", None)
        if isinstance(runtime, dict):
            runtime_library_revision = runtime.get("library_revision")
            runtime_geometry_revision = runtime.get("geometry_profile_revision")
        if (
            runtime_library_revision != record.revision
            or runtime_geometry_revision != geometry_revision
        ):
            runtime = None
        return replace(
            base_cache,
            geometry_profile=geometry_profile,
            geometry_profile_revision=geometry_revision,
            fast_runtime=runtime,
        )

    def fast_cache_status(self, workpiece_id: str) -> dict[str, Any]:
        job = self.fast_jobs.snapshot(workpiece_id)
        with self._lock:
            current = self._snapshots.get(workpiece_id)
            ready = current is not None and getattr(current.cache, "fast_runtime", None) is not None
            current_revision = None if current is None else current.record.revision
            geometry_revision = (
                None if current is None else getattr(current.cache, "geometry_profile_revision", None)
            )
        if job is not None and (
            current is None
            or job.library_revision == current_revision
            and job.geometry_profile_revision == geometry_revision
        ):
            state = "not_ready" if job.state == "stale" and current is not None else job.state
            return {
                "state": state,
                "completed": job.completed,
                "total": job.total,
                "elapsed_ms": job.elapsed_ms,
                "error": job.error,
            }
        return {
            "state": "ready" if ready else "not_ready",
            "completed": 0,
            "total": 0,
            "elapsed_ms": 0.0,
            "error": None,
        }

    def shutdown(self) -> None:
        self.fast_jobs.shutdown()

    def capture_snapshot(self, workpiece_id: str) -> ActiveWorkpieceSnapshot:
        with self._lock:
            return self._snapshots[workpiece_id]

    def set_geometry_profiles(self, geometry_profiles: Any | None) -> None:
        with self._lock:
            self.geometry_profiles = geometry_profiles

    def _idempotent(self, operation_id: str, action):
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a non-empty string")
        while True:
            with self._lock:
                if operation_id in self._operations:
                    return self._operations[operation_id]
                event = self._operation_events.get(operation_id)
                if event is None:
                    event = threading.Event()
                    self._operation_events[operation_id] = event
                    break
            event.wait()
        try:
            result = action()
        except Exception:
            with self._lock:
                self._operation_events.pop(operation_id, None)
                event.set()
            raise
        with self._lock:
            self._operations[operation_id] = result
            self._operation_events.pop(operation_id, None)
            event.set()
            return result

    def _load_template_cache(self, record: WorkpieceRecord):
        loader = getattr(self.classifier, "load_template_cache", None)
        return loader(record) if callable(loader) else None

    def _save_template_cache(self, record: WorkpieceRecord, cache) -> None:
        saver = getattr(self.classifier, "save_template_cache", None)
        if not callable(saver):
            return
        try:
            saver(record, cache)
        except Exception as exc:
            LOGGER.warning("Unable to persist template cache for %s: %s", record.id, exc)

    def register(self, name: str, front_images: Sequence[Path], back_images: Sequence[Path], replace: bool,
                 *, progress_callback=None):
        with self._lock:
            record, cache = self.library.register(
                name,
                front_images,
                back_images,
                replace,
                self.classifier.build_template_cache,
                progress_callback=progress_callback,
            )
            self._activate(record, cache)
            self._save_template_cache(record, cache)
            self._schedule_fast_cache(record, cache)
            return record, cache

    @staticmethod
    def _summary_from_snapshot(
        record: WorkpieceRecord,
        metadata: dict[str, Any],
        detectable: bool,
        geometry_profiles: Any | None,
    ) -> dict[str, Any]:
        geometry = geometry_profiles.snapshot(record.id) if geometry_profiles is not None else {}
        active = geometry.get("active") if isinstance(geometry, dict) else None
        rules = active.get("rules") if isinstance(active, dict) else None
        return {
            "id": record.id,
            "name": record.name,
            "revision": record.revision,
            "template_counts": {
                "front": len(record.front_images),
                "back": len(record.back_images),
            },
            "updated_at": metadata.get("updated_at"),
            "geometry_status": geometry.get("profile_status") or "not_configured"
            if isinstance(geometry, dict) else "not_configured",
            "geometry_rule_count": len(rules) if isinstance(rules, list) else 0,
            "detectable": detectable,
        }

    def list_workpiece_summaries(self) -> list[dict[str, Any]]:
        for _ in range(SUMMARY_READ_ATTEMPTS):
            with self._lock:
                geometry_profiles = self.geometry_profiles
                captured = [
                    (
                        record,
                        self.library.get_workpiece_metadata(record.id),
                        record.id in self._snapshots,
                    )
                    for item in self.library.list_workpieces()
                    for record in (self.library.get(item["id"]),)
                ]
                captured_signatures = [self._record_signature(record) for record, _, _ in captured]
            summaries = []
            stable = True
            for record, metadata, detectable in captured:
                try:
                    summary = self._summary_from_snapshot(
                        record,
                        metadata,
                        detectable,
                        geometry_profiles,
                    )
                except KeyError:
                    with self._lock:
                        if not self._record_is_current(record, geometry_profiles):
                            stable = False
                            break
                    raise
                with self._lock:
                    if not self._record_is_current(record, geometry_profiles):
                        stable = False
                        break
                summaries.append(summary)
                summary["fast_cache"] = self.fast_cache_status(record.id)
            if not stable:
                continue
            with self._lock:
                current_signatures = [
                    self._record_signature(self.library.get(item["id"]))
                    for item in self.library.list_workpieces()
                ]
                if (
                    current_signatures == captured_signatures
                    and self.geometry_profiles is geometry_profiles
                ):
                    return summaries
        raise StaleWorkpieceRevisionError(
            "Workpiece list changed repeatedly while reading summaries"
        )

    def list_workpieces(self):
        return self.list_workpiece_summaries()

    def get_workpiece_details(self, workpiece_id: str) -> dict[str, Any]:
        for _ in range(SUMMARY_READ_ATTEMPTS):
            with self._lock:
                record = self.library.get(workpiece_id)
                metadata = self.library.get_workpiece_metadata(workpiece_id)
                inventory = self.library.get_template_inventory(workpiece_id)
                detectable = record.id in self._snapshots
                geometry_profiles = self.geometry_profiles
            try:
                summary = self._summary_from_snapshot(
                    record,
                    metadata,
                    detectable,
                    geometry_profiles,
                )
                templates = self._template_details(record, inventory)
            except KeyError:
                with self._lock:
                    if not self._record_is_current(record, geometry_profiles):
                        continue
                raise
            with self._lock:
                if self._record_is_current(record, geometry_profiles):
                    summary["fast_cache"] = self.fast_cache_status(record.id)
                    return {**summary, "templates": templates}
        with self._lock:
            self.library.get(workpiece_id)
        raise StaleWorkpieceRevisionError(
            f"Workpiece changed repeatedly while reading details: {workpiece_id}"
        )

    @staticmethod
    def _record_signature(record: WorkpieceRecord) -> tuple[str, int, Path, str]:
        return record.id, record.revision, record.root, record.state

    def _record_is_current(
        self,
        record: WorkpieceRecord,
        geometry_profiles: Any | None,
    ) -> bool:
        if self.geometry_profiles is not geometry_profiles:
            return False
        try:
            current = self.library.get(record.id)
        except KeyError:
            return False
        return self._record_signature(current) == self._record_signature(record)

    @staticmethod
    def _template_details(
        record: WorkpieceRecord,
        inventory: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        template_paths = {
            (direction, path.name): path.resolve()
            for direction, paths in (("front", record.front_images), ("back", record.back_images))
            for path in paths
        }
        templates = []
        for item in inventory:
            direction = item["direction"]
            filename = item["filename"]
            path = template_paths.get((direction, filename))
            if path is None:
                label_dir = "0" if direction == "front" else "1"
                path = (record.root / label_dir / filename).resolve()
            templates.append(
                {
                    "template_id": item["template_id"],
                    "direction": direction,
                    "preview_path": str(path),
                    "source": item["source"],
                    "added_at": item.get("added_at"),
                    "readable": read_color_image(path) is not None,
                }
            )
        return templates

    def get(self, workpiece_id: str) -> WorkpieceRecord:
        with self._lock:
            return self.library.get(workpiece_id)

    def get_annotation_document(self, workpiece_id: str) -> dict[str, Any]:
        with self._lock:
            return self.library.get_annotation_document(workpiece_id)

    def _prepare_legacy_annotation_cache(
        self,
        record: WorkpieceRecord,
        *,
        base_cache: TemplateCache | None = None,
    ):
        """Build the pre-geometry active mask view, if one exists."""
        document = self.library.get_annotation_document(record.id)
        groups = document.get("active_groups", [])
        prepare = getattr(self.classifier, "prepare_template_masks", None)
        if not groups or not callable(prepare):
            return None
        masks = build_active_mask_map(len(record.front_images), len(record.back_images), groups)
        candidate, _ = prepare(record.id, masks, base_cache=base_cache)
        return candidate

    @staticmethod
    def _has_staged_active_geometry(record: WorkpieceRecord) -> bool:
        try:
            manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("geometry_mask_active_revision") is not None:
                return True
            profile_path = record.root / "geometry_masks" / "profile.json"
            if profile_path.exists():
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                return profile.get("active_revision") is not None
        except (OSError, ValueError, TypeError):
            return False
        return False

    def _prepare_effective_staged_cache(
        self,
        prepared: PreparedTemplateUpdate,
    ) -> TemplateCache:
        base_cache = prepared.candidate_cache
        geometry_profiles = self.geometry_profiles
        if geometry_profiles is not None:
            prepare = getattr(geometry_profiles, "prepare_staged_active_cache", None)
            candidate = prepare(prepared.staged_record, base_cache) if callable(prepare) else None
            if candidate is not None:
                return candidate
            if self._has_staged_active_geometry(prepared.staged_record):
                raise WorkpieceCatalogError("Unable to prepare the active geometry cache for appended templates")
        legacy = self._prepare_legacy_annotation_cache(
            prepared.staged_record,
            base_cache=base_cache,
        )
        if legacy is not None:
            return legacy
        current = self.capture_snapshot(prepared.workpiece_id).cache
        ignored_regions = getattr(current, "ignored_regions", None)
        prepare_masks = getattr(self.classifier, "prepare_template_masks", None)
        if ignored_regions and callable(prepare_masks):
            candidate, _ = prepare_masks(
                prepared.workpiece_id,
                ignored_regions,
                base_cache=base_cache,
            )
            return candidate
        return base_cache

    def restore_legacy_annotation_cache(
        self,
        workpiece_id: str,
        *,
        expected_revision: int,
        operation_id: str,
    ) -> WorkpieceRecord:
        """Restore the active legacy mask cache after geometry rollback."""
        def action():
            with self._lock:
                record = self.library.get(workpiece_id)
                if record.revision != expected_revision:
                    raise WorkpieceCatalogError(
                        f"Workpiece revision changed: expected {expected_revision}, current {record.revision}"
                    )
                current = self._snapshots.get(workpiece_id)
                if current is None or current.record.revision != expected_revision:
                    raise WorkpieceCatalogError("runtime revision changed before legacy rollback")
                target = replace(record, revision=expected_revision + 1)
                base_cache = current.cache
            candidate = self._prepare_legacy_annotation_cache(record, base_cache=base_cache)
            if candidate is None:
                raise WorkpieceCatalogError("no active legacy annotation cache is available")
            candidate = self._build_fast_candidate(target, candidate)
            runtime = getattr(candidate, "fast_runtime", None) if self._fast_cache_enabled() else None
            staged = self._stage_fast_runtime(target, runtime, None) if runtime is not None else None
            committed = None
            try:
                with self._lock:
                    record = self.library.get(workpiece_id)
                    current = self._snapshots.get(workpiece_id)
                    if (
                        record.revision != expected_revision
                        or current is None
                        or current.record.revision != expected_revision
                    ):
                        raise WorkpieceCatalogError(
                            f"Workpiece revision changed: expected {expected_revision}, current {record.revision}"
                        )
                    if runtime is not None:
                        committed = self.classifier.commit_staged_fast_runtime_cache(target, staged)
                    try:
                        updated = self.library.replace_geometry_profile_pointers(
                            workpiece_id,
                            expected_revision=expected_revision,
                            active_revision=None,
                            previous_active_revision=None,
                        )
                        self._activate(updated, candidate)
                    except Exception:
                        if committed is not None:
                            self.classifier.rollback_committed_fast_runtime_cache(committed)
                            committed = None
                        raise
                persisted = replace(candidate, fast_runtime=None) if runtime is not None else candidate
                self._save_template_cache(updated, persisted)
                if committed is not None:
                    try:
                        self.classifier.finalize_staged_fast_runtime_cache(committed)
                    except Exception as exc:
                        LOGGER.warning("Unable to clean committed fast cache staging: %s", exc)
                return updated
            finally:
                if staged is not None and committed is None:
                    self.classifier.discard_staged_fast_runtime_cache(staged)

        return self._idempotent(operation_id, action)

    @staticmethod
    def _template_id(orientation: str, path: Path) -> str:
        return f"{orientation}:{path.name}"

    @staticmethod
    def _keypoint_count(features: Any) -> int:
        keypoints = features.get("keypoints") if isinstance(features, dict) else None
        shape = tuple(getattr(keypoints, "shape", ()))
        if len(shape) == 3 and shape[0] == 1:
            return int(shape[1])
        if shape:
            return int(shape[0])
        return 0

    def _template_rows(self, record: WorkpieceRecord) -> list[dict[str, Any]]:
        cache = self.classifier.get_template_cache(record.id) if hasattr(self.classifier, "get_template_cache") else None
        raw = getattr(cache, "raw_local_features", None) if cache is not None else None
        local = getattr(cache, "local_features", None) if cache is not None else None
        rows: list[dict[str, Any]] = []
        for orientation, paths in (("front", record.front_images), ("back", record.back_images)):
            for index, path in enumerate(paths):
                image = read_color_image(path)
                readable = image is not None
                before_features = raw.get(orientation, [])[index] if raw and index < len(raw.get(orientation, [])) else {}
                after_features = local.get(orientation, [])[index] if local and index < len(local.get(orientation, [])) else before_features
                before = self._keypoint_count(before_features)
                after = self._keypoint_count(after_features)
                rows.append({
                    "template_id": self._template_id(orientation, path),
                    "orientation": orientation,
                    "index": index,
                    "preview_path": str(path),
                    "width": int(image.shape[1]) if readable else 0,
                    "height": int(image.shape[0]) if readable else 0,
                    "readable": readable,
                    "mask_effect": {
                        "keypoints_before": before,
                        "keypoints_after": after,
                        "remaining_ratio": float(after / before) if before else 1.0,
                    },
                })
        return rows

    def get_annotation_snapshot(self, workpiece_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.library.get(workpiece_id)
            document = self.library.get_annotation_document(workpiece_id)
            templates = self._template_rows(record)
            active_by_id = {
                str(group.get("group_id", group.get("name", ""))): group
                for group in document["active_groups"]
            }
            groups: list[dict[str, Any]] = []
            for source_group in document["draft_groups"]:
                group = json.loads(json.dumps(source_group, ensure_ascii=False))
                group_id = str(group.get("group_id", group.get("name", "")))
                annotations = []
                for annotation in group.get("annotations", []):
                    item = dict(annotation)
                    orientation = item.get("orientation")
                    index = item.get("index")
                    matching = next(
                        (row for row in templates
                         if row["orientation"] == orientation and row["index"] == index),
                        None,
                    )
                    if matching is not None:
                        item.setdefault("template_id", matching["template_id"])
                    annotations.append(item)
                group["annotations"] = annotations
                target_rows = []
                for template in templates:
                    target = next(
                        (item for item in annotations if item.get("template_id") == template["template_id"]
                         or (item.get("orientation") == template["orientation"]
                             and item.get("index") == template["index"])),
                        None,
                    )
                    unresolved = next(
                        (item for item in group.get("propagation", {}).get("unresolved", [])
                         if item.get("orientation") == template["orientation"]
                         and item.get("index") == template["index"]),
                        None,
                    )
                    target_rows.append({
                        "template_id": template["template_id"],
                        "orientation": template["orientation"],
                        "index": template["index"],
                        "state": target.get("status", "active") if target else (
                            "unresolved" if unresolved else "unresolved"
                        ),
                        "provenance": target.get("provenance", "manual") if target else "none",
                        "regions": target.get("regions", []) if target else [],
                        "diagnostics": target.get("diagnostics", unresolved or {}) if target else (unresolved or {}),
                    })
                manual_count = sum(1 for item in annotations if item.get("provenance", "manual") == "manual")
                automatic_count = sum(1 for item in annotations if item.get("provenance") == "automatic")
                review_count = sum(1 for item in annotations if item.get("status") == "needs_review")
                unresolved_count = sum(1 for item in target_rows if item["state"] == "unresolved")
                group["draft_state"] = group.get("propagation", {}).get("state", "needs_review")
                active_group = active_by_id.get(group_id)
                group["active_state"] = (
                    "active" if active_group is not None and group.get("enabled", True) else
                    "disabled" if not group.get("enabled", True) else "pending"
                )
                group["summary"] = {
                    "manual_count": manual_count,
                    "automatic_count": automatic_count,
                    "needs_review_count": review_count,
                    "unresolved_count": unresolved_count,
                }
                group["targets"] = target_rows
                groups.append(group)
            return {
                "workpiece_id": record.id,
                "revision": document["revision"],
                "annotation_revision": document["annotation_revision"],
                "active_annotation_revision": document["active_annotation_revision"],
                "legacy_archived": bool(
                    document["draft_groups"] or document["active_groups"]
                ),
                "templates": templates,
                "groups": groups,
            }

    def commit_annotation_document(
        self,
        workpiece_id: str,
        draft_groups: list[dict],
        *,
        expected_revision: int,
        operation_id: str,
        active_groups: list[dict] | None = None,
    ) -> dict[str, Any]:
        def action():
            with self._lock:
                record = self.library.get(workpiece_id)
                if record.revision != expected_revision:
                    raise StaleWorkpieceRevisionError(
                        f"Workpiece revision changed: expected {expected_revision}, current {record.revision}"
                    )
                snapshot = self._snapshots[workpiece_id]
                if snapshot.record.revision != expected_revision:
                    raise StaleWorkpieceRevisionError(
                        f"Runtime revision changed: expected {expected_revision}, "
                        f"current {snapshot.record.revision}"
                    )
                base_cache = snapshot.cache
            candidate_cache = None
            if active_groups is not None and self.geometry_profiles is None:
                masks = build_active_mask_map(
                    len(record.front_images), len(record.back_images), active_groups
                )
                prepare = getattr(self.classifier, "prepare_template_masks", None)
                if callable(prepare):
                    candidate_cache, _ = prepare(
                        workpiece_id, masks, base_cache=base_cache
                    )
            with self._lock:
                current = self._snapshots[workpiece_id]
                if current.record.revision != expected_revision:
                    raise StaleWorkpieceRevisionError(
                        f"Runtime revision changed: expected {expected_revision}, "
                        f"current {current.record.revision}"
                    )
                updated = self.library.replace_annotation_document(
                    workpiece_id,
                    draft_groups,
                    expected_revision=expected_revision,
                    active_groups=active_groups,
                )
                effective_cache = candidate_cache if candidate_cache is not None else current.cache
                self._activate(updated, effective_cache)
                if candidate_cache is not None:
                    self._save_template_cache(updated, candidate_cache)
                return self.get_annotation_snapshot(workpiece_id)

        return self._idempotent(operation_id, action)

    def recover(self):
        recovered = self.library.recover(
            self.classifier.build_template_cache,
            cache_loader=self._load_template_cache,
            cache_saver=self._save_template_cache,
        )
        with self._lock:
            for record, cache in recovered:
                self._activate(record, cache)
        geometry_profiles = self.geometry_profiles
        for record, base_cache in recovered:
            candidate = None
            if geometry_profiles is not None:
                try:
                    if getattr(self.classifier, "inference_mode", None) in {
                        "fast_geometry", "compare"
                    }:
                        candidate = self._materialize_fast_profile(record, base_cache)
                    else:
                        geometry_profiles.sync_library_revision(record)
                        candidate = geometry_profiles.rebuild_active_cache(record.id, record)
                        if candidate is None:
                            candidate = self._prepare_legacy_annotation_cache(
                                record,
                                base_cache=base_cache,
                            )
                except Exception as exc:
                    LOGGER.warning("Unable to restore active geometry profile for %s: %s", record.id, exc)
            else:
                document = self.library.get_annotation_document(record.id)
                if document["active_groups"] and hasattr(self.classifier, "prepare_template_masks"):
                    masks = build_active_mask_map(
                        len(record.front_images), len(record.back_images), document["active_groups"]
                    )
                    candidate, _ = self.classifier.prepare_template_masks(
                        record.id,
                        masks,
                        base_cache=base_cache,
                    )
            if candidate is not None:
                with self._lock:
                    current = self._snapshots.get(record.id)
                    if current is not None and current.record.revision == record.revision:
                        self._activate(record, candidate)
            with self._lock:
                current = self._snapshots.get(record.id)
            if current is not None and current.record.revision == record.revision:
                self._schedule_fast_cache(current.record, current.cache)
        return recovered

    def predict(self, workpiece_id: str, image_path: Path):
        snapshot = self.capture_snapshot(workpiece_id)
        return self.classifier.predict_with_cache(
            snapshot.cache,
            image_path,
            library_revision=snapshot.record.revision,
        )

    def commit_prepared_append(
        self,
        prepared: PreparedTemplateUpdate,
        effective_cache: TemplateCache,
    ) -> WorkpieceRecord:
        retired = None
        with self._lock:
            current = self._snapshots.get(prepared.workpiece_id)
            if current is None or current.record.revision != prepared.base_revision:
                actual = None if current is None else current.record.revision
                raise StaleWorkpieceRevisionError(
                    f"Workpiece revision changed: expected {prepared.base_revision}, current {actual}"
                )
            record, retired = self.library.commit_prepared(prepared)
            self._activate(record, effective_cache)
        try:
            self.library.remove_retired(retired)
        except Exception as exc:
            LOGGER.warning("Unable to remove retired workpiece directory %s: %s", retired, exc)
        return record

    def append_templates(
        self,
        workpiece_id: str,
        front_images: Sequence[Path],
        back_images: Sequence[Path],
        *,
        operation_id: str | None = None,
        progress_callback=None,
        source: str = "manual_append",
    ):
        base = self.capture_snapshot(workpiece_id)
        prepared = self.library.prepare_append(
            base.record,
            front_images,
            back_images,
            self.classifier.build_template_cache,
            operation_id=operation_id or uuid.uuid4().hex,
            progress_callback=progress_callback,
            source=source,
        )
        try:
            effective = self._prepare_effective_staged_cache(prepared)
            saver = getattr(self.classifier, "save_template_cache", None)
            if callable(saver):
                saver(prepared.staged_record, effective)
            total = len(prepared.staged_record.front_images) + len(prepared.staged_record.back_images)
            if progress_callback is not None:
                try:
                    progress_callback({"phase": "committing", "completed": total, "total": total})
                except Exception:
                    LOGGER.debug("Ignoring append progress callback failure", exc_info=True)
            record = self.commit_prepared_append(prepared, effective)
            self._schedule_fast_cache(record, effective)
            return record, effective
        except Exception:
            self.library.abort_prepared(prepared)
            raise

    def publish_geometry_profile(
        self,
        workpiece_id: str,
        candidate_cache: Any,
        *,
        profile_revision: int,
        previous_profile_revision: int | None,
        expected_revision: int,
        operation_id: str,
    ) -> WorkpieceRecord:
        """Switch the persistent geometry pointer and runtime cache together."""
        def action():
            with self._lock:
                current = self.library.get(workpiece_id)
                if current.revision != expected_revision:
                    raise StaleWorkpieceRevisionError(
                        f"Workpiece revision changed: expected {expected_revision}, current {current.revision}"
                    )
                target = replace(current, revision=expected_revision + 1)
            prepared_cache = self._build_fast_candidate(target, candidate_cache)
            runtime = getattr(prepared_cache, "fast_runtime", None) if self._fast_cache_enabled() else None
            staged = (
                self._stage_fast_runtime(target, runtime, profile_revision)
                if runtime is not None else None
            )
            committed = None
            try:
                with self._lock:
                    current = self.library.get(workpiece_id)
                    current_snapshot = self._snapshots.get(workpiece_id)
                    if (
                        current.revision != expected_revision
                        or current_snapshot is None
                        or current_snapshot.record.revision != expected_revision
                    ):
                        raise StaleWorkpieceRevisionError(
                            f"Workpiece revision changed: expected {expected_revision}, "
                            f"current {current.revision}"
                        )
                    if runtime is not None:
                        committed = self.classifier.commit_staged_fast_runtime_cache(target, staged)
                    try:
                        record = self.library.replace_geometry_profile_pointers(
                            workpiece_id,
                            expected_revision=expected_revision,
                            active_revision=profile_revision,
                            previous_active_revision=previous_profile_revision,
                        )
                        # set_template_cache is an in-memory reference swap; it does not
                        # run model inference.  The durable cache writer stores only the
                        # unmasked base cache so recovery can rebuild any active profile.
                        self._activate(record, prepared_cache)
                    except Exception:
                        if committed is not None:
                            self.classifier.rollback_committed_fast_runtime_cache(committed)
                            committed = None
                        raise
                persisted = replace(prepared_cache, fast_runtime=None) if runtime is not None else prepared_cache
                self._save_template_cache(record, persisted)
                if committed is not None:
                    try:
                        self.classifier.finalize_staged_fast_runtime_cache(committed)
                    except Exception as exc:
                        LOGGER.warning("Unable to clean committed fast cache staging: %s", exc)
                self._schedule_fast_cache(record, prepared_cache)
                return record
            finally:
                if staged is not None and committed is None:
                    self.classifier.discard_staged_fast_runtime_cache(staged)

        return self._idempotent(operation_id, action)

    def recycle(self, workpiece_id: str, *, operation_id: str):
        def action():
            with self._lock:
                record = self.library.recycle(workpiece_id)
                self.classifier.remove_template_cache(workpiece_id)
                self._snapshots.pop(workpiece_id, None)
                return {"id": record.id, "name": record.name, "revision": record.revision}

        return self._idempotent(operation_id, action)

    def list_recycled(self):
        with self._lock:
            return self.library.list_recycled()

    def restore(self, workpiece_id: str, *, operation_id: str) -> WorkpieceRecord:
        def action():
            with self._lock:
                recycled = self.library.get_recycled(workpiece_id)
            cache = self._load_template_cache(recycled)
            rebuilt_base = cache is None
            if rebuilt_base:
                cache = _call_cache_builder(
                    self.classifier.build_template_cache,
                    recycled.front_images,
                    recycled.back_images,
                    None,
                    library_revision=recycled.revision + 1,
                )
            with self._lock:
                try:
                    record = self.library.restore(workpiece_id)
                except Exception as exc:
                    if "conflict" in str(exc).casefold():
                        raise RestoreConflictError(str(exc)) from exc
                    raise
                self._activate(record, cache)
            effective = cache
            geometry_profiles = self.geometry_profiles
            if self._fast_cache_enabled():
                try:
                    effective = self._materialize_fast_profile(record, cache)
                except Exception as exc:
                    LOGGER.warning("Unable to restore active geometry profile for %s: %s", record.id, exc)
            elif geometry_profiles is not None:
                try:
                    candidate = geometry_profiles.rebuild_active_cache(record.id, record)
                    if candidate is None:
                        candidate = self._prepare_legacy_annotation_cache(
                            record,
                            base_cache=cache,
                        )
                    if candidate is not None:
                        effective = candidate
                    geometry_profiles.sync_library_revision(record)
                except Exception as exc:
                    LOGGER.warning("Unable to restore active geometry profile for %s: %s", record.id, exc)
            if effective is not cache:
                with self._lock:
                    current = self._snapshots.get(record.id)
                    if current is not None and current.record.revision == record.revision:
                        self._activate(record, effective)
            if rebuilt_base or not self._fast_cache_enabled():
                self._save_template_cache(record, effective)
            self._schedule_fast_cache(record, effective)
            return record

        return self._idempotent(operation_id, action)

    def purge(self, workpiece_id: str, *, operation_id: str):
        def action():
            with self._lock:
                self.library.purge(workpiece_id)
                self._snapshots.pop(workpiece_id, None)
                return {"id": workpiece_id}

        return self._idempotent(operation_id, action)

    def save_annotations(self, workpiece_id: str, groups: list[dict], *, operation_id: str):
        def action():
            with self._lock:
                current = self._snapshots[workpiece_id]
                record = self.library.save_annotation_groups(workpiece_id, groups)
                manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
                effective_groups = manifest.get("interference_groups", [])
                if self.geometry_profiles is not None:
                    self._activate(record, current.cache)
                    return {
                        "workpiece_id": workpiece_id,
                        "revision": record.revision,
                        "groups": effective_groups,
                        "archived": True,
                    }
                masks = {
                    "front": [[] for _ in record.front_images],
                    "back": [[] for _ in record.back_images],
                }
                has_unresolved_group = False
                for group in effective_groups:
                    for annotation in group.get("annotations", []):
                        for region_value in annotation.get("regions", []):
                            validate_region(region_value)
                    if group.get("propagation", {}).get("state") == "needs_review":
                        has_unresolved_group = True
                        continue
                    for annotation in group.get("annotations", []):
                        if not annotation.get("trusted", False) and annotation.get("status", "active") != "active":
                            continue
                        label = annotation.get("orientation")
                        index = annotation.get("index")
                        if label not in masks or type(index) is not int or not 0 <= index < len(masks[label]):
                            continue
                        masks[label][index].extend(annotation.get("regions", []))
                effective_cache = current.cache
                if not has_unresolved_group:
                    prepare = getattr(self.classifier, "prepare_template_masks", None)
                    if callable(prepare):
                        effective_cache, _ = prepare(
                            workpiece_id, masks, base_cache=current.cache
                        )
                    else:
                        self.classifier.set_template_masks(workpiece_id, masks)
                        effective_cache = self.classifier.get_template_cache(workpiece_id)
                self._activate(record, effective_cache)
                if effective_cache is not current.cache:
                    self._save_template_cache(record, effective_cache)
                return {"workpiece_id": workpiece_id, "revision": record.revision, "groups": effective_groups}

        return self._idempotent(operation_id, action)

    def get_annotations(self, workpiece_id: str):
        with self._lock:
            record = self.library.get(workpiece_id)
            manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
            return manifest.get("interference_groups", [])
