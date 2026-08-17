"""Coherent workpiece lifecycle and runtime-cache publication boundary."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import threading
from typing import Any, Sequence

from src.image_io import read_color_image
from src.interference_masks import build_active_mask_map, validate_region


LOGGER = logging.getLogger(__name__)


class WorkpieceCatalogError(RuntimeError):
    """Base catalog error."""


class RestoreConflictError(WorkpieceCatalogError):
    """Raised when restoring would collide with an active workpiece."""


class WorkpieceCatalog:
    """Own persistent lifecycle transitions and classifier cache publication."""

    def __init__(self, library: WorkpieceLibrary, classifier: Any, geometry_profiles: Any | None = None):
        self.library = library
        self.classifier = classifier
        self.geometry_profiles = geometry_profiles
        self._lock = threading.RLock()
        self._operations: dict[str, Any] = {}

    def set_geometry_profiles(self, geometry_profiles: Any | None) -> None:
        with self._lock:
            self.geometry_profiles = geometry_profiles

    def _idempotent(self, operation_id: str, action):
        if not isinstance(operation_id, str) or not operation_id:
            raise ValueError("operation_id must be a non-empty string")
        with self._lock:
            if operation_id in self._operations:
                return self._operations[operation_id]
            result = action()
            self._operations[operation_id] = result
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
            self.classifier.set_template_cache(record.id, cache)
            self._save_template_cache(record, cache)
            return record, cache

    def list_workpieces(self):
        with self._lock:
            return self.library.list_workpieces()

    def get(self, workpiece_id: str) -> WorkpieceRecord:
        with self._lock:
            return self.library.get(workpiece_id)

    def get_annotation_document(self, workpiece_id: str) -> dict[str, Any]:
        with self._lock:
            return self.library.get_annotation_document(workpiece_id)

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
                candidate_cache = None
                if active_groups is not None and self.geometry_profiles is None:
                    masks = build_active_mask_map(
                        len(record.front_images), len(record.back_images), active_groups
                    )
                    prepare = getattr(self.classifier, "prepare_template_masks", None)
                    if callable(prepare):
                        candidate_cache, _ = prepare(workpiece_id, masks)
                self.library.replace_annotation_document(
                    workpiece_id,
                    draft_groups,
                    expected_revision=expected_revision,
                    active_groups=active_groups,
                )
                if candidate_cache is not None and self.geometry_profiles is None:
                    self.classifier.set_template_cache(workpiece_id, candidate_cache)
                return self.get_annotation_snapshot(workpiece_id)

        return self._idempotent(operation_id, action)

    def recover(self):
        with self._lock:
            recovered = self.library.recover(
                self.classifier.build_template_cache,
                cache_loader=self._load_template_cache,
                cache_saver=self._save_template_cache,
            )
            for record, cache in recovered:
                self.classifier.set_template_cache(record.id, cache)
                if self.geometry_profiles is not None:
                    try:
                        candidate = self.geometry_profiles.rebuild_active_cache(record.id, record)
                        if candidate is not None:
                            self.classifier.set_template_cache(record.id, candidate)
                            self.geometry_profiles.sync_library_revision(record)
                    except Exception as exc:
                        LOGGER.warning("Unable to restore active geometry profile for %s: %s", record.id, exc)
                else:
                    document = self.library.get_annotation_document(record.id)
                    if document["active_groups"] and hasattr(self.classifier, "prepare_template_masks"):
                        masks = build_active_mask_map(
                            len(record.front_images), len(record.back_images), document["active_groups"]
                        )
                        filtered, _ = self.classifier.prepare_template_masks(record.id, masks)
                        self.classifier.set_template_cache(record.id, filtered)
            return recovered

    def predict(self, workpiece_id: str, image_path: Path):
        with self._lock:
            return self.classifier.predict(workpiece_id, image_path)

    def append_templates(self, workpiece_id: str, front_images: Sequence[Path], back_images: Sequence[Path],
                         *, progress_callback=None):
        with self._lock:
            previous = (self.classifier.get_template_cache(workpiece_id)
                        if hasattr(self.classifier, "get_template_cache") else None)
            record, cache = self.library.append_templates(
                workpiece_id,
                front_images,
                back_images,
                self.classifier.build_template_cache,
                progress_callback=progress_callback,
            )
            self.classifier.set_template_cache(record.id, cache)
            if self.geometry_profiles is not None:
                try:
                    candidate = self.geometry_profiles.rebuild_active_cache(record.id, record)
                    if candidate is not None:
                        self.classifier.set_template_cache(record.id, candidate)
                    self.geometry_profiles.sync_library_revision(record)
                except Exception as exc:
                    LOGGER.warning("Unable to rebuild active geometry profile for appended templates: %s", exc)
            elif previous is not None and previous.ignored_regions and hasattr(self.classifier, "set_template_masks"):
                self.classifier.set_template_masks(record.id, previous.ignored_regions)
            current = self.classifier.get_template_cache(record.id) if hasattr(self.classifier, "get_template_cache") else cache
            self._save_template_cache(record, current or cache)
            return record, cache

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
                record = self.library.replace_geometry_profile_pointers(
                    workpiece_id,
                    expected_revision=expected_revision,
                    active_revision=profile_revision,
                    previous_active_revision=previous_profile_revision,
                )
                # set_template_cache is an in-memory reference swap; it does not
                # run model inference.  The durable cache writer stores only the
                # unmasked base cache so recovery can rebuild any active profile.
                self.classifier.set_template_cache(workpiece_id, candidate_cache)
                self._save_template_cache(record, candidate_cache)
                return record

        return self._idempotent(operation_id, action)

    def recycle(self, workpiece_id: str, *, operation_id: str):
        def action():
            with self._lock:
                record = self.library.recycle(workpiece_id)
                self.classifier.remove_template_cache(workpiece_id)
                return {"id": record.id, "name": record.name, "revision": record.revision}

        return self._idempotent(operation_id, action)

    def list_recycled(self):
        with self._lock:
            return self.library.list_recycled()

    def restore(self, workpiece_id: str, *, operation_id: str) -> WorkpieceRecord:
        def action():
            with self._lock:
                try:
                    candidate = self.library.get_recycled(workpiece_id)
                    cache = self._load_template_cache(candidate)
                    if cache is None:
                        cache = self.classifier.build_template_cache(candidate.front_images, candidate.back_images, None)
                    record = self.library.restore(workpiece_id)
                except Exception as exc:
                    if "conflict" in str(exc).casefold():
                        raise RestoreConflictError(str(exc)) from exc
                    raise
                self.classifier.set_template_cache(record.id, cache)
                if self.geometry_profiles is not None:
                    try:
                        candidate = self.geometry_profiles.rebuild_active_cache(record.id, record)
                        if candidate is not None:
                            self.classifier.set_template_cache(record.id, candidate)
                        self.geometry_profiles.sync_library_revision(record)
                    except Exception as exc:
                        LOGGER.warning("Unable to restore active geometry profile for %s: %s", record.id, exc)
                self._save_template_cache(record, cache)
                return record

        return self._idempotent(operation_id, action)

    def purge(self, workpiece_id: str, *, operation_id: str):
        def action():
            with self._lock:
                self.library.purge(workpiece_id)
                return {"id": workpiece_id}

        return self._idempotent(operation_id, action)

    def save_annotations(self, workpiece_id: str, groups: list[dict], *, operation_id: str):
        def action():
            with self._lock:
                record = self.library.get(workpiece_id)
                record = self.library.save_annotation_groups(workpiece_id, groups)
                manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
                effective_groups = manifest.get("interference_groups", [])
                if self.geometry_profiles is not None:
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
                if not has_unresolved_group:
                    self.classifier.set_template_masks(workpiece_id, masks)
                return {"workpiece_id": workpiece_id, "revision": record.revision, "groups": effective_groups}

        return self._idempotent(operation_id, action)

    def get_annotations(self, workpiece_id: str):
        with self._lock:
            record = self.library.get(workpiece_id)
            manifest = json.loads((record.root / "manifest.json").read_text(encoding="utf-8"))
            return manifest.get("interference_groups", [])
