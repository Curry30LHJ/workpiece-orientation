"""Persistent, version-aware geometry-mask draft profiles.

This module deliberately owns only the small persistence boundary needed by
the editor.  Validation jobs and cache publication are added on top of this
document format by the catalog in the next implementation step.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Mapping

from src.geometry_calibration import SUPPORTED_MODES, SUPPORTED_SHAPES
from src.image_io import read_color_image
from src.workpiece_library import StaleWorkpieceRevisionError


LOGGER = logging.getLogger(__name__)
PROFILE_SCHEMA_VERSION = 1
PROFILE_DIRECTORY = "geometry_masks"


class GeometryProfileError(RuntimeError):
    """Base error for geometry profile operations."""


class InvalidGeometryProfileError(GeometryProfileError):
    """Raised when a draft cannot be represented by the profile schema."""


class StaleGeometryProfileError(GeometryProfileError):
    """Raised when a draft targets an older library or draft revision."""


class CorruptGeometryProfileError(GeometryProfileError):
    """Raised when an existing profile cannot be safely parsed or validated."""


class GeometryValidationError(GeometryProfileError):
    """Raised when a validation job cannot be created or completed."""


class GeometryValidationJobNotFoundError(GeometryValidationError):
    """Raised when a validation job id is unknown."""


class GeometryProfilePublishError(GeometryProfileError):
    """Raised when an explicit geometry profile publish cannot complete."""


class GeometryProfileNotReadyError(GeometryProfilePublishError):
    """Raised when a caller attempts to publish before validation completes."""


def _finite(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidGeometryProfileError(f"{field} must be numeric") from exc
    if not math.isfinite(converted):
        raise InvalidGeometryProfileError(f"{field} must be finite")
    return converted


def _non_empty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryProfileError(f"{field} must be non-empty text")
    return value.strip()


def _canonical_anchor(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvalidGeometryProfileError(f"{field} must be an object or null")
    shape = value.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise InvalidGeometryProfileError(f"{field}.shape is unsupported: {shape}")
    coarse = value.get("coarse")
    if not isinstance(coarse, Mapping):
        raise InvalidGeometryProfileError(f"{field}.coarse must be an object")
    result: dict[str, Any] = {
        "shape": shape,
        "coarse": {
            "cx": _finite(coarse.get("cx"), f"{field}.coarse.cx"),
            "cy": _finite(coarse.get("cy"), f"{field}.coarse.cy"),
            "angle_deg": _finite(coarse.get("angle_deg", 0.0), f"{field}.coarse.angle_deg"),
        },
    }
    if shape == "circle":
        radius = coarse.get("r", coarse.get("rx"))
        radius = _finite(radius, f"{field}.coarse.r")
        if radius <= 0:
            raise InvalidGeometryProfileError(f"{field}.coarse.r must be positive")
        result["coarse"]["r"] = radius
    elif shape == "ellipse":
        rx = _finite(coarse.get("rx"), f"{field}.coarse.rx")
        ry = _finite(coarse.get("ry"), f"{field}.coarse.ry")
        if rx <= 0 or ry <= 0:
            raise InvalidGeometryProfileError(f"{field}.coarse radii must be positive")
        result["coarse"].update({"rx": rx, "ry": ry})
    else:
        half_width = _finite(coarse.get("half_width"), f"{field}.coarse.half_width")
        half_height = _finite(coarse.get("half_height"), f"{field}.coarse.half_height")
        if half_width <= 0 or half_height <= 0:
            raise InvalidGeometryProfileError(f"{field}.coarse rectangle dimensions must be positive")
        result["coarse"].update({"half_width": half_width, "half_height": half_height})
    return result


def _canonical_rule(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidGeometryProfileError(f"{field} must be an object")
    rule_id = _non_empty_text(value.get("rule_id"), f"{field}.rule_id")
    name = _non_empty_text(value.get("name"), f"{field}.name")
    shape = value.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise InvalidGeometryProfileError(f"{field}.shape is unsupported: {shape}")
    mode = value.get("mode")
    if mode not in SUPPORTED_MODES:
        raise InvalidGeometryProfileError(f"{field}.mode is unsupported: {mode}")
    geometry = value.get("geometry")
    if not isinstance(geometry, Mapping):
        raise InvalidGeometryProfileError(f"{field}.geometry must be an object")
    normalized_geometry: dict[str, Any] = {
        "cx": _finite(geometry.get("cx", 0.0), f"{field}.geometry.cx"),
        "cy": _finite(geometry.get("cy", 0.0), f"{field}.geometry.cy"),
        "angle_deg": _finite(geometry.get("angle_deg", 0.0), f"{field}.geometry.angle_deg"),
    }
    if shape == "circle":
        radius = _finite(geometry.get("r"), f"{field}.geometry.r")
        if radius <= 0:
            raise InvalidGeometryProfileError(f"{field}.geometry.r must be positive")
        normalized_geometry["r"] = radius
    elif shape == "ellipse":
        rx = _finite(geometry.get("rx"), f"{field}.geometry.rx")
        ry = _finite(geometry.get("ry"), f"{field}.geometry.ry")
        if rx <= 0 or ry <= 0:
            raise InvalidGeometryProfileError(f"{field}.geometry radii must be positive")
        normalized_geometry.update({"rx": rx, "ry": ry})
    else:
        half_width = _finite(geometry.get("half_width"), f"{field}.geometry.half_width")
        half_height = _finite(geometry.get("half_height"), f"{field}.geometry.half_height")
        if half_width <= 0 or half_height <= 0:
            raise InvalidGeometryProfileError(f"{field}.geometry rectangle dimensions must be positive")
        normalized_geometry.update({"half_width": half_width, "half_height": half_height})
    margin_ratio = _finite(value.get("margin_ratio", 0.02), f"{field}.margin_ratio")
    if margin_ratio < 0 or margin_ratio >= 0.95:
        raise InvalidGeometryProfileError(f"{field}.margin_ratio must be in [0, 0.95)")
    enabled = value.get("enabled", True)
    if type(enabled) is not bool:
        raise InvalidGeometryProfileError(f"{field}.enabled must be boolean")
    return {
        "rule_id": rule_id,
        "name": name,
        "shape": shape,
        "geometry": normalized_geometry,
        "mode": mode,
        "margin_ratio": margin_ratio,
        "enabled": enabled,
    }


def normalize_geometry_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical JSON-compatible profile or raise a clear error."""
    if not isinstance(profile, Mapping):
        raise InvalidGeometryProfileError("geometry profile must be an object")
    directions = profile.get("directions")
    if not isinstance(directions, Mapping):
        raise InvalidGeometryProfileError("geometry profile.directions must be an object")
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "directions": {},
    }
    for direction in ("front", "back"):
        source = directions.get(direction, {})
        if not isinstance(source, Mapping):
            raise InvalidGeometryProfileError(f"directions.{direction} must be an object")
        anchor = _canonical_anchor(source.get("anchor"), f"directions.{direction}.anchor")
        rules_value = source.get("rules", [])
        if not isinstance(rules_value, list):
            raise InvalidGeometryProfileError(f"directions.{direction}.rules must be a list")
        rules: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for index, raw_rule in enumerate(rules_value):
            rule = _canonical_rule(raw_rule, f"directions.{direction}.rules[{index}]")
            if rule["rule_id"] in seen_ids:
                raise InvalidGeometryProfileError(f"duplicate rule_id: {rule['rule_id']}")
            seen_ids.add(rule["rule_id"])
            rules.append(rule)
        if anchor is None and any(rule["enabled"] for rule in rules):
            raise InvalidGeometryProfileError(
                f"directions.{direction}.anchor is required when an enabled rule exists"
            )
        side: dict[str, Any] = {"anchor": anchor, "rules": rules}
        if "fill_bgr" in source:
            fill = source["fill_bgr"]
            if not isinstance(fill, (list, tuple)) or len(fill) != 3:
                raise InvalidGeometryProfileError(f"directions.{direction}.fill_bgr must contain three values")
            channels = [_finite(item, f"directions.{direction}.fill_bgr[{index}]") for index, item in enumerate(fill)]
            if any(channel < 0 or channel > 255 for channel in channels):
                raise InvalidGeometryProfileError(f"directions.{direction}.fill_bgr must be in [0, 255]")
            side["fill_bgr"] = [int(round(channel)) for channel in channels]
        result["directions"][direction] = side
    return result


def _empty_profile() -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "directions": {
            "front": {"anchor": None, "rules": []},
            "back": {"anchor": None, "rules": []},
        },
    }


def _empty_document(library_revision: int) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "library_revision": int(library_revision),
        "draft_revision": 0,
        "active_revision": None,
        "previous_active_revision": None,
        "draft": _empty_profile(),
        "active": None,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class GeometryMaskProfiles:
    """Persist drafts without allowing stale editors to overwrite newer state."""

    def __init__(
        self,
        catalog_or_library: Any,
        calibrator: Any | None = None,
        *,
        start_worker: bool = True,
        storage_dir: Path | None = None,
    ):
        self.catalog = catalog_or_library
        self.calibrator = calibrator
        self.start_worker = bool(start_worker)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._operations: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_operations: dict[str, str] = {}
        self._candidate_caches: dict[str, Any] = {}
        library = getattr(self.catalog, "library", self.catalog)
        library_root = getattr(library, "library_dir", None)
        default_storage = Path(library_root) / ".geometry_mask_jobs" if library_root is not None else Path.cwd() / ".geometry_mask_jobs"
        self.storage_dir = Path(storage_dir) if storage_dir is not None else default_storage
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_path = self.storage_dir / "jobs.json"
        self._load_jobs()
        self._stop = False
        self._worker: threading.Thread | None = None
        if self.start_worker:
            self._worker = threading.Thread(target=self._worker_loop, name="geometry-mask-validation", daemon=True)
            self._worker.start()

    def _record(self, workpiece_id: str):
        getter = getattr(self.catalog, "get", None)
        if not callable(getter):
            raise GeometryProfileError("catalog must provide get(workpiece_id)")
        return getter(workpiece_id)

    @staticmethod
    def _profile_root(record: Any) -> Path:
        return Path(record.root) / PROFILE_DIRECTORY

    @staticmethod
    def _manifest(record: Any) -> dict[str, Any]:
        try:
            value = json.loads((Path(record.root) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            LOGGER.warning("Unable to read workpiece manifest for geometry profile: %s", exc)
            return {}
        return value if isinstance(value, dict) else {}

    def _load_document(self, record: Any) -> tuple[dict[str, Any], str, str | None]:
        path = self._profile_root(record) / "profile.json"
        if not path.exists():
            return _empty_document(record.revision), "empty", None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, Mapping) or raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
                raise ValueError("unsupported geometry profile schema")
            library_revision = raw.get("library_revision")
            draft_revision = raw.get("draft_revision")
            if type(library_revision) is not int or library_revision <= 0:
                raise ValueError("invalid library_revision")
            if type(draft_revision) is not int or draft_revision < 0:
                raise ValueError("invalid draft_revision")
            normalized_draft = normalize_geometry_profile(raw.get("draft", _empty_profile()))
            active_raw = raw.get("active")
            normalized_active = None if active_raw is None else normalize_geometry_profile(active_raw)
            active_revision = raw.get("active_revision")
            previous_active_revision = raw.get("previous_active_revision")
            for value, field in ((active_revision, "active_revision"), (previous_active_revision, "previous_active_revision")):
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(f"invalid {field}")
            document = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "library_revision": library_revision,
                "draft_revision": draft_revision,
                "active_revision": active_revision,
                "previous_active_revision": previous_active_revision,
                "draft": normalized_draft,
                "active": normalized_active,
            }
            return document, "ok", None
        except Exception as exc:
            return _empty_document(record.revision), "corrupt", str(exc)

    def _snapshot_for_record(self, record: Any) -> dict[str, Any]:
        document, status, error = self._load_document(record)
        if status == "ok" and document["library_revision"] != record.revision:
            status = "stale"
            error = (
                f"profile library revision {document['library_revision']} does not match "
                f"current revision {record.revision}"
            )
        manifest = self._manifest(record)
        snapshot = {
            "workpiece_id": record.id,
            "library_revision": int(record.revision),
            "draft_revision": int(document["draft_revision"]),
            "active_revision": document["active_revision"],
            "previous_active_revision": document["previous_active_revision"],
            "draft": deepcopy(document["draft"]),
            "active": deepcopy(document["active"]),
            "profile_status": status,
            "legacy_archived": bool(manifest.get("interference_groups") or manifest.get("active_interference_groups")),
        }
        if error:
            snapshot["profile_error"] = error
        return snapshot

    def snapshot(self, workpiece_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_for_record(self._record(workpiece_id))

    def save_draft(
        self,
        workpiece_id: str,
        draft: Mapping[str, Any],
        *,
        expected_library_revision: int,
        expected_draft_revision: int,
        operation_id: str,
    ) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        if type(expected_library_revision) is not int or expected_library_revision <= 0:
            raise StaleGeometryProfileError("expected_library_revision must be a positive integer")
        if type(expected_draft_revision) is not int or expected_draft_revision < 0:
            raise StaleGeometryProfileError("expected_draft_revision must be a non-negative integer")
        operation_key = f"{workpiece_id}:{operation_id}"
        with self._lock:
            if operation_key in self._operations:
                return deepcopy(self._operations[operation_key])
            record = self._record(workpiece_id)
            document, status, error = self._load_document(record)
            if status == "corrupt":
                raise CorruptGeometryProfileError(error or "geometry profile is corrupt")
            if record.revision != expected_library_revision:
                raise StaleGeometryProfileError(
                    f"workpiece revision changed: expected {expected_library_revision}, current {record.revision}"
                )
            if document["library_revision"] != expected_library_revision:
                raise StaleGeometryProfileError(
                    f"profile revision changed: expected {expected_library_revision}, current {document['library_revision']}"
                )
            if document["draft_revision"] != expected_draft_revision:
                raise StaleGeometryProfileError(
                    f"draft revision changed: expected {expected_draft_revision}, current {document['draft_revision']}"
                )
            normalized = normalize_geometry_profile(draft)
            next_revision = expected_draft_revision + 1
            document["draft_revision"] = next_revision
            document["draft"] = normalized
            root = self._profile_root(record)
            root.mkdir(parents=True, exist_ok=True)
            (root / "revisions").mkdir(exist_ok=True)
            (root / "previews").mkdir(exist_ok=True)
            _atomic_write_json(root / "profile.json", document)
            result = self._snapshot_for_record(record)
            self._operations[operation_key] = deepcopy(result)
            return result

    def sync_library_revision(self, record: Any) -> None:
        """Keep a profile document's library pointer current after template changes."""
        with self._lock:
            document, status, _ = self._load_document(record)
            if status != "ok" or document["library_revision"] == record.revision:
                return
            document["library_revision"] = int(record.revision)
            _atomic_write_json(self._profile_root(record) / "profile.json", document)

    def rebuild_active_cache(self, workpiece_id: str, record: Any | None = None) -> Any | None:
        """Build the active geometry view from its immutable revision file."""
        with self._lock:
            record = record or self._record(workpiece_id)
            manifest = self._manifest(record)
            active_revision = manifest.get("geometry_mask_active_revision")
            if active_revision is None:
                active_revision = self._load_document(record)[0].get("active_revision")
            if type(active_revision) is not int or active_revision <= 0:
                return None
            path = self._profile_root(record) / "revisions" / f"{active_revision}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                profile = payload["profile"]
            except Exception as exc:
                LOGGER.warning("Unable to load active geometry profile %s: %s", path, exc)
                return None
            classifier = getattr(self.catalog, "classifier", None)
            prepare = getattr(classifier, "prepare_geometry_cache", None)
            if not callable(prepare):
                return None
            candidate, _ = prepare(workpiece_id, record, profile, self.calibrator, None)
            return candidate

    def validate_new_template(self, workpiece_id: str, orientation: str, image_path: Path) -> dict[str, Any]:
        """Fit the active profile to one newly confirmed template."""
        if orientation not in {"front", "back"}:
            raise GeometryValidationError("orientation must be front or back")
        with self._lock:
            record = self._record(workpiece_id)
            snapshot = self._snapshot_for_record(record)
            if snapshot.get("active_revision") is None:
                return {"status": "not_configured", "needs_review": False}
            profile = snapshot.get("active") or {}
            directions = profile.get("directions", {}) if isinstance(profile, Mapping) else {}
            direction = directions.get(orientation) if isinstance(directions, Mapping) else None
            if not isinstance(direction, Mapping) or direction.get("anchor") is None or not direction.get("rules"):
                return {"status": "not_configured", "needs_review": False}
            image = read_color_image(Path(image_path))
            if image is None:
                return {"status": "unreadable", "needs_review": True}
            if self.calibrator is None:
                return {"status": "unavailable", "needs_review": True}
            fit = self.calibrator.fit(image, direction)
            result = {key: value for key, value in fit.items() if key != "ignore_mask"}
            result["status"] = fit.get("status", "low_confidence")
            result["needs_review"] = result["status"] != "active"
            result["profile_revision"] = snapshot["active_revision"]
            return result

    # ------------------------------------------------------------------
    # Background validation and immutable publication
    # ------------------------------------------------------------------

    def _load_jobs(self) -> None:
        if not self.jobs_path.exists():
            return
        try:
            payload = json.loads(self.jobs_path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("jobs document must be an object")
            operations = payload.get("operations", {})
            jobs = payload.get("jobs", [])
            if isinstance(operations, Mapping):
                self._job_operations = {str(key): str(value) for key, value in operations.items()}
            if isinstance(jobs, list):
                for item in jobs:
                    if not isinstance(item, Mapping) or not item.get("job_id"):
                        continue
                    job = deepcopy(dict(item))
                    if job.get("state") == "running":
                        job["state"] = "interrupted"
                        job["error"] = "backend restarted during geometry validation"
                    self._jobs[str(job["job_id"])] = job
        except Exception as exc:
            LOGGER.warning("Ignoring invalid geometry validation jobs: %s", exc)

    def _persist_jobs(self) -> None:
        payload = {
            "jobs": list(self._jobs.values()),
            "operations": dict(self._job_operations),
        }
        _atomic_write_json(self.jobs_path, payload)

    @staticmethod
    def _total_templates(record: Any) -> int:
        return len(record.front_images) + len(record.back_images)

    @staticmethod
    def _profile_for_revision(profile: Mapping[str, Any], revision: int) -> dict[str, Any]:
        candidate = deepcopy(dict(profile))
        candidate["schema_version"] = PROFILE_SCHEMA_VERSION
        candidate["profile_revision"] = int(revision)
        return candidate

    def _job_snapshot(self, job_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self._jobs[job_id])
        except KeyError as exc:
            raise GeometryValidationJobNotFoundError(f"Unknown geometry validation job: {job_id}") from exc

    def start_validation(
        self,
        workpiece_id: str,
        *,
        expected_library_revision: int,
        expected_draft_revision: int,
        operation_id: str,
    ) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        with self._condition:
            if operation_id in self._job_operations:
                return self._job_snapshot(self._job_operations[operation_id])
            record = self._record(workpiece_id)
            snapshot = self._snapshot_for_record(record)
            if snapshot["profile_status"] in {"corrupt", "stale"}:
                raise StaleGeometryProfileError(snapshot.get("profile_error", "geometry profile is not current"))
            if record.revision != expected_library_revision:
                raise StaleGeometryProfileError(
                    f"workpiece revision changed: expected {expected_library_revision}, current {record.revision}"
                )
            if snapshot["draft_revision"] != expected_draft_revision:
                raise StaleGeometryProfileError(
                    f"draft revision changed: expected {expected_draft_revision}, current {snapshot['draft_revision']}"
                )
            job_id = uuid.uuid4().hex
            now = time.time()
            total = self._total_templates(record)
            self._jobs[job_id] = {
                "job_id": job_id,
                "workpiece_id": workpiece_id,
                "base_library_revision": record.revision,
                "base_draft_revision": snapshot["draft_revision"],
                "state": "queued",
                "progress": {"completed": 0, "total": total, "phase": "queued"},
                "warnings": [],
                "regression": {"status": "not_run", "correct_to_wrong": 0},
                "error": None,
                "cancel_requested": False,
                "created_at": now,
                "updated_at": now,
                "report": None,
                "profile": self._profile_for_revision(snapshot["draft"], snapshot["draft_revision"]),
            }
            self._job_operations[operation_id] = job_id
            self._persist_jobs()
            self._condition.notify_all()
            return self._job_snapshot(job_id)

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._job_snapshot(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(job) for job in self._jobs.values()]

    def action(self, job_id: str, action: str) -> dict[str, Any]:
        if action not in {"cancel", "cleanup"}:
            raise GeometryValidationError(f"unsupported geometry validation action: {action}")
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise GeometryValidationJobNotFoundError(f"Unknown geometry validation job: {job_id}")
            if action == "cleanup":
                if job["state"] not in {"completed", "failed", "cancelled", "interrupted"}:
                    raise GeometryValidationError("only finished geometry validation jobs can be cleaned up")
                self._candidate_caches.pop(job_id, None)
                self._jobs.pop(job_id, None)
                self._persist_jobs()
                return {"job_id": job_id, "state": "removed"}
            if job["state"] == "queued":
                job["state"] = "cancelled"
                job["updated_at"] = time.time()
                job["error"] = "cancelled by operator"
            elif job["state"] == "running":
                job["cancel_requested"] = True
            self._persist_jobs()
            self._condition.notify_all()
            return self._job_snapshot(job_id)

    def _set_job_progress(self, job_id: str, label: str, completed: int, total: int) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return
            record = self._record(job["workpiece_id"])
            job["progress"] = {
                "completed": int(job["progress"].get("completed", 0)) + 1,
                "total": self._total_templates(record),
                "phase": "fitting",
                "label": label,
                "phase_completed": int(completed),
                "phase_total": int(total),
            }
            job["updated_at"] = time.time()
            self._persist_jobs()
            if job.get("cancel_requested"):
                raise GeometryValidationError("cancelled by operator")

    @staticmethod
    def _validation_warnings(report: Mapping[str, Any]) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for label, items in report.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                index = item.get("index")
                ignored_ratio = float(item.get("ignored_ratio", 0.0) or 0.0)
                remaining_ratio = float(item.get("remaining_ratio", 1.0) or 0.0)
                if ignored_ratio > 0.40:
                    warnings.append({
                        "orientation": label,
                        "index": index,
                        "code": "effective_area_low",
                        "ignored_ratio": ignored_ratio,
                    })
                if remaining_ratio < 0.30:
                    warnings.append({
                        "orientation": label,
                        "index": index,
                        "code": "keypoint_retention_low",
                        "remaining_ratio": remaining_ratio,
                    })
                if item.get("status") not in {"active", "not_configured"}:
                    warnings.append({
                        "orientation": label,
                        "index": index,
                        "code": "fit_not_active",
                        "status": item.get("status"),
                    })
        return warnings

    def _run_validation(self, job_id: str) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.get("state") != "queued":
                return
            job["state"] = "running"
            job["progress"]["phase"] = "fitting"
            job["updated_at"] = time.time()
            self._persist_jobs()
            workpiece_id = str(job["workpiece_id"])
            profile = deepcopy(job["profile"])
            expected_revision = int(job["base_library_revision"])
        try:
            record = self._record(workpiece_id)
            if record.revision != expected_revision:
                raise StaleGeometryProfileError(
                    f"workpiece revision changed: expected {expected_revision}, current {record.revision}"
                )
            classifier = getattr(self.catalog, "classifier", None)
            prepare = getattr(classifier, "prepare_geometry_cache", None)
            if not callable(prepare):
                raise GeometryValidationError("classifier does not support geometry validation")

            def progress(label: str, completed: int, total: int) -> None:
                self._set_job_progress(job_id, label, completed, total)

            candidate, report = prepare(
                workpiece_id,
                record,
                profile,
                self.calibrator,
                progress,
            )
            with self._condition:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                if job.get("cancel_requested"):
                    job["state"] = "cancelled"
                    job["error"] = "cancelled by operator"
                    self._candidate_caches.pop(job_id, None)
                else:
                    warnings = self._validation_warnings(report)
                    job["state"] = "completed"
                    job["progress"] = {
                        "completed": self._total_templates(record),
                        "total": self._total_templates(record),
                        "phase": "completed",
                    }
                    job["report"] = deepcopy(report)
                    job["warnings"] = warnings
                    job["candidate_profile"] = deepcopy(getattr(candidate, "geometry_profile", profile))
                    leave_one_out = getattr(classifier, "leave_one_out_report", None)
                    if callable(leave_one_out):
                        try:
                            job["regression"] = deepcopy(leave_one_out(record, candidate))
                        except Exception as exc:
                            LOGGER.warning("Geometry leave-one-out validation failed: %s", exc)
                            job["regression"] = {
                                "status": "failed",
                                "correct_to_wrong": 0,
                                "error": str(exc),
                            }
                    else:
                        job["regression"] = {
                            "status": "not_run",
                            "correct_to_wrong": 0,
                        }
                    if job["regression"].get("status") == "failed":
                        job["warnings"].append({
                            "code": "leave_one_out_failed",
                            "message": job["regression"].get("error", "留一验证失败"),
                        })
                    self._candidate_caches[job_id] = candidate
                job["updated_at"] = time.time()
                self._persist_jobs()
        except Exception as exc:
            with self._condition:
                job = self._jobs.get(job_id)
                if job is not None:
                    if job.get("cancel_requested") or "cancelled by operator" in str(exc):
                        job["state"] = "cancelled"
                    else:
                        job["state"] = "failed"
                    job["error"] = str(exc)
                    job["updated_at"] = time.time()
                    self._candidate_caches.pop(job_id, None)
                    self._persist_jobs()

    def run_next(self, *, force: bool = False) -> dict[str, Any] | None:
        """Run one queued job synchronously; useful for deterministic service tests."""
        with self._lock:
            queued = next((job["job_id"] for job in self._jobs.values() if job.get("state") == "queued"), None)
        if queued is None:
            return None
        self._run_validation(queued)
        return self.get_job(queued)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stop and not any(job.get("state") == "queued" for job in self._jobs.values()):
                    self._condition.wait(timeout=0.5)
                if self._stop:
                    return
                queued = next(job["job_id"] for job in self._jobs.values() if job.get("state") == "queued")
            self._run_validation(queued)

    def _require_publish_job(
        self,
        workpiece_id: str,
        job_id: str,
        expected_library_revision: int,
        expected_draft_revision: int,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        record = self._record(workpiece_id)
        snapshot = self._snapshot_for_record(record)
        job = self._jobs.get(job_id)
        if job is None:
            raise GeometryValidationJobNotFoundError(f"Unknown geometry validation job: {job_id}")
        if job.get("workpiece_id") != workpiece_id:
            raise GeometryProfilePublishError("validation job belongs to another workpiece")
        if job.get("state") != "completed":
            raise GeometryProfileNotReadyError(f"validation job is {job.get('state')}")
        if record.revision != expected_library_revision or snapshot["draft_revision"] != expected_draft_revision:
            raise StaleGeometryProfileError("geometry validation result is stale")
        if job.get("base_library_revision") != expected_library_revision or job.get("base_draft_revision") != expected_draft_revision:
            raise StaleGeometryProfileError("geometry validation result is stale")
        candidate = self._candidate_caches.get(job_id)
        if candidate is None:
            raise GeometryProfileNotReadyError("candidate cache is no longer available; validate again")
        return record, snapshot, job

    def _catalog_publish(
        self,
        workpiece_id: str,
        candidate: Any,
        *,
        profile_revision: int,
        previous_profile_revision: int | None,
        expected_revision: int,
        operation_id: str,
    ) -> Any:
        publisher = getattr(self.catalog, "publish_geometry_profile", None)
        if callable(publisher):
            return publisher(
                workpiece_id,
                candidate,
                profile_revision=profile_revision,
                previous_profile_revision=previous_profile_revision,
                expected_revision=expected_revision,
                operation_id=operation_id,
            )
        library = getattr(self.catalog, "library", self.catalog)
        record = library.replace_geometry_profile_pointers(
            workpiece_id,
            expected_revision=expected_revision,
            active_revision=profile_revision,
            previous_active_revision=previous_profile_revision,
        )
        classifier = getattr(self.catalog, "classifier", None)
        if classifier is not None:
            classifier.set_template_cache(workpiece_id, candidate)
        return record

    @staticmethod
    def _restore_bytes(path: Path, content: bytes | None) -> None:
        if content is None:
            if path.exists():
                path.unlink()
            return
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.restore")
        try:
            temporary.write_bytes(content)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def publish(
        self,
        workpiece_id: str,
        job_id: str,
        *,
        expected_library_revision: int,
        expected_draft_revision: int,
        operation_id: str,
        override_reason: str = "",
    ) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        operation_key = f"publish:{workpiece_id}:{operation_id}"
        with self._condition:
            cached = self._operations.get(operation_key)
            if cached is not None:
                return deepcopy(cached)
            record, snapshot, job = self._require_publish_job(
                workpiece_id, job_id, expected_library_revision, expected_draft_revision
            )
            if (job.get("warnings") or job.get("regression", {}).get("correct_to_wrong", 0)) and not str(override_reason).strip():
                raise GeometryProfilePublishError("validation warnings or regressions require override_reason")
            candidate = self._candidate_caches[job_id]
            profile_revision = int(expected_draft_revision)
            previous_revision = snapshot["active_revision"]
            root = self._profile_root(record)
            profile_path = root / "profile.json"
            previous_bytes = profile_path.read_bytes() if profile_path.exists() else None
            document, status, error = self._load_document(record)
            if status == "corrupt":
                raise CorruptGeometryProfileError(error or "geometry profile is corrupt")
            candidate_profile = self._profile_for_revision(
                getattr(candidate, "geometry_profile", None) or job.get("candidate_profile") or job["profile"],
                profile_revision,
            )
            revision_payload = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "profile_revision": profile_revision,
                "library_revision": expected_library_revision + 1,
                "previous_active_revision": previous_revision,
                "profile": deepcopy(candidate_profile),
                "report": deepcopy(job.get("report") or {}),
                "warnings": deepcopy(job.get("warnings") or []),
                "regression": deepcopy(job.get("regression") or {}),
                "published_at": time.time(),
                "operation_id": operation_id,
                "override_reason": str(override_reason),
            }
            revision_path = root / "revisions" / f"{profile_revision}.json"
            _atomic_write_json(revision_path, revision_payload)
            document.update(
                {
                    "library_revision": expected_library_revision + 1,
                    "active_revision": profile_revision,
                    "previous_active_revision": previous_revision,
                    "active": deepcopy(candidate_profile),
                }
            )
            _atomic_write_json(profile_path, document)
            try:
                published_record = self._catalog_publish(
                    workpiece_id,
                    candidate,
                    profile_revision=profile_revision,
                    previous_profile_revision=previous_revision,
                    expected_revision=expected_library_revision,
                    operation_id=operation_id,
                )
            except Exception as exc:
                self._restore_bytes(profile_path, previous_bytes)
                raise GeometryProfilePublishError(str(exc)) from exc
            job["published"] = True
            job["published_revision"] = profile_revision
            job["updated_at"] = time.time()
            result = self._snapshot_for_record(published_record)
            result["job_id"] = job_id
            result["override_reason"] = str(override_reason)
            self._operations[operation_key] = deepcopy(result)
            self._persist_jobs()
            return result

    def rollback(
        self,
        workpiece_id: str,
        *,
        expected_library_revision: int,
        operation_id: str,
    ) -> dict[str, Any]:
        if not isinstance(operation_id, str) or not operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        operation_key = f"rollback:{workpiece_id}:{operation_id}"
        with self._condition:
            cached = self._operations.get(operation_key)
            if cached is not None:
                return deepcopy(cached)
            record = self._record(workpiece_id)
            snapshot = self._snapshot_for_record(record)
            previous_revision = snapshot["previous_active_revision"]
            if record.revision != expected_library_revision:
                raise StaleGeometryProfileError("workpiece revision changed before rollback")
            if previous_revision is None:
                raise GeometryProfilePublishError("no previous geometry profile revision is available")
            revision_path = self._profile_root(record) / "revisions" / f"{previous_revision}.json"
            try:
                payload = json.loads(revision_path.read_text(encoding="utf-8"))
                profile = payload["profile"]
            except Exception as exc:
                raise GeometryProfilePublishError(f"unable to load rollback revision {previous_revision}") from exc
            classifier = getattr(self.catalog, "classifier", None)
            prepare = getattr(classifier, "prepare_geometry_cache", None)
            if not callable(prepare):
                raise GeometryProfilePublishError("classifier does not support geometry cache preparation")
            candidate, report = prepare(workpiece_id, record, profile, self.calibrator, None)
            current_active = snapshot["active_revision"]
            document, status, error = self._load_document(record)
            if status == "corrupt":
                raise CorruptGeometryProfileError(error or "geometry profile is corrupt")
            profile_path = self._profile_root(record) / "profile.json"
            previous_bytes = profile_path.read_bytes() if profile_path.exists() else None
            document.update(
                {
                    "library_revision": expected_library_revision + 1,
                    "active_revision": previous_revision,
                    "previous_active_revision": payload.get("previous_active_revision"),
                    "active": deepcopy(profile),
                }
            )
            _atomic_write_json(profile_path, document)
            try:
                published_record = self._catalog_publish(
                    workpiece_id,
                    candidate,
                    profile_revision=int(previous_revision),
                    previous_profile_revision=payload.get("previous_active_revision"),
                    expected_revision=expected_library_revision,
                    operation_id=operation_id,
                )
            except Exception as exc:
                self._restore_bytes(profile_path, previous_bytes)
                raise GeometryProfilePublishError(str(exc)) from exc
            result = self._snapshot_for_record(published_record)
            result["rollback_revision"] = previous_revision
            result["report"] = report
            self._operations[operation_key] = deepcopy(result)
            self._persist_jobs()
            return result

    def shutdown(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
