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
import uuid
from typing import Any, Mapping

from src.geometry_calibration import SUPPORTED_MODES, SUPPORTED_SHAPES
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

    def __init__(self, catalog_or_library: Any, calibrator: Any | None = None, *, start_worker: bool = True):
        self.catalog = catalog_or_library
        self.calibrator = calibrator
        # The worker is intentionally not started in this persistence step.  The
        # flag is accepted now so the later validation implementation can be
        # introduced without changing Qt/TCP construction code.
        self.start_worker = bool(start_worker)
        self._lock = threading.RLock()
        self._operations: dict[str, dict[str, Any]] = {}

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
