"""Canonical schema and compatibility helpers for geometry-mask profiles."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import time
from typing import Any, Mapping

from src.geometry_calibration import SUPPORTED_MODES, SUPPORTED_SHAPES


PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_SCHEMA_VERSION = 1
SIGNED_MARGIN_SEMANTICS = "signed_boundary_v2"
DIRECTIONS = ("front", "back")
CALIBRATION_STATES = {"missing", "ready", "low_confidence", "needs_review"}
REVIEW_STATES = {"included", "review", "excluded"}


class GeometryProfileSchemaError(ValueError):
    """Raised when a profile cannot be represented safely."""


class DuplicateLogicalRuleSchemaError(GeometryProfileSchemaError):
    """Raised when a schema-v2 logical rule identifier is duplicated."""


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryProfileSchemaError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise GeometryProfileSchemaError(f"{field} must be finite")
    return result


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryProfileSchemaError(f"{field} must be non-empty text")
    return value.strip()


def _margin(rule: Mapping[str, Any], field: str) -> float:
    marker = rule.get("margin_semantics")
    if marker is None:
        raw = _finite(rule.get("margin_ratio", 0.02), f"{field}.margin_ratio")
        if raw < 0 or raw >= 0.95:
            raise GeometryProfileSchemaError(
                f"{field}.margin_ratio must be in [0, 0.95) for legacy rules"
            )
        return raw if rule.get("mode") == "inside" else -raw
    if marker != SIGNED_MARGIN_SEMANTICS:
        raise GeometryProfileSchemaError(f"{field}.margin_semantics is unsupported")
    result = _finite(rule.get("margin_ratio", 0.0), f"{field}.margin_ratio")
    if result < -0.94 or result > 0.94:
        raise GeometryProfileSchemaError(f"{field}.margin_ratio must be in [-0.94, 0.94]")
    return result


def _geometry(value: Any, shape: str, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object")
    result = {
        "cx": _finite(value.get("cx", 0.0), f"{field}.cx"),
        "cy": _finite(value.get("cy", 0.0), f"{field}.cy"),
        "angle_deg": _finite(value.get("angle_deg", 0.0), f"{field}.angle_deg"),
    }
    if shape == "circle":
        radius = _finite(value.get("r"), f"{field}.r")
        if radius <= 0:
            raise GeometryProfileSchemaError(f"{field}.r must be positive")
        result["r"] = radius
    elif shape == "ellipse":
        rx = _finite(value.get("rx"), f"{field}.rx")
        ry = _finite(value.get("ry"), f"{field}.ry")
        if rx <= 0 or ry <= 0:
            raise GeometryProfileSchemaError(f"{field} radii must be positive")
        result.update({"rx": rx, "ry": ry})
    else:
        half_width = _finite(value.get("half_width"), f"{field}.half_width")
        half_height = _finite(value.get("half_height"), f"{field}.half_height")
        if half_width <= 0 or half_height <= 0:
            raise GeometryProfileSchemaError(f"{field} dimensions must be positive")
        result.update({"half_width": half_width, "half_height": half_height})
    return result


def _anchor(value: Any, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object or null")
    shape = value.get("shape")
    if shape not in SUPPORTED_SHAPES:
        raise GeometryProfileSchemaError(f"{field}.shape is unsupported: {shape}")
    coarse = value.get("coarse")
    if not isinstance(coarse, Mapping):
        raise GeometryProfileSchemaError(f"{field}.coarse must be an object")
    normalized = _geometry(coarse, shape, f"{field}.coarse")
    mode = value.get("mode", "manual")
    if mode not in {"manual", "auto"}:
        raise GeometryProfileSchemaError(f"{field}.mode is unsupported")
    return {"shape": shape, "mode": mode, "coarse": normalized}


def _reference_template(value: Any, direction: str, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object or null")
    template_id = _text(value.get("template_id"), f"{field}.template_id")
    if not template_id.startswith(f"{direction}:"):
        raise GeometryProfileSchemaError(f"{field}.template_id does not match direction")
    width = value.get("width")
    height = value.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        raise GeometryProfileSchemaError(f"{field}.width and height must be positive integers")
    return {
        "template_id": template_id,
        "direction": direction,
        "width": width,
        "height": height,
    }


def _template_reviews(value: Any, direction: str, field: str) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_template_id, raw_review in value.items():
        template_id = _text(raw_template_id, f"{field}.template_id")
        if not template_id.startswith(f"{direction}:"):
            raise GeometryProfileSchemaError(f"{field} contains another direction")
        if not isinstance(raw_review, Mapping):
            raise GeometryProfileSchemaError(f"{field}.{template_id} must be an object")
        state = _text(raw_review.get("state"), f"{field}.{template_id}.state")
        reason = str(raw_review.get("reason", "")).strip()
        if state not in REVIEW_STATES:
            raise GeometryProfileSchemaError(f"unsupported review state: {state}")
        if state != "included" and not reason:
            raise GeometryProfileSchemaError("review and excluded templates require a reason")
        result[template_id] = {"state": state, "reason": reason}
    return result


def _logical_rule(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object")
    shape = value.get("shape")
    mode = value.get("mode")
    if shape not in SUPPORTED_SHAPES:
        raise GeometryProfileSchemaError(f"{field}.shape is unsupported: {shape}")
    if mode not in SUPPORTED_MODES:
        raise GeometryProfileSchemaError(f"{field}.mode is unsupported: {mode}")
    enabled = value.get("enabled", True)
    if type(enabled) is not bool:
        raise GeometryProfileSchemaError(f"{field}.enabled must be boolean")
    return {
        "rule_id": _text(value.get("rule_id"), f"{field}.rule_id"),
        "name": _text(value.get("name"), f"{field}.name"),
        "shape": shape,
        "mode": mode,
        "margin_ratio": _margin(value, field),
        "margin_semantics": SIGNED_MARGIN_SEMANTICS,
        "enabled": enabled,
    }


def _calibration(value: Any, logical: Mapping[str, Any], direction: str,
                 field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryProfileSchemaError(f"{field} must be an object")
    state = value.get("state", "ready")
    if state not in CALIBRATION_STATES:
        raise GeometryProfileSchemaError(f"{field}.state is unsupported")
    result: dict[str, Any] = {"state": state}
    geometry = value.get("geometry")
    if state == "ready" and geometry is None:
        raise GeometryProfileSchemaError(f"{field}.geometry is required when ready")
    if geometry is not None:
        result["geometry"] = _geometry(geometry, logical["shape"], f"{field}.geometry")
    seed = value.get("seed_geometry")
    if seed is not None:
        result["seed_geometry"] = _geometry(
            seed, logical["shape"], f"{field}.seed_geometry"
        )
    reference = _reference_template(
        value.get("reference_template"), direction, f"{field}.reference_template"
    )
    if reference is not None:
        result["reference_template"] = reference
    diagnostics = value.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise GeometryProfileSchemaError(f"{field}.diagnostics must be an object")
    result["diagnostics"] = deepcopy(dict(diagnostics))
    if "updated_at" in value:
        result["updated_at"] = _text(value.get("updated_at"), f"{field}.updated_at")
    return result


def normalize_profile_v2(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a canonical schema-v2 geometry profile."""
    if not isinstance(profile, Mapping):
        raise GeometryProfileSchemaError("geometry profile must be an object")
    if profile.get("schema_version", PROFILE_SCHEMA_VERSION) != PROFILE_SCHEMA_VERSION:
        raise GeometryProfileSchemaError("geometry profile is not schema v2")
    raw_rules = profile.get("rules", [])
    if not isinstance(raw_rules, list):
        raise GeometryProfileSchemaError("geometry profile.rules must be a list")
    rules: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw_rule in enumerate(raw_rules):
        rule = _logical_rule(raw_rule, f"rules[{index}]")
        if rule["rule_id"] in by_id:
            raise DuplicateLogicalRuleSchemaError(f"duplicate rule_id: {rule['rule_id']}")
        by_id[rule["rule_id"]] = rule
        rules.append(rule)
    raw_directions = profile.get("directions")
    if not isinstance(raw_directions, Mapping):
        raise GeometryProfileSchemaError("geometry profile.directions must be an object")
    directions: dict[str, Any] = {}
    for direction in DIRECTIONS:
        raw_side = raw_directions.get(direction, {})
        if not isinstance(raw_side, Mapping):
            raise GeometryProfileSchemaError(f"directions.{direction} must be an object")
        anchor = _anchor(raw_side.get("anchor"), f"directions.{direction}.anchor")
        raw_calibrations = raw_side.get("calibrations", {})
        if not isinstance(raw_calibrations, Mapping):
            raise GeometryProfileSchemaError(
                f"directions.{direction}.calibrations must be an object"
            )
        calibrations: dict[str, Any] = {}
        for raw_rule_id, raw_calibration in raw_calibrations.items():
            rule_id = _text(raw_rule_id, f"directions.{direction}.calibrations.rule_id")
            logical = by_id.get(rule_id)
            if logical is None:
                raise GeometryProfileSchemaError(f"orphan calibration: {rule_id}")
            calibrations[rule_id] = _calibration(
                raw_calibration,
                logical,
                direction,
                f"directions.{direction}.calibrations.{rule_id}",
            )
        if anchor is None and any(item.get("state") == "ready" for item in calibrations.values()):
            raise GeometryProfileSchemaError(
                f"directions.{direction}.anchor is required for ready calibrations"
            )
        side: dict[str, Any] = {
            "anchor": anchor,
            "calibrations": calibrations,
            "template_reviews": _template_reviews(
                raw_side.get("template_reviews"),
                direction,
                f"directions.{direction}.template_reviews",
            ),
        }
        fill = raw_side.get("fill_bgr")
        if fill is not None:
            if not isinstance(fill, (list, tuple)) or len(fill) != 3:
                raise GeometryProfileSchemaError(
                    f"directions.{direction}.fill_bgr must contain three values"
                )
            channels = [int(round(_finite(item, f"fill_bgr[{index}]")))
                        for index, item in enumerate(fill)]
            if any(item < 0 or item > 255 for item in channels):
                raise GeometryProfileSchemaError("fill_bgr values must be in [0, 255]")
            side["fill_bgr"] = channels
        directions[direction] = side
    migration = profile.get("migration", {})
    if not isinstance(migration, Mapping):
        raise GeometryProfileSchemaError("migration must be an object")
    conflicts = migration.get("conflicts", [])
    resolutions = migration.get("resolutions", [])
    if not isinstance(conflicts, list) or not isinstance(resolutions, list):
        raise GeometryProfileSchemaError("migration conflicts and resolutions must be lists")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "rules": rules,
        "directions": directions,
        "migration": {
            "source_schema_version": migration.get("source_schema_version"),
            "conflicts": deepcopy(conflicts),
            "resolutions": deepcopy(resolutions),
        },
    }


def _legacy_source(rule: Mapping[str, Any], direction: str, index: int,
                   reference_template: Any) -> dict[str, Any]:
    logical = _logical_rule(rule, f"directions.{direction}.rules[{index}]")
    geometry = _geometry(rule.get("geometry"), logical["shape"], "legacy.geometry")
    seed = _geometry(
        rule.get("seed_geometry", rule.get("geometry")),
        logical["shape"],
        "legacy.seed_geometry",
    )
    state = "needs_review" if rule.get("editor_state") == "needs_reseed" else "ready"
    calibration: dict[str, Any] = {
        "state": state,
        "geometry": geometry,
        "seed_geometry": seed,
        "diagnostics": {},
    }
    if reference_template is not None:
        calibration["reference_template"] = deepcopy(reference_template)
    signature = (
        logical["name"].casefold(), logical["shape"], logical["mode"],
        round(float(logical["margin_ratio"]), 9), bool(logical["enabled"]),
    )
    return {
        "direction": direction,
        "index": index,
        "source_rule_id": logical["rule_id"],
        "logical": logical,
        "calibration": calibration,
        "signature": signature,
    }


def _conflict_id(signature: tuple[Any, ...], sources: list[dict[str, Any]]) -> str:
    raw = repr((signature, [(item["direction"], item["source_rule_id"]) for item in sources]))
    return f"legacy-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def migrate_profile_v1(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Create a lossless v2 draft from a legacy direction-owned profile."""
    if not isinstance(profile, Mapping):
        raise GeometryProfileSchemaError("legacy profile must be an object")
    raw_directions = profile.get("directions")
    if not isinstance(raw_directions, Mapping):
        raise GeometryProfileSchemaError("legacy directions must be an object")
    anchors: dict[str, Any] = {}
    reviews: dict[str, Any] = {}
    fills: dict[str, Any] = {}
    sources: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        side = raw_directions.get(direction, {})
        if not isinstance(side, Mapping):
            raise GeometryProfileSchemaError(f"legacy {direction} direction must be an object")
        anchors[direction] = _anchor(side.get("anchor"), f"directions.{direction}.anchor")
        reviews[direction] = _template_reviews(
            side.get("template_reviews"), direction, f"directions.{direction}.template_reviews"
        )
        if "fill_bgr" in side:
            fills[direction] = deepcopy(side["fill_bgr"])
        raw_rules = side.get("rules", [])
        if not isinstance(raw_rules, list):
            raise GeometryProfileSchemaError(f"legacy {direction}.rules must be a list")
        reference = side.get("reference_template")
        if reference is not None:
            reference = _reference_template(
                reference, direction, f"directions.{direction}.reference_template"
            )
        for index, raw_rule in enumerate(raw_rules):
            sources.append(_legacy_source(raw_rule, direction, index, reference))

    paired: list[list[dict[str, Any]]] = []
    remaining = list(sources)
    for source in list(remaining):
        matches = [item for item in remaining
                   if item is not source
                   and item["direction"] != source["direction"]
                   and item["source_rule_id"] == source["source_rule_id"]]
        same_id = [item for item in remaining if item["source_rule_id"] == source["source_rule_id"]]
        if len(matches) == 1 and len(same_id) == 2:
            pair = [source, matches[0]]
            paired.append(pair)
            remaining = [item for item in remaining if item not in pair]
    signatures = {item["signature"] for item in remaining}
    for signature in signatures:
        group = [item for item in remaining if item["signature"] == signature]
        front = [item for item in group if item["direction"] == "front"]
        back = [item for item in group if item["direction"] == "back"]
        if len(front) == 1 and len(back) == 1:
            paired.append([front[0], back[0]])
            remaining = [item for item in remaining if item not in group]

    logical_rules: list[dict[str, Any]] = []
    calibrations = {direction: {} for direction in DIRECTIONS}
    used_ids: set[str] = set()

    def add_sources(group: list[dict[str, Any]]) -> str:
        preferred = group[0]["source_rule_id"]
        rule_id = preferred
        suffix = 2
        while rule_id in used_ids:
            rule_id = f"{preferred}-{suffix}"
            suffix += 1
        used_ids.add(rule_id)
        logical = deepcopy(group[0]["logical"])
        logical["rule_id"] = rule_id
        logical_rules.append(logical)
        for item in group:
            calibrations[item["direction"]][rule_id] = deepcopy(item["calibration"])
            item["logical_rule_id"] = rule_id
        return rule_id

    for group in paired:
        add_sources(sorted(group, key=lambda item: DIRECTIONS.index(item["direction"])))
    for source in remaining:
        add_sources([source])

    conflicts: list[dict[str, Any]] = []
    for signature in {item["signature"] for item in remaining}:
        group = [item for item in remaining if item["signature"] == signature]
        directions_present = {item["direction"] for item in group}
        conflict_type = "same_direction_duplicate" if len(group) > 1 else "missing_direction"
        if len(directions_present) > 1:
            conflict_type = "ambiguous_pair"
        conflicts.append({
            "conflict_id": _conflict_id(signature, group),
            "type": conflict_type,
            "sources": [
                {
                    "direction": item["direction"],
                    "source_rule_id": item["source_rule_id"],
                    "logical_rule_id": item["logical_rule_id"],
                    "name": item["logical"]["name"],
                }
                for item in group
            ],
        })

    result = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "rules": logical_rules,
        "directions": {},
        "migration": {
            "source_schema_version": LEGACY_PROFILE_SCHEMA_VERSION,
            "conflicts": conflicts,
            "resolutions": [],
        },
    }
    for direction in DIRECTIONS:
        side = {
            "anchor": anchors[direction],
            "calibrations": calibrations[direction],
            "template_reviews": reviews[direction],
        }
        if direction in fills:
            side["fill_bgr"] = fills[direction]
        result["directions"][direction] = side
    return normalize_profile_v2(result)


def resolve_migration_conflict(
    profile: Mapping[str, Any],
    conflict_id: str,
    resolution: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply one explicit migration decision without touching unrelated rules."""
    normalized = normalize_profile_v2(profile)
    if not isinstance(conflict_id, str) or not conflict_id.strip():
        raise GeometryProfileSchemaError("conflict_id must be non-empty text")
    if not isinstance(resolution, Mapping):
        raise GeometryProfileSchemaError("resolution must be an object")
    conflicts = normalized["migration"]["conflicts"]
    conflict = next(
        (item for item in conflicts if item.get("conflict_id") == conflict_id), None
    )
    if conflict is None:
        raise GeometryProfileSchemaError(f"unknown migration conflict: {conflict_id}")
    action = resolution.get("action")
    if action not in {"keep_only", "pair", "keep_separate"}:
        raise GeometryProfileSchemaError(f"unsupported migration resolution: {action}")
    sources = conflict.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise GeometryProfileSchemaError("migration conflict has no sources")

    remove_ids: set[str] = set()
    if action == "keep_only":
        survivor = _text(resolution.get("survivor_rule_id"), "survivor_rule_id")
        survivor_source = next(
            (
                item for item in sources
                if item.get("source_rule_id") == survivor
                or item.get("logical_rule_id") == survivor
            ),
            None,
        )
        if survivor_source is None:
            raise GeometryProfileSchemaError("survivor_rule_id is not part of conflict")
        survivor_logical_id = survivor_source["logical_rule_id"]
        remove_ids = {
            item["logical_rule_id"] for item in sources
            if item.get("logical_rule_id") != survivor_logical_id
        }
    elif action == "pair":
        front_id = _text(resolution.get("front_rule_id"), "front_rule_id")
        back_id = _text(resolution.get("back_rule_id"), "back_rule_id")
        front_source = next(
            (item for item in sources if item.get("direction") == "front"
             and item.get("logical_rule_id") == front_id),
            None,
        )
        back_source = next(
            (item for item in sources if item.get("direction") == "back"
             and item.get("logical_rule_id") == back_id),
            None,
        )
        if front_source is None or back_source is None:
            raise GeometryProfileSchemaError("pair requires one front and one back source")
        logical_by_id = {item["rule_id"]: item for item in normalized["rules"]}
        front_rule = logical_by_id[front_id]
        back_rule = logical_by_id[back_id]
        signature_fields = ("name", "shape", "mode", "margin_ratio", "enabled")
        if any(front_rule[field] != back_rule[field] for field in signature_fields):
            raise GeometryProfileSchemaError("paired rules must share one logical definition")
        normalized["directions"]["back"]["calibrations"][front_id] = deepcopy(
            normalized["directions"]["back"]["calibrations"][back_id]
        )
        remove_ids = {back_id}

    if remove_ids:
        normalized["rules"] = [
            item for item in normalized["rules"] if item["rule_id"] not in remove_ids
        ]
        for direction in DIRECTIONS:
            calibrations = normalized["directions"][direction]["calibrations"]
            for rule_id in remove_ids:
                calibrations.pop(rule_id, None)

    remaining_conflicts = []
    for item in conflicts:
        if item.get("conflict_id") == conflict_id:
            continue
        updated = deepcopy(item)
        updated["sources"] = [
            source for source in updated.get("sources", [])
            if source.get("logical_rule_id") not in remove_ids
        ]
        if updated["sources"]:
            remaining_conflicts.append(updated)
    normalized["migration"]["conflicts"] = remaining_conflicts
    normalized["migration"]["resolutions"].append({
        "conflict_id": conflict_id,
        "resolution": deepcopy(dict(resolution)),
        "resolved_at": time.time(),
    })
    return normalize_profile_v2(normalized)


def materialize_direction_profile(profile: Mapping[str, Any], direction: str) -> dict[str, Any]:
    """Convert one v2 calibration branch to the legacy runtime shape."""
    if direction not in DIRECTIONS:
        raise GeometryProfileSchemaError("direction must be front or back")
    normalized = normalize_profile_v2(profile)
    side = normalized["directions"][direction]
    rules: list[dict[str, Any]] = []
    for logical in normalized["rules"]:
        calibration = side["calibrations"].get(logical["rule_id"])
        if calibration is None or calibration["state"] != "ready":
            continue
        runtime_rule = deepcopy(logical)
        runtime_rule["geometry"] = deepcopy(calibration["geometry"])
        runtime_rule["seed_geometry"] = deepcopy(
            calibration.get("seed_geometry", calibration["geometry"])
        )
        runtime_rule["editor_state"] = "ready"
        rules.append(runtime_rule)
    result = {
        "anchor": deepcopy(side.get("anchor")),
        "rules": rules,
        "template_reviews": deepcopy(side.get("template_reviews", {})),
    }
    if "fill_bgr" in side:
        result["fill_bgr"] = deepcopy(side["fill_bgr"])
    return result


def materialize_runtime_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the direction-owned shape consumed by the current calibrator."""
    version = profile.get("schema_version", LEGACY_PROFILE_SCHEMA_VERSION)
    if version == LEGACY_PROFILE_SCHEMA_VERSION:
        # Schema v1 is already the runtime direction-owned representation.
        # Preserve it byte-for-byte apart from the explicit version marker so
        # old active libraries keep working until a v2 draft is published.
        legacy = deepcopy(dict(profile))
        legacy.setdefault("schema_version", LEGACY_PROFILE_SCHEMA_VERSION)
        return legacy
    normalized = normalize_profile_v2(profile)
    return {
        "schema_version": LEGACY_PROFILE_SCHEMA_VERSION,
        "directions": {
            direction: materialize_direction_profile(normalized, direction)
            for direction in DIRECTIONS
        },
    }
