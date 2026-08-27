"""Loopback TCP JSON-lines service for the orientation classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any, Callable, Mapping

# Qt launches this file by path. Add the repository root for that entrypoint
# so absolute ``src.*`` imports work both by file path and by ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orientation_classifier import (
    DEFAULT_INFERENCE_MODE,
    DEFAULT_LOCAL_SEARCH_MODE,
    ImageUnreadableError,
    INFERENCE_MODES,
    LOCAL_SEARCH_MODES,
    OrientationClassifier,
    OrientationClassifierError,
    PropagationModelError,
    WorkpieceNotFoundError,
)
from src.workpiece_library import (
    FeatureBuildError,
    InvalidTemplateSetError,
    InvalidWorkpieceNameError,
    StaleWorkpieceRevisionError,
    WorkpieceExistsError,
    WorkpieceLibrary,
)
from src.workpiece_catalog import RestoreConflictError, WorkpieceCatalog
from src.geometry_mask_profiles import (
    CorruptGeometryProfileError,
    DuplicateLogicalRuleError,
    FittedGeometryMissingError,
    GeometryCacheRevisionMismatchError,
    GeometryContextMismatchError,
    GeometryMaskProfiles,
    GeometryProfileMigrationConflictError,
    GeometryProfileNotReadyError,
    GeometryProfilePublishError,
    GeometryValidationError,
    GeometryValidationJobNotFoundError,
    InvalidGeometryProfileError,
    MissingDirectionCalibrationError,
    StaleGeometryProfileError,
)
from src.geometry_calibration import GeometryCalibrationError
from src.template_evolution import (
    AnnotationGroupNotFoundError,
    DuplicateTemplateError,
    InvalidAnnotationReviewError,
    TemplateEvolution,
    TemplateEvolutionError,
)
from src.interference_masks import InvalidMaskError


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
SERVICE_NAME = "workpiece-orientation"
MAX_MESSAGE_BYTES = 1024 * 1024
GEOMETRY_ERROR_CODES = {
    MissingDirectionCalibrationError: "MISSING_DIRECTION_CALIBRATION",
    FittedGeometryMissingError: "FITTED_GEOMETRY_MISSING",
    GeometryProfileMigrationConflictError: "MIGRATION_CONFLICT",
    DuplicateLogicalRuleError: "DUPLICATE_LOGICAL_RULE",
    GeometryContextMismatchError: "GEOMETRY_CONTEXT_MISMATCH",
    GeometryCacheRevisionMismatchError: "PROFILE_CACHE_REVISION_MISMATCH",
}
FAST_HARD_ERROR_CODES = {
    "FAST_CACHE_NOT_READY",
    "FAST_CACHE_REVISION_MISMATCH",
    "FAST_CACHE_BUILD_FAILED",
    "FAST_FEATURE_INVALID",
    "FAST_CACHE_CAPABILITY_UNAVAILABLE",
}


def _prefixed_fast_error_code(error: object) -> str | None:
    code = str(error).partition(":")[0].strip()
    return code if code in FAST_HARD_ERROR_CODES else None


def configure_diagnostic_logging(library_dir: Path) -> logging.Handler:
    """Persist backend diagnostics next to the workpiece library.

    The handler is marked so repeated startup/configuration calls replace only
    this service's handler and leave application/test handlers untouched.
    """
    diagnostics_dir = Path(library_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    log_path = (diagnostics_dir / "orientation-service.log").resolve()
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if not getattr(existing, "_workpiece_orientation_diagnostic", False):
            continue
        existing_path = getattr(existing, "baseFilename", None)
        if existing_path and Path(existing_path).resolve() == log_path:
            return existing
        root_logger.removeHandler(existing)
        existing.close()

    handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler._workpiece_orientation_diagnostic = True
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    return handler


def _prepare_windows_torch_dll_path() -> None:
    """Make the bundled PyTorch CUDA DLLs discoverable before importing torch."""
    if os.name != "nt":
        return
    torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
    if not torch_lib.is_dir():
        return
    os.add_dll_directory(str(torch_lib))
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")


class ServiceStartupError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ProtocolError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RuntimeSnapshot:
    status: str
    classifier: Any | None = None
    library: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    catalog: Any | None = None
    evolution: Any | None = None
    geometry_profiles: Any | None = None


def _shutdown_runtime_components(
    catalog: Any | None,
    evolution: Any | None,
    geometry_profiles: Any | None,
) -> None:
    for name, component in (
        ("fast cache jobs", catalog),
        ("template evolution", evolution),
        ("geometry profiles", geometry_profiles),
    ):
        shutdown = getattr(component, "shutdown", None)
        if not callable(shutdown):
            continue
        try:
            shutdown()
        except Exception:
            LOGGER.exception("Unable to shut down %s", name)


class ServiceRuntime:
    """Thread-safe model lifecycle state shared by the TCP dispatcher and loader."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._snapshot = RuntimeSnapshot(status="loading")
        self._shutdown_requested = False

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    def request_shutdown(self) -> RuntimeSnapshot:
        with self._lock:
            self._shutdown_requested = True
            return self._snapshot

    def set_ready(self, classifier: Any, library: Any, catalog: Any | None = None) -> bool:
        resolved_catalog = catalog or WorkpieceCatalog(library, classifier)
        evolution = None
        library_dir = getattr(library, "library_dir", None)
        geometry_profiles = getattr(resolved_catalog, "geometry_profiles", None)
        if geometry_profiles is None and library_dir is not None:
            geometry_profiles = GeometryMaskProfiles(
                resolved_catalog,
                getattr(classifier, "geometry_calibrator", None),
                storage_dir=Path(library_dir) / ".geometry-mask-jobs",
                start_worker=False,
            )
            setter = getattr(resolved_catalog, "set_geometry_profiles", None)
            if callable(setter):
                setter(geometry_profiles)
        with self._lock:
            shutdown_requested = self._shutdown_requested
        if shutdown_requested:
            _shutdown_runtime_components(resolved_catalog, None, geometry_profiles)
            return False
        if library_dir is not None:
            evolution = TemplateEvolution(
                resolved_catalog,
                Path(library_dir) / ".evolution",
                geometry_profiles=geometry_profiles,
                start_worker=False,
            )
        with self._lock:
            if self._shutdown_requested:
                shutdown_requested = True
            else:
                shutdown_requested = False
                self._snapshot = RuntimeSnapshot(
                    status="ready", classifier=classifier, library=library,
                    catalog=resolved_catalog, evolution=evolution, geometry_profiles=geometry_profiles,
                )
                for worker in (geometry_profiles, evolution):
                    start = getattr(worker, "start", None)
                    if callable(start):
                        start()
        if shutdown_requested:
            _shutdown_runtime_components(resolved_catalog, evolution, geometry_profiles)
            return False
        return True

    def set_failed(self, code: str, message: str) -> None:
        with self._lock:
            self._snapshot = RuntimeSnapshot(
                status="failed",
                error_code=code,
                error_message=message or "模型加载失败",
            )


class JsonLineConnection:
    """Read and write bounded UTF-8 JSON frames over a connected socket."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buffer = bytearray()
        self._send_lock = threading.Lock()

    def read_message(self) -> dict[str, object] | None:
        while b"\n" not in self._buffer:
            if len(self._buffer) > MAX_MESSAGE_BYTES:
                raise ProtocolError("MESSAGE_TOO_LARGE", "JSON line exceeds 1 MiB")
            chunk = self.sock.recv(65536)
            if not chunk:
                if self._buffer:
                    raise ProtocolError("INVALID_REQUEST", "JSON line is not terminated")
                return None
            self._buffer.extend(chunk)
        line, remainder = bytes(self._buffer).split(b"\n", 1)
        self._buffer = bytearray(remainder)
        if len(line) > MAX_MESSAGE_BYTES:
            raise ProtocolError("MESSAGE_TOO_LARGE", "JSON line exceeds 1 MiB")
        try:
            decoded = line.decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("INVALID_REQUEST", "Request is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ProtocolError("INVALID_REQUEST", "Request JSON must be an object")
        return payload

    def send(self, payload: Mapping[str, object]) -> None:
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._send_lock:
            self.sock.sendall(encoded)


class OrientationCommandDispatcher:
    """Validate protocol commands and translate domain failures to stable codes."""

    def __init__(self, runtime_or_classifier: ServiceRuntime | OrientationClassifier,
                 library: WorkpieceLibrary | None = None):
        if isinstance(runtime_or_classifier, ServiceRuntime):
            self.runtime = runtime_or_classifier
        else:
            if library is None:
                raise TypeError("library is required when constructing a dispatcher from a classifier")
            self.runtime = ServiceRuntime()
            self.runtime.set_ready(runtime_or_classifier, library)

    @staticmethod
    def _response(request_id: object, *, ok: bool, **fields: object) -> dict[str, object]:
        return {"version": PROTOCOL_VERSION, "request_id": request_id, "ok": ok, **fields}

    @classmethod
    def _error(cls, request_id: object, code: str, message: str) -> dict[str, object]:
        return cls._response(request_id, ok=False, error={"code": code, "message": message})

    @classmethod
    def _fast_prediction_error(
        cls,
        request_id: object,
        catalog: Any,
        workpiece_id: str,
        exc: BaseException,
    ) -> dict[str, object] | None:
        code = _prefixed_fast_error_code(exc)
        if code != "FAST_CACHE_NOT_READY":
            return None if code is None else cls._error(request_id, code, str(exc))

        status = None
        status_reader = getattr(catalog, "fast_cache_status", None)
        if callable(status_reader):
            try:
                status = status_reader(workpiece_id)
            except Exception:
                LOGGER.debug("Unable to read fast cache status after prediction failure", exc_info=True)
        if isinstance(status, Mapping):
            status_error = status.get("error")
            status_code = _prefixed_fast_error_code(status_error) if status_error else None
            if status_code == "FAST_CACHE_CAPABILITY_UNAVAILABLE":
                return cls._error(request_id, status_code, str(status_error))
            if status.get("state") == "failed":
                return cls._error(
                    request_id,
                    "FAST_CACHE_BUILD_FAILED",
                    str(status_error or exc),
                )
        return cls._error(request_id, code, str(exc))

    def dispatch(
        self,
        request: Mapping[str, object],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        request_id = request.get("request_id") if isinstance(request, Mapping) else None
        if not isinstance(request, Mapping):
            return self._error(request_id, "INVALID_REQUEST", "Request must be a JSON object")
        if request.get("version") != PROTOCOL_VERSION:
            return self._error(request_id, "UNSUPPORTED_PROTOCOL_VERSION", "Only protocol version 1 is supported")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            return self._error(request_id, "INVALID_REQUEST", "request_id must be a non-empty string")
        command = request.get("command")
        if not isinstance(command, str):
            return self._error(request_id, "INVALID_REQUEST", "command must be a string")
        runtime = self.runtime.snapshot()
        try:
            if command == "hello":
                if runtime.status == "loading":
                    return self._response(
                        request_id,
                        ok=True,
                        service=SERVICE_NAME,
                        ready=False,
                        status="loading",
                        message="模型加载中",
                    )
                if runtime.status == "failed":
                    return self._response(
                        request_id,
                        ok=False,
                        service=SERVICE_NAME,
                        ready=False,
                        status="failed",
                        error={
                            "code": runtime.error_code or "MODEL_LOAD_FAILED",
                            "message": runtime.error_message or "模型加载失败",
                        },
                    )
                return self._response(
                    request_id,
                    ok=True,
                    service=SERVICE_NAME,
                    ready=True,
                )
            if command == "list_workpieces":
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                return self._response(request_id, ok=True, workpieces=catalog.list_workpiece_summaries())
            if command == "get_workpiece_details":
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                if not isinstance(workpiece_id, str) or not workpiece_id.strip():
                    return self._error(
                        request_id,
                        "INVALID_REQUEST",
                        "workpiece_id must be a non-empty string",
                    )
                catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                return self._response(
                    request_id,
                    ok=True,
                    workpiece=catalog.get_workpiece_details(workpiece_id),
                )
            if command == "list_recycled_workpieces":
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                return self._response(request_id, ok=True, workpieces=catalog.list_recycled())
            if command in {"recycle_workpiece", "restore_workpiece", "purge_workpiece"}:
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                operation_id = request.get("operation_id")
                if not isinstance(workpiece_id, str) or not workpiece_id:
                    return self._error(request_id, "INVALID_REQUEST", "workpiece_id must be a non-empty string")
                if not isinstance(operation_id, str) or not operation_id:
                    return self._error(request_id, "INVALID_REQUEST", "operation_id must be a non-empty string")
                catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                if command == "recycle_workpiece":
                    if runtime.evolution is not None and hasattr(runtime.evolution, "cancel_for_workpiece"):
                        runtime.evolution.cancel_for_workpiece(workpiece_id)
                    result = catalog.recycle(workpiece_id, operation_id=operation_id)
                    return self._response(request_id, ok=True, workpiece=result)
                if command == "restore_workpiece":
                    record = catalog.restore(workpiece_id, operation_id=operation_id)
                    return self._response(
                        request_id,
                        ok=True,
                        workpiece={"id": record.id, "name": record.name, "revision": record.revision},
                    )
                result = catalog.purge(workpiece_id, operation_id=operation_id)
                return self._response(request_id, ok=True, workpiece=result)
            if command == "submit_confirmation":
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                orientation = request.get("orientation")
                image_path = request.get("image_path")
                operation_id = request.get("operation_id")
                if not all(isinstance(value, str) and value for value in (workpiece_id, orientation, image_path, operation_id)):
                    return self._error(request_id, "INVALID_REQUEST", "submit_confirmation requires workpiece_id, orientation, image_path and operation_id")
                job = runtime.evolution.submit_confirmation(
                    workpiece_id, orientation, Path(image_path), operation_id=operation_id
                )
                return self._response(request_id, ok=True, job=job)
            if command == "list_evolution_jobs":
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                return self._response(request_id, ok=True, jobs=runtime.evolution.list_jobs())
            if command == "evolution_job_action":
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                job_id = request.get("job_id")
                action = request.get("action")
                if not isinstance(job_id, str) or not isinstance(action, str):
                    return self._error(request_id, "INVALID_REQUEST", "evolution_job_action requires job_id and action")
                return self._response(request_id, ok=True, job=runtime.evolution.action(job_id, action))
            if command == "get_workpiece_annotations":
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                if not isinstance(workpiece_id, str) or not workpiece_id:
                    return self._error(request_id, "INVALID_REQUEST", "workpiece_id must be a non-empty string")
                return self._response(
                    request_id, ok=True,
                    annotations=runtime.evolution.get_annotations(workpiece_id),
                )
            if command == "save_workpiece_annotations":
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                groups = request.get("groups")
                base_revision = request.get("base_revision")
                operation_id = request.get("operation_id")
                progress_events = request.get("progress_events", False)
                if (
                    not isinstance(workpiece_id, str) or not workpiece_id
                    or not isinstance(groups, list)
                    or type(base_revision) is not int or base_revision < 0
                    or not isinstance(operation_id, str) or not operation_id
                    or not isinstance(progress_events, bool)
                ):
                    return self._error(
                        request_id, "INVALID_REQUEST",
                        "save_workpiece_annotations requires workpiece_id, groups, base_revision and operation_id",
                    )
                return self._response(
                    request_id, ok=True,
                    annotations=runtime.evolution.save_annotations(
                        workpiece_id, groups, expected_revision=base_revision, operation_id=operation_id,
                        progress_callback=progress_callback if progress_events else None,
                    ),
                )
            if command in {"set_workpiece_annotation_group_enabled", "delete_workpiece_annotation_group"}:
                if runtime.status != "ready" or runtime.evolution is None:
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                group_id = request.get("group_id")
                base_revision = request.get("base_revision")
                operation_id = request.get("operation_id")
                if (
                    not isinstance(workpiece_id, str) or not workpiece_id
                    or not isinstance(group_id, str) or not group_id
                    or type(base_revision) is not int or base_revision < 0
                    or not isinstance(operation_id, str) or not operation_id
                ):
                    return self._error(
                        request_id, "INVALID_REQUEST",
                        f"{command} requires workpiece_id, group_id, base_revision and operation_id",
                    )
                if command == "set_workpiece_annotation_group_enabled":
                    enabled = request.get("enabled")
                    if type(enabled) is not bool:
                        return self._error(request_id, "INVALID_REQUEST", "enabled must be boolean")
                    annotations = runtime.evolution.set_group_enabled(
                        workpiece_id, group_id, enabled,
                        expected_revision=base_revision, operation_id=operation_id,
                    )
                else:
                    annotations = runtime.evolution.delete_group(
                        workpiece_id, group_id,
                        expected_revision=base_revision, operation_id=operation_id,
                    )
                return self._response(request_id, ok=True, annotations=annotations)
            if command in {
                "get_geometry_mask_profile",
                "preview_geometry_mask_rule",
                "save_geometry_mask_draft",
                "validate_geometry_mask_draft",
                "get_geometry_mask_validation_job",
                "geometry_mask_validation_job_action",
                "resolve_geometry_mask_migration",
                "publish_geometry_mask_profile",
                "rollback_geometry_mask_profile",
            }:
                if runtime.status != "ready" or runtime.geometry_profiles is None:
                    return self._runtime_error(request_id, runtime)
                profiles = runtime.geometry_profiles
                if command == "get_geometry_mask_profile":
                    workpiece_id = request.get("workpiece_id")
                    if not isinstance(workpiece_id, str) or not workpiece_id:
                        return self._error(request_id, "INVALID_REQUEST", "get_geometry_mask_profile requires workpiece_id")
                    return self._response(request_id, ok=True, profile=profiles.snapshot(workpiece_id))
                if command == "get_geometry_mask_validation_job":
                    job_id = request.get("job_id")
                    if not isinstance(job_id, str) or not job_id:
                        return self._error(request_id, "INVALID_REQUEST", "get_geometry_mask_validation_job requires job_id")
                    return self._response(request_id, ok=True, job=profiles.get_job(job_id))
                if command == "geometry_mask_validation_job_action":
                    job_id = request.get("job_id")
                    action = request.get("action")
                    if not isinstance(job_id, str) or not isinstance(action, str):
                        return self._error(request_id, "INVALID_REQUEST", "geometry_mask_validation_job_action requires job_id and action")
                    return self._response(request_id, ok=True, job=profiles.action(job_id, action))
                workpiece_id = request.get("workpiece_id")
                if command == "preview_geometry_mask_rule":
                    base_library_revision = request.get("base_library_revision")
                    rule_id = request.get("rule_id")
                    direction = request.get("direction")
                    template_id = request.get("template_id")
                    seed_shape = request.get("seed_shape")
                    mode = request.get("mode")
                    margin_ratio = request.get("margin_ratio", 0.02)
                    anchor_candidate_index = request.get("anchor_candidate_index")
                    rule_candidate_index = request.get("rule_candidate_index")
                    if not isinstance(rule_id, str) or not rule_id.strip():
                        return self._error(
                            request_id, "NO_SELECTED_RULE",
                            "preview_geometry_mask_rule requires a selected rule_id",
                        )
                    if (
                        not isinstance(workpiece_id, str) or not workpiece_id
                        or type(base_library_revision) is not int or base_library_revision <= 0
                        or direction not in {"front", "back"}
                        or not isinstance(template_id, str) or not template_id
                        or not template_id.startswith(f"{direction}:")
                        or not isinstance(seed_shape, Mapping)
                        or seed_shape.get("shape") not in {"circle", "ellipse", "rotated_rectangle"}
                        or mode not in {"inside", "outside"}
                    ):
                        return self._error(request_id, "INVALID_REQUEST", "preview_geometry_mask_rule has invalid workpiece, direction, template, shape or mode")
                    try:
                        margin_value = float(margin_ratio)
                    except (TypeError, ValueError):
                        return self._error(request_id, "INVALID_REQUEST", "margin_ratio must be numeric")
                    if not math.isfinite(margin_value) or margin_value < -0.94 or margin_value > 0.94:
                        return self._error(request_id, "INVALID_REQUEST", "margin_ratio must be in [-0.94, 0.94]")
                    for name, value in (("anchor_candidate_index", anchor_candidate_index),
                                        ("rule_candidate_index", rule_candidate_index)):
                        if value is not None and type(value) is not int:
                            return self._error(request_id, "INVALID_REQUEST", f"{name} must be an integer")
                    preview = profiles.preview_rule(
                        workpiece_id,
                        expected_library_revision=base_library_revision,
                        rule_id=rule_id,
                        direction=direction,
                        template_id=template_id,
                        seed_shape=seed_shape,
                        mode=mode,
                        margin_ratio=margin_value,
                        anchor_candidate_index=anchor_candidate_index,
                        rule_candidate_index=rule_candidate_index,
                    )
                    return self._response(request_id, ok=True, preview=preview)
                operation_id = request.get("operation_id")
                if not isinstance(workpiece_id, str) or not workpiece_id:
                    return self._error(request_id, "INVALID_REQUEST", f"{command} requires workpiece_id")
                if not isinstance(operation_id, str) or not operation_id:
                    return self._error(request_id, "INVALID_REQUEST", f"{command} requires operation_id")
                if command == "resolve_geometry_mask_migration":
                    base_library_revision = request.get("base_library_revision")
                    base_draft_revision = request.get("base_draft_revision")
                    conflict_id = request.get("conflict_id")
                    resolution = request.get("resolution")
                    if (
                        type(base_library_revision) is not int or base_library_revision <= 0
                        or type(base_draft_revision) is not int or base_draft_revision < 0
                        or not isinstance(conflict_id, str) or not conflict_id.strip()
                        or not isinstance(resolution, Mapping)
                    ):
                        return self._error(
                            request_id,
                            "INVALID_REQUEST",
                            "resolve_geometry_mask_migration requires base revisions, conflict_id and resolution",
                        )
                    resolved = profiles.resolve_migration(
                        workpiece_id,
                        conflict_id,
                        resolution,
                        expected_library_revision=base_library_revision,
                        expected_draft_revision=base_draft_revision,
                        operation_id=operation_id,
                    )
                    return self._response(request_id, ok=True, profile=resolved)
                if command == "save_geometry_mask_draft":
                    base_library_revision = request.get("base_library_revision")
                    base_draft_revision = request.get("base_draft_revision")
                    draft = request.get("draft")
                    if type(base_library_revision) is not int or type(base_draft_revision) is not int or not isinstance(draft, Mapping):
                        return self._error(request_id, "INVALID_REQUEST", "save_geometry_mask_draft requires base revisions and draft")
                    saved = profiles.save_draft(
                        workpiece_id,
                        draft,
                        expected_library_revision=base_library_revision,
                        expected_draft_revision=base_draft_revision,
                        operation_id=operation_id,
                    )
                    return self._response(request_id, ok=True, profile=saved)
                if command == "validate_geometry_mask_draft":
                    base_library_revision = request.get("base_library_revision")
                    base_draft_revision = request.get("base_draft_revision")
                    if type(base_library_revision) is not int or type(base_draft_revision) is not int:
                        return self._error(request_id, "INVALID_REQUEST", "validate_geometry_mask_draft requires base revisions")
                    job = profiles.start_validation(
                        workpiece_id,
                        expected_library_revision=base_library_revision,
                        expected_draft_revision=base_draft_revision,
                        operation_id=operation_id,
                    )
                    return self._response(request_id, ok=True, job=job)
                if command == "publish_geometry_mask_profile":
                    job_id = request.get("job_id")
                    base_library_revision = request.get("base_library_revision")
                    base_draft_revision = request.get("base_draft_revision")
                    override_reason = request.get("override_reason", "")
                    if (
                        not isinstance(job_id, str) or not job_id
                        or type(base_library_revision) is not int
                        or type(base_draft_revision) is not int
                        or not isinstance(override_reason, str)
                    ):
                        return self._error(request_id, "INVALID_REQUEST", "publish_geometry_mask_profile requires job_id, base revisions and operation_id")
                    published = profiles.publish(
                        workpiece_id,
                        job_id,
                        expected_library_revision=base_library_revision,
                        expected_draft_revision=base_draft_revision,
                        operation_id=operation_id,
                        override_reason=override_reason,
                    )
                    return self._response(request_id, ok=True, profile=published)
                if command == "rollback_geometry_mask_profile":
                    base_library_revision = request.get("base_library_revision")
                    if type(base_library_revision) is not int:
                        return self._error(request_id, "INVALID_REQUEST", "rollback_geometry_mask_profile requires base_library_revision")
                    rolled_back = profiles.rollback(
                        workpiece_id,
                        expected_library_revision=base_library_revision,
                        operation_id=operation_id,
                    )
                    return self._response(request_id, ok=True, profile=rolled_back)
            if command == "register":
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                name = request.get("name")
                front_images = request.get("front_images")
                back_images = request.get("back_images")
                replace = request.get("replace", False)
                progress_events = request.get("progress_events", False)
                if not isinstance(name, str) or not isinstance(front_images, list) or not isinstance(back_images, list):
                    return self._error(request_id, "INVALID_REQUEST", "register requires name and image arrays")
                if not isinstance(replace, bool):
                    return self._error(request_id, "INVALID_REQUEST", "replace must be boolean")
                if not isinstance(progress_events, bool):
                    return self._error(request_id, "INVALID_REQUEST", "progress_events must be boolean")
                if any(not isinstance(item, str) for item in (*front_images, *back_images)):
                    return self._error(request_id, "INVALID_REQUEST", "template image paths must be strings")
                started = time.perf_counter()
                catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                record, cache = catalog.register(
                    name,
                    [Path(item) for item in front_images],
                    [Path(item) for item in back_images],
                    replace,
                    progress_callback=progress_callback if progress_events else None,
                )
                fast_status_reader = getattr(catalog, "fast_cache_status", None)
                fast_status = fast_status_reader(record.id) if callable(fast_status_reader) else {}
                response_cache = cache
                if isinstance(fast_status, Mapping) and fast_status.get("state") == "ready":
                    capture_snapshot = getattr(catalog, "capture_snapshot", None)
                    if callable(capture_snapshot):
                        current = capture_snapshot(record.id)
                        if getattr(current.record, "revision", None) == record.revision:
                            response_cache = current.cache
                fast_runtime = getattr(response_cache, "fast_runtime", None)
                if isinstance(response_cache, Mapping):
                    fast_runtime = response_cache.get("fast_runtime")
                fast_cache_revision = getattr(fast_runtime, "cache_revision", None)
                training_summary = getattr(fast_runtime, "training_summary", None)
                if isinstance(fast_runtime, Mapping):
                    fast_cache_revision = fast_runtime.get("cache_revision")
                    training_summary = fast_runtime.get("training_summary")
                return self._response(
                    request_id,
                    ok=True,
                    workpiece={"id": record.id, "name": record.name},
                    template_counts={
                        "front": len(record.front_images),
                        "back": len(record.back_images),
                    },
                    fast_cache_state=fast_status.get("state") if isinstance(fast_status, Mapping) else None,
                    fast_cache_revision=fast_cache_revision,
                    training_summary=training_summary,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            if command == "predict":
                if runtime.status != "ready":
                    return self._runtime_error(request_id, runtime)
                workpiece_id = request.get("workpiece_id")
                image_path = request.get("image_path")
                if not isinstance(workpiece_id, str) or not isinstance(image_path, str):
                    return self._error(request_id, "INVALID_REQUEST", "predict requires workpiece_id and image_path")
                try:
                    catalog = runtime.catalog or WorkpieceCatalog(runtime.library, runtime.classifier)
                    prediction = catalog.predict(workpiece_id, Path(image_path))
                except Exception as exc:
                    fast_error = self._fast_prediction_error(
                        request_id,
                        catalog,
                        workpiece_id,
                        exc,
                    )
                    if fast_error is not None:
                        return fast_error
                    if isinstance(exc, OrientationClassifierError):
                        raise
                    LOGGER.exception("Orientation classifier failed")
                    return self._error(request_id, "MODEL_ERROR", str(exc))
                return self._response(request_id, ok=True, **prediction)
            if command == "shutdown":
                return self._response(request_id, ok=True)
            return self._error(request_id, "INVALID_REQUEST", f"Unknown command: {command}")
        except WorkpieceExistsError as exc:
            return self._error(request_id, "WORKPIECE_EXISTS", str(exc))
        except (InvalidWorkpieceNameError, InvalidTemplateSetError) as exc:
            return self._error(request_id, "INVALID_TEMPLATE_SET", str(exc))
        except FeatureBuildError as exc:
            return self._error(request_id, "MODEL_ERROR", str(exc))
        except WorkpieceNotFoundError as exc:
            return self._error(request_id, "WORKPIECE_NOT_FOUND", str(exc))
        except RestoreConflictError as exc:
            return self._error(request_id, "RESTORE_CONFLICT", str(exc))
        except tuple(GEOMETRY_ERROR_CODES) as exc:
            code = next(
                value for error_type, value in GEOMETRY_ERROR_CODES.items()
                if isinstance(exc, error_type)
            )
            return self._error(request_id, code, str(exc))
        except (InvalidGeometryProfileError, CorruptGeometryProfileError) as exc:
            return self._error(request_id, "INVALID_GEOMETRY_PROFILE", str(exc))
        except StaleGeometryProfileError as exc:
            return self._error(request_id, "STALE_GEOMETRY_PROFILE", str(exc))
        except GeometryProfileNotReadyError as exc:
            return self._error(request_id, "GEOMETRY_VALIDATION_NOT_READY", str(exc))
        except GeometryValidationJobNotFoundError as exc:
            return self._error(request_id, "GEOMETRY_VALIDATION_FAILED", str(exc))
        except GeometryProfilePublishError as exc:
            code = "GEOMETRY_OVERRIDE_REQUIRED" if "override_reason" in str(exc) else "GEOMETRY_VALIDATION_FAILED"
            return self._error(request_id, code, str(exc))
        except GeometryValidationError as exc:
            return self._error(request_id, "GEOMETRY_VALIDATION_FAILED", str(exc))
        except GeometryCalibrationError as exc:
            return self._error(request_id, "INVALID_GEOMETRY", str(exc))
        except StaleWorkpieceRevisionError as exc:
            return self._error(request_id, "STALE_WORKPIECE_REVISION", str(exc))
        except DuplicateTemplateError as exc:
            return self._error(request_id, "DUPLICATE_TEMPLATE", str(exc))
        except InvalidMaskError as exc:
            return self._error(request_id, "INVALID_MASK", str(exc))
        except AnnotationGroupNotFoundError as exc:
            return self._error(request_id, "ANNOTATION_GROUP_NOT_FOUND", str(exc))
        except InvalidAnnotationReviewError as exc:
            return self._error(request_id, "INVALID_ANNOTATION_REVIEW", str(exc))
        except TemplateEvolutionError as exc:
            return self._error(request_id, "JOB_NOT_ACTIONABLE", str(exc))
        except KeyError as exc:
            return self._error(request_id, "WORKPIECE_NOT_FOUND", str(exc))
        except ImageUnreadableError as exc:
            return self._error(request_id, "IMAGE_UNREADABLE", str(exc))
        except PropagationModelError as exc:
            LOGGER.error(
                "Propagation model failure request_id=%s operation_id=%s workpiece_id=%s",
                request_id,
                request.get("operation_id"),
                request.get("workpiece_id"),
                exc_info=True,
            )
            return self._error(request_id, "MODEL_ERROR", str(exc))
        except OrientationClassifierError as exc:
            return self._error(request_id, "MODEL_ERROR", str(exc))
        except Exception:
            LOGGER.exception("Unhandled orientation command failure")
            return self._error(request_id, "INTERNAL_ERROR", "Internal server error")

    @classmethod
    def _runtime_error(cls, request_id: object, runtime: RuntimeSnapshot) -> dict[str, object]:
        if runtime.status == "loading":
            return cls._error(request_id, "MODEL_LOADING", "模型加载中")
        return cls._error(
            request_id,
            runtime.error_code or "MODEL_LOAD_FAILED",
            runtime.error_message or "模型加载失败",
        )


class OrientationTcpServer:
    """Single-client loopback TCP server with serial business command handling."""

    def __init__(
        self,
        dispatcher: OrientationCommandDispatcher,
        *,
        host: str = "127.0.0.1",
        port: int = 37651,
        handshake_timeout_seconds: float = 5.0,
    ):
        if host != "127.0.0.1":
            raise ServiceStartupError("INVALID_BIND_ADDRESS", "only 127.0.0.1 is allowed")
        if not math.isfinite(float(handshake_timeout_seconds)) or float(handshake_timeout_seconds) <= 0:
            raise ServiceStartupError("INVALID_HANDSHAKE_TIMEOUT", "handshake timeout must be positive")
        self.dispatcher = dispatcher
        self.handshake_timeout_seconds = float(handshake_timeout_seconds)
        self._stop_event = threading.Event()
        self._shutdown_started = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._listener_close_lock = threading.Lock()
        self._listener_closed = False
        self._client_state_lock = threading.Lock()
        self._handshake_in_progress = False
        self._active_client = False
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._listener.bind((host, port))
            self._listener.listen(8)
        except OSError as exc:
            self._listener.close()
            raise ServiceStartupError("PORT_IN_USE", f"cannot bind {host}:{port}: {exc}") from exc
        self.address = self._listener.getsockname()

    def _close_listener_once(self) -> None:
        with self._listener_close_lock:
            if self._listener_closed:
                return
            self._listener_closed = True
        try:
            self._listener.close()
        except OSError:
            pass

    def _claim_handshake(self) -> bool:
        with self._client_state_lock:
            if self._active_client or self._handshake_in_progress:
                return False
            self._handshake_in_progress = True
            return True

    def _finish_handshake(self, success: bool) -> None:
        with self._client_state_lock:
            self._handshake_in_progress = False
            self._active_client = success

    def _release_client(self) -> None:
        with self._client_state_lock:
            self._active_client = False

    def _handle_client(self, sock: socket.socket) -> None:
        connection = JsonLineConnection(sock)
        progress_active = True

        def dispatch_request(request: Mapping[str, object]) -> dict[str, object]:
            nonlocal progress_active
            callback = None
            command = request.get("command")
            if command in {"register", "save_workpiece_annotations"} and request.get("progress_events") is True:
                def send_progress(progress: dict[str, object]) -> None:
                    nonlocal progress_active
                    if not progress_active:
                        return
                    try:
                        connection.send(
                            {
                                "version": PROTOCOL_VERSION,
                                "request_id": request.get("request_id"),
                                "event": "progress",
                                "command": command,
                                "progress": progress,
                            }
                        )
                    except OSError:
                        progress_active = False

                callback = send_progress
            return self.dispatcher.dispatch(request, progress_callback=callback)

        if not self._claim_handshake():
            try:
                sock.settimeout(1.0)
                request_id = None
                try:
                    pending = connection.read_message()
                    if isinstance(pending, dict):
                        request_id = pending.get("request_id")
                except (ProtocolError, socket.timeout, OSError):
                    pass
                connection.send(self._error_response(request_id, "SERVER_BUSY", "Another client is active"))
            finally:
                sock.close()
            return
        handshake_ok = False
        try:
            sock.settimeout(self.handshake_timeout_seconds)
            while True:
                try:
                    first = connection.read_message()
                except ProtocolError as exc:
                    connection.send(self._error_response(None, exc.code, exc.message))
                    if exc.code == "MESSAGE_TOO_LARGE":
                        return
                    continue
                if first is None:
                    return
                response = dispatch_request(first)
                if first.get("command") == "hello" and response.get("ok") and response.get("ready") is True:
                    break
                connection.send(response)
                if first.get("command") == "shutdown" and response.get("ok"):
                    self.request_shutdown()
                    return
            sock.settimeout(None)
            self._finish_handshake(True)
            handshake_ok = True
            connection.send(response)
            while not self._stop_event.is_set():
                try:
                    request = connection.read_message()
                except ProtocolError as exc:
                    connection.send(self._error_response(None, exc.code, exc.message))
                    if exc.code == "MESSAGE_TOO_LARGE":
                        break
                    continue
                if request is None:
                    break
                response = dispatch_request(request)
                connection.send(response)
                if request.get("command") == "shutdown" and response.get("ok"):
                    self.request_shutdown()
                    break
        except ProtocolError as exc:
            try:
                connection.send(self._error_response(None, exc.code, exc.message))
            except OSError:
                pass
        except (ConnectionError, OSError):
            pass
        finally:
            if not handshake_ok:
                self._finish_handshake(False)
            else:
                self._release_client()
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _error_response(request_id: object, code: str, message: str) -> dict[str, object]:
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": False,
            "error": {"code": code, "message": message},
        }

    def serve_forever(self) -> None:
        self._listener.settimeout(0.2)
        try:
            while not self._stop_event.is_set():
                if self._shutdown_started.is_set():
                    self._stop_event.wait()
                    break
                try:
                    client, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise
                if self._shutdown_started.is_set():
                    client.close()
                    self._stop_event.wait()
                    break
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
        finally:
            self._close_listener_once()

    def request_shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self._shutdown_started.set()
        runtime = self.dispatcher.runtime.request_shutdown()
        profiles = runtime.geometry_profiles or getattr(runtime.catalog, "geometry_profiles", None)
        try:
            _shutdown_runtime_components(runtime.catalog, runtime.evolution, profiles)
        finally:
            self._stop_event.set()
            self._close_listener_once()


def _load_runtime(
    runtime: ServiceRuntime,
    project_root: Path,
    model_dir: Path,
    library_dir: Path,
    local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
    inference_mode: str = DEFAULT_INFERENCE_MODE,
) -> None:
    try:
        classifier = OrientationClassifier.load(
            project_root,
            model_dir,
            local_search_mode=local_search_mode,
            inference_mode=inference_mode,
        )
        library = WorkpieceLibrary(library_dir)
        catalog = WorkpieceCatalog(library, classifier)
        profiles = GeometryMaskProfiles(
            catalog,
            getattr(classifier, "geometry_calibrator", None),
            storage_dir=Path(library_dir) / ".geometry-mask-jobs",
            start_worker=False,
        )
        catalog.set_geometry_profiles(profiles)
        catalog.recover()
        runtime.set_ready(classifier, library, catalog)
    except Exception as exc:
        LOGGER.exception("Orientation service model loading failed")
        runtime.set_failed("MODEL_LOAD_FAILED", str(exc) or type(exc).__name__)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workpiece orientation loopback service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37651)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument(
        "--local-search-mode",
        choices=LOCAL_SEARCH_MODES,
        default=DEFAULT_LOCAL_SEARCH_MODE,
    )
    parser.add_argument(
        "--inference-mode",
        choices=INFERENCE_MODES,
        default=DEFAULT_INFERENCE_MODE,
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("INVALID_BIND_ADDRESS: only 127.0.0.1 is allowed")
    configure_diagnostic_logging(args.library_dir)
    _prepare_windows_torch_dll_path()
    runtime = ServiceRuntime()
    try:
        server = OrientationTcpServer(OrientationCommandDispatcher(runtime), host=args.host, port=args.port)
    except ServiceStartupError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}") from exc
    loader = threading.Thread(
        target=_load_runtime,
        args=(
            runtime,
            args.project_root,
            args.model_dir,
            args.library_dir,
            args.local_search_mode,
            args.inference_mode,
        ),
        name="orientation-model-loader",
        daemon=True,
    )
    loader.start()
    try:
        server.serve_forever()
    finally:
        server.request_shutdown()
        loader.join()


if __name__ == "__main__":
    main()
