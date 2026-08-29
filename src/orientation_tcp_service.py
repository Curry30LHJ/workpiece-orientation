"""Loopback TCP JSON-lines service for the orientation classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any, Callable, Mapping

_WINDOWS_DLL_DIRECTORY_HANDLES: list[Any] = []


def _windows_short_path(path: Path) -> Path | None:
    if os.name != "nt":
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
        if length <= 0 or length >= len(buffer):
            return None
        value = buffer.value
        try:
            value.encode("ascii")
        except UnicodeEncodeError:
            return None
        return Path(value)
    except (AttributeError, OSError):
        return None


def _add_windows_dll_directory(path: Path, seen: set[str]) -> None:
    selected = _windows_short_path(path) or path
    key = str(selected).casefold()
    if key in seen:
        return
    handle = os.add_dll_directory(str(selected))
    # ``os.add_dll_directory`` unregisters the path when its handle is
    # garbage-collected.  Keep the handles alive for the lifetime of the
    # frozen service so extension modules can resolve their dependencies.
    _WINDOWS_DLL_DIRECTORY_HANDLES.append(handle)
    os.environ["PATH"] = str(selected) + os.pathsep + os.environ.get("PATH", "")
    seen.add(key)


def _prepare_windows_numpy_dll_path() -> None:
    """Expose the frozen backend runtime directories before importing NumPy.

    When the packaged service is launched from a long or non-ASCII install
    path, the Windows loader can fail to resolve NumPy's extension DLLs unless
    the frozen runtime directories are visible ahead of the first ``numpy``
    import.  Keep this narrow to the bundled runtime trees used by the frozen
    backend.
    """

    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    runtime_root = Path(sys.executable).resolve().parent
    seen: set[str] = set()
    for candidate in (
        runtime_root,
        runtime_root / "_internal",
        runtime_root / "_internal" / "numpy" / ".libs",
        runtime_root / "_internal" / "paddle" / "libs",
        # Keep compatibility with older/custom collectors that flatten these
        # directories beside the package trees.
        runtime_root / "_internal" / "numpy.libs",
        runtime_root / "_internal" / "paddle.libs",
        runtime_root / "_internal" / "cv2",
    ):
        if candidate.is_dir():
            _add_windows_dll_directory(candidate, seen)


_prepare_windows_numpy_dll_path()

# Qt launches this file by path. Add the repository root for that entrypoint
# so absolute ``src.*`` imports work both by file path and by ``python -m``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orientation_classifier import (
    COMPUTE_DEVICES,
    DEFAULT_COMPUTE_DEVICE,
    DEFAULT_INFERENCE_MODE,
    DEFAULT_LOCAL_SEARCH_MODE,
    ComputeDeviceError,
    ImageUnreadableError,
    INFERENCE_MODES,
    LOCAL_SEARCH_MODES,
    ModelFingerprintError,
    OrientationClassifier,
    OrientationClassifierError,
    PropagationModelError,
    WorkpieceNotFoundError,
)
from src.runtime_data import RuntimeDataError, legacy_runtime_data, prepare_runtime_data
from src.windows_parent_watchdog import start_parent_watchdog
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
STARTUP_PHASE_ORDER = {
    "loading_runtime": 0,
    "preparing_data": 1,
    "loading_model": 2,
    "restoring_library": 3,
    "ready": 4,
}
STARTUP_ERROR_ACTIONS = {
    "MODEL_LOAD_FAILED": "检查模型文件和后端依赖是否完整，然后重启程序",
    "MODEL_FINGERPRINT_MISMATCH": "恢复本版本随附的完整模型目录，然后重启程序",
    "GPU_UNAVAILABLE": "确认 NVIDIA 显卡和兼容驱动可用；无 NVIDIA GPU 时请改用 CPU 版",
    "DEVICE_MISMATCH": "关闭程序，确认安装包版本与计算设备一致后重新启动",
    "RUNTIME_SELF_CHECK_FAILED": "检查对应版本运行库和驱动，保留日志并联系技术支持",
    "DATA_DIRECTORY_NOT_WRITABLE": "将程序解压到当前用户可写目录，确认 data 未被占用后重启",
    "DATA_VERSION_UNSUPPORTED": "使用支持该 data 版本的软件，或恢复兼容备份",
    "DATA_LAYOUT_AMBIGUOUS": "备份 data 后移除无法识别的文件，禁止合并两个非空 data",
    "DATA_MIGRATION_FAILED": "保留现有 data 和自动备份，查看日志后重试或联系技术支持",
    "FAST_CACHE_BUILD_FAILED": "保留工件库并重启；仍失败时查看日志并重新建立该工件缓存",
}
DEFAULT_STARTUP_ERROR_ACTION = "查看后端日志并联系技术支持"


def _prefixed_fast_error_code(error: object) -> str | None:
    code = str(error).partition(":")[0].strip()
    return code if code in FAST_HARD_ERROR_CODES else None


def _diagnostic_log_series(handler: TimedRotatingFileHandler) -> list[Path]:
    log_path = Path(handler.baseFilename)
    backup_prefix = f"{log_path.name}."
    files = []
    for child in log_path.parent.iterdir():
        if child.is_symlink() or not child.is_file():
            continue
        if child.name == log_path.name:
            files.append(child)
            continue
        if not child.name.startswith(backup_prefix):
            continue
        suffix = child.name[len(backup_prefix):]
        if handler.extMatch.fullmatch(suffix):
            files.append(child)
    return files


def _prune_diagnostic_logs(
    handler: TimedRotatingFileHandler,
    maximum_bytes: int = 30 * 1024 * 1024,
) -> None:
    log_path = Path(handler.baseFilename)
    backups = sorted(path for path in _diagnostic_log_series(handler) if path != log_path)
    excess = max(0, len(backups) - handler.backupCount)
    for path in backups[:excess]:
        path.unlink()

    files = []
    total = 0
    for path in _diagnostic_log_series(handler):
        stat = path.stat()
        files.append((stat.st_mtime_ns, path, stat.st_size))
        total += stat.st_size
    for _modified, path, size in sorted(files):
        if total <= maximum_bytes:
            break
        path.unlink()
        total -= size


def configure_diagnostic_logging(logs_dir: Path) -> logging.Handler:
    """Persist rotating backend diagnostics in the exact supplied log directory.

    The handler is marked so repeated startup/configuration calls replace only
    this service's handler and leave application/test handlers untouched.
    """
    logs_dir = Path(logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "orientation-service.log"
    root_logger = logging.getLogger()
    for existing in list(root_logger.handlers):
        if not getattr(existing, "_workpiece_orientation_diagnostic", False):
            continue
        existing_path = getattr(existing, "baseFilename", None)
        if existing_path and Path(existing_path).resolve() == log_path:
            return existing
        root_logger.removeHandler(existing)
        existing.close()

    retention_probe = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
        delay=True,
    )
    try:
        _prune_diagnostic_logs(retention_probe)
    finally:
        retention_probe.close()
    handler = TimedRotatingFileHandler(
        log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        utc=False,
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
    phase: str = "loading_runtime"
    message: str = "正在加载运行环境"
    progress: int = 5
    package_version: str = "dev"
    edition: str = "dev"
    compute_device: str = "gpu"
    model_fingerprint: str = ""
    instance_token: str = "external"
    error_action: str = ""
    log_path: str = ""
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

    def __init__(
        self,
        *,
        package_version: str = "dev",
        edition: str = "dev",
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        model_fingerprint: str = "",
        instance_token: str = "external",
        log_path: str = "",
    ) -> None:
        self._lock = threading.RLock()
        self._snapshot = RuntimeSnapshot(
            status="loading",
            package_version=package_version,
            edition=edition,
            compute_device=compute_device,
            model_fingerprint=model_fingerprint,
            instance_token=instance_token,
            log_path=log_path,
        )
        self._shutdown_requested = False

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    def request_shutdown(self) -> RuntimeSnapshot:
        with self._lock:
            self._shutdown_requested = True
            return self._snapshot

    def update_loading(self, phase: str, message: str, progress: int) -> None:
        if phase not in STARTUP_PHASE_ORDER:
            raise ValueError(f"unknown loading phase: {phase}")
        if not 0 <= progress < 100:
            raise ValueError("loading progress must be between 0 and 99")
        with self._lock:
            if self._snapshot.status != "loading" or self._shutdown_requested:
                return
            if STARTUP_PHASE_ORDER[phase] < STARTUP_PHASE_ORDER[self._snapshot.phase]:
                return
            if progress < self._snapshot.progress:
                return
            self._snapshot = replace(
                self._snapshot,
                phase=phase,
                message=message,
                progress=progress,
            )

    def set_ready(self, classifier: Any, library: Any, catalog: Any | None = None) -> bool:
        owns_catalog = catalog is None
        resolved_catalog = None
        owns_geometry_profiles = False
        evolution = None
        geometry_profiles = None
        transferred = False
        try:
            resolved_catalog = catalog if catalog is not None else WorkpieceCatalog(library, classifier)
            library_dir = getattr(library, "library_dir", None)
            geometry_profiles = getattr(resolved_catalog, "geometry_profiles", None)
            if geometry_profiles is None and library_dir is not None:
                geometry_profiles = GeometryMaskProfiles(
                    resolved_catalog,
                    getattr(classifier, "geometry_calibrator", None),
                    storage_dir=Path(library_dir) / ".geometry-mask-jobs",
                    start_worker=False,
                )
                owns_geometry_profiles = True
                setter = getattr(resolved_catalog, "set_geometry_profiles", None)
                if callable(setter):
                    setter(geometry_profiles)
            with self._lock:
                if self._shutdown_requested:
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
                    return False
                previous_snapshot = self._snapshot
                self._snapshot = replace(
                    self._snapshot,
                    status="ready",
                    phase="ready",
                    message="后端已就绪",
                    progress=100,
                    compute_device=getattr(classifier, "compute_device", self._snapshot.compute_device),
                    model_fingerprint=getattr(classifier, "model_fingerprint", self._snapshot.model_fingerprint),
                    classifier=classifier,
                    library=library,
                    catalog=resolved_catalog, evolution=evolution, geometry_profiles=geometry_profiles,
                )
                try:
                    for worker in (geometry_profiles, evolution):
                        start = getattr(worker, "start", None)
                        if callable(start):
                            start()
                except Exception:
                    self._snapshot = previous_snapshot
                    raise
                transferred = True
            return True
        finally:
            if not transferred:
                _shutdown_runtime_components(
                    resolved_catalog if owns_catalog else None,
                    evolution,
                    geometry_profiles if owns_geometry_profiles else None,
                )

    def set_failed(self, code: str, message: str) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                status="failed",
                message=message or "模型加载失败",
                error_code=code,
                error_message=message or "模型加载失败",
                error_action=STARTUP_ERROR_ACTIONS.get(code, DEFAULT_STARTUP_ERROR_ACTION),
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
                hello = {
                    "service": SERVICE_NAME,
                    "ready": runtime.status == "ready",
                    "status": runtime.status,
                    "phase": runtime.phase,
                    "message": runtime.message,
                    "progress": runtime.progress,
                    "package_version": runtime.package_version,
                    "edition": runtime.edition,
                    "compute_device": runtime.compute_device,
                    "model_fingerprint": runtime.model_fingerprint,
                    "instance_token": runtime.instance_token,
                    "error_action": runtime.error_action,
                    "log_path": runtime.log_path,
                }
                if runtime.status == "loading":
                    return self._response(request_id, ok=True, **hello)
                if runtime.status == "failed":
                    return self._response(
                        request_id,
                        ok=False,
                        **hello,
                        error={
                            "code": runtime.error_code or "MODEL_LOAD_FAILED",
                            "message": runtime.error_message or "模型加载失败",
                        },
                    )
                return self._response(request_id, ok=True, **hello)
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
                if (
                    runtime.instance_token != "external"
                    and request.get("instance_token") != runtime.instance_token
                ):
                    return self._error(
                        request_id,
                        "INSTANCE_TOKEN_MISMATCH",
                        "instance_token does not match this backend instance",
                    )
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
        self._shutdown_complete = threading.Event()
        self._lifecycle_lock = threading.Lock()
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

    def _start_client_handler(self, client: socket.socket) -> bool:
        try:
            handler = threading.Thread(
                target=self._handle_client,
                args=(client,),
                daemon=True,
            )
        except Exception:
            client.close()
            raise
        try:
            with self._lifecycle_lock:
                if not self._shutdown_requested:
                    handler.start()
                    return True
        except Exception:
            client.close()
            raise
        client.close()
        return False

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
        try:
            self._listener.settimeout(0.2)
            while not self._stop_event.is_set():
                if self._shutdown_started.is_set():
                    break
                try:
                    client, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._shutdown_started.is_set():
                        break
                    raise
                if not self._start_client_handler(client):
                    break
        finally:
            self.request_shutdown()

    def request_shutdown(self) -> None:
        with self._lifecycle_lock:
            if self._shutdown_requested:
                owns_shutdown = False
                runtime = None
            else:
                runtime = self.dispatcher.runtime.request_shutdown()
                self._shutdown_requested = True
                self._shutdown_started.set()
                owns_shutdown = True
        if not owns_shutdown:
            self._shutdown_complete.wait()
            return
        assert runtime is not None
        try:
            profiles = runtime.geometry_profiles or getattr(runtime.catalog, "geometry_profiles", None)
            _shutdown_runtime_components(runtime.catalog, runtime.evolution, profiles)
        finally:
            self._stop_event.set()
            try:
                self._close_listener_once()
            finally:
                self._shutdown_complete.set()


def _load_runtime(
    runtime: ServiceRuntime,
    project_root: Path,
    model_dir: Path,
    library_dir: Path | None,
    local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
    inference_mode: str = DEFAULT_INFERENCE_MODE,
    *,
    data_root: Path | None = None,
    paddle_config_path: Path | None = None,
    compute_device: str = DEFAULT_COMPUTE_DEVICE,
    model_sha256: str | None = None,
) -> None:
    catalog = None
    profiles = None
    transferred = False
    try:
        runtime.update_loading("preparing_data", "正在检查数据目录", 15)
        if data_root is not None:
            paths = prepare_runtime_data(data_root)
        elif library_dir is not None:
            paths = legacy_runtime_data(library_dir)
        else:
            raise RuntimeDataError("DATA_LAYOUT_AMBIGUOUS", "No runtime data path was provided")
        runtime.update_loading("loading_model", "正在加载 PP-ShiTu 模型", 35)
        classifier = OrientationClassifier.load(
            project_root,
            model_dir,
            paddle_config_path=paddle_config_path,
            compute_device=compute_device,
            expected_model_fingerprint=model_sha256,
            local_search_mode=local_search_mode,
            inference_mode=inference_mode,
        )
        runtime.update_loading("restoring_library", "正在恢复工件库和快速缓存", 80)
        library = WorkpieceLibrary(paths.workpieces)
        catalog = WorkpieceCatalog(library, classifier)
        profiles = GeometryMaskProfiles(
            catalog,
            getattr(classifier, "geometry_calibrator", None),
            storage_dir=paths.workpieces / ".geometry-mask-jobs",
            start_worker=False,
        )
        catalog.set_geometry_profiles(profiles)
        catalog.recover()
        transferred = runtime.set_ready(classifier, library, catalog)
    except (RuntimeDataError, ComputeDeviceError, ModelFingerprintError) as exc:
        LOGGER.exception("Orientation service startup failed with code %s", exc.code)
        runtime.set_failed(exc.code, str(exc) or type(exc).__name__)
    except Exception as exc:
        LOGGER.exception("Orientation service model loading failed")
        runtime.set_failed("MODEL_LOAD_FAILED", str(exc) or type(exc).__name__)
    finally:
        if not transferred:
            _shutdown_runtime_components(catalog, None, profiles)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workpiece orientation loopback service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37651)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-root", type=Path)
    data_group.add_argument("--library-dir", type=Path)
    parser.add_argument("--paddle-config", type=Path)
    parser.add_argument("--compute-device", choices=COMPUTE_DEVICES, default=DEFAULT_COMPUTE_DEVICE)
    parser.add_argument("--model-sha256")
    parser.add_argument("--package-version", default="dev")
    parser.add_argument("--edition", default="dev")
    parser.add_argument("--instance-token", default="external")
    parser.add_argument("--parent-pid", type=int)
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


def _validate_packaged_arguments(args: argparse.Namespace) -> None:
    if getattr(args, "data_root", None) is None:
        return
    instance_token = str(getattr(args, "instance_token", "")).strip()
    if not instance_token or instance_token == "external":
        raise SystemExit("INVALID_INSTANCE_TOKEN: packaged startup requires a non-empty instance token")
    model_sha256 = str(getattr(args, "model_sha256", "") or "")
    if re.fullmatch(r"[0-9a-f]{64}", model_sha256) is None:
        raise SystemExit("INVALID_MODEL_SHA256: packaged startup requires a lowercase SHA-256")


def main() -> None:
    args = _build_argument_parser().parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("INVALID_BIND_ADDRESS: only 127.0.0.1 is allowed")
    _validate_packaged_arguments(args)
    data_root = getattr(args, "data_root", None)
    library_dir = getattr(args, "library_dir", None)
    if data_root is not None:
        logs_dir = Path(data_root).resolve() / "logs"
    else:
        logs_dir = legacy_runtime_data(library_dir).logs
    configure_diagnostic_logging(logs_dir)
    log_path = str((logs_dir / "orientation-service.log").resolve())
    _prepare_windows_torch_dll_path()
    runtime = ServiceRuntime(
        package_version=getattr(args, "package_version", "dev"),
        edition=getattr(args, "edition", "dev"),
        compute_device=getattr(args, "compute_device", DEFAULT_COMPUTE_DEVICE),
        model_fingerprint=getattr(args, "model_sha256", None) or "",
        instance_token=getattr(args, "instance_token", "external"),
        log_path=log_path,
    )
    try:
        server = OrientationTcpServer(OrientationCommandDispatcher(runtime), host=args.host, port=args.port)
    except ServiceStartupError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}") from exc
    start_parent_watchdog(getattr(args, "parent_pid", None), server.request_shutdown)
    loader = threading.Thread(
        target=_load_runtime,
        args=(
            runtime,
            args.project_root,
            args.model_dir,
            library_dir,
        ),
        kwargs={
            "data_root": data_root,
            "paddle_config_path": getattr(args, "paddle_config", None),
            "compute_device": getattr(args, "compute_device", DEFAULT_COMPUTE_DEVICE),
            "model_sha256": getattr(args, "model_sha256", None),
            "local_search_mode": args.local_search_mode,
            "inference_mode": args.inference_mode,
        },
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
