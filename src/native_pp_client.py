"""Persistent subprocess client for the native PP-ShiTu feature service.

The client deliberately owns the process boundary.  A failed native process is
poisoned for the rest of its lifetime so callers cannot accidentally retry the
same request through a different backend or observe partially ordered results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import BinaryIO

import numpy as np

from src.model_fingerprint import model_directory_sha256
from src.native_pp_protocol import (
    CLOSE,
    DEFAULT_MAX_FRAME_BYTES,
    ERROR,
    Frame,
    HELLO,
    MAX_BATCH,
    NativeHello,
    NativeProtocolError,
    NativeResult,
    RESULT,
    decode_error,
    decode_hello,
    decode_result,
    encode_frame,
    encode_predict,
    read_frame,
    write_frame,
)


MIN_STARTUP_TIMEOUT_S = 5.0


class NativePPError(RuntimeError):
    """A stable, service-facing native backend failure."""

    def __init__(self, code: str, message: str, *, details: Mapping[str, object] | None = None):
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")


class _ReaderFailure:
    def __init__(self, error: BaseException):
        self.error = error


class _ReaderEof:
    pass


def _format_float(value: object, name: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must contain finite numbers")
    return repr(number)


def _format_triplet(value: object, name: str) -> str:
    if isinstance(value, (str, bytes, bytearray)):
        parts = str(value).split(",")
    else:
        try:
            parts = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(f"{name} must contain three values") from exc
    if len(parts) != 3:
        raise ValueError(f"{name} must contain three values")
    return ",".join(_format_float(part, name) for part in parts)


class NativePPClient:
    """Own one long-lived ``ppshitu_rec_service`` child process."""

    def __init__(
        self,
        executable: Path,
        model_dir: Path,
        *,
        threads: int = 1,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        max_batch: int = 256,
        request_timeout_s: float = 30.0,
        preprocess: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.executable = Path(executable)
        self.model_dir = Path(model_dir)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("threads must be a positive integer")
        if (
            isinstance(max_frame_bytes, bool)
            or not isinstance(max_frame_bytes, int)
            or max_frame_bytes <= 0
            or max_frame_bytes > DEFAULT_MAX_FRAME_BYTES
        ):
            raise ValueError("max_frame_bytes is out of range")
        if (
            isinstance(max_batch, bool)
            or not isinstance(max_batch, int)
            or max_batch <= 0
            or max_batch > MAX_BATCH
        ):
            raise ValueError("max_batch is out of range")
        try:
            timeout = float(request_timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("request_timeout_s must be finite and positive") from exc
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("request_timeout_s must be finite and positive")

        self.threads = threads
        self.max_frame_bytes = max_frame_bytes
        self.max_batch = max_batch
        self.request_timeout_s = timeout
        self.preprocess = dict(preprocess or {})
        self._environment = os.environ.copy()
        if env is not None:
            self._environment.update({str(key): str(value) for key, value in env.items()})

        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._stdin: BinaryIO | None = None
        self._stdout: BinaryIO | None = None
        self._stderr: BinaryIO | None = None
        self._frames: queue.Queue[object] = queue.Queue()
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stderr_buffer = bytearray()
        self._hello: NativeHello | None = None
        self._poison: NativePPError | None = None
        self._closed = False
        self._next_request_id = 1

    @property
    def ready(self) -> bool:
        with self._lock:
            return (
                self._hello is not None
                and self._poison is None
                and not self._closed
                and self._process is not None
                and self._process.poll() is None
            )

    @property
    def hello(self) -> NativeHello | None:
        with self._lock:
            return self._hello

    @property
    def feature_dim(self) -> int | None:
        hello = self.hello
        return None if hello is None else hello.feature_dim

    @property
    def last_stderr(self) -> str:
        with self._lock:
            return bytes(self._stderr_buffer).decode("utf-8", errors="replace")

    def _command(self) -> list[str]:
        if self.executable.suffix.lower() == ".py":
            command = [sys.executable, str(self.executable)]
        else:
            command = [str(self.executable)]
        command.extend(
            [
                "--serve",
                "--model-dir",
                str(self.model_dir),
                "--threads",
                str(self.threads),
                "--max-frame-bytes",
                str(self.max_frame_bytes),
                "--max-batch",
                str(self.max_batch),
            ]
        )
        aliases = {
            "input_width": "input_width",
            "width": "input_width",
            "input_height": "input_height",
            "height": "input_height",
            "scale": "scale",
            "mean_rgb": "mean_rgb",
            "mean": "mean_rgb",
            "std_rgb": "std_rgb",
            "std": "std_rgb",
        }
        values: dict[str, object] = {}
        for key, value in self.preprocess.items():
            canonical = aliases.get(str(key))
            if canonical is not None:
                values[canonical] = value
        for key, option in (
            ("input_width", "--input-width"),
            ("input_height", "--input-height"),
        ):
            if key in values:
                try:
                    number = int(values[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{key} must be a positive integer") from exc
                if number <= 0:
                    raise ValueError(f"{key} must be a positive integer")
                command.extend([option, str(number)])
        if "scale" in values:
            scale = _format_float(values["scale"], "scale")
            if float(scale) <= 0.0:
                raise ValueError("scale must be finite and positive")
            command.extend(["--scale", scale])
        if "mean_rgb" in values:
            command.extend(["--mean-rgb", _format_triplet(values["mean_rgb"], "mean_rgb")])
        if "std_rgb" in values:
            std = _format_triplet(values["std_rgb"], "std_rgb")
            if any(float(part) <= 0.0 for part in std.split(",")):
                raise ValueError("std_rgb must contain three positive values")
            command.extend(["--std-rgb", std])
        return command

    def _append_stderr(self, chunk: bytes) -> None:
        with self._lock:
            self._stderr_buffer.extend(chunk)
            del self._stderr_buffer[:-8192]

    def _read_stdout(self) -> None:
        stream = self._stdout
        if stream is None:
            return
        try:
            while True:
                frame = read_frame(stream, max_payload_bytes=self.max_frame_bytes)
                if frame is None:
                    self._frames.put(_ReaderEof())
                    return
                self._frames.put(frame)
        except BaseException as exc:  # delivered to the request owner
            self._frames.put(_ReaderFailure(exc))

    def _read_stderr(self) -> None:
        stream = self._stderr
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    return
                self._append_stderr(bytes(chunk))
        except (OSError, ValueError):
            return

    def _service_error(self, code: str, message: str) -> NativePPError:
        details = {"stderr": self.last_stderr}
        return NativePPError(code, message, details=details)

    def _terminate_process(self) -> None:
        process = self._process
        self._process = None
        stdin = self._stdin
        self._stdin = None
        if stdin is not None:
            try:
                stdin.close()
            except (OSError, ValueError):
                pass
        if process is not None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
        for stream in (self._stdout, self._stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self._stdout = None
        self._stderr = None
        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=0.2)
        self._reader_thread = None
        self._stderr_thread = None

    def _poison_client(self, failure: NativePPError) -> NativePPError:
        if self._poison is None:
            self._poison = failure
        self._hello = None
        self._terminate_process()
        return self._poison

    def _raise_if_unusable(self) -> None:
        if self._poison is not None:
            raise self._poison
        if self._closed:
            raise NativePPError("NATIVE_PP_CLOSED", "native client is closed")

    def _wait_frame(self, timeout: float, *, phase: str) -> Frame:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise self._service_error(
                    "NATIVE_PP_TIMEOUT",
                    f"native service {phase} timed out after {timeout:.3f}s",
                )
            try:
                item = self._frames.get(timeout=min(0.1, remaining))
            except queue.Empty:
                process = self._process
                if process is not None and process.poll() is not None:
                    code = "NATIVE_PP_STARTUP_FAILED" if phase == "startup" else "NATIVE_PP_SERVICE_EXITED"
                    raise self._service_error(
                        code,
                        f"native service exited with code {process.returncode}",
                    )
                continue
            if isinstance(item, _ReaderEof):
                code = "NATIVE_PP_STARTUP_FAILED" if phase == "startup" else "NATIVE_PP_SERVICE_EXITED"
                raise self._service_error(code, "native service closed stdout")
            if isinstance(item, _ReaderFailure):
                if isinstance(item.error, NativeProtocolError):
                    raise self._service_error(
                        "NATIVE_PP_PROTOCOL_ERROR", str(item.error)
                    )
                raise self._service_error("NATIVE_PP_SERVICE_EXITED", str(item.error))
            if not isinstance(item, Frame):
                raise self._service_error("NATIVE_PP_PROTOCOL_ERROR", "invalid frame reader event")
            return item

    def start(self) -> NativeHello:
        with self._lock:
            if self._hello is not None and self._poison is None and not self._closed:
                return self._hello
            self._raise_if_unusable()
            if not self.executable.is_file() or self.executable.is_symlink():
                failure = self._service_error(
                    "NATIVE_PP_CONFIG_INVALID",
                    f"native executable is not a regular file: {self.executable}",
                )
                self._poison_client(failure)
                raise failure
            if not self.model_dir.is_dir():
                failure = self._service_error(
                    "NATIVE_PP_CONFIG_INVALID",
                    f"native model directory is missing: {self.model_dir}",
                )
                self._poison_client(failure)
                raise failure
            try:
                command = self._command()
            except ValueError as exc:
                failure = self._service_error("NATIVE_PP_CONFIG_INVALID", str(exc))
                self._poison_client(failure)
                raise failure
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=self._environment,
                    creationflags=creationflags,
                )
            except (OSError, ValueError) as exc:
                failure = self._service_error("NATIVE_PP_STARTUP_FAILED", str(exc))
                self._poison_client(failure)
                raise failure
            self._process = process
            self._stdin = process.stdin
            self._stdout = process.stdout
            self._stderr = process.stderr
            self._frames = queue.Queue()
            self._stderr_buffer.clear()
            self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self._reader_thread.start()
            self._stderr_thread.start()
            try:
                # Process creation and Paddle DLL/model loading are not part
                # of the per-request SLA.  Keep a small floor so callers can
                # use a short prediction timeout without making every Python
                # interpreter launch fail its HELLO handshake.
                frame = self._wait_frame(
                    max(self.request_timeout_s, MIN_STARTUP_TIMEOUT_S),
                    phase="startup",
                )
                if frame.kind != HELLO:
                    raise self._service_error(
                        "NATIVE_PP_PROTOCOL_ERROR", "native service did not send HELLO"
                    )
                hello = decode_hello(frame)
                expected_digest = model_directory_sha256(self.model_dir)
                if hello.model_sha256 != expected_digest:
                    raise self._service_error(
                        "NATIVE_PP_MODEL_MISMATCH",
                        "native service model digest does not match the requested directory",
                    )
                self._hello = hello
                return hello
            except NativePPError as failure:
                stored = self._poison_client(failure)
                raise stored
            except NativeProtocolError as exc:
                failure = self._service_error("NATIVE_PP_PROTOCOL_ERROR", str(exc))
                stored = self._poison_client(failure)
                raise stored
            except OSError as exc:
                failure = self._service_error("NATIVE_PP_STARTUP_FAILED", str(exc))
                stored = self._poison_client(failure)
                raise stored

    def _next_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        if self._next_request_id >= (1 << 64):
            self._next_request_id = 1
        return request_id

    def _write_bytes(self, data: bytes) -> None:
        stream = self._stdin
        if stream is None:
            raise BrokenPipeError("native service stdin is closed")
        offset = 0
        while offset < len(data):
            written = stream.write(data[offset:])
            if written is None:
                written = len(data) - offset
            if written <= 0:
                raise BrokenPipeError("native service stdin write failed")
            offset += int(written)
        stream.flush()

    def predict(self, images: Sequence[np.ndarray]) -> NativeResult:
        with self._lock:
            self._raise_if_unusable()
            if self._hello is None or self._process is None:
                raise NativePPError("NATIVE_PP_NOT_READY", "native client has not completed HELLO")
            # Encode before taking any process-side action.  Invalid arrays
            # therefore cannot leave a partial request in the pipe.
            request_id = self._next_id()
            encoded = encode_predict(
                images,
                request_id,
                max_payload_bytes=self.max_frame_bytes,
            )
            started = time.perf_counter()
            try:
                self._write_bytes(encoded)
                frame = self._wait_frame(self.request_timeout_s, phase="prediction")
                if frame.kind == ERROR:
                    code, message, diagnostic = decode_error(
                        frame, expected_request_id=request_id
                    )
                    failure = self._service_error(code, message)
                    failure.details["diagnostic"] = diagnostic
                    stored = self._poison_client(failure)
                    raise stored
                if frame.kind != RESULT:
                    raise self._service_error(
                        "NATIVE_PP_PROTOCOL_ERROR", "native service returned an unexpected frame"
                    )
                result = decode_result(frame, expected_request_id=request_id)
                if result.embeddings.shape[1] != self._hello.feature_dim:
                    raise self._service_error(
                        "NATIVE_PP_DIMENSION_MISMATCH",
                        "native service result dimension differs from HELLO",
                    )
                norms = np.linalg.norm(result.embeddings.astype(np.float64), axis=1)
                if (
                    not np.isfinite(norms).all()
                    or np.any(norms <= 0.0)
                    or np.any(np.abs(norms - 1.0) > 1e-4)
                ):
                    raise self._service_error(
                        "NATIVE_PP_PROTOCOL_ERROR",
                        "native service returned non-unit embeddings",
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                timings = dict(result.timings_ms)
                service_ms = sum(
                    float(timings[name])
                    for name in ("preprocess_ms", "inference_ms", "postprocess_ms")
                )
                timings["transport_ms"] = max(0.0, elapsed_ms - service_ms)
                return NativeResult(embeddings=result.embeddings, timings_ms=timings)
            except NativePPError as failure:
                stored = self._poison_client(failure)
                raise stored
            except NativeProtocolError as exc:
                failure = self._service_error("NATIVE_PP_PROTOCOL_ERROR", str(exc))
                stored = self._poison_client(failure)
                raise stored
            except (BrokenPipeError, OSError, ValueError) as exc:
                process = self._process
                code = (
                    "NATIVE_PP_SERVICE_EXITED"
                    if process is None or process.poll() is not None
                    else "NATIVE_PP_PROTOCOL_ERROR"
                )
                failure = self._service_error(code, str(exc))
                stored = self._poison_client(failure)
                raise stored

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process is None:
                return
            if self._poison is None and self._stdin is not None:
                try:
                    self._write_bytes(encode_frame(CLOSE, 0, b"", max_payload_bytes=self.max_frame_bytes))
                except (BrokenPipeError, OSError, ValueError):
                    pass
            self._terminate_process()

    def __enter__(self) -> "NativePPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["NativePPClient", "NativePPError"]
