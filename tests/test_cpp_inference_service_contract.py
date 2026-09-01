"""Process-level contract tests for the persistent native PP-ShiTu service."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np
import pytest

from src.native_pp_protocol import (
    CLOSE,
    ERROR,
    HEADER,
    HELLO,
    decode_error,
    decode_hello,
    decode_result,
    encode_frame,
    encode_predict,
    read_frame,
)


@pytest.fixture
def native_service():
    configured = os.environ.get("WORKPIECE_CPP_SERVICE_EXE")
    if not configured:
        pytest.skip("WORKPIECE_CPP_SERVICE_EXE is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(
            "WORKPIECE_CPP_SERVICE_EXE points to a missing file: " f"{path}"
        )
    return path


@pytest.fixture
def native_model_dir():
    configured = os.environ.get("WORKPIECE_CPP_MODEL_DIR")
    if not configured:
        pytest.skip("WORKPIECE_CPP_MODEL_DIR is not configured")
    path = Path(configured)
    missing = [
        str(path / name)
        for name in ("inference.pdmodel", "inference.pdiparams")
        if not (path / name).is_file()
    ]
    if missing:
        pytest.fail("WORKPIECE_CPP_MODEL_DIR is incomplete: " + ", ".join(missing))
    return path


class NativeRawProcess:
    def __init__(self, executable: Path, model_dir: Path, *, max_batch=256):
        self.executable = executable
        self.model_dir = model_dir
        self.max_batch = max_batch
        self.process = None
        self._frames = queue.Queue()
        self._stderr = bytearray()
        self._stderr_lock = threading.Lock()
        self._reader = None
        self._stderr_reader = None

    def __enter__(self):
        self.process = subprocess.Popen(
            [
                str(self.executable),
                "--serve",
                "--model-dir",
                str(self.model_dir),
                "--threads",
                "1",
                "--max-frame-bytes",
                str(16 * 1024 * 1024),
                "--max-batch",
                str(self.max_batch),
                "--input-width",
                "224",
                "--input-height",
                "224",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._reader = threading.Thread(target=self._read_frames, daemon=True)
        self._stderr_reader = threading.Thread(
            target=self._read_stderr, daemon=True
        )
        self._reader.start()
        self._stderr_reader.start()
        return self

    def _read_frames(self):
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            while True:
                frame = read_frame(self.process.stdout, max_payload_bytes=16 * 1024 * 1024)
                if frame is None:
                    self._frames.put(EOFError("native service closed stdout"))
                    return
                self._frames.put(frame)
        except BaseException as exc:  # transfer parser errors to the test thread
            self._frames.put(exc)

    def _read_stderr(self):
        assert self.process is not None
        assert self.process.stderr is not None
        while True:
            chunk = self.process.stderr.read(4096)
            if not chunk:
                return
            with self._stderr_lock:
                self._stderr.extend(chunk)
                del self._stderr[:-8192]

    def read_frame(self, timeout=30.0):
        item = self._frames.get(timeout=timeout)
        if isinstance(item, BaseException):
            raise item
        return item

    def write_predict(self, images, request_id):
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(
            encode_predict(images, request_id, max_payload_bytes=16 * 1024 * 1024)
        )
        self.process.stdin.flush()

    def write_raw(self, value: bytes):
        assert self.process is not None and self.process.stdin is not None
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def close_stdin(self):
        if self.process is not None and self.process.stdin is not None:
            self.process.stdin.close()
            self.process.stdin = None

    def stderr_tail(self):
        with self._stderr_lock:
            return bytes(self._stderr).decode(errors="replace")

    def wait(self, timeout=10.0):
        assert self.process is not None
        return self.process.wait(timeout=timeout)

    def __exit__(self, exc_type, exc, tb):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                try:
                    self.write_raw(encode_frame(CLOSE, 0, b"", max_payload_bytes=256))
                    self.close_stdin()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3.0)
        finally:
            if self.process.stdin is not None:
                self.process.stdin.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
            if self.process.stderr is not None:
                self.process.stderr.close()


def _synthetic_images():
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    first[:, :, 0] = 32
    second = np.zeros((8, 8, 3), dtype=np.uint8)
    second[:, :, 1] = 224
    return first, second


@pytest.mark.integration
def test_service_hello_and_ordered_batch(native_service, native_model_dir):
    first, second = _synthetic_images()
    with NativeRawProcess(native_service, native_model_dir, max_batch=8) as process:
        hello_frame = process.read_frame()
        assert hello_frame.kind == HELLO
        hello = decode_hello(hello_frame)
        assert hello.feature_dim == 512
        assert hello.max_batch == 8
        assert hello.threads == 1

        process.write_predict([second, first], request_id=12)
        result_frame = process.read_frame()
        result = decode_result(result_frame, expected_request_id=12)
        assert result.embeddings.shape == (2, hello.feature_dim)
        np.testing.assert_allclose(
            np.linalg.norm(result.embeddings, axis=1), np.ones(2), atol=1e-4
        )

        # Compare each row with a single-image request to make ordering
        # observable, rather than merely checking the row count.
        process.write_predict([second], request_id=13)
        second_result = decode_result(process.read_frame(), expected_request_id=13)
        process.write_predict([first], request_id=14)
        first_result = decode_result(process.read_frame(), expected_request_id=14)
        np.testing.assert_allclose(result.embeddings[0], second_result.embeddings[0], atol=1e-5)
        np.testing.assert_allclose(result.embeddings[1], first_result.embeddings[0], atol=1e-5)


@pytest.mark.integration
def test_service_returns_request_scoped_error_for_malformed_frame(
    native_service, native_model_dir
):
    with NativeRawProcess(native_service, native_model_dir) as process:
        assert decode_hello(process.read_frame()).feature_dim == 512
        # A complete PREDICT frame with count=0 is malformed but still has a
        # trustworthy request id, so the service must answer with ERROR.
        malformed = encode_frame(2, 77, b"\0\0\0\0", max_payload_bytes=256)
        process.write_raw(malformed)
        error_frame = process.read_frame()
        assert error_frame.kind == ERROR
        code, message, diagnostic = decode_error(error_frame, expected_request_id=77)
        assert code == "INVALID_PREDICT"
        assert "batch" in message.lower()
        assert diagnostic


@pytest.mark.integration
def test_service_truncated_frame_exits_nonzero_without_text_stdout(
    native_service, native_model_dir
):
    process = NativeRawProcess(native_service, native_model_dir).__enter__()
    try:
        hello = process.read_frame()
        assert hello.kind == HELLO
        # Header prefix only: request id and payload cannot be trusted.
        process.write_raw(HEADER.pack(b"PPSH", 1, 2, 0, 91)[:7])
        process.close_stdin()
        assert process.wait(timeout=10.0) != 0
        assert "truncated" in process.stderr_tail().lower()
    finally:
        process.__exit__(None, None, None)
