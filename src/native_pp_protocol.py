"""Versioned binary protocol shared by the native PP-ShiTu service and client.

The module intentionally has no Paddle/OpenCV dependency.  Keeping framing and
payload validation here makes the subprocess boundary deterministic and easy to
exercise with byte-level tests.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Mapping, Sequence

import numpy as np


MAGIC = b"PPSH"
PROTOCOL_VERSION = 1

KIND_HELLO = 1
KIND_PREDICT = 2
KIND_RESULT = 3
KIND_ERROR = 4
KIND_CLOSE = 5

# Short aliases make the protocol constants convenient for callers/tests.
HELLO = KIND_HELLO
PREDICT = KIND_PREDICT
RESULT = KIND_RESULT
ERROR = KIND_ERROR
CLOSE = KIND_CLOSE

HEADER = struct.Struct("<4sHHIQ")
IMAGE_HEADER = struct.Struct("<IIH2xI")
RESULT_HEADER = struct.Struct("<IIfff")
HELLO_PREFIX = struct.Struct("<HH")
HELLO_NUMBERS = struct.Struct("<III")
ERROR_LENGTHS = struct.Struct("<HHH")

DEFAULT_MAX_FRAME_BYTES = 256 * 1024 * 1024
MAX_BATCH = 4096
MAX_IMAGE_WIDTH = 8192
MAX_IMAGE_HEIGHT = 8192
MAX_FEATURE_DIMENSION = 65536
MAX_STRING_BYTES = 65535
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class NativeProtocolError(ValueError):
    """Raised when a frame or payload violates the native protocol contract."""


@dataclass(frozen=True)
class Frame:
    kind: int
    request_id: int
    payload: bytes


@dataclass(frozen=True)
class NativeHello:
    service_version: str
    model_sha256: str
    feature_dim: int
    max_batch: int
    threads: int


@dataclass(frozen=True)
class NativeResult:
    embeddings: np.ndarray
    timings_ms: dict[str, float]


def _require_uint(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise NativeProtocolError(f"{name} must be an unsigned integer")
    return int(value)


def _require_limit(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > _UINT32_MAX:
        raise NativeProtocolError(f"{name} must be a positive uint32")
    return int(value)


def _checked_add(total: int, increment: int, *, name: str) -> int:
    if increment < 0 or total > _UINT32_MAX - increment:
        raise NativeProtocolError(f"{name} exceeds uint32 payload limit")
    return total + increment


def _validate_frame_limit(max_payload_bytes: int) -> int:
    return _require_limit(max_payload_bytes, name="max_payload_bytes")


def _validate_kind(kind: int) -> int:
    kind = _require_uint(kind, name="kind", maximum=0xFFFF)
    if kind not in {HELLO, PREDICT, RESULT, ERROR, CLOSE}:
        raise NativeProtocolError(f"unknown frame kind: {kind}")
    return kind


def encode_frame(
    kind: int,
    request_id: int,
    payload: bytes,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    """Encode one complete frame after validating its bounded length."""

    kind = _validate_kind(kind)
    request_id = _require_uint(request_id, name="request_id", maximum=_UINT64_MAX)
    max_payload_bytes = _validate_frame_limit(max_payload_bytes)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise NativeProtocolError("payload must be bytes-like")
    payload_bytes = bytes(payload)
    if len(payload_bytes) > max_payload_bytes:
        raise NativeProtocolError("payload exceeds frame limit")
    if len(payload_bytes) > _UINT32_MAX:
        raise NativeProtocolError("payload exceeds uint32 length")
    return HEADER.pack(MAGIC, PROTOCOL_VERSION, kind, len(payload_bytes), request_id) + payload_bytes


def decode_frame_header(
    header: bytes,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> tuple[int, int, int]:
    """Validate a 20-byte header and return ``(kind, request_id, payload_len)``."""

    max_payload_bytes = _validate_frame_limit(max_payload_bytes)
    if not isinstance(header, (bytes, bytearray, memoryview)) or len(header) != HEADER.size:
        raise NativeProtocolError("frame header is truncated")
    magic, version, kind, payload_len, request_id = HEADER.unpack(bytes(header))
    if magic != MAGIC:
        raise NativeProtocolError("frame magic is invalid")
    if version != PROTOCOL_VERSION:
        raise NativeProtocolError("frame version is unsupported")
    kind = _validate_kind(kind)
    if payload_len > max_payload_bytes:
        raise NativeProtocolError("payload exceeds frame limit")
    return kind, int(request_id), int(payload_len)


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise NativeProtocolError("truncated frame payload")
        chunk = bytes(chunk)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(
    stream: BinaryIO,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> Frame | None:
    """Read one frame; return ``None`` only for clean EOF before any header byte."""

    header = stream.read(HEADER.size)
    if not header:
        return None
    header = bytes(header)
    if len(header) != HEADER.size:
        raise NativeProtocolError("truncated frame header")
    kind, request_id, payload_len = decode_frame_header(
        header, max_payload_bytes=max_payload_bytes
    )
    return Frame(
        kind=kind,
        request_id=request_id,
        payload=_read_exact(stream, payload_len),
    )


def write_frame(
    stream: BinaryIO,
    frame: Frame,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Write a complete frame, handling streams that perform short writes."""

    encoded = encode_frame(
        frame.kind,
        frame.request_id,
        frame.payload,
        max_payload_bytes=max_payload_bytes,
    )
    offset = 0
    while offset < len(encoded):
        written = stream.write(encoded[offset:])
        if written is None:
            written = len(encoded) - offset
        if written <= 0:
            raise NativeProtocolError("frame stream write failed")
        offset += int(written)
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def _encode_utf8(value: str, *, name: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise NativeProtocolError(f"{name} must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise NativeProtocolError(f"{name} is not valid UTF-8") from exc
    if len(encoded) > MAX_STRING_BYTES:
        raise NativeProtocolError(f"{name} exceeds UTF-8 length limit")
    return encoded


def _decode_utf8(payload: bytes, offset: int, *, name: str) -> tuple[str, int]:
    if offset + 2 > len(payload):
        raise NativeProtocolError(f"{name} length is truncated")
    length = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    if length > MAX_STRING_BYTES or offset + length > len(payload):
        raise NativeProtocolError(f"{name} is truncated or too long")
    try:
        value = payload[offset : offset + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NativeProtocolError(f"{name} is not valid UTF-8") from exc
    if not value:
        raise NativeProtocolError(f"{name} must not be empty")
    return value, offset + length


def encode_hello(
    hello: NativeHello,
    *,
    request_id: int = 0,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    if not isinstance(hello, NativeHello):
        raise NativeProtocolError("hello must be NativeHello")
    service = _encode_utf8(hello.service_version, name="service_version")
    digest = _encode_utf8(hello.model_sha256, name="model_sha256")
    feature_dim = _require_limit(hello.feature_dim, name="feature_dim")
    max_batch = _require_limit(hello.max_batch, name="max_batch")
    threads = _require_limit(hello.threads, name="threads")
    payload = (
        HELLO_PREFIX.pack(PROTOCOL_VERSION, len(service))
        + service
        + struct.pack("<H", len(digest))
        + digest
        + HELLO_NUMBERS.pack(feature_dim, max_batch, threads)
    )
    return encode_frame(HELLO, request_id, payload, max_payload_bytes=max_payload_bytes)


def decode_hello(frame: Frame) -> NativeHello:
    if not isinstance(frame, Frame) or frame.kind != HELLO:
        raise NativeProtocolError("expected HELLO frame")
    payload = bytes(frame.payload)
    if len(payload) < HELLO_PREFIX.size:
        raise NativeProtocolError("HELLO payload is truncated")
    payload_version, service_length = HELLO_PREFIX.unpack_from(payload, 0)
    if payload_version != PROTOCOL_VERSION:
        raise NativeProtocolError("HELLO payload version is unsupported")
    offset = HELLO_PREFIX.size
    if offset + service_length > len(payload):
        raise NativeProtocolError("service_version is truncated")
    try:
        service_version = payload[offset : offset + service_length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NativeProtocolError("service_version is not valid UTF-8") from exc
    if not service_version:
        raise NativeProtocolError("service_version must not be empty")
    offset += service_length
    model_sha256, offset = _decode_utf8(payload, offset, name="model_sha256")
    if offset + HELLO_NUMBERS.size != len(payload):
        raise NativeProtocolError("HELLO payload has unexpected trailing bytes")
    feature_dim, max_batch, threads = HELLO_NUMBERS.unpack_from(payload, offset)
    if not (0 < feature_dim <= MAX_FEATURE_DIMENSION):
        raise NativeProtocolError("HELLO feature dimension is invalid")
    if not (0 < max_batch <= MAX_BATCH):
        raise NativeProtocolError("HELLO max batch is invalid")
    if threads <= 0:
        raise NativeProtocolError("HELLO thread count is invalid")
    return NativeHello(
        service_version=service_version,
        model_sha256=model_sha256,
        feature_dim=int(feature_dim),
        max_batch=int(max_batch),
        threads=int(threads),
    )


def _validate_image(image: object, index: int) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise NativeProtocolError(f"image {index} must be a numpy array")
    if image.ndim != 3:
        raise NativeProtocolError(f"image {index} must be HWC")
    height, width, channels = (int(value) for value in image.shape)
    if channels != 3:
        raise NativeProtocolError(f"image {index} must have 3 channels")
    if image.dtype != np.uint8:
        raise NativeProtocolError(f"image {index} must be uint8")
    if not image.flags.c_contiguous:
        raise NativeProtocolError(f"image {index} must be contiguous")
    if not (0 < width <= MAX_IMAGE_WIDTH and 0 < height <= MAX_IMAGE_HEIGHT):
        raise NativeProtocolError(f"image {index} dimensions are invalid")
    expected = width * height * 3
    if expected != image.nbytes:
        raise NativeProtocolError(f"image {index} byte count does not match dimensions")
    return image


def encode_predict(
    images: Sequence[np.ndarray],
    request_id: int,
    *,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    if isinstance(images, (str, bytes, bytearray)):
        raise NativeProtocolError("images must be a sequence")
    try:
        image_list = list(images)
    except TypeError as exc:
        raise NativeProtocolError("images must be a sequence") from exc
    if not image_list:
        raise NativeProtocolError("empty batch")
    if len(image_list) > MAX_BATCH:
        raise NativeProtocolError("batch exceeds maximum")
    payload = bytearray(struct.pack("<I", len(image_list)))
    for index, image in enumerate(image_list):
        image = _validate_image(image, index)
        height, width, channels = (int(value) for value in image.shape)
        raw = image.tobytes(order="C")
        if len(raw) != width * height * channels:
            raise NativeProtocolError(f"image {index} byte count does not match dimensions")
        payload.extend(IMAGE_HEADER.pack(width, height, channels, len(raw)))
        payload.extend(raw)
        if len(payload) > max_payload_bytes:
            raise NativeProtocolError("payload exceeds frame limit")
    return encode_frame(
        PREDICT,
        _require_uint(request_id, name="request_id", maximum=_UINT64_MAX),
        bytes(payload),
        max_payload_bytes=max_payload_bytes,
    )


def decode_predict(frame: Frame) -> list[np.ndarray]:
    if not isinstance(frame, Frame) or frame.kind != PREDICT:
        raise NativeProtocolError("expected PREDICT frame")
    payload = bytes(frame.payload)
    if len(payload) < 4:
        raise NativeProtocolError("PREDICT payload is truncated")
    count = struct.unpack_from("<I", payload, 0)[0]
    if not (0 < count <= MAX_BATCH):
        raise NativeProtocolError("PREDICT batch count is invalid")
    offset = 4
    images: list[np.ndarray] = []
    for index in range(count):
        if offset + IMAGE_HEADER.size > len(payload):
            raise NativeProtocolError("PREDICT image header is truncated")
        width, height, channels, data_len = IMAGE_HEADER.unpack_from(payload, offset)
        offset += IMAGE_HEADER.size
        if channels != 3:
            raise NativeProtocolError(f"image {index} must have 3 channels")
        if not (0 < width <= MAX_IMAGE_WIDTH and 0 < height <= MAX_IMAGE_HEIGHT):
            raise NativeProtocolError(f"image {index} dimensions are invalid")
        expected = int(width) * int(height) * 3
        if data_len != expected or offset + data_len > len(payload):
            raise NativeProtocolError(f"image {index} byte count does not match dimensions")
        image = np.frombuffer(payload, dtype=np.uint8, count=data_len, offset=offset)
        images.append(image.reshape((height, width, 3)).copy())
        offset += data_len
    if offset != len(payload):
        raise NativeProtocolError("PREDICT payload has unexpected trailing bytes")
    return images


def _timing_values(timings_ms: Mapping[str, float]) -> tuple[float, float, float]:
    if not isinstance(timings_ms, Mapping):
        raise NativeProtocolError("timings_ms must be a mapping")
    values = []
    for name in ("preprocess_ms", "inference_ms", "postprocess_ms"):
        try:
            value = float(timings_ms[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise NativeProtocolError(f"missing {name} timing") from exc
        if not np.isfinite(value) or value < 0:
            raise NativeProtocolError(f"timing {name} must be finite and non-negative")
        values.append(value)
    return values[0], values[1], values[2]


def _validate_embeddings(embeddings: object) -> np.ndarray:
    try:
        array = np.asarray(embeddings, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise NativeProtocolError("embeddings must be numeric") from exc
    if array.ndim != 2:
        raise NativeProtocolError("embeddings must be a 2-D matrix")
    rows, columns = (int(value) for value in array.shape)
    if not (0 < rows <= MAX_BATCH and 0 < columns <= MAX_FEATURE_DIMENSION):
        raise NativeProtocolError("embedding dimensions are invalid")
    array = np.ascontiguousarray(array, dtype=np.dtype("<f4"))
    if not np.isfinite(array).all():
        raise NativeProtocolError("embeddings must contain finite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0) or np.any(np.abs(norms - 1.0) > 1e-3):
        raise NativeProtocolError("embeddings must have unit norm")
    return array


def encode_result(
    embeddings: object,
    timings_ms: Mapping[str, float],
    *,
    request_id: int,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    array = _validate_embeddings(embeddings)
    preprocess_ms, inference_ms, postprocess_ms = _timing_values(timings_ms)
    rows, columns = (int(value) for value in array.shape)
    payload = RESULT_HEADER.pack(
        rows,
        columns,
        preprocess_ms,
        inference_ms,
        postprocess_ms,
    ) + array.tobytes(order="C")
    return encode_frame(RESULT, request_id, payload, max_payload_bytes=max_payload_bytes)


def decode_result(frame: Frame, *, expected_request_id: int) -> NativeResult:
    if not isinstance(frame, Frame) or frame.kind != RESULT:
        raise NativeProtocolError("expected RESULT frame")
    expected_request_id = _require_uint(
        expected_request_id, name="expected_request_id", maximum=_UINT64_MAX
    )
    if frame.request_id != expected_request_id:
        raise NativeProtocolError("RESULT request_id does not match")
    payload = bytes(frame.payload)
    if len(payload) < RESULT_HEADER.size:
        raise NativeProtocolError("RESULT payload is truncated")
    rows, columns, preprocess_ms, inference_ms, postprocess_ms = RESULT_HEADER.unpack_from(payload, 0)
    if not (0 < rows <= MAX_BATCH):
        raise NativeProtocolError("RESULT count is invalid")
    if not (0 < columns <= MAX_FEATURE_DIMENSION):
        raise NativeProtocolError("RESULT dimension is invalid")
    timings = {
        "preprocess_ms": float(preprocess_ms),
        "inference_ms": float(inference_ms),
        "postprocess_ms": float(postprocess_ms),
    }
    _timing_values(timings)
    expected_bytes = int(rows) * int(columns) * 4
    actual_bytes = len(payload) - RESULT_HEADER.size
    if actual_bytes != expected_bytes:
        raise NativeProtocolError("embedding byte count does not match dimensions")
    array = np.frombuffer(payload, dtype="<f4", offset=RESULT_HEADER.size).reshape(
        (rows, columns)
    ).copy()
    array = _validate_embeddings(array)
    return NativeResult(embeddings=array, timings_ms=timings)


def encode_error(
    code: str,
    message: str,
    diagnostic: str = "",
    *,
    request_id: int,
    max_payload_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    code_bytes = _encode_utf8(code, name="error code")
    message_bytes = _encode_utf8(message, name="error message")
    if diagnostic:
        diagnostic_bytes = _encode_utf8(diagnostic, name="error diagnostic")
    else:
        diagnostic_bytes = b""
    payload = (
        ERROR_LENGTHS.pack(len(code_bytes), len(message_bytes), len(diagnostic_bytes))
        + code_bytes
        + message_bytes
        + diagnostic_bytes
    )
    return encode_frame(ERROR, request_id, payload, max_payload_bytes=max_payload_bytes)


def decode_error(
    frame: Frame,
    *,
    expected_request_id: int | None = None,
) -> tuple[str, str, str]:
    if not isinstance(frame, Frame) or frame.kind != ERROR:
        raise NativeProtocolError("expected ERROR frame")
    if expected_request_id is not None and frame.request_id != _require_uint(
        expected_request_id, name="expected_request_id", maximum=_UINT64_MAX
    ):
        raise NativeProtocolError("ERROR request_id does not match")
    payload = bytes(frame.payload)
    if len(payload) < ERROR_LENGTHS.size:
        raise NativeProtocolError("ERROR payload is truncated")
    code_len, message_len, diagnostic_len = ERROR_LENGTHS.unpack_from(payload, 0)
    offset = ERROR_LENGTHS.size
    values: list[str] = []
    for length, name, allow_empty in (
        (code_len, "error code", False),
        (message_len, "error message", False),
        (diagnostic_len, "error diagnostic", True),
    ):
        if length > MAX_STRING_BYTES or offset + length > len(payload):
            raise NativeProtocolError(f"{name} is truncated or too long")
        try:
            value = payload[offset : offset + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NativeProtocolError(f"{name} is not valid UTF-8") from exc
        if not value and not allow_empty:
            raise NativeProtocolError(f"{name} must not be empty")
        values.append(value)
        offset += length
    if offset != len(payload):
        raise NativeProtocolError("ERROR payload has unexpected trailing bytes")
    return values[0], values[1], values[2]


__all__ = [
    "CLOSE",
    "DEFAULT_MAX_FRAME_BYTES",
    "ERROR",
    "Frame",
    "HEADER",
    "HELLO",
    "IMAGE_HEADER",
    "KIND_CLOSE",
    "KIND_ERROR",
    "KIND_HELLO",
    "KIND_PREDICT",
    "KIND_RESULT",
    "MAGIC",
    "NativeHello",
    "NativeProtocolError",
    "NativeResult",
    "PREDICT",
    "PROTOCOL_VERSION",
    "RESULT",
    "decode_error",
    "decode_frame_header",
    "decode_hello",
    "decode_predict",
    "decode_result",
    "encode_error",
    "encode_frame",
    "encode_hello",
    "encode_predict",
    "encode_result",
    "read_frame",
    "write_frame",
]
