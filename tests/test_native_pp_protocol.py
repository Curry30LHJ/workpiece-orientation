import io
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from src.native_pp_protocol import (
    CLOSE,
    ERROR,
    HELLO,
    IMAGE_HEADER,
    MAGIC,
    NativeHello,
    NativeProtocolError,
    PREDICT,
    PROTOCOL_VERSION,
    RESULT,
    Frame,
    decode_error,
    decode_hello,
    decode_result,
    encode_error,
    encode_frame,
    encode_hello,
    encode_predict,
    encode_result,
    read_frame,
    write_frame,
)


HEADER = struct.Struct("<4sHHIQ")


def make_result_frame(request_id: int, rows: list[list[float]]) -> Frame:
    embeddings = np.asarray(rows, dtype=np.float32)
    encoded = encode_result(
        embeddings,
        {"preprocess_ms": 1.0, "inference_ms": 2.0, "postprocess_ms": 3.0},
        request_id=request_id,
        max_payload_bytes=1024,
    )
    return read_frame(io.BytesIO(encoded), max_payload_bytes=1024)


def test_frame_header_is_fixed_and_little_endian():
    encoded = encode_frame(2, 7, b"abc", max_payload_bytes=1024)
    assert encoded[:20] == HEADER.pack(MAGIC, PROTOCOL_VERSION, 2, 3, 7)
    assert len(encoded) == 23


def test_predict_rejects_empty_bad_channels_and_oversized_payload():
    with pytest.raises(NativeProtocolError, match="empty batch"):
        encode_predict([], 1, max_payload_bytes=1024)
    with pytest.raises(NativeProtocolError, match="3 channels"):
        encode_predict([np.zeros((2, 2, 1), np.uint8)], 1, max_payload_bytes=1024)
    with pytest.raises(NativeProtocolError, match="frame"):
        encode_predict([np.zeros((64, 64, 3), np.uint8)], 1, max_payload_bytes=32)


def test_predict_record_uses_the_declared_16_byte_header():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    encoded = encode_predict([image], 9, max_payload_bytes=1024)
    payload = encoded[HEADER.size:]
    assert struct.unpack_from("<I", payload, 0)[0] == 1
    width, height, channels, data_len = IMAGE_HEADER.unpack_from(payload, 4)
    assert (width, height, channels, data_len) == (2, 2, 3, 12)
    assert payload[4 + IMAGE_HEADER.size : 4 + IMAGE_HEADER.size + data_len] == image.tobytes()


def test_result_decode_preserves_rows_and_checks_request_id():
    frame = make_result_frame(9, [[1.0, 0.0], [0.0, 1.0]])
    result = decode_result(frame, expected_request_id=9)
    np.testing.assert_allclose(result.embeddings, np.eye(2, dtype=np.float32))
    assert result.timings_ms == {
        "preprocess_ms": pytest.approx(1.0),
        "inference_ms": pytest.approx(2.0),
        "postprocess_ms": pytest.approx(3.0),
    }
    with pytest.raises(NativeProtocolError, match="request_id"):
        decode_result(frame, expected_request_id=10)


def test_hello_and_error_round_trip():
    hello = NativeHello("service-1", "a" * 64, 512, 256, 2)
    hello_frame = read_frame(
        io.BytesIO(encode_hello(hello, request_id=0, max_payload_bytes=1024)),
        max_payload_bytes=1024,
    )
    assert decode_hello(hello_frame) == hello

    error_frame_bytes = encode_error(
        "INVALID_IMAGE", "bad dimensions", "request=4", request_id=4, max_payload_bytes=1024
    )
    error_frame = read_frame(io.BytesIO(error_frame_bytes), max_payload_bytes=1024)
    assert decode_error(error_frame, expected_request_id=4) == (
        "INVALID_IMAGE",
        "bad dimensions",
        "request=4",
    )


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda data: b"BAD!" + data[4:], "magic"),
        (lambda data: data[:4] + struct.pack("<H", 99) + data[6:], "version"),
    ],
)
def test_read_frame_rejects_bad_header(mutate, message):
    encoded = encode_frame(2, 1, b"x", max_payload_bytes=32)
    with pytest.raises(NativeProtocolError, match=message):
        read_frame(io.BytesIO(mutate(encoded)), max_payload_bytes=32)


def test_read_frame_rejects_truncation_and_payload_limit():
    encoded = encode_frame(2, 1, b"abcd", max_payload_bytes=32)
    with pytest.raises(NativeProtocolError, match="truncated"):
        read_frame(io.BytesIO(encoded[:-1]), max_payload_bytes=32)
    oversized_header = HEADER.pack(MAGIC, PROTOCOL_VERSION, 2, 33, 1)
    with pytest.raises(NativeProtocolError, match="payload"):
        read_frame(io.BytesIO(oversized_header), max_payload_bytes=32)


def test_result_decoder_rejects_zero_shape_nonfinite_timing_and_wrong_byte_count():
    base = bytearray(
        HEADER.pack(MAGIC, PROTOCOL_VERSION, RESULT, 20 + 4, 5)
        + struct.pack("<IIfff", 1, 0, 1.0, 2.0, 3.0)
        + struct.pack("<f", 1.0)
    )
    with pytest.raises(NativeProtocolError, match="dimension"):
        decode_result(read_frame(io.BytesIO(base), max_payload_bytes=1024), expected_request_id=5)

    nan_payload = struct.pack("<IIfff", 1, 1, float("nan"), 1.0, 1.0) + struct.pack("<f", 1.0)
    nan_frame = Frame(RESULT, 5, nan_payload)
    with pytest.raises(NativeProtocolError, match="timing"):
        decode_result(nan_frame, expected_request_id=5)

    wrong_size = struct.pack("<IIfff", 1, 2, 1.0, 1.0, 1.0) + struct.pack("<f", 1.0)
    with pytest.raises(NativeProtocolError, match="embedding byte"):
        decode_result(Frame(RESULT, 5, wrong_size), expected_request_id=5)


def test_result_decoder_rejects_non_unit_or_nonfinite_embeddings():
    non_unit = struct.pack("<IIfff", 1, 2, 1.0, 1.0, 1.0) + np.array(
        [[2.0, 0.0]], dtype="<f4"
    ).tobytes()
    with pytest.raises(NativeProtocolError, match="norm"):
        decode_result(Frame(RESULT, 3, non_unit), expected_request_id=3)

    nonfinite = struct.pack("<IIfff", 1, 1, 1.0, 1.0, 1.0) + np.array(
        [[np.nan]], dtype="<f4"
    ).tobytes()
    with pytest.raises(NativeProtocolError, match="finite"):
        decode_result(Frame(RESULT, 3, nonfinite), expected_request_id=3)


def test_write_frame_flushes_and_read_frame_returns_clean_eof():
    stream = io.BytesIO()
    write_frame(stream, Frame(HELLO, 0, b"ok"), max_payload_bytes=32)
    stream.seek(0)
    frame = read_frame(stream, max_payload_bytes=32)
    assert frame == Frame(HELLO, 0, b"ok")
    assert read_frame(stream, max_payload_bytes=32) is None


@pytest.fixture
def cpp_protocol_probe():
    configured = os.environ.get("WORKPIECE_CPP_PROTOCOL_PROBE_EXE")
    if not configured:
        pytest.skip("WORKPIECE_CPP_PROTOCOL_PROBE_EXE is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"WORKPIECE_CPP_PROTOCOL_PROBE_EXE points to a missing file: {path}")
    return path


@pytest.mark.integration
def test_cpp_probe_accepts_python_close_frame(cpp_protocol_probe):
    encoded = encode_frame(CLOSE, 4, b"", max_payload_bytes=256)
    result = subprocess.run(
        [str(cpp_protocol_probe), "--max-payload", "256"],
        input=encoded,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""


@pytest.mark.integration
def test_cpp_probe_rejects_payload_over_limit(cpp_protocol_probe):
    encoded = HEADER.pack(MAGIC, PROTOCOL_VERSION, PREDICT, 257, 4)
    result = subprocess.run(
        [str(cpp_protocol_probe), "--max-payload", "256"],
        input=encoded,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert result.stdout == b""
