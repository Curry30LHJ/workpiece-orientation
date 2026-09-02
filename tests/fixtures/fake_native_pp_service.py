"""Small protocol-only fake used by NativePPClient unit tests."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model_fingerprint import model_directory_sha256
from src.native_pp_protocol import (
    CLOSE,
    ERROR,
    HELLO,
    PREDICT,
    NativeHello,
    decode_predict,
    encode_error,
    encode_hello,
    encode_result,
    read_frame,
)


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-frame-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=224)
    parser.add_argument("--input-height", type=int, default=224)
    parser.add_argument("--scale", default="0.00392156862745098")
    parser.add_argument("--mean-rgb", default="0.485,0.456,0.406")
    parser.add_argument("--std-rgb", default="0.229,0.224,0.225")
    return parser.parse_args()


def main() -> int:
    args = _args()
    if not args.serve:
        return 2
    mode = os.environ.get("FAKE_NATIVE_PP_MODE", "normal")
    max_frame = int(args.max_frame_bytes)
    model_sha256 = model_directory_sha256(Path(args.model_dir))
    if mode == "model-mismatch":
        model_sha256 = "0" * 64
    hello = NativeHello(
        "fake-native/1",
        model_sha256,
        2,
        int(args.max_batch),
        int(args.threads),
    )
    sys.stdout.buffer.write(encode_hello(hello, request_id=0, max_payload_bytes=max_frame))
    sys.stdout.buffer.flush()
    if mode == "exit":
        return 9

    while True:
        frame = read_frame(sys.stdin.buffer, max_payload_bytes=max_frame)
        if frame is None:
            return 0
        if frame.kind == CLOSE:
            return 0
        if frame.kind != PREDICT:
            sys.stdout.buffer.write(
                encode_error(
                    "INVALID_REQUEST",
                    "expected PREDICT",
                    "fake service",
                    request_id=frame.request_id,
                    max_payload_bytes=max_frame,
                )
            )
            sys.stdout.buffer.flush()
            continue
        if mode == "sleep":
            time.sleep(10.0)
        images = decode_predict(frame)
        dimension = 3 if mode == "bad-dimension" else 2
        rows = []
        for image in images:
            value = float(image.reshape(-1, 3)[0, 0])
            if dimension == 2:
                rows.append([value, 1.0])
            else:
                rows.append([value, 1.0, 0.0])
        array = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / norms
        response = encode_result(
            array,
            {
                "preprocess_ms": 1.0,
                "inference_ms": 2.0,
                "postprocess_ms": 3.0,
            },
            request_id=frame.request_id,
            max_payload_bytes=max_frame,
        )
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
