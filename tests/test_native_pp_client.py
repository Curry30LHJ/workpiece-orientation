from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

from src.native_pp_client import NativePPClient, NativePPError
from src.native_pp_protocol import NativeProtocolError


FAKE_SERVICE = Path(__file__).parent / "fixtures" / "fake_native_pp_service.py"


@pytest.fixture
def fake_service():
    assert FAKE_SERVICE.is_file()
    return FAKE_SERVICE


def _client(fake_service, model_dir, **kwargs):
    # The client accepts a .py executable and invokes it with the current
    # interpreter, which keeps this fixture independent of platform launchers.
    return NativePPClient(fake_service, model_dir, **kwargs)


def test_client_waits_for_hello_and_preserves_order(fake_service, tmp_path):
    client = _client(fake_service, tmp_path, request_timeout_s=5.0)
    hello = client.start()
    assert hello.feature_dim == 2
    result = client.predict(
        [
            np.full((2, 2, 3), 2, np.uint8),
            np.full((2, 2, 3), 1, np.uint8),
        ]
    )
    assert result.embeddings.shape == (2, 2)
    assert result.embeddings[0, 0] > result.embeddings[1, 0]
    assert result.timings_ms["transport_ms"] >= 0.0
    client.close()


@pytest.mark.parametrize(
    "mode,code",
    [
        ("sleep", "NATIVE_PP_TIMEOUT"),
        ("exit", "NATIVE_PP_SERVICE_EXITED"),
        ("bad-dimension", "NATIVE_PP_DIMENSION_MISMATCH"),
    ],
)
def test_client_poisoned_process_is_not_reused(
    fake_service, tmp_path, monkeypatch, mode, code
):
    monkeypatch.setenv("FAKE_NATIVE_PP_MODE", mode)
    client = _client(fake_service, tmp_path, request_timeout_s=0.05)
    client.start()
    with pytest.raises(NativePPError) as raised:
        client.predict([np.zeros((2, 2, 3), np.uint8)])
    assert raised.value.code == code
    with pytest.raises(NativePPError) as reused:
        client.predict([np.zeros((2, 2, 3), np.uint8)])
    assert reused.value.code == code
    client.close()
    client.close()


def test_client_rejects_invalid_images_before_writing(fake_service, tmp_path):
    client = _client(fake_service, tmp_path)
    client.start()
    with pytest.raises(NativeProtocolError, match="uint8|3 channels|contiguous"):
        client.predict([np.zeros((2, 2, 3), np.float32)])
    client.close()


def test_client_detects_model_digest_mismatch(fake_service, tmp_path, monkeypatch):
    # Alter the fixture's advertised digest through a dedicated fake mode so
    # the client fails before sending a prediction request.
    monkeypatch.setenv("FAKE_NATIVE_PP_MODE", "model-mismatch")
    client = _client(fake_service, tmp_path)
    with pytest.raises(NativePPError) as raised:
        client.start()
    assert raised.value.code == "NATIVE_PP_MODEL_MISMATCH"
    client.close()

