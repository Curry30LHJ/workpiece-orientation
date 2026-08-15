import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import pytest

from src.orientation_tcp_service import (
    OrientationCommandDispatcher,
    OrientationTcpServer,
    ServiceStartupError,
)


class FakeLibrary:
    def __init__(self):
        self.register_calls = []

    def list_workpieces(self):
        return [{"id": "m7", "name": "M7"}]

    def register(self, name, front_images, back_images, replace, build_cache):
        self.register_calls.append((name, front_images, back_images, replace))
        return SimpleNamespace(id="m7", name=name), {"fake": "cache"}


class FakeClassifier:
    def __init__(self):
        self.predict_calls = []
        self.predict_error = None

    def set_template_cache(self, workpiece_id, cache):
        assert workpiece_id == "m7"

    def build_template_cache(self, front_images, back_images):
        return {"front": list(front_images), "back": list(back_images)}

    def predict(self, workpiece_id, image_path):
        self.predict_calls.append((workpiece_id, image_path))
        if self.predict_error is not None:
            raise self.predict_error
        return {
            "label": "front",
            "global_prediction": "front",
            "global_scores": {"front": 0.9, "back": 0.8},
            "global_margin": 0.1,
            "local_prediction": "front",
            "local_scores": {"front": 8.0, "back": 1.0},
            "local_margin": 7.0,
            "decision_source": "global",
            "needs_review": False,
            "elapsed_ms": 1.0,
        }


class TcpTestClient:
    def __init__(self, address):
        self.socket = socket.create_connection(address, timeout=3)
        self.socket.settimeout(3)
        self._buffer = b""

    def send_raw(self, data):
        self.socket.sendall(data)

    def read(self):
        while b"\n" not in self._buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise AssertionError("server closed before a JSON response")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def request(self, command, request_id=None, version=1, **fields):
        payload = {"version": version, "request_id": request_id or command, "command": command, **fields}
        self.send_raw(json.dumps(payload).encode("utf-8") + b"\n")
        return self.read()

    def close(self):
        self.socket.close()

    def wait_closed(self):
        self.socket.settimeout(2)
        try:
            return self.socket.recv(1) == b""
        except ConnectionResetError:
            return True


class RunningServer:
    def __init__(self):
        self.library = FakeLibrary()
        self.classifier = FakeClassifier()
        dispatcher = OrientationCommandDispatcher(self.classifier, self.library)
        self.server = OrientationTcpServer(dispatcher, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.address = self.server.address

    def stop(self):
        self.server.request_shutdown()
        self.thread.join(timeout=3)


@pytest.fixture
def running_server():
    instance = RunningServer()
    yield instance
    if instance.thread.is_alive():
        instance.stop()


@pytest.fixture
def client(running_server):
    value = TcpTestClient(running_server.address)
    yield value
    value.close()


def test_partial_and_multiple_json_lines_are_decoded(client):
    client.send_raw(b'{"version":1,"request_id":"1","command":"hel')
    client.send_raw(b'lo"}\n{"version":1,"request_id":"2","command":"list_workpieces"}\n')

    assert client.read()["request_id"] == "1"
    assert client.read()["request_id"] == "2"


def test_bad_json_does_not_stop_following_request(client):
    client.send_raw(b"{bad}\n")

    assert client.read()["error"]["code"] == "INVALID_REQUEST"
    assert client.request("hello")["ok"] is True


def test_wrong_version_returns_unsupported_protocol_version(client):
    response = client.request("hello", version=2)

    assert response["error"]["code"] == "UNSUPPORTED_PROTOCOL_VERSION"


def test_second_client_receives_server_busy(running_server):
    active = TcpTestClient(running_server.address)
    assert active.request("hello")["ok"] is True
    second = TcpTestClient(running_server.address)
    try:
        assert second.request("hello")["error"]["code"] == "SERVER_BUSY"
        assert second.wait_closed()
    finally:
        active.close()
        second.close()


def test_register_and_predict_responses_preserve_request_id(client, running_server):
    assert client.request("hello")["ok"] is True
    registered = client.request(
        "register",
        request_id="register-7",
        name="M7",
        replace=False,
        front_images=["正面-1.png"] * 5,
        back_images=["反面-1.png"] * 5,
    )
    assert registered["request_id"] == "register-7"
    assert registered["ok"] is True
    response = client.request(
        "predict", request_id="predict-8", workpiece_id="m7", image_path="测试图.png"
    )
    assert response["request_id"] == "predict-8"
    assert response["label"] == "front"
    assert running_server.classifier.predict_calls == [("m7", Path("测试图.png"))]


def test_message_over_one_mib_returns_message_too_large_and_closes(client):
    client.send_raw(b'{"x":"' + b"x" * (1024 * 1024) + b'"}\n')

    assert client.read()["error"]["code"] == "MESSAGE_TOO_LARGE"
    assert client.wait_closed()


def test_invalid_utf8_returns_invalid_request(client):
    client.send_raw(b"\xff\n")

    assert client.read()["error"]["code"] == "INVALID_REQUEST"


def test_first_disconnect_allows_a_new_client(running_server):
    first = TcpTestClient(running_server.address)
    assert first.request("hello")["ok"] is True
    first.close()
    time.sleep(0.05)
    replacement = TcpTestClient(running_server.address)
    try:
        assert replacement.request("hello")["ok"] is True
    finally:
        replacement.close()


def test_classifier_exception_returns_model_error_and_server_survives(client, running_server):
    assert client.request("hello")["ok"] is True
    running_server.classifier.predict_error = RuntimeError("gpu")

    error = client.request("predict", workpiece_id="m7", image_path="q.png")
    assert error["error"]["code"] == "MODEL_ERROR"
    assert client.request("list_workpieces")["ok"] is True


def test_shutdown_response_arrives_before_server_exits(running_server):
    client = TcpTestClient(running_server.address)
    assert client.request("hello")["ok"] is True

    assert client.request("shutdown")["ok"] is True
    running_server.thread.join(timeout=2)
    assert not running_server.thread.is_alive()
    client.close()


def test_rejects_non_loopback_bind_address():
    with pytest.raises(ServiceStartupError) as error:
        OrientationTcpServer(OrientationCommandDispatcher(FakeClassifier(), FakeLibrary()), host="0.0.0.0", port=0)

    assert error.value.code == "INVALID_BIND_ADDRESS"


def test_reports_port_in_use(running_server):
    with pytest.raises(ServiceStartupError) as error:
        OrientationTcpServer(
            OrientationCommandDispatcher(FakeClassifier(), FakeLibrary()),
            host="127.0.0.1",
            port=running_server.address[1],
        )

    assert error.value.code == "PORT_IN_USE"
