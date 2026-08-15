"""Loopback TCP JSON-lines service for the orientation classifier."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any, Mapping

from src.orientation_classifier import (
    ImageUnreadableError,
    OrientationClassifier,
    OrientationClassifierError,
    WorkpieceNotFoundError,
)
from src.workpiece_library import (
    FeatureBuildError,
    InvalidTemplateSetError,
    InvalidWorkpieceNameError,
    WorkpieceExistsError,
    WorkpieceLibrary,
)


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = 1
SERVICE_NAME = "workpiece-orientation"
MAX_MESSAGE_BYTES = 1024 * 1024


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

    def __init__(self, classifier: OrientationClassifier, library: WorkpieceLibrary):
        self.classifier = classifier
        self.library = library

    @staticmethod
    def _response(request_id: object, *, ok: bool, **fields: object) -> dict[str, object]:
        return {"version": PROTOCOL_VERSION, "request_id": request_id, "ok": ok, **fields}

    @classmethod
    def _error(cls, request_id: object, code: str, message: str) -> dict[str, object]:
        return cls._response(request_id, ok=False, error={"code": code, "message": message})

    def dispatch(self, request: Mapping[str, object]) -> dict[str, object]:
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
        try:
            if command == "hello":
                return self._response(
                    request_id,
                    ok=True,
                    service=SERVICE_NAME,
                    ready=True,
                )
            if command == "list_workpieces":
                return self._response(request_id, ok=True, workpieces=self.library.list_workpieces())
            if command == "register":
                name = request.get("name")
                front_images = request.get("front_images")
                back_images = request.get("back_images")
                replace = request.get("replace", False)
                if not isinstance(name, str) or not isinstance(front_images, list) or not isinstance(back_images, list):
                    return self._error(request_id, "INVALID_REQUEST", "register requires name and image arrays")
                if not isinstance(replace, bool):
                    return self._error(request_id, "INVALID_REQUEST", "replace must be boolean")
                record, cache = self.library.register(
                    name,
                    [Path(item) for item in front_images if isinstance(item, str)],
                    [Path(item) for item in back_images if isinstance(item, str)],
                    replace,
                    self.classifier.build_template_cache,
                )
                self.classifier.set_template_cache(record.id, cache)
                return self._response(
                    request_id,
                    ok=True,
                    workpiece={"id": record.id, "name": record.name},
                )
            if command == "predict":
                workpiece_id = request.get("workpiece_id")
                image_path = request.get("image_path")
                if not isinstance(workpiece_id, str) or not isinstance(image_path, str):
                    return self._error(request_id, "INVALID_REQUEST", "predict requires workpiece_id and image_path")
                try:
                    prediction = self.classifier.predict(workpiece_id, Path(image_path))
                except OrientationClassifierError:
                    raise
                except Exception as exc:
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
        except ImageUnreadableError as exc:
            return self._error(request_id, "IMAGE_UNREADABLE", str(exc))
        except OrientationClassifierError as exc:
            return self._error(request_id, "MODEL_ERROR", str(exc))
        except Exception:
            LOGGER.exception("Unhandled orientation command failure")
            return self._error(request_id, "INTERNAL_ERROR", "Internal server error")


class OrientationTcpServer:
    """Single-client loopback TCP server with serial business command handling."""

    def __init__(self, dispatcher: OrientationCommandDispatcher, *, host: str = "127.0.0.1", port: int = 37651):
        if host != "127.0.0.1":
            raise ServiceStartupError("INVALID_BIND_ADDRESS", "only 127.0.0.1 is allowed")
        self.dispatcher = dispatcher
        self._stop_event = threading.Event()
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
                response = self.dispatcher.dispatch(first)
                if first.get("command") == "hello" and response.get("ok"):
                    break
                connection.send(response)
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
                response = self.dispatcher.dispatch(request)
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
        while not self._stop_event.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                raise
            threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
        self._listener.close()

    def request_shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._listener.close()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Workpiece orientation loopback service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37651)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise SystemExit("INVALID_BIND_ADDRESS: only 127.0.0.1 is allowed")
    _prepare_windows_torch_dll_path()
    classifier = OrientationClassifier.load(args.project_root, args.model_dir)
    library = WorkpieceLibrary(args.library_dir)
    for record, cache in library.recover(classifier.build_template_cache):
        classifier.set_template_cache(record.id, cache)
    try:
        server = OrientationTcpServer(OrientationCommandDispatcher(classifier, library), host=args.host, port=args.port)
    except ServiceStartupError as exc:
        raise SystemExit(f"{exc.code}: {exc.message}") from exc
    server.serve_forever()


if __name__ == "__main__":
    main()
