import json
from dataclasses import replace
import logging
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import cv2
import numpy as np
import src.orientation_tcp_service as service_module

from src.orientation_tcp_service import (
    OrientationCommandDispatcher,
    OrientationTcpServer,
    ServiceRuntime,
    ServiceStartupError,
    configure_diagnostic_logging,
)
from src.orientation_classifier import OrientationClassifierError, PropagationModelError
from src.workpiece_catalog import WorkpieceCatalog
from src.workpiece_library import WorkpieceLibrary
from src.geometry_mask_profiles import (
    DuplicateLogicalRuleError,
    FittedGeometryMissingError,
    GeometryCacheRevisionMismatchError,
    GeometryContextMismatchError,
    GeometryProfileMigrationConflictError,
    MissingDirectionCalibrationError,
    StaleGeometryProfileError,
)


class FakeLibrary:
    def __init__(self):
        self.register_calls = []
        self.recycled = []

    def list_workpieces(self):
        return [] if self.recycled else [{"id": "m7", "name": "M7"}]

    def get(self, workpiece_id):
        if workpiece_id != "m7" or self.recycled:
            raise KeyError(workpiece_id)
        front = (Path("fake-front-00.png").resolve(),)
        back = tuple(Path(f"fake-back-{index:02d}.png").resolve() for index in range(12))
        return SimpleNamespace(
            id="m7",
            name="M7",
            root=Path.cwd().resolve(),
            front_images=front,
            back_images=back,
            revision=1,
            state="active",
        )

    def get_workpiece_metadata(self, workpiece_id):
        self.get(workpiece_id)
        return {"updated_at": "2026-08-25T00:00:00+00:00"}

    def get_template_inventory(self, workpiece_id):
        record = self.get(workpiece_id)
        inventory = [{
            "template_id": "front:00.png",
            "direction": "front",
            "filename": "fake-front-00.png",
            "source": "initial_registration",
            "added_at": "2026-08-25T00:00:00+00:00",
            "preview_path": str(record.front_images[0]),
            "readable": False,
        }]
        inventory.extend({
            "template_id": f"back:{index:02d}.png",
            "direction": "back",
            "filename": f"fake-back-{index:02d}.png",
            "source": "initial_registration",
            "added_at": "2026-08-25T00:00:00+00:00",
            "preview_path": str(record.back_images[index]),
            "readable": False,
        } for index in range(12))
        return inventory

    def list_recycled(self):
        return list(self.recycled)

    def recycle(self, workpiece_id):
        value = {"id": workpiece_id, "name": "M7", "revision": 2}
        self.recycled.append(value)
        return SimpleNamespace(**value)

    def restore(self, workpiece_id):
        value = self.recycled.pop(0)
        return SimpleNamespace(id=value["id"], name=value["name"], revision=3,
                               front_images=(), back_images=())

    def purge(self, workpiece_id):
        self.recycled = [item for item in self.recycled if item["id"] != workpiece_id]
        return {"id": workpiece_id}

    def register(self, name, front_images, back_images, replace, build_cache, *, progress_callback=None):
        self.register_calls.append((name, front_images, back_images, replace))
        if progress_callback is not None:
            progress_callback({"phase": "features", "completed": 1, "total": len(front_images) + len(back_images)})
            progress_callback({"phase": "fast_originals", "completed": 1, "total": 3})
            progress_callback({"phase": "fast_originals", "completed": 3, "total": 3})
            progress_callback({"phase": "fast_augmentation", "completed": 11, "total": 33})
            progress_callback({"phase": "fast_augmentation", "completed": 33, "total": 33})
            progress_callback({"phase": "fast_ridge", "completed": 0, "total": 1})
            progress_callback({"phase": "fast_ridge", "completed": 1, "total": 1})
        return (
            SimpleNamespace(
                id="m7",
                name=name,
                front_images=tuple(front_images),
                back_images=tuple(back_images),
                revision=1,
            ),
            {"fake": "cache"},
        )


class FakeClassifier:
    def __init__(self):
        self.predict_calls = []
        self.predict_error = None

    def set_template_cache(self, workpiece_id, cache):
        assert workpiece_id == "m7"
        self.active_workpiece_id = workpiece_id

    def remove_template_cache(self, workpiece_id):
        assert workpiece_id == "m7"

    def build_template_cache(self, front_images, back_images, progress_callback=None):
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

    def predict_with_cache(self, cache, image_path, *, library_revision=None):
        result = self.predict(self.active_workpiece_id, image_path)
        if library_revision is not None:
            result["library_revision"] = library_revision
        return result


class FakeEvolution:
    def __init__(self):
        self.jobs = []
        self.snapshot = {
            "workpiece_id": "m7",
            "revision": 3,
            "annotation_revision": 2,
            "active_annotation_revision": 1,
            "templates": [],
            "groups": [],
        }
        self.delete_calls = []
        self.enable_calls = []
        self.save_calls = []
        self.propagation_error = None

    def submit_confirmation(self, workpiece_id, orientation, image_path, *, operation_id):
        job = {"job_id": operation_id, "workpiece_id": workpiece_id, "orientation": orientation, "state": "queued"}
        self.jobs.append(job)
        return job

    def list_jobs(self):
        return list(self.jobs)

    def action(self, job_id, action):
        for job in self.jobs:
            if job["job_id"] == job_id:
                job["state"] = action
                return job
        raise KeyError(job_id)

    def get_annotations(self, workpiece_id):
        return self.snapshot

    def save_annotations(self, workpiece_id, groups, *, expected_revision=None, operation_id,
                         progress_callback=None):
        self.save_calls.append((workpiece_id, groups, expected_revision, operation_id))
        if self.propagation_error is not None:
            raise self.propagation_error
        if progress_callback is not None:
            progress_callback({"phase": "propagating_annotations", "completed": 0, "total": 2})
            progress_callback({"phase": "propagating_annotations", "completed": 2, "total": 2})
        return self.snapshot

    def set_group_enabled(self, workpiece_id, group_id, enabled, *, expected_revision, operation_id):
        self.enable_calls.append((workpiece_id, group_id, enabled, expected_revision, operation_id))
        return self.snapshot

    def delete_group(self, workpiece_id, group_id, *, expected_revision, operation_id):
        self.delete_calls.append((workpiece_id, group_id, expected_revision, operation_id))
        return self.snapshot


class FakeGeometryProfiles:
    def __init__(self):
        self.profile = {
            "workpiece_id": "m7", "library_revision": 1, "draft_revision": 0,
            "active_revision": None, "previous_active_revision": None,
            "draft": {"schema_version": 1, "directions": {"front": {"anchor": None, "rules": []},
                                                              "back": {"anchor": None, "rules": []}}},
            "active": None,
        }
        self.jobs = {}
        self.preview_calls = []
        self.last_resolution = None
        self.publish_error = None

    def preview_rule(self, workpiece_id, *, expected_library_revision, rule_id, direction, template_id,
                     seed_shape, mode, margin_ratio, anchor_candidate_index=None,
                     rule_candidate_index=None):
        self.preview_calls.append({
            "workpiece_id": workpiece_id,
            "expected_library_revision": expected_library_revision,
            "rule_id": rule_id,
            "direction": direction,
            "template_id": template_id,
            "seed_shape": seed_shape,
            "mode": mode,
            "margin_ratio": margin_ratio,
            "anchor_candidate_index": anchor_candidate_index,
            "rule_candidate_index": rule_candidate_index,
        })
        return {
            "rule_id": rule_id,
            "template_id": template_id,
            "direction": direction,
            "margin_ratio": margin_ratio,
            "rule_fit": {"effective_shape": {"shape": "ellipse", "cx": 1.0, "cy": 1.0,
                                                "rx": 1.0, "ry": 1.0, "angle_deg": 0.0}},
            "profile_patch": {
                "reference_template": {"direction": direction},
                "margin_ratio": margin_ratio,
                "margin_semantics": "signed_boundary_v2",
            },
        }

    def resolve_migration(self, workpiece_id, conflict_id, resolution, *,
                          expected_library_revision, expected_draft_revision, operation_id):
        self.last_resolution = {
            "workpiece_id": workpiece_id,
            "conflict_id": conflict_id,
            "resolution": resolution,
            "expected_library_revision": expected_library_revision,
            "expected_draft_revision": expected_draft_revision,
            "operation_id": operation_id,
        }
        return self.snapshot(workpiece_id)

    def snapshot(self, workpiece_id):
        return dict(self.profile)

    def save_draft(self, workpiece_id, draft, *, expected_library_revision, expected_draft_revision, operation_id):
        if expected_draft_revision != self.profile["draft_revision"]:
            raise StaleGeometryProfileError("stale draft")
        self.profile["draft_revision"] += 1
        self.profile["draft"] = draft
        return self.snapshot(workpiece_id)

    def start_validation(self, workpiece_id, *, expected_library_revision, expected_draft_revision, operation_id):
        job = {"job_id": operation_id, "state": "queued", "base_library_revision": expected_library_revision,
               "base_draft_revision": expected_draft_revision}
        self.jobs[operation_id] = job
        return dict(job)

    def get_job(self, job_id):
        return dict(self.jobs[job_id])

    def action(self, job_id, action):
        self.jobs[job_id]["state"] = "cancelled"
        return dict(self.jobs[job_id])

    def publish(self, workpiece_id, job_id, *, expected_library_revision, expected_draft_revision,
                operation_id, override_reason=""):
        if self.publish_error is not None:
            raise self.publish_error
        if expected_draft_revision != 99:
            raise StaleGeometryProfileError("stale publish")
        self.profile["active_revision"] = expected_draft_revision
        return self.snapshot(workpiece_id)

    def rollback(self, workpiece_id, *, expected_library_revision, operation_id):
        return self.snapshot(workpiece_id)

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
        snapshot = dispatcher.runtime.snapshot()
        dispatcher.runtime._snapshot = replace(
            snapshot, evolution=FakeEvolution(), geometry_profiles=FakeGeometryProfiles()
        )
        self.dispatcher = dispatcher
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


def test_workpiece_list_and_details_are_additive(client):
    listing = client.request("list_workpieces")
    item = listing["workpieces"][0]

    assert item["id"] == "m7"
    assert item["name"] == "M7"
    assert item["template_counts"] == {"front": 1, "back": 12}
    details = client.request("get_workpiece_details", workpiece_id="m7")
    assert details["ok"] is True
    assert len(details["workpiece"]["templates"]) == 13


def test_unknown_workpiece_details_return_stable_code(client):
    response = client.request("get_workpiece_details", workpiece_id="missing")

    assert response["error"]["code"] == "WORKPIECE_NOT_FOUND"


def test_malformed_inventory_is_not_reported_as_missing_workpiece(tmp_path):
    class AnyIdClassifier(FakeClassifier):
        def set_template_cache(self, workpiece_id, cache):
            self.active_workpiece_id = workpiece_id

    classifier = AnyIdClassifier()
    library = WorkpieceLibrary(tmp_path / "malformed-inventory-library")
    catalog = WorkpieceCatalog(library, classifier)
    front_path = tmp_path / "malformed-front.png"
    back_path = tmp_path / "malformed-back.png"
    assert cv2.imwrite(str(front_path), np.full((8, 8, 3), 10, dtype=np.uint8))
    assert cv2.imwrite(str(back_path), np.full((8, 8, 3), 20, dtype=np.uint8))
    record, _ = catalog.register("M-malformed", [front_path], [back_path], False)
    manifest_path = record.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["template_inventory"][0].pop("source")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dispatcher = OrientationCommandDispatcher(FakeClassifier(), FakeLibrary())
    snapshot = dispatcher.runtime.snapshot()
    dispatcher.runtime._snapshot = replace(
        snapshot,
        classifier=classifier,
        library=library,
        catalog=catalog,
    )

    response = dispatcher.dispatch({
        "version": 1,
        "request_id": "malformed-inventory",
        "command": "get_workpiece_details",
        "workpiece_id": record.id,
    })

    assert response["error"]["code"] == "INVALID_TEMPLATE_SET"


def test_unstable_workpiece_list_returns_stale_revision_error(tmp_path):
    class AnyIdClassifier(FakeClassifier):
        def set_template_cache(self, workpiece_id, cache):
            self.active_workpiece_id = workpiece_id

    class MutatingProfiles:
        def __init__(self):
            self.catalog = None
            self.calls = 0

        def snapshot(self, workpiece_id):
            self.calls += 1
            current = self.catalog.capture_snapshot(workpiece_id)
            self.catalog.publish_geometry_profile(
                workpiece_id,
                current.cache,
                profile_revision=self.calls,
                previous_profile_revision=None,
                expected_revision=current.record.revision,
                operation_id=f"protocol-summary-churn-{self.calls}",
            )
            return {"profile_status": "ok", "active": {"rules": []}}

    classifier = AnyIdClassifier()
    profiles = MutatingProfiles()
    library = WorkpieceLibrary(tmp_path / "protocol-summary-churn-library")
    catalog = WorkpieceCatalog(library, classifier, profiles)
    profiles.catalog = catalog
    front_path = tmp_path / "protocol-summary-churn-front.png"
    back_path = tmp_path / "protocol-summary-churn-back.png"
    assert cv2.imwrite(str(front_path), np.full((8, 8, 3), 10, dtype=np.uint8))
    assert cv2.imwrite(str(back_path), np.full((8, 8, 3), 20, dtype=np.uint8))
    catalog.register("M-protocol-summary-churn", [front_path], [back_path], False)
    dispatcher = OrientationCommandDispatcher(FakeClassifier(), FakeLibrary())
    snapshot = dispatcher.runtime.snapshot()
    dispatcher.runtime._snapshot = replace(
        snapshot,
        classifier=classifier,
        library=library,
        catalog=catalog,
        geometry_profiles=profiles,
    )

    response = dispatcher.dispatch({
        "version": 1,
        "request_id": "protocol-summary-churn",
        "command": "list_workpieces",
    })

    assert response["ok"] is False
    assert response["error"]["code"] == "STALE_WORKPIECE_REVISION"
    assert "workpieces" not in response


@pytest.mark.parametrize("workpiece_id", [None, "", "   ", 7])
def test_workpiece_details_reject_invalid_ids(client, workpiece_id):
    response = client.request("get_workpiece_details", workpiece_id=workpiece_id)

    assert response["error"]["code"] == "INVALID_REQUEST"


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


def test_silent_handshake_times_out_and_releases_server_slot():
    dispatcher = OrientationCommandDispatcher(FakeClassifier(), FakeLibrary())
    server = OrientationTcpServer(
        dispatcher,
        host="127.0.0.1",
        port=0,
        handshake_timeout_seconds=0.1,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    silent = socket.create_connection(server.address, timeout=1)
    try:
        time.sleep(0.25)
        replacement = TcpTestClient(server.address)
        try:
            response = replacement.request("hello")
            assert response["ok"] is True
            assert response["ready"] is True
        finally:
            replacement.close()
    finally:
        silent.close()
        server.request_shutdown()
        thread.join(timeout=2)


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


def test_register_rejects_non_string_template_item(client, running_server):
    assert client.request("hello")["ok"] is True

    response = client.request(
        "register",
        name="M7",
        front_images=["front.png", 7],
        back_images=["back.png"],
    )

    assert response["error"]["code"] == "INVALID_REQUEST"
    assert running_server.library.register_calls == []


def test_register_without_progress_flag_returns_one_final_response(client):
    assert client.request("hello")["ok"] is True

    response = client.request(
        "register",
        name="M7",
        front_images=["front.png"],
        back_images=["back.png"],
    )

    assert response["ok"] is True
    assert "event" not in response
    assert response["template_counts"] == {"front": 1, "back": 1}


def test_register_with_progress_flag_streams_before_final_response(client):
    assert client.request("hello")["ok"] is True
    payload = {
        "version": 1,
        "request_id": "register-progress",
        "command": "register",
        "name": "M7",
        "front_images": ["front-1.png", "front-2.png"],
        "back_images": ["back-1.png"],
        "progress_events": True,
    }
    client.send_raw(json.dumps(payload).encode("utf-8") + b"\n")

    messages = []
    while True:
        messages.append(client.read())
        if messages[-1].get("ok") is True:
            break

    assert any(message.get("event") == "progress" for message in messages)
    assert messages[0]["request_id"] == "register-progress"
    assert messages[-1]["template_counts"] == {"front": 2, "back": 1}


def test_register_forwards_fast_original_augmentation_and_ridge_progress(client):
    assert client.request("hello")["ok"] is True
    client.send_raw(json.dumps({
        "version": 1,
        "request_id": "fast-progress",
        "command": "register",
        "name": "M7",
        "front_images": ["front-1.png", "front-2.png"],
        "back_images": ["back-1.png"],
        "progress_events": True,
    }).encode("utf-8") + b"\n")

    progress = []
    while True:
        message = client.read()
        if message.get("event") == "progress" and message["progress"]["phase"].startswith("fast_"):
            progress.append(message["progress"])
        if message.get("ok") is True:
            break

    assert progress == [
        {"phase": "fast_originals", "completed": 1, "total": 3},
        {"phase": "fast_originals", "completed": 3, "total": 3},
        {"phase": "fast_augmentation", "completed": 11, "total": 33},
        {"phase": "fast_augmentation", "completed": 33, "total": 33},
        {"phase": "fast_ridge", "completed": 0, "total": 1},
        {"phase": "fast_ridge", "completed": 1, "total": 1},
    ]


def _dispatcher_with_catalog(catalog):
    dispatcher = OrientationCommandDispatcher(FakeClassifier(), FakeLibrary())
    dispatcher.runtime._snapshot = replace(dispatcher.runtime.snapshot(), catalog=catalog)
    return dispatcher


def _predict_request(request_id="fast-predict"):
    return {
        "version": 1,
        "request_id": request_id,
        "command": "predict",
        "workpiece_id": "m7",
        "image_path": "query.png",
    }


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        ("FAST_CACHE_NOT_READY: cache is queued", "FAST_CACHE_NOT_READY"),
        ("FAST_CACHE_REVISION_MISMATCH: library revision differs", "FAST_CACHE_REVISION_MISMATCH"),
        ("FAST_FEATURE_INVALID: embedding must be finite", "FAST_FEATURE_INVALID"),
    ],
)
def test_fast_hard_failures_map_to_stable_error_without_label(message, expected_code):
    class FailingCatalog:
        def predict(self, workpiece_id, image_path):
            raise OrientationClassifierError(message)

        def fast_cache_status(self, workpiece_id):
            return {"state": "not_ready", "completed": 0, "total": 0, "elapsed_ms": 0.0, "error": None}

    response = _dispatcher_with_catalog(FailingCatalog()).dispatch(_predict_request(expected_code))

    assert response["ok"] is False
    assert response["error"]["code"] == expected_code
    assert "label" not in response


def test_fast_cache_build_failure_maps_to_stable_error_code():
    class FailedCatalog:
        def predict(self, workpiece_id, image_path):
            raise OrientationClassifierError("FAST_CACHE_NOT_READY: runtime is unavailable")

        def fast_cache_status(self, workpiece_id):
            return {
                "state": "failed",
                "completed": 7,
                "total": 10,
                "elapsed_ms": 12.5,
                "error": "GPU allocation failed; retry registration",
            }

    response = _dispatcher_with_catalog(FailedCatalog()).dispatch(_predict_request("failed-job"))

    assert response["ok"] is False
    assert response["error"]["code"] == "FAST_CACHE_BUILD_FAILED"
    assert "GPU allocation failed; retry registration" in response["error"]["message"]
    assert "label" not in response


def test_fast_cache_capability_unavailable_is_not_reported_as_build_failure():
    class CapabilityCatalog:
        def predict(self, workpiece_id, image_path):
            raise OrientationClassifierError("FAST_CACHE_NOT_READY: runtime is unavailable")

        def fast_cache_status(self, workpiece_id):
            return {
                "state": "not_ready",
                "completed": 0,
                "total": 0,
                "elapsed_ms": 0.0,
                "error": "FAST_CACHE_CAPABILITY_UNAVAILABLE: persistence is unavailable",
            }

    response = _dispatcher_with_catalog(CapabilityCatalog()).dispatch(_predict_request("capability"))

    assert response["error"]["code"] == "FAST_CACHE_CAPABILITY_UNAVAILABLE"


@pytest.mark.parametrize(
    "reason_code",
    ["FAST_GEOMETRY_LOW_CONFIDENCE", "FAST_CLASSIFIER_LOW_MARGIN", "FAST_MODE_NOT_VALIDATED"],
)
def test_fast_review_conditions_remain_predictions_with_labels(reason_code):
    class ReviewCatalog:
        def predict(self, workpiece_id, image_path):
            return {
                "label": "back",
                "needs_review": True,
                "review_reason_codes": [reason_code],
            }

    response = _dispatcher_with_catalog(ReviewCatalog()).dispatch(_predict_request(reason_code))

    assert response["ok"] is True
    assert response["label"] == "back"
    assert response["needs_review"] is True
    assert response["review_reason_codes"] == [reason_code]


def test_register_returns_actual_fast_cache_state_and_training_metadata():
    fast_runtime = SimpleNamespace(
        cache_revision="fast-revision-9",
        template_counts={"front": 5, "back": 12},
        training_summary={"validation_status": "validated", "original_samples": 17},
    )

    class RegisterCatalog:
        def register(self, name, front_images, back_images, replace, *, progress_callback=None):
            return (
                SimpleNamespace(id="m9", name=name, front_images=tuple(range(5)), back_images=tuple(range(12))),
                SimpleNamespace(fast_runtime=fast_runtime),
            )

        def fast_cache_status(self, workpiece_id):
            return {"state": "ready", "completed": 17, "total": 17, "elapsed_ms": 8.0, "error": None}

    response = _dispatcher_with_catalog(RegisterCatalog()).dispatch({
        "version": 1,
        "request_id": "register-fast-state",
        "command": "register",
        "name": "M9",
        "front_images": [f"front-{index}.png" for index in range(5)],
        "back_images": [f"back-{index}.png" for index in range(12)],
    })

    assert response["ok"] is True
    assert response["template_counts"] == {"front": 5, "back": 12}
    assert response["fast_cache_state"] == "ready"
    assert response["fast_cache_revision"] == "fast-revision-9"
    assert response["training_summary"] == {
        "validation_status": "validated",
        "original_samples": 17,
    }


def test_list_workpieces_preserves_complete_fast_cache_object():
    fast_cache = {
        "state": "running",
        "completed": 19,
        "total": 44,
        "elapsed_ms": 123.5,
        "error": None,
    }

    class ListingCatalog:
        def list_workpiece_summaries(self):
            return [{"id": "m7", "name": "M7", "fast_cache": fast_cache}]

    response = _dispatcher_with_catalog(ListingCatalog()).dispatch({
        "version": 1,
        "request_id": "list-fast-state",
        "command": "list_workpieces",
    })

    assert response["ok"] is True
    assert response["workpieces"][0]["fast_cache"] == fast_cache


def test_shutdown_stops_fast_cache_jobs_once_before_listener_close():
    events = []

    class ShutdownCatalog:
        geometry_profiles = None

        def shutdown(self):
            events.append("fast_jobs")

    class RecordingListener:
        def close(self):
            events.append("listener")

    runtime = ServiceRuntime()
    runtime._snapshot = replace(runtime.snapshot(), status="ready", catalog=ShutdownCatalog())
    server = OrientationTcpServer(OrientationCommandDispatcher(runtime), host="127.0.0.1", port=0)
    server._listener.close()
    server._listener = RecordingListener()

    server.request_shutdown()
    server.request_shutdown()

    assert events == ["fast_jobs", "listener"]


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


def test_recycle_command_is_idempotent(client, running_server):
    assert client.request("hello")["ok"] is True

    first = client.request(
        "recycle_workpiece", request_id="recycle-request", operation_id="delete-1", workpiece_id="m7"
    )
    assert first["ok"] is True
    assert first["workpiece"]["id"] == "m7"
    replay = client.request(
        "recycle_workpiece", request_id="recycle-retry", operation_id="delete-1", workpiece_id="m7"
    )
    assert replay["ok"] is True
    assert replay["workpiece"] == first["workpiece"]


def test_list_recycled_command_is_additive(client):
    assert client.request("hello")["ok"] is True
    response = client.request("list_recycled_workpieces")
    assert response["ok"] is True
    assert isinstance(response["workpieces"], list)


def test_confirmation_and_job_commands_are_additive(client):
    assert client.request("hello")["ok"] is True
    submitted = client.request(
        "submit_confirmation", operation_id="confirm-1", workpiece_id="m7", orientation="front", image_path="q.png"
    )
    assert submitted["ok"] is True
    jobs = client.request("list_evolution_jobs")
    assert jobs["ok"] is True
    assert jobs["jobs"][0]["job_id"] == "confirm-1"
    action = client.request("evolution_job_action", job_id="confirm-1", action="cancel")
    assert action["ok"] is True


def test_annotation_snapshot_and_group_mutations_use_revision_and_operation_ids(client, running_server):
    assert client.request("hello")["ok"] is True

    snapshot = client.request("get_workpiece_annotations", workpiece_id="m7")
    assert snapshot["ok"] is True
    assert snapshot["annotations"]["active_annotation_revision"] == 1

    saved = client.request(
        "save_workpiece_annotations",
        workpiece_id="m7",
        groups=[{"group_id": "glare", "annotations": []}],
        base_revision=2,
        operation_id="annotation-save-1",
    )
    assert saved["ok"] is True
    assert running_server.dispatcher.runtime.snapshot().evolution.save_calls[-1] == (
        "m7", [{"group_id": "glare", "annotations": []}], 2, "annotation-save-1"
    )

    enabled = client.request(
        "set_workpiece_annotation_group_enabled",
        workpiece_id="m7", group_id="glare", enabled=False,
        base_revision=2, operation_id="annotation-enable-1",
    )
    assert enabled["ok"] is True
    assert running_server.dispatcher.runtime.snapshot().evolution.enable_calls[-1] == (
        "m7", "glare", False, 2, "annotation-enable-1"
    )

    deleted = client.request(
        "delete_workpiece_annotation_group",
        workpiece_id="m7", group_id="glare", base_revision=2,
        operation_id="annotation-delete-1",
    )
    assert deleted["ok"] is True
    assert running_server.dispatcher.runtime.snapshot().evolution.delete_calls[-1] == (
        "m7", "glare", 2, "annotation-delete-1"
    )


def test_annotation_save_progress_is_streamed_when_requested(client):
    assert client.request("hello")["ok"] is True
    client.send_raw(json.dumps({
        "version": 1,
        "request_id": "annotation-progress",
        "command": "save_workpiece_annotations",
        "workpiece_id": "m7",
        "groups": [{"group_id": "glare", "annotations": []}],
        "base_revision": 2,
        "operation_id": "annotation-progress-1",
        "progress_events": True,
    }).encode("utf-8") + b"\n")

    first = client.read()
    second = client.read()
    final = client.read()
    assert first["event"] == "progress"
    assert first["command"] == "save_workpiece_annotations"
    assert second["progress"]["completed"] == 2
    assert final["ok"] is True


def test_propagation_model_failure_returns_stable_model_error(client, running_server):
    assert client.request("hello")["ok"] is True
    running_server.dispatcher.runtime.snapshot().evolution.propagation_error = PropagationModelError(
        "递推失败次数过多，已中止本次保存，请检查后端日志"
    )

    response = client.request(
        "save_workpiece_annotations",
        workpiece_id="m7",
        groups=[{"group_id": "glare", "annotations": []}],
        base_revision=2,
        operation_id="annotation-model-failure-1",
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "MODEL_ERROR"
    assert "递推失败次数过多" in response["error"]["message"]


def test_diagnostic_logging_persists_propagation_context(tmp_path):
    handler = configure_diagnostic_logging(tmp_path)
    try:
        logging.getLogger("src.template_evolution").warning(
            "annotation propagation pair failed operation_id=%s workpiece_id=%s group_id=%s source=%s target=%s",
            "op-1", "m7", "glare", "front:00.png", "front:02.png",
        )
        handler.flush()
        log_path = tmp_path / "diagnostics" / "orientation-service.log"
        content = log_path.read_text(encoding="utf-8")
        assert "operation_id=op-1" in content
        assert "workpiece_id=m7" in content
        assert "target=front:02.png" in content
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def test_annotation_commands_reject_missing_or_wrong_revision_fields(client):
    assert client.request("hello")["ok"] is True
    missing = client.request(
        "delete_workpiece_annotation_group", workpiece_id="m7", group_id="glare",
        operation_id="annotation-delete-missing-revision",
    )
    assert missing["error"]["code"] == "INVALID_REQUEST"
    wrong_type = client.request(
        "set_workpiece_annotation_group_enabled", workpiece_id="m7", group_id="glare",
        enabled="false", base_revision=True, operation_id="annotation-enable-wrong-type",
    )
    assert wrong_type["error"]["code"] == "INVALID_REQUEST"


def test_geometry_profile_commands_round_trip_and_return_job_without_streaming(client):
    profile = client.request("get_geometry_mask_profile", workpiece_id="m7")
    assert profile["ok"] is True
    saved = client.request(
        "save_geometry_mask_draft",
        workpiece_id="m7",
        base_library_revision=1,
        base_draft_revision=0,
        draft=profile["profile"]["draft"],
        operation_id="geometry-draft-1",
    )
    assert saved["ok"] is True
    job = client.request(
        "validate_geometry_mask_draft",
        workpiece_id="m7",
        base_library_revision=1,
        base_draft_revision=1,
        operation_id="geometry-validate-1",
    )
    assert job["ok"] is True
    assert job["job"]["state"] == "queued"
    polled = client.request("get_geometry_mask_validation_job", job_id="geometry-validate-1")
    assert polled["ok"] is True
    assert polled["job"]["job_id"] == "geometry-validate-1"
    cancelled = client.request(
        "geometry_mask_validation_job_action", job_id="geometry-validate-1", action="cancel"
    )
    assert cancelled["job"]["state"] == "cancelled"


def test_preview_geometry_mask_rule_dispatches_without_operation_id(client, running_server):
    response = client.request(
        "preview_geometry_mask_rule",
        workpiece_id="m7",
        base_library_revision=4,
        rule_id="glare",
        direction="front",
        template_id="front:00.png",
        seed_shape={"shape": "circle", "cx": 120.0, "cy": 130.0, "r": 44.0},
        mode="inside",
        margin_ratio=-0.02,
    )

    assert response["ok"] is True
    assert response["preview"]["template_id"] == "front:00.png"
    assert response["preview"]["rule_id"] == "glare"
    assert response["preview"]["direction"] == "front"
    assert response["preview"]["margin_ratio"] == pytest.approx(-0.02)
    assert "effective_shape" in response["preview"]["rule_fit"]
    assert running_server.dispatcher.runtime.snapshot().geometry_profiles.preview_calls


def test_preview_geometry_rule_requires_rule_id(client):
    response = client.request(
        "preview_geometry_mask_rule",
        workpiece_id="m7",
        base_library_revision=1,
        direction="front",
        template_id="front:00.png",
        seed_shape={"shape": "circle", "cx": 100.0, "cy": 100.0, "r": 60.0},
        mode="inside",
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "NO_SELECTED_RULE"


def test_resolve_geometry_migration_forwards_revisioned_resolution(client, running_server):
    response = client.request(
        "resolve_geometry_mask_migration",
        workpiece_id="m7",
        base_library_revision=1,
        base_draft_revision=3,
        operation_id="resolve-op-1",
        conflict_id="duplicate-back-glare",
        resolution={"action": "keep_only", "survivor_rule_id": "back-glare-a"},
    )

    assert response["ok"] is True
    profiles = running_server.dispatcher.runtime.snapshot().geometry_profiles
    assert profiles.last_resolution["conflict_id"] == "duplicate-back-glare"
    assert profiles.last_resolution["expected_draft_revision"] == 3


@pytest.mark.parametrize(
    "fields",
    [
        {"base_library_revision": True},
        {"base_library_revision": 4, "direction": "side"},
        {"base_library_revision": 4, "direction": "front", "template_id": ""},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "triangle"}},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "circle"}, "mode": "sideways"},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "circle"}, "mode": "inside", "margin_ratio": 1.0},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "circle"}, "mode": "inside", "margin_ratio": -0.95},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "circle"}, "mode": "inside", "margin_ratio": 0.95},
        {"base_library_revision": 4, "direction": "front", "template_id": "front:00.png",
         "seed_shape": {"shape": "circle"}, "mode": "inside", "rule_candidate_index": "1"},
    ],
)
def test_preview_geometry_mask_rule_rejects_invalid_request(client, fields):
    defaults = {
        "workpiece_id": "m7",
        "base_library_revision": 4,
        "rule_id": "glare",
        "direction": "front",
        "template_id": "front:00.png",
        "seed_shape": {"shape": "circle", "cx": 120.0, "cy": 130.0, "r": 44.0},
        "mode": "inside",
        "margin_ratio": 0.02,
    }
    defaults.update(fields)
    response = client.request("preview_geometry_mask_rule", **defaults)
    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_REQUEST"


def test_geometry_profile_maps_stale_publish_to_stable_error_code(client):
    response = client.request(
        "publish_geometry_mask_profile",
        workpiece_id="m7",
        job_id="geometry-validate-1",
        base_library_revision=1,
        base_draft_revision=1,
        operation_id="geometry-publish-stale",
    )
    assert response["ok"] is False
    assert response["error"]["code"] == "STALE_GEOMETRY_PROFILE"


@pytest.mark.parametrize(
    ("error_type", "code"),
    [
        (MissingDirectionCalibrationError, "MISSING_DIRECTION_CALIBRATION"),
        (FittedGeometryMissingError, "FITTED_GEOMETRY_MISSING"),
        (GeometryProfileMigrationConflictError, "MIGRATION_CONFLICT"),
        (DuplicateLogicalRuleError, "DUPLICATE_LOGICAL_RULE"),
        (GeometryContextMismatchError, "GEOMETRY_CONTEXT_MISMATCH"),
        (GeometryCacheRevisionMismatchError, "PROFILE_CACHE_REVISION_MISMATCH"),
    ],
)
def test_geometry_profile_maps_typed_errors_to_stable_codes(
    client, running_server, error_type, code,
):
    profiles = running_server.dispatcher.runtime.snapshot().geometry_profiles
    profiles.publish_error = error_type("typed geometry failure")

    response = client.request(
        "publish_geometry_mask_profile",
        workpiece_id="m7",
        job_id="geometry-validate-typed",
        base_library_revision=1,
        base_draft_revision=99,
        operation_id=f"publish-{code}",
    )

    assert response["ok"] is False
    assert response["error"]["code"] == code


def test_listener_reports_loading_then_accepts_ready_handshake():
    runtime = ServiceRuntime()
    dispatcher = OrientationCommandDispatcher(runtime)
    server = OrientationTcpServer(dispatcher, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = TcpTestClient(server.address)
    try:
        loading = client.request("hello")
        assert loading["ok"] is True
        assert loading["ready"] is False
        assert loading["status"] == "loading"
        assert client.request("list_workpieces")["error"]["code"] == "MODEL_LOADING"

        runtime.set_ready(FakeClassifier(), FakeLibrary())
        ready = client.request("hello")
        assert ready["ok"] is True
        assert ready["ready"] is True
    finally:
        client.close()
        server.request_shutdown()
        thread.join(timeout=2)


def test_local_search_mode_argument_defaults_to_adaptive(tmp_path):
    parser = service_module._build_argument_parser()
    args = parser.parse_args([
        "--project-root", str(tmp_path),
        "--model-dir", str(tmp_path / "models"),
        "--library-dir", str(tmp_path / "library"),
    ])

    assert args.local_search_mode == "adaptive"


def test_local_search_mode_argument_rejects_unknown_value(tmp_path):
    parser = service_module._build_argument_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--project-root", str(tmp_path),
            "--model-dir", str(tmp_path / "models"),
            "--library-dir", str(tmp_path / "library"),
            "--local-search-mode", "fast",
        ])


def test_runtime_loader_starts_workers_only_after_recovery_and_ready_publication(monkeypatch, tmp_path):
    runtime = ServiceRuntime()
    events = []

    class LoadedClassifier:
        geometry_calibrator = None

        @classmethod
        def load(cls, project_root, model_dir, *, local_search_mode, inference_mode):
            events.append(f"model-loaded:{local_search_mode}:{inference_mode}")
            return cls()

    class LoadedLibrary:
        def __init__(self, library_dir):
            self.library_dir = Path(library_dir)

    class LoadedCatalog:
        def __init__(self, library, classifier):
            self.library = library
            self.classifier = classifier
            self.geometry_profiles = None

        def set_geometry_profiles(self, profiles):
            self.geometry_profiles = profiles

        def recover(self):
            assert runtime.snapshot().status == "loading"
            events.append("recovered")

    class LoadedProfiles:
        def __init__(self, catalog, calibrator, *, storage_dir, start_worker=True):
            assert start_worker is False
            events.append("profiles-created")

        def start(self):
            assert runtime.snapshot().status == "ready"
            assert "recovered" in events
            events.append("profiles-started")

    class LoadedEvolution:
        def __init__(self, catalog, storage_dir, *, geometry_profiles, start_worker=True):
            assert start_worker is False
            events.append("evolution-created")

        def start(self):
            assert runtime.snapshot().status == "ready"
            assert "recovered" in events
            events.append("evolution-started")

    monkeypatch.setattr(service_module, "OrientationClassifier", LoadedClassifier)
    monkeypatch.setattr(service_module, "WorkpieceLibrary", LoadedLibrary)
    monkeypatch.setattr(service_module, "WorkpieceCatalog", LoadedCatalog)
    monkeypatch.setattr(service_module, "GeometryMaskProfiles", LoadedProfiles)
    monkeypatch.setattr(service_module, "TemplateEvolution", LoadedEvolution)

    service_module._load_runtime(
        runtime,
        tmp_path,
        tmp_path / "models",
        tmp_path / "library",
        local_search_mode="exhaustive",
        inference_mode="fast_geometry",
    )

    assert runtime.snapshot().status == "ready"
    assert events == [
        "model-loaded:exhaustive:fast_geometry",
        "profiles-created",
        "recovered",
        "evolution-created",
        "profiles-started",
        "evolution-started",
    ]


def test_loading_runtime_accepts_shutdown_before_model_ready():
    runtime = ServiceRuntime()
    server = OrientationTcpServer(OrientationCommandDispatcher(runtime), host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = TcpTestClient(server.address)
    try:
        assert client.request("shutdown")["ok"] is True
        thread.join(timeout=2)
        assert not thread.is_alive()
    finally:
        client.close()


def test_script_entrypoint_resolves_src_package():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "src" / "orientation_tcp_service.py"

    result = subprocess.run(
        [sys.executable, "-u", str(script), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
