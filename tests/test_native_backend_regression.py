"""Cross-backend regression coverage for the opt-in native PP-ShiTu path.

The tests deliberately use deterministic, process-free doubles.  This keeps
the correctness gate runnable on machines that do not have the production
Paddle/OpenCV native bundle, while still exercising the real classifier,
FastRuntimeCache, and BatchInferencePool code paths.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.fast_geometry import FastGeometryProcessor
from src.fast_orientation import FastOrientationEngine
from src.model_fingerprint import model_directory_sha256
from src.native_pp_client import NativePPError
from src.native_pp_protocol import NativeHello, NativeResult
from src.orientation_classifier import OrientationClassifier, TemplateCache


MODEL_FINGERPRINT = "native-regression-model"


def _embedding_from_rgb(image: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector from an RGB image."""

    values = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) / 255.0
    # Keep a small positive offset so synthetic dark images also have a valid
    # non-zero vector.  The exact same function is used by both doubles.
    vector = np.asarray(
        [values[0] + 0.01, values[1] + 0.02, values[2] + 0.03],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    return vector / norm


class _DeterministicPythonPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        return [_embedding_from_rgb(image) for image in images]


class _DeterministicNativeClient:
    """A native-client-shaped double with the production result contract."""

    instances: list["_DeterministicNativeClient"] = []

    def __init__(self, executable, model_dir, *, threads=1, request_timeout_s=30.0, **_kwargs):
        self.executable = Path(executable)
        self.model_dir = Path(model_dir)
        self.threads = int(threads)
        self.request_timeout_s = float(request_timeout_s)
        self.hello: NativeHello | None = None
        self.closed = 0
        self.predict_calls = 0
        type(self).instances.append(self)

    @property
    def ready(self) -> bool:
        return self.hello is not None and self.closed == 0

    def start(self) -> NativeHello:
        self.hello = NativeHello(
            service_version="regression-native/1",
            model_sha256=model_directory_sha256(self.model_dir),
            feature_dim=3,
            max_batch=256,
            threads=self.threads,
        )
        return self.hello

    def predict(self, images) -> NativeResult:
        self.predict_calls += 1
        return NativeResult(
            embeddings=np.stack([_embedding_from_rgb(image) for image in images]).astype(
                np.float32
            ),
            timings_ms={
                "preprocess_ms": 0.0,
                "inference_ms": 0.0,
                "postprocess_ms": 0.0,
            },
        )

    def close(self) -> None:
        self.closed += 1


class _ExitNativeClient(_DeterministicNativeClient):
    """Simulate a poisoned native child after a successful HELLO."""

    def predict(self, _images) -> NativeResult:
        self.predict_calls += 1
        raise NativePPError(
            "NATIVE_PP_SERVICE_EXITED",
            "native service exited during batch prediction",
        )


class _PythonFallbackSentinel:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, _images):  # pragma: no cover - failure is the assertion
        self.calls += 1
        raise AssertionError("native mode must never call the Python predictor")


def _write_template(path: Path, *, side: str, index: int) -> Path:
    """Write a unique lossless BGR image for one template slot."""

    image = np.zeros((24, 24, 3), dtype=np.uint8)
    if side == "front":
        image[:, :, 2] = 210
        image[:, :, 0] = 20
    else:
        image[:, :, 0] = 210
        image[:, :, 2] = 20
    # A deterministic stripe makes every path's pixel digest unique without
    # changing the broad orientation appearance.
    image[index % image.shape[0] :, :, 1] = np.uint8(10 + (index % 25))
    assert cv2.imwrite(str(path), image)
    return path


def _write_query(path: Path, *, side: str = "front") -> Path:
    return _write_template(path, side=side, index=23)


def _make_engine(classifier: OrientationClassifier) -> FastOrientationEngine:
    # All regression profiles are intentionally unconfigured.  This exercises
    # the same fast path used by a freshly created library without requiring a
    # geometry calibrator or local feature model.
    return FastOrientationEngine(
        classifier._global_embeddings,
        FastGeometryProcessor(object()),
        image_reader=lambda path: cv2.imread(str(path), cv2.IMREAD_COLOR),
        deduplicate_identical_slots=False,
    )


def _make_python_classifier() -> tuple[OrientationClassifier, _DeterministicPythonPredictor]:
    predictor = _DeterministicPythonPredictor()
    classifier = OrientationClassifier(
        predictor,
        None,
        None,
        "cpu",
        compute_device="cpu",
        inference_mode="fast_geometry",
        model_fingerprint=MODEL_FINGERPRINT,
    )
    classifier.fast_engine = _make_engine(classifier)
    return classifier, predictor


def _make_native_classifier(
    model_dir: Path,
    executable: Path,
    *,
    client: _DeterministicNativeClient | None = None,
    fallback_sentinel: _PythonFallbackSentinel | None = None,
) -> OrientationClassifier:
    if client is None:
        client = _DeterministicNativeClient(executable, model_dir, threads=1)
        client.start()
    classifier = OrientationClassifier(
        fallback_sentinel,
        None,
        None,
        "cpu",
        compute_device="cpu",
        inference_mode="fast_geometry",
        pp_backend="native_cpp",
        native_pp_client=client,
        model_fingerprint=MODEL_FINGERPRINT,
    )
    classifier.native_pp_executable = executable
    classifier.native_model_dir = model_dir
    classifier.fast_engine = _make_engine(classifier)
    return classifier


def _decision_view(result: dict[str, object]) -> tuple[object, ...]:
    """Select parity fields whose values must not depend on backend."""

    return (
        result.get("label"),
        result.get("needs_review"),
        result.get("review_reason_codes"),
        result.get("geometry_status"),
        result.get("decision_source"),
    )


def run_backend_pair(
    tmp_path: Path,
    *,
    front_count: int,
    back_count: int,
) -> tuple[dict[str, object], dict[str, object], TemplateCache, OrientationClassifier, OrientationClassifier]:
    """Build one cache and run equivalent Python/native predictions."""

    assert front_count > 0 and back_count > 0
    model_dir = tmp_path / "native-model"
    model_dir.mkdir(exist_ok=True)
    executable = tmp_path / "ppshitu_rec_service.exe"
    executable.write_bytes(b"test executable")
    front = [
        _write_template(tmp_path / f"front-{index:02d}.png", side="front", index=index)
        for index in range(front_count)
    ]
    back = [
        _write_template(tmp_path / f"back-{index:02d}.png", side="back", index=index)
        for index in range(back_count)
    ]
    query = _write_query(tmp_path / "query.png")

    python_classifier, _python_predictor = _make_python_classifier()
    cache = python_classifier.build_template_cache(front, back, library_revision=7)
    native_client = _DeterministicNativeClient(executable, model_dir, threads=1)
    native_client.start()
    native_classifier = _make_native_classifier(
        model_dir,
        executable,
        client=native_client,
    )
    python_result = python_classifier.predict_fast_with_cache(
        cache, query, library_revision=7
    )
    native_result = native_classifier.predict_fast_with_cache(
        cache, query, library_revision=7
    )
    return python_result, native_result, cache, python_classifier, native_classifier


@pytest.mark.parametrize("front_count,back_count", [(1, 1), (5, 10), (10, 15), (35, 35)])
def test_unequal_template_counts_have_native_python_parity(
    tmp_path, front_count: int, back_count: int
):
    python_result, native_result, cache, python_classifier, native_classifier = run_backend_pair(
        tmp_path,
        front_count=front_count,
        back_count=back_count,
    )
    try:
        assert cache.fast_runtime is not None
        assert cache.fast_runtime.template_counts == {
            "front": front_count,
            "back": back_count,
        }
        assert _decision_view(native_result) == _decision_view(python_result)
        assert native_result["inference_engine"] == "fast_geometry"
    finally:
        python_classifier.close()
        native_classifier.close()


def test_serialized_five_plus_five_fast_cache_recovers_without_rebuild(tmp_path):
    model_dir = tmp_path / "native-model"
    model_dir.mkdir()
    executable = tmp_path / "service.exe"
    executable.write_bytes(b"test executable")
    front = [
        _write_template(tmp_path / f"front-{index}.png", side="front", index=index)
        for index in range(5)
    ]
    back = [
        _write_template(tmp_path / f"back-{index}.png", side="back", index=index)
        for index in range(5)
    ]
    record = type(
        "Record",
        (),
        {
            "root": tmp_path / "library" / "m7",
            "front_images": tuple(front),
            "back_images": tuple(back),
            "revision": 7,
            "geometry_profile_revision": None,
        },
    )()
    record.root.mkdir(parents=True)

    builder, _ = _make_python_classifier()
    cache = builder.build_template_cache(front, back, library_revision=7)
    builder.save_template_cache(record, cache)

    # Load through a new classifier to model an already-established library;
    # no template extraction callback is allowed during recovery.
    restored, _ = _make_python_classifier()
    restored_cache = restored.load_template_cache(record)
    assert restored_cache is not None
    assert restored_cache.fast_runtime is not None
    assert restored_cache.fast_runtime.template_counts == {"front": 5, "back": 5}
    query = _write_query(tmp_path / "restored-query.png")
    result = restored.predict_fast_with_cache(restored_cache, query, library_revision=7)
    assert result["inference_engine"] == "fast_geometry"

    builder.close()
    restored.close()


def test_native_batch_child_exit_is_reported_per_item_without_python_fallback(tmp_path, monkeypatch):
    # Build a valid cache with the deterministic Python path first.
    builder, _ = _make_python_classifier()
    front = [_write_template(tmp_path / "front.png", side="front", index=0)]
    back = [_write_template(tmp_path / "back.png", side="back", index=0)]
    cache = builder.build_template_cache(front, back, library_revision=1)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    executable = tmp_path / "service.exe"
    executable.write_bytes(b"test executable")
    scalar = _DeterministicNativeClient(executable, model_dir, threads=1)
    scalar.start()
    _ExitNativeClient.instances = []
    monkeypatch.setattr("src.orientation_classifier.NativePPClient", _ExitNativeClient)
    fallback = _PythonFallbackSentinel()
    classifier = _make_native_classifier(
        model_dir,
        executable,
        client=scalar,
        fallback_sentinel=fallback,
    )
    try:
        status = classifier.prepare_batch_pool(worker_count=2, threads_per_worker=1)
        assert status["batch_ready"] is True
        paths = [
            _write_query(tmp_path / f"query-{index}.png", side="front")
            for index in range(4)
        ]
        results = classifier.predict_many_with_cache(cache, paths, library_revision=1)
        assert results.execution == {
            "batch_mode": "batch",
            "worker_count": 2,
            "fallback": None,
        }
        assert [item["index"] for item in results] == [0, 1, 2, 3]
        assert all(
            item["error_type"] == "NativePPError"
            and "NATIVE_PP_SERVICE_EXITED" in str(item["error"])
            for item in results
        )
        assert fallback.calls == 0
        classifier.close()
        assert all(client.closed == 1 for client in _ExitNativeClient.instances)
    finally:
        classifier.close()
        builder.close()
