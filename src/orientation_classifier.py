"""Reusable PP-ShiTuV2 and ALIKED/LightGlue orientation classifier."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import cv2
import numpy as np

from src.image_io import read_color_image
from src.shitu_baseline import classify_embedding
from src.interference_masks import filter_template_features
from src.interference_masks import project_region
from src.geometry_calibration import apply_geometry_fit
from src.geometry_calibration import apply_ignore_mask
from src.geometry_calibration import filter_features_by_mask
from src.geometry_calibration import geometry_feature_mask
from src.geometry_profile_schema import materialize_runtime_profile
from src.model_execution_gate import PriorityModelGate
from src.model_fingerprint import model_directory_sha256
from src.paddleclas_inference_compat import create_rec_predictor, install_optional_sklearn_stubs
from src.fast_geometry import FastGeometryProcessor
from src.fast_orientation import FastOrientationEngine, FastRuntimeCache


ROI_RATIO = 1.0
MAX_NUM_KEYPOINTS = 512
GLOBAL_MARGIN_THRESHOLD = 0.05
LOCAL_MIN_SCORE = 4.0
LOCAL_MIN_MARGIN = 0.5
LOCAL_OVERRIDE_MARGIN = 3.0
LOCAL_SEARCH_MODES = ("adaptive", "exhaustive")
DEFAULT_LOCAL_SEARCH_MODE = "adaptive"
LOCAL_SEARCH_STAGE_LIMITS = (("top5", 5), ("top10", 10))
TEMPLATE_CACHE_FORMAT_VERSION = 2
TEMPLATE_CACHE_FILE_NAME = ".template_cache.pkl"
FAST_RUNTIME_CACHE_FILE_NAME = ".fast_runtime_cache.pkl"
INFERENCE_MODES = ("legacy", "fast_geometry", "compare")
DEFAULT_INFERENCE_MODE = "legacy"
COMPUTE_DEVICES = ("gpu", "cpu")
DEFAULT_COMPUTE_DEVICE = "gpu"
CPU_THREADS_ENV = "WORKPIECE_CPU_THREADS"
CPU_SLOT_DEDUP_ENV = "WORKPIECE_CPU_DEDUPLICATE_SLOTS"
DEFAULT_CPU_NUM_THREADS = 10
_GEOMETRY_REVISION_FROM_RECORD = object()


LOGGER = logging.getLogger(__name__)


class OrientationClassifierError(RuntimeError):
    """Base error for service-facing classifier failures."""


class ComputeDeviceError(OrientationClassifierError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ModelFingerprintError(OrientationClassifierError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class WorkpieceNotFoundError(OrientationClassifierError):
    """Raised when prediction is requested for an unknown workpiece."""


class ImageUnreadableError(OrientationClassifierError):
    """Raised when OpenCV cannot decode an input image."""


class PropagationModelError(OrientationClassifierError):
    """Raised when the local feature model cannot safely continue propagation."""


def _is_fatal_local_model_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(token in text for token in ("cuda out of memory", "cuda error", "cudnn", "device-side"))


@dataclass(frozen=True)
class TemplateCache:
    """CPU-resident features for one workpiece."""

    global_vectors: dict[str, np.ndarray]
    local_features: dict[str, list[dict[str, Any]]]
    raw_global_vectors: dict[str, np.ndarray] | None = None
    raw_local_features: dict[str, list[dict[str, Any]]] | None = None
    ignored_regions: dict[str, list[dict[str, float]]] | None = None
    geometry_profile: dict[str, Any] | None = None
    geometry_profile_revision: int | None = None
    geometry_template_report: dict[str, Any] | None = None
    geometry_template_indices: dict[str, list[int]] | None = None
    geometry_unsafe: bool = False
    fast_runtime: FastRuntimeCache | None = None
    fast_template_signature: dict[str, Any] | None = None


@dataclass
class _FastRuntimeTransaction:
    lock: threading.Lock = field(repr=False, compare=False)
    state: str = "staged"
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.lock.release()


@dataclass(frozen=True)
class StagedFastRuntimeCache:
    record_root: Path
    temporary_root: Path
    temporary_path: Path
    cache: FastRuntimeCache
    staged_digest: str
    previous_digest: str | None
    transaction: _FastRuntimeTransaction = field(repr=False, compare=False)


@dataclass(frozen=True)
class CommittedFastRuntimeCache:
    target_path: Path
    backup_path: Path | None
    temporary_root: Path
    committed_digest: str
    previous_digest: str | None
    transaction: _FastRuntimeTransaction = field(repr=False, compare=False)


@dataclass(frozen=True)
class LocalSearchResult:
    scores: dict[str, float]
    diagnostics: dict[str, object]
    matching_ms: float
    trace: dict[str, object]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_tensors(value: Any, device: Any) -> Any:
    """Move nested Torch tensors while leaving test doubles and scalars unchanged."""
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, np.ndarray) or value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    return value


def _to_cpu(value: Any) -> Any:
    """Detach nested Torch tensors so template caches do not retain GPU memory."""
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _read_image(path: Path) -> np.ndarray:
    image = read_color_image(path)
    if image is None:
        raise ImageUnreadableError(f"Unable to read image: {path}")
    return image


def _default_extract_features(image, extractor, device, roi_ratio=ROI_RATIO):
    from src.aliked_lightglue_matcher import extract_features

    return extract_features(image, extractor, device, roi_ratio=roi_ratio)


def _default_score_feature_pair(query_features, template_features, image_shape, matcher):
    from src.soft_center_matcher import score_feature_pair_soft

    return score_feature_pair_soft(query_features, template_features, image_shape, matcher)


def _decide_local_label(scores: dict[str, float]) -> tuple[str, float]:
    ordered = sorted(scores, key=scores.get, reverse=True)
    margin = float(scores[ordered[0]] - scores[ordered[1]])
    if scores[ordered[0]] < LOCAL_MIN_SCORE or margin < LOCAL_MIN_MARGIN:
        return "uncertain", margin
    return ordered[0], margin


def _validate_local_search_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in LOCAL_SEARCH_MODES:
        raise ValueError("local_search_mode must be 'adaptive' or 'exhaustive'")
    return mode


def _validate_inference_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in INFERENCE_MODES:
        raise ValueError("inference_mode must be 'legacy', 'fast_geometry', or 'compare'")
    return mode


def _validate_compute_device(value: str) -> str:
    device = str(value).strip().lower()
    if device not in COMPUTE_DEVICES:
        raise ValueError("compute_device must be 'gpu' or 'cpu'")
    return device


def _select_paddle_device(paddle: Any, device: str) -> str:
    if device == "gpu":
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
            raise ComputeDeviceError("GPU_UNAVAILABLE", "未检测到可用的 NVIDIA GPU/Paddle GPU 运行时")
        paddle.set_device("gpu:0")
        if not str(paddle.device.get_device()).startswith("gpu"):
            raise ComputeDeviceError("DEVICE_MISMATCH", "Paddle 未实际使用 GPU")
        return "gpu"
    paddle.set_device("cpu")
    if not str(paddle.device.get_device()).startswith("cpu"):
        raise ComputeDeviceError("DEVICE_MISMATCH", "Paddle 未实际使用 CPU")
    return "cpu"


def _validate_model_fingerprint(value: str) -> str:
    fingerprint = str(value).strip()
    if not fingerprint:
        raise ValueError("model_fingerprint must be a non-empty string")
    return fingerprint


def _resolve_cpu_num_threads(global_config: Any) -> int:
    """Resolve the configured CPU thread count, allowing a process override."""
    configured_value = getattr(global_config, "cpu_num_threads", DEFAULT_CPU_NUM_THREADS)
    try:
        if isinstance(configured_value, bool):
            raise ValueError
        if isinstance(configured_value, int):
            configured = configured_value
        elif isinstance(configured_value, str):
            configured = int(configured_value)
        else:
            raise ValueError
        if configured <= 0:
            raise ValueError
    except (TypeError, ValueError):
        LOGGER.warning(
            "Ignoring invalid YAML cpu_num_threads=%r; using default=%s",
            configured_value,
            DEFAULT_CPU_NUM_THREADS,
        )
        configured = DEFAULT_CPU_NUM_THREADS
    override = os.environ.get(CPU_THREADS_ENV)
    if override is None:
        return configured
    if override.isdecimal() and int(override) > 0:
        return int(override)
    LOGGER.warning("Ignoring invalid %s=%r; keeping YAML cpu_num_threads=%s", CPU_THREADS_ENV, override, configured)
    return configured


def _resolve_cpu_slot_dedup(compute_device: str) -> bool:
    """Resolve exact query-slot deduplication, enabled by default on CPU only."""
    default = compute_device == "cpu"
    override = os.environ.get(CPU_SLOT_DEDUP_ENV)
    if override is None:
        return default
    normalized = override.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return default if compute_device == "gpu" else True
    if normalized in {"0", "false", "no"}:
        return False if compute_device == "cpu" else default
    LOGGER.warning("Ignoring invalid %s=%r; using device default=%s", CPU_SLOT_DEDUP_ENV, override, default)
    return default


class OrientationClassifier:
    """Fuse global retrieval with decisive soft-center local evidence."""

    def __init__(
        self,
        global_predictor,
        extractor,
        matcher,
        device,
        *,
        extract_features_fn: Callable[..., dict] | None = None,
        score_feature_pair_fn: Callable[..., dict] | None = None,
        geometry_calibrator: Any | None = None,
        model_gate: PriorityModelGate | None = None,
        local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
        inference_mode: str = DEFAULT_INFERENCE_MODE,
        model_fingerprint: str = "unconfigured",
        fast_engine: FastOrientationEngine | None = None,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
    ) -> None:
        self.global_predictor = global_predictor
        self.extractor = extractor
        self.matcher = matcher
        self.device = device
        self._extract_features = extract_features_fn or _default_extract_features
        self._score_feature_pair = score_feature_pair_fn or _default_score_feature_pair
        self.geometry_calibrator = geometry_calibrator
        self.model_gate = model_gate or PriorityModelGate()
        self.local_search_mode = _validate_local_search_mode(local_search_mode)
        self.inference_mode = _validate_inference_mode(inference_mode)
        self.model_fingerprint = _validate_model_fingerprint(model_fingerprint)
        self.compute_device = _validate_compute_device(compute_device)
        self.cpu_num_threads = None
        self.fast_engine = fast_engine
        self._inference_lock = threading.RLock()
        self._template_caches: dict[str, TemplateCache] = {}
        self._fast_runtime_transaction_guard = threading.Lock()
        self._fast_runtime_transaction_locks: dict[Path, threading.Lock] = {}

    def _fast_runtime_transaction_lock(self, root: Path) -> threading.Lock:
        guard = getattr(self, "_fast_runtime_transaction_guard", None)
        if guard is None:
            guard = threading.Lock()
            self._fast_runtime_transaction_guard = guard
            self._fast_runtime_transaction_locks = {}
        key = Path(root).resolve()
        with guard:
            return self._fast_runtime_transaction_locks.setdefault(key, threading.Lock())

    @staticmethod
    def _model_directory_fingerprint(model_dir: Path) -> str:
        return model_directory_sha256(model_dir)

    @classmethod
    def load(
        cls,
        project_root: Path,
        model_dir: Path,
        *,
        local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
        inference_mode: str = DEFAULT_INFERENCE_MODE,
        paddle_config_path: Path | None = None,
        compute_device: str = DEFAULT_COMPUTE_DEVICE,
        expected_model_fingerprint: str | None = None,
    ) -> "OrientationClassifier":
        """Load the production Paddle and Torch models lazily at service startup."""
        inference_mode = _validate_inference_mode(inference_mode)
        compute_device = _validate_compute_device(compute_device)
        model_fingerprint = cls._model_directory_fingerprint(model_dir)
        if (
            expected_model_fingerprint is not None
            and str(expected_model_fingerprint).strip() != model_fingerprint
        ):
            raise ModelFingerprintError(
                "MODEL_FINGERPRINT_MISMATCH",
                "模型目录指纹与期望值不一致",
            )
        extractor = matcher = device = None
        if inference_mode in {"legacy", "compare"}:
            # On Windows, Paddle and PyTorch can expose incompatible DLLs when
            # Paddle is imported first. Load the local stack first when needed.
            from src.aliked_lightglue_matcher import build_models

            extractor, matcher, device = build_models(MAX_NUM_KEYPOINTS)
        import paddle
        install_optional_sklearn_stubs()
        from paddleclas.deploy.python.predict_rec import RecPredictor
        from paddleclas.deploy.utils import config as paddle_config
        from src.paddleclas_inference_compat import prepare_paddle_model_path

        selected_compute_device = _select_paddle_device(paddle, compute_device)
        config_path = paddle_config_path or (
            project_root / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml"
        )
        config = paddle_config.get_config(str(config_path), show=False)
        cpu_num_threads = (
            _resolve_cpu_num_threads(config.Global)
            if compute_device == "cpu"
            else getattr(config.Global, "cpu_num_threads", DEFAULT_CPU_NUM_THREADS)
        )
        try:
            paddle_model_dir = prepare_paddle_model_path(model_dir)
        except Exception as exc:
            raise ComputeDeviceError(
                "MODEL_PATH_UNSUPPORTED",
                "Paddle 无法处理模型路径；请将程序解压到纯 ASCII 路径后重试",
            ) from exc
        config.Global.rec_inference_model_dir = str(paddle_model_dir)
        config.Global.use_gpu = compute_device == "gpu"
        config.Global.enable_mkldnn = compute_device == "cpu"
        if compute_device == "cpu":
            config.Global.cpu_num_threads = cpu_num_threads
        config.Global.enable_benchmark = False
        config.Global.gpu_mem = 1024
        global_predictor = create_rec_predictor(RecPredictor, config, paddle, paddle_model_dir)
        try:
            embeddings = list(global_predictor.predict([np.zeros((512, 512, 3), dtype=np.uint8)]))
            embedding = np.asarray(embeddings[0], dtype=np.float32) if len(embeddings) == 1 else None
            if embedding is None or embedding.size == 0 or not np.all(np.isfinite(embedding)):
                raise ValueError("Paddle returned an invalid embedding")
        except Exception as exc:
            raise ComputeDeviceError(
                "RUNTIME_SELF_CHECK_FAILED",
                "Paddle 推理运行时自检失败",
            ) from exc
        from src.geometry_calibration import GeometryCalibrator

        calibrator = GeometryCalibrator()
        classifier = cls(
            global_predictor,
            extractor,
            matcher,
            device,
            geometry_calibrator=calibrator,
            local_search_mode=local_search_mode,
            inference_mode=inference_mode,
            model_fingerprint=model_fingerprint,
            compute_device=selected_compute_device,
        )
        classifier.cpu_num_threads = cpu_num_threads
        classifier.fast_engine = FastOrientationEngine(
            classifier._global_embeddings,
            FastGeometryProcessor(calibrator),
            image_reader=_read_image,
            deduplicate_identical_slots=_resolve_cpu_slot_dedup(selected_compute_device),
        )
        return classifier

    def _global_embeddings(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not images:
            return []
        rgb_images = [image[:, :, ::-1] for image in images]
        with self._inference_lock:
            embeddings = self.global_predictor.predict(rgb_images)
        if len(embeddings) != len(images):
            raise OrientationClassifierError("global predictor returned unexpected batch size")
        return [np.asarray(embedding, dtype=np.float32) for embedding in embeddings]

    def _global_embedding(self, image: np.ndarray) -> np.ndarray:
        return self._global_embeddings([image])[0]

    def _extract_local(self, image: np.ndarray) -> dict:
        with self._inference_lock:
            features = self._extract_features(image, self.extractor, self.device, roi_ratio=ROI_RATIO)
        return _to_cpu(features)

    def _extract_template_features(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """Extract one template as a bounded, low-priority model step."""
        return self.model_gate.run_background_step(
            lambda: (self._global_embedding(image), self._extract_local(image))
        )

    def build_template_cache(
        self,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        library_revision: int = 1,
    ) -> TemplateCache:
        if type(library_revision) is not int or library_revision < 0:
            raise ValueError("library_revision must be a non-negative integer")
        return self._build_template_cache(
            front_paths,
            back_paths,
            progress_callback,
            include_local=self.inference_mode != "fast_geometry",
            attach_fast=self.inference_mode in {"fast_geometry", "compare"},
            library_revision=library_revision,
        )

    def _build_template_cache(
        self,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
        progress_callback: Callable[[str, int, int], None] | None,
        *,
        include_local: bool,
        attach_fast: bool,
        library_revision: int,
    ) -> TemplateCache:
        if not front_paths or not back_paths:
            raise ValueError("Each orientation requires at least one template")
        global_vectors: dict[str, np.ndarray] = {}
        local_features: dict[str, list[dict[str, Any]]] = {}
        for label, paths in (("front", front_paths), ("back", back_paths)):
            embeddings = []
            features = []
            total = len(paths)
            for completed, path in enumerate(paths, start=1):
                image = _read_image(Path(path))
                if include_local:
                    embedding, local = self._extract_template_features(image)
                else:
                    embedding = self.model_gate.run_background_step(
                        lambda: self._global_embedding(image)
                    )
                    local = {}
                embeddings.append(embedding)
                features.append(local)
                if progress_callback is not None:
                    progress_callback(label, completed, total)
            global_vectors[label] = np.stack(embeddings).astype(np.float32)
            local_features[label] = features
        cache = TemplateCache(
            global_vectors=global_vectors,
            local_features=local_features,
            raw_global_vectors={label: vectors.copy() for label, vectors in global_vectors.items()},
            raw_local_features=local_features,
            ignored_regions={},
            fast_template_signature=self._fast_template_signature(front_paths, back_paths),
        )
        if attach_fast and self.fast_engine is not None:
            fast_runtime = self._fast_build_engine().build_cache(
                front_paths,
                back_paths,
                geometry_profile=None,
                library_revision=library_revision,
                model_fingerprint=self.model_fingerprint,
                progress_callback=self._adapt_fast_progress(progress_callback),
            )
            cache = replace(cache, fast_runtime=fast_runtime)
        return cache

    @staticmethod
    def _adapt_fast_progress(
        progress_callback: Callable[..., None] | None,
    ) -> Callable[[dict[str, Any]], None] | None:
        if progress_callback is None:
            return None

        def report(event: dict[str, Any]) -> None:
            try:
                progress_callback(event)
            except TypeError:
                # Existing builders accept (label, completed, total). They
                # already received legacy template progress above.
                return

        return report

    def _fast_build_engine(self) -> FastOrientationEngine:
        if self.fast_engine is None:
            raise OrientationClassifierError("FAST_CACHE_NOT_READY: fast engine is unavailable")
        engine = self.fast_engine
        if not isinstance(engine, FastOrientationEngine):
            return engine

        def background_embed(images: Sequence[np.ndarray]) -> list[np.ndarray]:
            return self.model_gate.run_background_step(lambda: engine.embed_batch(images))

        return FastOrientationEngine(
            background_embed,
            engine.geometry,
            image_reader=engine.image_reader,
            deduplicate_identical_slots=engine.deduplicate_identical_slots,
        )

    @staticmethod
    def _image_fingerprint(path: Path) -> str:
        image = read_color_image(Path(path))
        if image is None:
            raise ImageUnreadableError(f"Unable to read image: {path}")
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    @classmethod
    def _template_cache_signature(
        cls,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
    ) -> dict[str, Any]:
        def side_signature(paths: Sequence[Path]) -> list[dict[str, str]]:
            return [
                {
                    "name": Path(path).name,
                    "fingerprint": cls._image_fingerprint(Path(path)),
                }
                for path in paths
            ]

        return {
            "format_version": TEMPLATE_CACHE_FORMAT_VERSION,
            "max_num_keypoints": MAX_NUM_KEYPOINTS,
            "roi_ratio": ROI_RATIO,
            "front": side_signature(front_paths),
            "back": side_signature(back_paths),
        }

    @staticmethod
    def _fast_image_fingerprint(path: Path) -> str:
        image = _read_image(Path(path))
        normalized = np.ascontiguousarray(image)
        digest = hashlib.sha256()
        digest.update(str(normalized.shape).encode("ascii"))
        digest.update(normalized.dtype.str.encode("ascii"))
        digest.update(normalized.tobytes())
        return digest.hexdigest()

    @classmethod
    def _fast_template_signature(
        cls,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
    ) -> dict[str, Any]:
        signature: dict[str, Any] = {}
        for label, paths in (("front", front_paths), ("back", back_paths)):
            signature[label] = [
                {
                    "path": str(Path(path)),
                    "sha256": cls._fast_image_fingerprint(Path(path)),
                }
                for path in paths
            ]
        encoded = json.dumps(
            {"front": signature["front"], "back": signature["back"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        signature["combined_sha256"] = hashlib.sha256(encoded).hexdigest()
        return signature

    @staticmethod
    def _fast_template_signatures_match(
        cached: Any,
        actual: Any,
    ) -> bool:
        if not isinstance(cached, Mapping) or not isinstance(actual, Mapping):
            return False
        for label in ("front", "back"):
            cached_entries = cached.get(label)
            actual_entries = actual.get(label)
            if not isinstance(cached_entries, list) or not isinstance(actual_entries, list):
                return False
            if len(cached_entries) != len(actual_entries):
                return False
            for cached_entry, actual_entry in zip(cached_entries, actual_entries):
                if not isinstance(cached_entry, Mapping) or not isinstance(actual_entry, Mapping):
                    return False
                if cached_entry.get("sha256") != actual_entry.get("sha256"):
                    return False
                if Path(str(cached_entry.get("path", ""))).name != Path(
                    str(actual_entry.get("path", ""))
                ).name:
                    return False
        return True

    @staticmethod
    def _cache_for_persistence(cache: TemplateCache) -> TemplateCache:
        """Persist the unfiltered base cache; active masks are restored separately."""
        raw = getattr(cache, "raw_local_features", None) or cache.local_features
        raw_global = getattr(cache, "raw_global_vectors", None) or cache.global_vectors
        return TemplateCache(
            global_vectors=raw_global,
            local_features=raw,
            raw_global_vectors=raw_global,
            raw_local_features=raw,
            ignored_regions={},
            fast_runtime=None,
            fast_template_signature=getattr(cache, "fast_template_signature", None),
        )

    @staticmethod
    def _normalize_template_cache(cache: Any) -> TemplateCache:
        if not isinstance(cache, TemplateCache):
            raise ValueError("cache payload has an unexpected type")
        return TemplateCache(
            global_vectors=cache.global_vectors,
            local_features=getattr(cache, "local_features", {}) or {},
            raw_global_vectors=getattr(cache, "raw_global_vectors", None),
            raw_local_features=getattr(cache, "raw_local_features", None),
            ignored_regions=getattr(cache, "ignored_regions", None),
            geometry_profile=getattr(cache, "geometry_profile", None),
            geometry_profile_revision=getattr(cache, "geometry_profile_revision", None),
            geometry_template_report=getattr(cache, "geometry_template_report", None),
            geometry_template_indices=getattr(cache, "geometry_template_indices", None),
            geometry_unsafe=bool(getattr(cache, "geometry_unsafe", False)),
            fast_runtime=getattr(cache, "fast_runtime", None),
            fast_template_signature=getattr(cache, "fast_template_signature", None),
        )

    @staticmethod
    def _validate_cached_shape(
        cache: TemplateCache,
        front_count: int,
        back_count: int,
    ) -> None:
        if not isinstance(cache, TemplateCache):
            raise ValueError("cache payload has an unexpected type")
        for label, expected in (("front", front_count), ("back", back_count)):
            vectors = cache.global_vectors.get(label)
            if vectors is None or len(getattr(vectors, "shape", ())) < 1 or vectors.shape[0] != expected:
                raise ValueError(f"cache global vector count mismatch for {label}")
            raw_vectors = getattr(cache, "raw_global_vectors", None)
            if raw_vectors is not None:
                raw_label = raw_vectors.get(label)
                if raw_label is None or raw_label.shape[0] != expected:
                    raise ValueError(f"cache raw global vector count mismatch for {label}")
            features = cache.local_features.get(label, [])
            if not isinstance(features, list) or len(features) not in {0, expected}:
                raise ValueError(f"cache local feature count mismatch for {label}")
            if cache.raw_local_features is not None:
                raw_features = cache.raw_local_features.get(label, [])
                if not isinstance(raw_features, list) or len(raw_features) not in {0, expected}:
                    raise ValueError(f"cache raw feature count mismatch for {label}")

    @staticmethod
    def _has_legacy_local_features(cache: TemplateCache, record: Any) -> bool:
        local = getattr(cache, "raw_local_features", None) or cache.local_features
        if not isinstance(local, dict):
            return False
        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            features = local.get(label)
            if not isinstance(features, list) or len(features) != len(paths):
                return False
            if any(not isinstance(feature, Mapping) or not feature for feature in features):
                return False
        return True

    @staticmethod
    def _record_geometry_revision(record: Any) -> int | None:
        revision = getattr(record, "geometry_profile_revision", None)
        if revision is not None:
            return int(revision)
        profile = getattr(record, "geometry_profile", None)
        if isinstance(profile, Mapping) and isinstance(profile.get("profile_revision"), int):
            return int(profile["profile_revision"])
        manifest_path = Path(record.root) / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if "geometry_mask_active_revision" in manifest:
                    active_revision = manifest["geometry_mask_active_revision"]
                    return None if active_revision is None else int(active_revision)
            except (OSError, ValueError, TypeError):
                return None
        profile_path = Path(record.root) / "geometry_masks" / "profile.json"
        if profile_path.is_file():
            try:
                document = json.loads(profile_path.read_text(encoding="utf-8"))
                active_revision = document.get("active_revision")
                if active_revision is not None:
                    return int(active_revision)
            except (OSError, ValueError, TypeError):
                return None
        return None

    def _fast_persistence_signature(
        self,
        record: Any,
        cache: FastRuntimeCache,
    ) -> dict[str, Any]:
        return {
            "format_version": cache.format_version,
            "library_revision": int(record.revision),
            "geometry_profile_revision": cache.geometry_profile_revision,
            "model_fingerprint": self.model_fingerprint,
            "template_content": self._template_cache_signature(
                record.front_images, record.back_images
            ),
            "template_counts": dict(cache.template_counts),
            "feature_layout": tuple(cache.feature_layout),
            "feature_dim": int(cache.ridge_head.feature_dim),
        }

    def _validate_fast_runtime_for_record(
        self,
        record: Any,
        cache: FastRuntimeCache,
        *,
        geometry_profile_revision: int | None | object = _GEOMETRY_REVISION_FROM_RECORD,
    ) -> None:
        FastOrientationEngine._validate_cache(cache)
        if cache.library_revision != int(record.revision):
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: library revision differs")
        if cache.model_fingerprint != self.model_fingerprint:
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: model fingerprint differs")
        geometry_revision = (
            self._record_geometry_revision(record)
            if geometry_profile_revision is _GEOMETRY_REVISION_FROM_RECORD
            else geometry_profile_revision
        )
        if cache.geometry_profile_revision != geometry_revision:
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: geometry revision differs")
        expected_counts = {
            "front": len(record.front_images),
            "back": len(record.back_images),
        }
        if cache.template_counts != expected_counts:
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: template counts differ")
        try:
            actual_signature = self._fast_template_signature(
                record.front_images,
                record.back_images,
            )
        except Exception as exc:
            raise ValueError(
                "FAST_CACHE_REVISION_MISMATCH: template content is unavailable"
            ) from exc
        if not self._fast_template_signatures_match(
            cache.template_signature,
            actual_signature,
        ):
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: template content differs")

    def save_template_cache(self, record: Any, cache: TemplateCache) -> None:
        """Atomically persist a base template cache for one workpiece."""
        root = Path(record.root)
        cache_path = root / TEMPLATE_CACHE_FILE_NAME
        payload = {
            "signature": self._template_cache_signature(record.front_images, record.back_images),
            "cache": self._cache_for_persistence(cache),
        }
        temporary = root / f".{TEMPLATE_CACHE_FILE_NAME}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        fast_runtime = getattr(cache, "fast_runtime", None)
        if fast_runtime is not None:
            self.save_fast_runtime_cache(record, fast_runtime)

    def save_fast_runtime_cache(self, record: Any, cache: FastRuntimeCache) -> None:
        """Validate and atomically replace one workpiece's fast payload."""
        staged = self.stage_fast_runtime_cache(record, cache)
        try:
            committed = self.commit_staged_fast_runtime_cache(record, staged)
        except Exception:
            self.discard_staged_fast_runtime_cache(staged)
            raise
        self.finalize_staged_fast_runtime_cache(committed)

    def stage_fast_runtime_cache(
        self,
        record: Any,
        cache: FastRuntimeCache,
        *,
        geometry_profile_revision: int | None | object = _GEOMETRY_REVISION_FROM_RECORD,
    ) -> StagedFastRuntimeCache:
        """Validate and serialize a fast payload without changing the live sidecar."""
        self._validate_fast_runtime_for_record(
            record,
            cache,
            geometry_profile_revision=geometry_profile_revision,
        )
        root = Path(record.root)
        if not root.is_dir():
            raise FileNotFoundError(f"workpiece root is unavailable: {root}")
        transaction_lock = self._fast_runtime_transaction_lock(root)
        transaction_lock.acquire()
        transaction = _FastRuntimeTransaction(transaction_lock)
        temporary_root = root.parent / f".fast-runtime-stage-{root.name}-{uuid.uuid4().hex}"
        temporary_root_created = False
        staged = False
        try:
            temporary_root.mkdir()
            temporary_root_created = True
            temporary = temporary_root / FAST_RUNTIME_CACHE_FILE_NAME
            payload = {
                "signature": self._fast_persistence_signature(record, cache),
                "cache": cache,
            }
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            staged_digest = _file_sha256(temporary)
            target = root / FAST_RUNTIME_CACHE_FILE_NAME
            previous_digest = None
            if target.is_file():
                backup = temporary_root / ".previous-fast-runtime-cache.pkl"
                with target.open("rb") as source, backup.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                    destination.flush()
                    os.fsync(destination.fileno())
                previous_digest = _file_sha256(backup)
            staged = True
            return StagedFastRuntimeCache(
                root,
                temporary_root,
                temporary,
                cache,
                staged_digest,
                previous_digest,
                transaction,
            )
        finally:
            if not staged:
                try:
                    if temporary_root_created and temporary_root.exists():
                        shutil.rmtree(temporary_root)
                except Exception:
                    LOGGER.warning(
                        "Unable to clean failed fast cache staging %s",
                        temporary_root,
                        exc_info=True,
                    )
                finally:
                    transaction.release()

    @staticmethod
    def commit_staged_fast_runtime_cache(
        record: Any,
        staged: StagedFastRuntimeCache,
    ) -> CommittedFastRuntimeCache:
        root = Path(record.root)
        if root != staged.record_root:
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: workpiece root changed")
        target = root / FAST_RUNTIME_CACHE_FILE_NAME
        backup = staged.temporary_root / ".previous-fast-runtime-cache.pkl"
        backup_path = backup if staged.previous_digest is not None else None
        transaction = staged.transaction
        if transaction.state == "staged":
            transaction.state = "committing"
            try:
                os.replace(staged.temporary_path, target)
            except Exception:
                transaction.state = (
                    "committed" if not staged.temporary_path.exists() else "staged"
                )
                raise
            transaction.state = "committed"
        elif transaction.state not in {"committed", "finalized"}:
            raise ValueError(
                f"FAST_CACHE_STAGE_STATE_CHANGED: transaction is {transaction.state}"
            )
        return CommittedFastRuntimeCache(
            target,
            backup_path,
            staged.temporary_root,
            staged.staged_digest,
            staged.previous_digest,
            transaction,
        )

    @staticmethod
    def finalize_staged_fast_runtime_cache(committed: CommittedFastRuntimeCache) -> None:
        transaction = committed.transaction
        if transaction.state in {
            "finalized", "rolled_back", "rollback_failed", "discarded"
        }:
            return
        if transaction.state != "committed":
            raise ValueError(
                f"FAST_CACHE_STAGE_STATE_CHANGED: transaction is {transaction.state}"
            )
        try:
            if committed.temporary_root.exists():
                shutil.rmtree(committed.temporary_root)
        finally:
            transaction.state = "finalized"
            transaction.release()

    @staticmethod
    def rollback_committed_fast_runtime_cache(committed: CommittedFastRuntimeCache) -> None:
        transaction = committed.transaction
        if transaction.state in {"rolled_back", "rollback_failed", "discarded"}:
            return
        if transaction.state != "committed":
            raise ValueError(
                f"FAST_CACHE_STAGE_STATE_CHANGED: transaction is {transaction.state}"
            )
        transaction.state = "rolling_back"
        try:
            if committed.previous_digest is not None:
                if committed.backup_path is None:
                    raise ValueError(
                        "FAST_CACHE_STAGE_STATE_CHANGED: rollback payload is unavailable"
                    )
                os.replace(committed.backup_path, committed.target_path)
            elif committed.target_path.exists():
                committed.target_path.unlink()
        except Exception:
            rollback_completed = (
                committed.previous_digest is not None
                and committed.backup_path is not None
                and not committed.backup_path.exists()
            ) or (
                committed.previous_digest is None
                and not committed.target_path.exists()
            )
            transaction.state = (
                "rolled_back" if rollback_completed else "rollback_failed"
            )
            raise
        else:
            try:
                if committed.temporary_root.exists():
                    shutil.rmtree(committed.temporary_root)
            finally:
                transaction.state = "rolled_back"
        finally:
            transaction.release()

    @staticmethod
    def discard_staged_fast_runtime_cache(staged: StagedFastRuntimeCache) -> None:
        transaction = staged.transaction
        if transaction.state in {
            "discarded", "rolled_back", "rollback_failed", "finalized"
        }:
            return
        if transaction.state == "committed":
            backup = staged.temporary_root / ".previous-fast-runtime-cache.pkl"
            OrientationClassifier.rollback_committed_fast_runtime_cache(
                CommittedFastRuntimeCache(
                    staged.record_root / FAST_RUNTIME_CACHE_FILE_NAME,
                    backup if staged.previous_digest is not None else None,
                    staged.temporary_root,
                    staged.staged_digest,
                    staged.previous_digest,
                    transaction,
                )
            )
            return
        if transaction.state != "staged":
            raise ValueError(
                f"FAST_CACHE_STAGE_STATE_CHANGED: transaction is {transaction.state}"
            )
        try:
            if staged.temporary_root.exists():
                shutil.rmtree(staged.temporary_root)
        finally:
            transaction.state = "discarded"
            transaction.release()

    def _read_fast_runtime_cache(self, record: Any, cache_path: Path) -> FastRuntimeCache:
        with cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or set(payload) != {"signature", "cache"}:
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: payload is incomplete")
        cache = payload["cache"]
        self._validate_fast_runtime_for_record(record, cache)
        if payload["signature"] != self._fast_persistence_signature(record, cache):
            raise ValueError("FAST_CACHE_REVISION_MISMATCH: persistence signature differs")
        return cache

    @staticmethod
    def _fast_runtime_staging_directories(
        record: Any,
        staging_parent: Path | None,
    ) -> list[Path]:
        root = Path(record.root)
        parent = root.parent if staging_parent is None else Path(staging_parent)
        resolved_parent = parent.resolve()
        allowed_parents = {root.parent.resolve()}
        if root.parent.name == ".recycled":
            allowed_parents.add(root.parent.parent.resolve())
        if resolved_parent not in allowed_parents:
            raise ValueError("FAST_CACHE_STAGE_PATH_INVALID: staging parent is unrelated")
        if staging_parent is not None and str(record.id) != root.name:
            raise ValueError("FAST_CACHE_STAGE_PATH_INVALID: record path does not match its id")

        prefix = f".fast-runtime-stage-{root.name}-"
        directories = []
        candidates = sorted(parent.iterdir()) if parent.is_dir() else []
        for candidate in candidates:
            if not candidate.name.startswith(prefix):
                continue
            suffix = candidate.name[len(prefix):]
            if (
                candidate.parent.resolve() != resolved_parent
                or candidate.resolve().parent != resolved_parent
                or candidate.is_symlink()
                or not candidate.is_dir()
                or len(suffix) != 32
                or any(character not in "0123456789abcdef" for character in suffix)
            ):
                continue
            directories.append(candidate)
        return directories

    def recover_fast_runtime_cache_staging(
        self,
        record: Any,
        *,
        staging_parent: Path | None = None,
    ) -> None:
        """Resolve interrupted sidecar replacements for the recovered manifest revision."""
        root = Path(record.root)
        target = root / FAST_RUNTIME_CACHE_FILE_NAME
        for temporary_root in self._fast_runtime_staging_directories(record, staging_parent):
            temporary = temporary_root / FAST_RUNTIME_CACHE_FILE_NAME
            if temporary.is_file():
                shutil.rmtree(temporary_root)
                continue
            try:
                self._read_fast_runtime_cache(record, target)
                target_is_valid = True
            except Exception:
                target_is_valid = False
            if not target_is_valid:
                backup = temporary_root / ".previous-fast-runtime-cache.pkl"
                try:
                    self._read_fast_runtime_cache(record, backup)
                    backup_is_valid = True
                except Exception:
                    backup_is_valid = False
                if backup_is_valid:
                    os.replace(backup, target)
                elif target.exists():
                    target.unlink()
            shutil.rmtree(temporary_root)

    def recover_recycled_fast_runtime_cache_staging(
        self,
        record: Any,
        library_root: Path,
    ) -> None:
        """Resolve this recycled record's exact sibling stages at the library root."""
        self.recover_fast_runtime_cache_staging(
            record,
            staging_parent=library_root,
        )

    def load_fast_runtime_cache(self, record: Any) -> FastRuntimeCache | None:
        cache_path = Path(record.root) / FAST_RUNTIME_CACHE_FILE_NAME
        if not cache_path.is_file():
            return None
        try:
            return self._read_fast_runtime_cache(record, cache_path)
        except Exception as exc:
            LOGGER.warning("Ignoring fast runtime cache %s: %s", cache_path, exc)
            return None

    def build_fast_runtime_cache(
        self,
        record: Any,
        geometry_profile: Mapping[str, Any] | None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> FastRuntimeCache:
        if self.fast_engine is None:
            raise OrientationClassifierError("FAST_CACHE_NOT_READY: fast engine is unavailable")
        return self._fast_build_engine().build_cache(
            record.front_images,
            record.back_images,
            geometry_profile=geometry_profile,
            library_revision=int(record.revision),
            model_fingerprint=self.model_fingerprint,
            progress_callback=progress_callback,
        )

    def load_template_cache(self, record: Any) -> TemplateCache | None:
        """Load a cache only when its signature and feature shapes still match."""
        cache_path = Path(record.root) / TEMPLATE_CACHE_FILE_NAME
        if not cache_path.is_file():
            return None
        try:
            with cache_path.open("rb") as stream:
                payload = pickle.load(stream)
            if not isinstance(payload, dict) or "signature" not in payload or "cache" not in payload:
                raise ValueError("cache payload is incomplete")
            expected_signature = self._template_cache_signature(record.front_images, record.back_images)
            if payload["signature"] != expected_signature:
                raise ValueError("template cache signature mismatch")
            cache = self._normalize_template_cache(payload["cache"])
            self._validate_cached_shape(cache, len(record.front_images), len(record.back_images))
            cache = self._cache_for_persistence(cache)
            if (
                getattr(self, "inference_mode", DEFAULT_INFERENCE_MODE) in {"legacy", "compare"}
                and hasattr(self, "model_gate")
                and hasattr(self, "global_predictor")
                and hasattr(self, "extractor")
                and not self._has_legacy_local_features(cache, record)
            ):
                cache = self._build_template_cache(
                    record.front_images,
                    record.back_images,
                    None,
                    include_local=True,
                    attach_fast=False,
                    library_revision=int(getattr(record, "revision", 1)),
                )
            cache = replace(
                cache,
                fast_template_signature=self._fast_template_signature(
                    record.front_images,
                    record.back_images,
                ),
            )
            fast_runtime = self.load_fast_runtime_cache(record)
            if fast_runtime is not None:
                cache = replace(cache, fast_runtime=fast_runtime)
            return cache
        except Exception as exc:
            LOGGER.warning("Ignoring template cache %s: %s", cache_path, exc)
            return None

    def set_template_cache(self, workpiece_id: str, cache: TemplateCache) -> None:
        self._template_caches[workpiece_id] = cache

    def get_template_cache(self, workpiece_id: str) -> TemplateCache | None:
        return self._template_caches.get(workpiece_id)

    @staticmethod
    def _geometry_direction(profile: dict[str, Any], label: str) -> dict[str, Any] | None:
        directions = profile.get("directions", {}) if isinstance(profile, dict) else {}
        direction = directions.get(label) if isinstance(directions, dict) else None
        return direction if isinstance(direction, dict) else None

    @staticmethod
    def _neutral_fill(image: np.ndarray, mask: np.ndarray) -> list[int]:
        valid = image[mask == 0]
        if valid.size == 0:
            valid = image.reshape(-1, 3)
        return [int(round(value)) for value in np.median(valid, axis=0)]

    def prepare_geometry_cache(
        self,
        workpiece_id: str,
        record: Any,
        profile: dict[str, Any],
        calibrator: Any | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        base_cache: TemplateCache | None = None,
    ) -> tuple[TemplateCache, dict[str, list[dict[str, Any]]]]:
        """Build a geometry-masked candidate cache without publishing it."""
        base = base_cache if base_cache is not None else self._template_caches.get(workpiece_id)
        if base is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        calibrator = calibrator or self.geometry_calibrator
        if calibrator is None:
            raise ValueError("geometry calibrator is not configured")
        raw_global = getattr(base, "raw_global_vectors", None) or base.global_vectors
        raw_local = getattr(base, "raw_local_features", None) or base.local_features
        fast_only = self.inference_mode == "fast_geometry"
        runtime_profile = materialize_runtime_profile(profile)
        candidate_profile = deepcopy(profile)
        candidate_globals: dict[str, np.ndarray] = {}
        candidate_locals: dict[str, list[dict[str, Any]]] = {}
        report: dict[str, list[dict[str, Any]]] = {}

        geometry_indices: dict[str, list[int]] = {}
        geometry_unsafe = False
        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            direction = self._geometry_direction(runtime_profile, label)
            candidate_direction = self._geometry_direction(candidate_profile, label)
            if direction is None or direction.get("anchor") is None or not direction.get("rules"):
                candidate_globals[label] = raw_global[label]
                candidate_locals[label] = (
                    [{} for _ in paths] if fast_only else raw_local[label]
                )
                geometry_indices[label] = list(range(len(paths)))
                report[label] = [{"status": "not_configured", "index": index}
                                 for index in range(len(paths))]
                continue
            # A copied rule has no trustworthy object-relative geometry until
            # it has been previewed on this direction.  Keep the raw feature
            # cache active instead of silently applying a stale mask from the
            # source direction/template.
            if any(rule.get("editor_state") == "needs_reseed"
                   for rule in direction.get("rules", [])
                   if isinstance(rule, Mapping)):
                geometry_unsafe = True
                candidate_globals[label] = raw_global[label]
                candidate_locals[label] = (
                    [{} for _ in paths] if fast_only else raw_local[label]
                )
                geometry_indices[label] = list(range(len(paths)))
                report[label] = [
                    {"index": index,
                     "template_id": f"{label}:{Path(path).name}",
                     "status": "needs_reseed",
                     "review_state": "included"}
                    for index, path in enumerate(paths)
                ]
                if progress_callback is not None:
                    for index in range(len(paths)):
                        progress_callback(label, index + 1, len(paths))
                continue
            reviews = direction.get("template_reviews", {})
            fits: list[tuple[int, np.ndarray, dict[str, Any], dict[str, Any]]] = []
            fills: list[list[int]] = []
            label_report: list[dict[str, Any]] = []
            for index, path in enumerate(paths):
                template_id = f"{label}:{Path(path).name}"
                review = reviews.get(template_id, {}) if isinstance(reviews, dict) else {}
                review_state = review.get("state", "included") if isinstance(review, dict) else "included"
                review_reason = str(review.get("reason", "")) if isinstance(review, dict) else ""
                if review_state == "excluded":
                    label_report.append({
                        "index": index,
                        "template_id": template_id,
                        "status": "excluded",
                        "review_state": review_state,
                        "review_reason": review_reason,
                    })
                    if progress_callback is not None:
                        progress_callback(label, index + 1, len(paths))
                    continue
                image = _read_image(Path(path))
                fit = calibrator.fit(image, direction)
                item = {
                    "index": index,
                    "template_id": template_id,
                    "status": fit.get("status", "low_confidence"),
                    "review_state": review_state,
                    "review_reason": review_reason,
                }
                item.update(self._serializable_geometry_fit(fit))
                if fit.get("status") == "active":
                    mask = fit["ignore_mask"]
                    fills.append(direction.get("fill_bgr") or self._neutral_fill(image, mask))
                    item.update({
                        "ignored_ratio": float(fit.get("ignored_ratio", np.count_nonzero(mask) / mask.size)),
                    })
                    if float(item["ignored_ratio"]) >= 0.55:
                        geometry_unsafe = True
                else:
                    item["reason_code"] = fit.get("reason_code", "boundary_not_found")
                    geometry_unsafe = True
                fits.append((index, image, fit, item))
                label_report.append(item)
            fill_bgr = direction.get("fill_bgr")
            if fill_bgr is None:
                fill_bgr = [int(round(value)) for value in np.median(np.asarray(fills), axis=0)] if fills else [0, 0, 0]
            if isinstance(candidate_direction, dict):
                candidate_direction["fill_bgr"] = list(fill_bgr)
            embeddings: list[np.ndarray] = []
            features: list[dict[str, Any]] = []
            included_indices: list[int] = []
            for index, image, fit, item in fits:
                included_indices.append(index)
                if fit.get("status") != "active":
                    embeddings.append(raw_global[label][index])
                    if fast_only:
                        features.append({})
                    else:
                        features.append(raw_local[label][index])
                        before = self._feature_keypoint_count(raw_local[label][index])
                        item.update({"keypoints_before": before, "keypoints_after": before, "remaining_ratio": 1.0})
                else:
                    mask = fit["ignore_mask"]
                    masked = apply_geometry_fit(image, fit, fill_bgr)
                    embedding = self.model_gate.run_background_step(
                        lambda: self._global_embedding(masked)
                    )
                    embeddings.append(embedding)
                    if fast_only:
                        features.append({})
                    else:
                        extracted = raw_local[label][index]
                        before = self._feature_keypoint_count(extracted)
                        feature_mask = geometry_feature_mask(fit)
                        filtered = (
                            filter_features_by_mask(extracted, feature_mask)
                            if np.any(feature_mask) else extracted
                        )
                        after = self._feature_keypoint_count(filtered)
                        features.append(filtered)
                        item.update({"keypoints_before": before, "keypoints_after": after,
                                     "remaining_ratio": float(after / before) if before else 1.0,
                                     "feature_mask_mode": "all_ignored_regions" if np.any(feature_mask) else "none"})
                if progress_callback is not None:
                    progress_callback(label, index + 1, len(paths))
            if not embeddings:
                raise ValueError(f"{label} has no included templates")
            candidate_globals[label] = np.stack(embeddings).astype(np.float32)
            candidate_locals[label] = features
            geometry_indices[label] = included_indices
            report[label] = label_report
        candidate = TemplateCache(
            global_vectors=candidate_globals,
            local_features=candidate_locals,
            raw_global_vectors={label: vectors.copy() for label, vectors in raw_global.items()},
            raw_local_features=raw_local,
            geometry_profile=candidate_profile,
            geometry_profile_revision=profile.get("profile_revision"),
            geometry_template_report=report,
            geometry_template_indices=geometry_indices,
            ignored_regions={},
            geometry_unsafe=geometry_unsafe,
            fast_template_signature=self._fast_template_signature(
                record.front_images,
                record.back_images,
            ),
        )
        if self.inference_mode in {"fast_geometry", "compare"} and self.fast_engine is not None:
            fast_runtime = self.build_fast_runtime_cache(
                record,
                candidate_profile,
                self._adapt_fast_progress(progress_callback),
            )
            candidate = replace(candidate, fast_runtime=fast_runtime)
        return candidate, report

    @staticmethod
    def _serializable_geometry_fit(fit: dict[str, Any]) -> dict[str, Any]:
        def clean(value: Any, key: str = "") -> Any:
            if key in {"ignore_mask", "contour"} or isinstance(value, np.ndarray):
                return None
            if isinstance(value, dict):
                return {
                    child_key: clean(child_value, str(child_key))
                    for child_key, child_value in value.items()
                    if child_key not in {"ignore_mask", "contour"}
                }
            if isinstance(value, list):
                return [clean(item) for item in value]
            return deepcopy(value)

        return clean(fit)

    @staticmethod
    def _cache_without_template(cache: TemplateCache, label: str, position: int) -> TemplateCache:
        """Return a validation-only cache with one candidate removed."""
        geometry_indices = deepcopy(cache.geometry_template_indices)
        if geometry_indices is None:
            geometry_indices = {
                side: list(range(len(cache.global_vectors.get(side, []))))
                for side in ("front", "back")
            }
        original_index = geometry_indices.get(label, [])[position]
        updated_indices = {side: list(indices) for side, indices in geometry_indices.items()}
        updated_indices[label].pop(position)

        vectors: dict[str, np.ndarray] = {}
        for side, values in cache.global_vectors.items():
            array = np.asarray(values)
            vectors[side] = np.delete(array, position, axis=0) if side == label else array
        features = {
            side: ([item for item_index, item in enumerate(values) if item_index != position]
                   if side == label else list(values))
            for side, values in cache.local_features.items()
        }

        raw_vectors = None
        if cache.raw_global_vectors is not None:
            raw_vectors = {}
            for side, values in cache.raw_global_vectors.items():
                array = np.asarray(values)
                raw_vectors[side] = (
                    np.delete(array, original_index, axis=0)
                    if side == label else array
                )
        raw_features = None
        if cache.raw_local_features is not None:
            raw_features = {
                side: ([item for item_index, item in enumerate(values)
                        if not (side == label and item_index == original_index)])
                for side, values in cache.raw_local_features.items()
            }
        return TemplateCache(
            global_vectors=vectors,
            local_features=features,
            raw_global_vectors=raw_vectors,
            raw_local_features=raw_features,
            ignored_regions=deepcopy(cache.ignored_regions),
            geometry_profile=deepcopy(cache.geometry_profile),
            geometry_profile_revision=cache.geometry_profile_revision,
            geometry_template_report=deepcopy(cache.geometry_template_report),
            geometry_template_indices=updated_indices,
            geometry_unsafe=getattr(cache, "geometry_unsafe", False),
        )

    def leave_one_out_report(self, record: Any, cache: TemplateCache) -> dict[str, Any]:
        """Re-extract every held-out template against a cache without itself."""
        correct_to_correct = 0
        correct_to_wrong = 0
        wrong_to_correct = 0
        wrong_to_wrong = 0
        evaluated = 0
        skipped = 0
        excluded = 0
        fit_failures: list[dict[str, Any]] = []
        excluded_templates: list[str] = []
        changed_predictions: list[dict[str, Any]] = []
        geometry_indices = cache.geometry_template_indices or {
            label: list(range(len(getattr(record, f"{label}_images"))))
            for label in ("front", "back")
        }
        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            side_vectors = cache.global_vectors.get(label)
            side_features = cache.local_features.get(label)
            if side_vectors is None or not isinstance(side_features, list):
                skipped += len(paths)
                continue
            side_indices = geometry_indices.get(label, list(range(len(paths))))
            report_items = ((cache.geometry_template_report or {}).get(label, [])
                            if isinstance(cache.geometry_template_report, dict) else [])
            excluded_indices = {
                int(item.get("index"))
                for item in report_items
                if isinstance(item, dict)
                and item.get("index") is not None
                and (item.get("status") == "excluded" or item.get("review_state") == "excluded")
            }
            for index, path in enumerate(paths):
                template_id = f"{label}:{Path(path).name}"
                if index not in side_indices:
                    if index in excluded_indices:
                        excluded += 1
                        excluded_templates.append(template_id)
                    else:
                        skipped += 1
                        fit_failures.append({"template_id": template_id, "reason": "template_excluded"})
                    continue
                candidate_index = side_indices.index(index)
                if candidate_index >= len(side_features) or len(side_features) <= 1:
                    skipped += 1
                    fit_failures.append({"template_id": template_id, "reason": "insufficient_same_side_templates"})
                    continue
                try:
                    image = _read_image(Path(path))
                    candidate_cache = self._cache_without_template(cache, label, candidate_index)
                    started = time.perf_counter()
                    if cache.geometry_profile is not None:
                        baseline_result, result = self.model_gate.run_background_step(
                            lambda: (
                                self._predict_baseline(image, candidate_cache, started),
                                self._predict_geometry(image, candidate_cache, started),
                            )
                        )
                    else:
                        result = self.model_gate.run_background_step(
                            lambda: self._predict_baseline(image, candidate_cache, started)
                        )
                        baseline_result = result
                    if cache.geometry_profile is not None:
                        geometry_mask = result.get("geometry_mask", {})
                        if geometry_mask.get("status") != "active":
                            skipped += 1
                            fit_failures.append({
                                "template_id": template_id,
                                "reason": geometry_mask.get("status", "geometry_fit_failed"),
                                "geometry_mask": geometry_mask,
                            })
                            continue
                    evaluated += 1
                    baseline_predicted = baseline_result.get("label")
                    candidate_predicted = result.get("label")
                    baseline_correct = baseline_predicted == label
                    candidate_correct = candidate_predicted == label
                    if baseline_correct and candidate_correct:
                        correct_to_correct += 1
                    elif baseline_correct:
                        correct_to_wrong += 1
                        candidate_global = result.get("global_prediction")
                        candidate_local = result.get("local_prediction")
                        decision_source = result.get("decision_source")
                        cause = (
                            "geometry_local_override"
                            if decision_source == "local_override"
                            and candidate_global == label
                            and candidate_local != label
                            else "geometry_global_shift"
                            if candidate_global != baseline_result.get("global_prediction")
                            else "geometry_fusion_change"
                        )
                        changed_predictions.append({
                            "template_id": template_id,
                            "expected": label,
                            "predicted": candidate_predicted,
                            "baseline_predicted": baseline_predicted,
                            "candidate_predicted": candidate_predicted,
                            "baseline_global_prediction": baseline_result.get("global_prediction"),
                            "baseline_local_prediction": baseline_result.get("local_prediction"),
                            "candidate_global_prediction": candidate_global,
                            "candidate_local_prediction": candidate_local,
                            "candidate_decision_source": decision_source,
                            "candidate_global_margin": float(result.get("global_margin", 0.0) or 0.0),
                            "candidate_local_margin": float(result.get("local_margin", 0.0) or 0.0),
                            "cause": cause,
                        })
                    elif candidate_correct:
                        wrong_to_correct += 1
                    else:
                        wrong_to_wrong += 1
                except (ImageUnreadableError, AttributeError, IndexError, KeyError,
                        TypeError, ValueError, RuntimeError) as exc:
                    skipped += 1
                    fit_failures.append({"template_id": template_id, "reason": str(exc)})
        return {
            "status": "completed" if skipped == 0 else "incomplete",
            "correct_to_correct": int(correct_to_correct),
            "correct_to_wrong": int(correct_to_wrong),
            "wrong_to_correct": int(wrong_to_correct),
            "wrong_to_wrong": int(wrong_to_wrong),
            "evaluated": int(evaluated),
            "skipped": int(skipped),
            "excluded": int(excluded),
            "excluded_templates": excluded_templates,
            "fit_failures": fit_failures,
            "changed_predictions": changed_predictions,
        }

    @staticmethod
    def _feature_keypoint_count(features: dict[str, Any]) -> int:
        keypoints = features.get("keypoints")
        if keypoints is None:
            return 0
        shape = tuple(getattr(keypoints, "shape", ()))
        if len(shape) == 3 and shape[0] == 1:
            return int(shape[1])
        if len(shape) >= 1:
            return int(shape[0])
        return 0

    def prepare_template_masks(
        self,
        workpiece_id: str,
        ignored_regions: dict[str, Any],
        *,
        base_cache: TemplateCache | None = None,
    ) -> tuple[TemplateCache, dict[str, list[dict[str, float | int]]]]:
        """Build a filtered candidate cache without publishing it."""
        cache = base_cache if base_cache is not None else self._template_caches.get(workpiece_id)
        if cache is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        raw = cache.raw_local_features or cache.local_features
        filtered: dict[str, list[dict[str, Any]]] = {}
        statistics: dict[str, list[dict[str, float | int]]] = {}
        for label, templates in raw.items():
            label_regions = ignored_regions.get(label, [])
            if label_regions and isinstance(label_regions[0], dict):
                label_regions = [label_regions for _ in templates]
            filtered[label] = []
            statistics[label] = []
            for index, features in enumerate(templates):
                regions = label_regions[index] if index < len(label_regions) else []
                before = self._feature_keypoint_count(features)
                candidate = filter_template_features(features, regions)
                after = self._feature_keypoint_count(candidate)
                filtered[label].append(candidate)
                statistics[label].append({
                    "keypoints_before": before,
                    "keypoints_after": after,
                    "remaining_ratio": float(after / before) if before else 1.0,
                })
        return TemplateCache(
            global_vectors=cache.global_vectors,
            local_features=filtered,
            raw_local_features=raw,
            ignored_regions=deepcopy(ignored_regions),
        ), statistics

    def set_template_masks(self, workpiece_id: str, ignored_regions: dict[str, Any]) -> None:
        candidate, _ = self.prepare_template_masks(workpiece_id, ignored_regions)
        self._template_caches[workpiece_id] = candidate

    def project_region_between_templates(self, source_path: Path, target_path: Path,
                                         region: dict[str, float]) -> dict[str, float] | None:
        """Project one template-native region through existing ALIKED/LightGlue matches."""
        try:
            def project():
                import torch

                # ALIKED returns inference tensors. LightGlue must run in the same
                # inference context, otherwise PyTorch rejects its Linear layers
                # for attempting to save inference tensors for backward.
                with torch.inference_mode():
                    source_image = _read_image(Path(source_path))
                    target_image = _read_image(Path(target_path))
                    source_features = self._extract_features(
                        source_image, self.extractor, self.device, roi_ratio=ROI_RATIO
                    )
                    target_features = self._extract_features(
                        target_image, self.extractor, self.device, roi_ratio=ROI_RATIO
                    )
                    match_data = self.matcher({"image0": source_features, "image1": target_features})
                    matches = match_data["matches"][0].detach().cpu().numpy()
                    if len(matches) < 4:
                        return None
                    source_points = source_features["keypoints"][0][matches[:, 0]].detach().cpu().numpy()
                    target_points = target_features["keypoints"][0][matches[:, 1]].detach().cpu().numpy()
                    return project_region(region, source_points, target_points)

            return self.model_gate.run_background_step(project)
        except PropagationModelError:
            raise
        except RuntimeError as exc:
            if _is_fatal_local_model_error(exc):
                raise PropagationModelError("局部特征模型不可用，请检查后端日志后重试") from exc
            raise
        except (ValueError, TypeError, IndexError, KeyError, cv2.error):
            return None

    def remove_template_cache(self, workpiece_id: str) -> None:
        self._template_caches.pop(workpiece_id, None)

    @staticmethod
    def _local_expansion_reason(
        global_scores: dict[str, float], local_scores: dict[str, float]
    ) -> str:
        local_prediction, _ = _decide_local_label(local_scores)
        if local_prediction == "uncertain":
            if max(local_scores.values()) < LOCAL_MIN_SCORE:
                return "local_uncertain"
            return "local_margin_low"
        global_prediction = max(global_scores, key=global_scores.get)
        if local_prediction != global_prediction:
            return "local_conflict"
        return "local_uncertain"

    @staticmethod
    def _validate_local_cache_alignment(
        global_vectors: Mapping[str, np.ndarray],
        local_features: Mapping[str, Sequence[dict[str, Any]]],
    ) -> dict[str, np.ndarray]:
        normalized_vectors: dict[str, np.ndarray] = {}
        for label in ("front", "back"):
            if label not in global_vectors or label not in local_features:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    "missing vectors or local features"
                )
            try:
                vectors = np.asarray(global_vectors[label], dtype=np.float32)
                candidate_count = len(local_features[label])
            except (TypeError, ValueError):
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    "invalid vectors or local features"
                ) from None
            if vectors.ndim != 2:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    "global vectors must be a 2D array"
                )
            if vectors.shape[0] == 0 or candidate_count == 0:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    "at least one local template is required"
                )
            if vectors.shape[0] != candidate_count:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    f"{vectors.shape[0]} global vectors != "
                    f"{candidate_count} local templates"
                )
            normalized_vectors[label] = vectors
        return normalized_vectors

    def _rank_local_candidates(
        self,
        global_vectors: Mapping[str, np.ndarray],
        query_embeddings: Mapping[str, np.ndarray],
        local_features: Mapping[str, Sequence[dict[str, Any]]],
    ) -> dict[str, list[int]]:
        vectors_by_label = self._validate_local_cache_alignment(
            global_vectors, local_features
        )
        rankings = {}
        for label in ("front", "back"):
            vectors = vectors_by_label[label]
            query = np.asarray(query_embeddings[label], dtype=np.float32)
            similarities = np.asarray(vectors @ query, dtype=np.float32).reshape(-1)
            rankings[label] = np.argsort(-similarities, kind="stable").tolist()
        return rankings

    def _search_local(
        self,
        *,
        query_features_by_label: Mapping[str, dict[str, Any]],
        global_vectors: Mapping[str, np.ndarray],
        query_embeddings: Mapping[str, np.ndarray],
        local_features: Mapping[str, Sequence[dict[str, Any]]],
        image_shape: tuple[int, int],
        global_scores: dict[str, float],
    ) -> LocalSearchResult:
        for label in ("front", "back"):
            if label not in query_embeddings:
                raise OrientationClassifierError(
                    f"local search input error: missing query embedding for {label}"
                )
            if label not in query_features_by_label:
                raise OrientationClassifierError(
                    f"local search input error: missing query local features for {label}"
                )
            if label not in global_scores:
                raise OrientationClassifierError(
                    f"local search input error: missing global score for {label}"
                )
        rankings = self._rank_local_candidates(
            global_vectors, query_embeddings, local_features
        )
        available_counts = {label: len(rankings[label]) for label in ("front", "back")}
        score_cache: dict[str, dict[int, float]] = {"front": {}, "back": {}}
        query_features = {
            label: _move_tensors(query_features_by_label[label], self.device)
            for label in ("front", "back")
        }
        matching_ms = 0.0
        trace: dict[str, object] = {"ranked_indices": rankings, "stages": []}

        def scores() -> dict[str, float]:
            return {
                label: max(score_cache[label].values())
                for label in ("front", "back")
            }

        def score_to(limit: int) -> None:
            nonlocal matching_ms
            for label in ("front", "back"):
                for index in rankings[label][:min(limit, available_counts[label])]:
                    if index in score_cache[label]:
                        continue
                    template_features = _move_tensors(
                        local_features[label][index], self.device
                    )
                    started = time.perf_counter()
                    score_cache[label][index] = float(self._score_feature_pair(
                        query_features[label],
                        template_features,
                        image_shape,
                        self.matcher,
                    )["score"])
                    matching_ms += (time.perf_counter() - started) * 1000.0

        def record_stage(stage: str) -> tuple[dict[str, float], dict[str, object]]:
            current_scores = scores()
            fusion = self._fuse_scores(global_scores, current_scores, time.perf_counter())
            trace["stages"].append({
                "stage": stage,
                "scores": dict(current_scores),
                "fusion": fusion,
            })
            return current_scores, fusion

        ordered_globals = sorted(global_scores, key=global_scores.get, reverse=True)
        global_margin = float(
            global_scores[ordered_globals[0]] - global_scores[ordered_globals[1]]
        )
        low_global_margin = (
            global_margin <= GLOBAL_MARGIN_THRESHOLD
            or math.isclose(
                global_margin,
                GLOBAL_MARGIN_THRESHOLD,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        global_margin_gate = (
            "adaptive_low_margin"
            if self.local_search_mode == "adaptive" and low_global_margin
            else "preserve_review_semantics"
        )
        direct_full = (
            self.local_search_mode == "exhaustive"
            or not low_global_margin
        )
        expanded_because: str | None = (
            "exhaustive_mode" if self.local_search_mode == "exhaustive"
            else "preserve_review_semantics" if direct_full else None
        )

        if direct_full:
            score_to(max(available_counts.values()))
            stage = "full"
            current_scores, _ = record_stage(stage)
        else:
            stage, limit = LOCAL_SEARCH_STAGE_LIMITS[0]
            score_to(limit)
            current_scores, fusion = record_stage(stage)
            if bool(fusion["needs_review"]) and not all(
                len(score_cache[label]) == available_counts[label]
                for label in ("front", "back")
            ):
                expanded_because = self._local_expansion_reason(global_scores, current_scores)
                stage, limit = LOCAL_SEARCH_STAGE_LIMITS[1]
                score_to(limit)
                current_scores, fusion = record_stage(stage)
                if bool(fusion["needs_review"]) and not all(
                    len(score_cache[label]) == available_counts[label]
                    for label in ("front", "back")
                ):
                    expanded_because = self._local_expansion_reason(global_scores, current_scores)
                    score_to(max(available_counts.values()))
                    stage = "full"
                    current_scores, _ = record_stage(stage)

        diagnostics = {
            "stage": stage,
            "mode": self.local_search_mode,
            "matched_counts": {
                label: len(score_cache[label]) for label in ("front", "back")
            },
            "available_counts": available_counts,
            "expanded_because": expanded_because,
            "exhaustive": all(
                len(score_cache[label]) == available_counts[label]
                for label in ("front", "back")
            ),
            "global_margin_gate": global_margin_gate,
        }
        return LocalSearchResult(current_scores, diagnostics, matching_ms, trace)

    @staticmethod
    def _fuse_scores(global_scores: dict[str, float], local_scores: dict[str, float],
                     started: float, *, geometry_mask: dict[str, Any] | None = None) -> dict[str, object]:
        ordered = sorted(global_scores, key=global_scores.get, reverse=True)
        global_prediction = ordered[0]
        global_margin = float(global_scores[ordered[0]] - global_scores[ordered[1]])
        local_prediction, local_margin = _decide_local_label(local_scores)
        local_is_decisive = local_prediction != "uncertain"
        use_local = (
            global_margin <= GLOBAL_MARGIN_THRESHOLD
            and local_is_decisive
            and local_margin >= LOCAL_OVERRIDE_MARGIN
        )
        label = local_prediction if use_local else global_prediction
        decision_source = "local_override" if use_local else "global"
        needs_review = (
            (global_margin <= GLOBAL_MARGIN_THRESHOLD and not local_is_decisive)
            or (local_is_decisive and local_prediction != global_prediction)
        )
        result: dict[str, object] = {
            "label": label,
            "global_prediction": global_prediction,
            "global_scores": {key: float(value) for key, value in global_scores.items()},
            "global_margin": global_margin,
            "local_prediction": local_prediction,
            "local_scores": {key: float(value) for key, value in local_scores.items()},
            "local_margin": float(local_margin),
            "decision_source": decision_source,
            "needs_review": needs_review,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        if geometry_mask is not None:
            result["geometry_mask"] = geometry_mask
            result["needs_review"] = bool(result["needs_review"] or geometry_mask.get("needs_review", False))
        return result

    def _predict_baseline(
        self,
        image: np.ndarray,
        cache: TemplateCache,
        started: float,
        geometry_mask: dict[str, Any] | None = None,
        timings: dict[str, float] | None = None,
    ) -> dict[str, object]:
        raw_globals = getattr(cache, "raw_global_vectors", None) or cache.global_vectors
        raw_locals = getattr(cache, "raw_local_features", None) or cache.local_features
        self._validate_local_cache_alignment(raw_globals, raw_locals)
        global_started = time.perf_counter()
        query_embedding = self._global_embedding(image)
        global_prediction, global_scores, _ = classify_embedding(query_embedding, raw_globals)
        if timings is not None:
            timings["global_batch"] += (time.perf_counter() - global_started) * 1000.0
        del global_prediction
        local_started = time.perf_counter()
        with self._inference_lock:
            query_features = self._extract_features(
                image, self.extractor, self.device, roi_ratio=ROI_RATIO
            )
        if timings is not None:
            timings["local_features"] += (time.perf_counter() - local_started) * 1000.0
        search = self._search_local(
            query_features_by_label={"front": query_features, "back": query_features},
            global_vectors=raw_globals,
            query_embeddings={"front": query_embedding, "back": query_embedding},
            local_features=raw_locals,
            image_shape=image.shape[:2],
            global_scores=global_scores,
        )
        if timings is not None:
            timings["local_matching"] += search.matching_ms
        fusion_started = time.perf_counter()
        result = self._fuse_scores(
            global_scores, search.scores, started, geometry_mask=geometry_mask
        )
        result["local_search"] = search.diagnostics
        if timings is not None:
            timings["fusion"] += (time.perf_counter() - fusion_started) * 1000.0
            if geometry_mask is not None:
                geometry_mask["timings_ms"] = timings
        return result

    @staticmethod
    def _geometry_report(fit: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in fit.items() if key != "ignore_mask"}
        result["rules"] = [
            {key: value for key, value in rule.items() if key != "ignore_mask"}
            for rule in fit.get("rules", [])
        ]
        return result

    @staticmethod
    def _geometry_timings() -> dict[str, float]:
        return {
            "fit_context": 0.0,
            "fit_directions": 0.0,
            "mask_build": 0.0,
            "global_batch": 0.0,
            "local_features": 0.0,
            "local_matching": 0.0,
            "fusion": 0.0,
        }

    def _prepare_geometry_queries(
        self,
        image: np.ndarray,
        profile: Mapping[str, Any],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        """Build front/back query variants while fitting image contours once."""
        calibrator = self.geometry_calibrator
        if calibrator is None:
            return [], {
                "status": "unavailable",
                "fallback_reason": "geometry_calibrator_unavailable",
                "directions": {},
                "timings_ms": self._geometry_timings(),
            }
        timings = self._geometry_timings()
        directions = profile.get("directions", {}) if isinstance(profile, Mapping) else {}
        configured = [
            label for label in ("front", "back")
            if isinstance(directions.get(label), Mapping)
            and directions[label].get("anchor") is not None
            and directions[label].get("rules")
        ]
        context = None
        if configured:
            context_started = time.perf_counter()
            context = calibrator.prepare_context(image)
            timings["fit_context"] = (time.perf_counter() - context_started) * 1000.0

        processed: list[np.ndarray] = []
        reports: dict[str, Any] = {}
        fits: dict[str, Any] = {}
        for label in ("front", "back"):
            direction = directions.get(label) if isinstance(directions, Mapping) else None
            if not isinstance(direction, Mapping) or label not in configured:
                processed.append(image.copy())
                reports[label] = {"status": "not_configured"}
                continue
            fit_started = time.perf_counter()
            fit = calibrator.fit(image, direction, context=context)
            timings["fit_directions"] += (time.perf_counter() - fit_started) * 1000.0
            reports[label] = self._geometry_report(fit)
            fits[label] = fit
            if fit.get("status") != "active":
                return [], {
                    "status": fit.get("status", "low_confidence"),
                    "fallback_reason": fit.get("reason_code", "boundary_not_found"),
                    "directions": reports,
                    "timings_ms": timings,
                }
            if float(fit.get("ignored_ratio", 0.0) or 0.0) >= 0.55:
                reports[label]["reason_code"] = "geometry_mask_too_large"
                return [], {
                    "status": "low_confidence",
                    "fallback_reason": "geometry_mask_too_large",
                    "directions": reports,
                    "timings_ms": timings,
                }
            mask_started = time.perf_counter()
            mask = fit["ignore_mask"]
            fill = direction.get("fill_bgr") or self._neutral_fill(image, mask)
            processed.append(apply_geometry_fit(image, fit, fill))
            timings["mask_build"] += (time.perf_counter() - mask_started) * 1000.0
        return processed, {
            "status": "active",
            "directions": reports,
            "fits": fits,
            "timings_ms": timings,
        }

    def _predict_geometry(self, image: np.ndarray, cache: TemplateCache,
                          started: float) -> dict[str, object]:
        profile = materialize_runtime_profile(cache.geometry_profile or {})
        calibrator = self.geometry_calibrator
        raw_globals = getattr(cache, "raw_global_vectors", None) or cache.global_vectors
        raw_locals = getattr(cache, "raw_local_features", None) or cache.local_features
        if calibrator is None:
            timings = self._geometry_timings()
            return self._predict_baseline(image, cache, started, {
                "status": "unavailable", "needs_review": True,
                "profile_revision": cache.geometry_profile_revision,
                "reason_code": "GEOMETRY_FALLBACK_TO_BASELINE",
                "fallback_reason": "geometry_calibrator_unavailable",
                "fallback": "raw_baseline",
            }, timings)
        directions = profile.get("directions", {}) if isinstance(profile, Mapping) else {}
        needs_reseed = [
            label for label in ("front", "back")
            if isinstance(directions.get(label), Mapping)
            and any(rule.get("editor_state") == "needs_reseed"
                    for rule in directions[label].get("rules", [])
                    if isinstance(rule, Mapping))
        ]
        if needs_reseed:
            reports = {
                label: {
                    "status": "needs_reseed" if label in needs_reseed else "not_configured",
                    "needs_review": label in needs_reseed,
                    "reason_code": "preview_required" if label in needs_reseed else None,
                }
                for label in ("front", "back")
            }
            timings = self._geometry_timings()
            return self._predict_baseline(image, TemplateCache(
                global_vectors=raw_globals,
                local_features=raw_locals,
                raw_global_vectors=raw_globals,
                raw_local_features=raw_locals,
            ), started, {
                "status": "needs_reseed",
                "needs_review": True,
                "profile_revision": cache.geometry_profile_revision,
                "directions": reports,
                "reason_code": "GEOMETRY_FALLBACK_TO_BASELINE",
                "fallback_reason": "preview_required",
                "fallback": "raw_baseline",
            }, timings)
        if getattr(cache, "geometry_unsafe", False):
            timings = self._geometry_timings()
            return self._predict_baseline(image, TemplateCache(
                global_vectors=raw_globals,
                local_features=raw_locals,
                raw_global_vectors=raw_globals,
                raw_local_features=raw_locals,
            ), started, {
                "status": "unsafe_template_geometry",
                "needs_review": True,
                "profile_revision": cache.geometry_profile_revision,
                "reason_code": "GEOMETRY_FALLBACK_TO_BASELINE",
                "fallback_reason": "template_geometry_validation_failed",
                "fallback": "raw_baseline",
            }, timings)

        processed, prepared = self._prepare_geometry_queries(image, profile)
        timings = prepared["timings_ms"]
        reports = prepared.get("directions", {})
        if prepared.get("status") != "active":
            geometry_mask = {
                "status": prepared.get("status", "low_confidence"),
                "needs_review": True,
                "profile_revision": cache.geometry_profile_revision,
                "directions": reports,
                "reason_code": "GEOMETRY_FALLBACK_TO_BASELINE",
                "fallback_reason": prepared.get("fallback_reason", "geometry_fit_failed"),
                "fallback": "raw_baseline",
            }
            return self._predict_baseline(image, TemplateCache(
                global_vectors=raw_globals,
                local_features=raw_locals,
                raw_global_vectors=raw_globals,
                raw_local_features=raw_locals,
            ), started, geometry_mask, timings)

        self._validate_local_cache_alignment(
            cache.global_vectors, cache.local_features
        )
        global_started = time.perf_counter()
        batch_embeddings = self._global_embeddings(processed)
        timings["global_batch"] = (time.perf_counter() - global_started) * 1000.0
        query_embeddings = dict(zip(("front", "back"), batch_embeddings))
        local_started = time.perf_counter()
        fits = prepared.get("fits", {})
        extracted = self._extract_local(image)
        query_features: dict[str, dict[str, Any]] = {}
        for label in ("front", "back"):
            fit = fits.get(label)
            if not isinstance(fit, Mapping):
                query_features[label] = extracted
                continue
            feature_mask = geometry_feature_mask(fit)
            query_features[label] = (
                filter_features_by_mask(extracted, feature_mask)
                if np.any(feature_mask) else extracted
            )
            reports[label]["feature_mask_mode"] = (
                "all_ignored_regions" if np.any(feature_mask) else "none"
            )
        timings["local_features"] = (time.perf_counter() - local_started) * 1000.0
        global_scores = {
            label: float(np.max(cache.global_vectors[label] @ query_embeddings[label]))
            for label in ("front", "back")
        }
        search = self._search_local(
            query_features_by_label=query_features,
            global_vectors=cache.global_vectors,
            query_embeddings=query_embeddings,
            local_features=cache.local_features,
            image_shape=image.shape[:2],
            global_scores=global_scores,
        )
        timings["local_matching"] = search.matching_ms
        geometry_mask = {
            "status": "active",
            "needs_review": False,
            "profile_revision": cache.geometry_profile_revision,
            "directions": reports,
            "ignored_ratio": {
                label: float(reports[label].get("ignored_ratio", 0.0)) for label in ("front", "back")
            },
            "timings_ms": timings,
        }
        fusion_started = time.perf_counter()
        result = self._fuse_scores(
            global_scores, search.scores, started, geometry_mask=geometry_mask
        )
        result["local_search"] = search.diagnostics
        timings["fusion"] = (time.perf_counter() - fusion_started) * 1000.0
        return result

    def predict_with_cache(
        self,
        cache: TemplateCache,
        image_path: Path | np.ndarray,
        *,
        library_revision: int | None = None,
    ) -> dict[str, object]:
        """Predict exclusively from a caller-owned immutable cache snapshot."""
        if self.inference_mode == "fast_geometry":
            return self.predict_fast_with_cache(
                cache,
                image_path,
                library_revision=library_revision,
            )

        def run() -> dict[str, object]:
            started = time.perf_counter()
            image = (
                image_path
                if isinstance(image_path, np.ndarray)
                else _read_image(Path(image_path))
            )
            result = (
                self._predict_geometry(image, cache, started)
                if getattr(cache, "geometry_profile", None) is not None
                else self._predict_baseline(image, cache, started)
            )
            if library_revision is not None:
                result["library_revision"] = int(library_revision)
            return result

        return self.model_gate.run_online(run)

    def predict_fast_with_cache(
        self,
        cache: TemplateCache,
        image_path: Path | np.ndarray,
        *,
        library_revision: int | None = None,
    ) -> dict[str, object]:
        """Run only the lightweight geometry/Ridge path for a cache snapshot."""
        def run() -> dict[str, object]:
            started = time.perf_counter()
            runtime = getattr(cache, "fast_runtime", None)
            if runtime is None or self.fast_engine is None:
                raise OrientationClassifierError(
                    "FAST_CACHE_NOT_READY: fast runtime cache is unavailable"
                )
            if runtime.model_fingerprint != self.model_fingerprint:
                raise OrientationClassifierError(
                    "FAST_CACHE_REVISION_MISMATCH: model fingerprint differs"
                )
            if library_revision is not None and runtime.library_revision != int(library_revision):
                raise OrientationClassifierError(
                    "FAST_CACHE_REVISION_MISMATCH: library revision differs"
                )
            geometry_revision = getattr(cache, "geometry_profile_revision", None)
            if runtime.geometry_profile_revision != geometry_revision:
                raise OrientationClassifierError(
                    "FAST_CACHE_REVISION_MISMATCH: geometry revision differs"
                )
            if not self._fast_template_signatures_match(
                runtime.template_signature,
                getattr(cache, "fast_template_signature", None),
            ):
                raise OrientationClassifierError(
                    "FAST_CACHE_REVISION_MISMATCH: template content differs"
                )
            try:
                FastOrientationEngine._validate_cache(runtime)
            except ValueError as exc:
                raise OrientationClassifierError(str(exc)) from exc

            if isinstance(image_path, np.ndarray):
                image = image_path
                decode_ms = 0.0
            else:
                decode_started = time.perf_counter()
                image = _read_image(Path(image_path))
                decode_ms = (time.perf_counter() - decode_started) * 1000.0
            try:
                result = self.fast_engine.predict(image, runtime)
            except ValueError as exc:
                message = str(exc)
                if "FAST_" in message:
                    raise OrientationClassifierError(message) from exc
                raise
            timings = dict(result.get("timings_ms") or {})
            timings["decode"] = decode_ms
            total_ms = (time.perf_counter() - started) * 1000.0
            timings["total"] = total_ms
            result["timings_ms"] = timings
            result["elapsed_ms"] = total_ms
            if library_revision is not None:
                result["library_revision"] = int(library_revision)
            return result

        return self.model_gate.run_online(run)

    def predict(self, workpiece_id: str, image_path: Path) -> dict[str, object]:
        cache = self._template_caches.get(workpiece_id)
        if cache is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        return self.predict_with_cache(cache, image_path)
