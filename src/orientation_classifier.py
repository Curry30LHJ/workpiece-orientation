"""Reusable PP-ShiTuV2 and ALIKED/LightGlue orientation classifier."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path
import pickle
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
from src.geometry_calibration import apply_ignore_mask
from src.geometry_calibration import filter_features_by_mask


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


LOGGER = logging.getLogger(__name__)


class OrientationClassifierError(RuntimeError):
    """Base error for service-facing classifier failures."""


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

@dataclass(frozen=True)
class LocalSearchResult:
    scores: dict[str, float]
    diagnostics: dict[str, object]
    matching_ms: float
    trace: dict[str, object]


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
        local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
    ) -> None:
        self.global_predictor = global_predictor
        self.extractor = extractor
        self.matcher = matcher
        self.device = device
        self._extract_features = extract_features_fn or _default_extract_features
        self._score_feature_pair = score_feature_pair_fn or _default_score_feature_pair
        self.geometry_calibrator = geometry_calibrator
        self.local_search_mode = _validate_local_search_mode(local_search_mode)
        self._inference_lock = threading.RLock()
        self._template_caches: dict[str, TemplateCache] = {}

    @classmethod
    def load(
        cls,
        project_root: Path,
        model_dir: Path,
        *,
        local_search_mode: str = DEFAULT_LOCAL_SEARCH_MODE,
    ) -> "OrientationClassifier":
        """Load the production Paddle and Torch models lazily at service startup."""
        # On Windows, Paddle and PyTorch can expose incompatible DLLs when Paddle
        # is imported first. Load the Torch/ALIKED stack before PaddleClas.
        from src.aliked_lightglue_matcher import build_models
        from paddleclas.deploy.python.predict_rec import RecPredictor
        from paddleclas.deploy.utils import config as paddle_config

        config_path = project_root / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml"
        config = paddle_config.get_config(str(config_path), show=False)
        config.Global.rec_inference_model_dir = str(model_dir)
        config.Global.use_gpu = True
        config.Global.enable_mkldnn = False
        config.Global.enable_benchmark = False
        config.Global.gpu_mem = 1024
        global_predictor = RecPredictor(config)
        extractor, matcher, device = build_models(MAX_NUM_KEYPOINTS)
        from src.geometry_calibration import GeometryCalibrator

        return cls(global_predictor, extractor, matcher, device,
                   geometry_calibrator=GeometryCalibrator(),
                   local_search_mode=local_search_mode)

    def _global_embedding(self, image: np.ndarray) -> np.ndarray:
        with self._inference_lock:
            embedding = self.global_predictor.predict([image[:, :, ::-1]])[0]
        return np.asarray(embedding, dtype=np.float32)

    def _extract_local(self, image: np.ndarray) -> dict:
        with self._inference_lock:
            features = self._extract_features(image, self.extractor, self.device, roi_ratio=ROI_RATIO)
        return _to_cpu(features)

    def build_template_cache(
        self,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
        progress_callback: Callable[[str, int, int], None] | None = None,
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
                embeddings.append(self._global_embedding(image))
                features.append(self._extract_local(image))
                if progress_callback is not None:
                    progress_callback(label, completed, total)
            global_vectors[label] = np.stack(embeddings).astype(np.float32)
            local_features[label] = features
        return TemplateCache(
            global_vectors=global_vectors,
            local_features=local_features,
            raw_global_vectors={label: vectors.copy() for label, vectors in global_vectors.items()},
            raw_local_features=local_features,
            ignored_regions={},
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
            features = cache.local_features.get(label)
            if not isinstance(features, list) or len(features) != expected:
                raise ValueError(f"cache local feature count mismatch for {label}")
            if cache.raw_local_features is not None:
                raw_features = cache.raw_local_features.get(label)
                if not isinstance(raw_features, list) or len(raw_features) != expected:
                    raise ValueError(f"cache raw feature count mismatch for {label}")

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
            cache = payload["cache"]
            self._validate_cached_shape(cache, len(record.front_images), len(record.back_images))
            return self._cache_for_persistence(cache)
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
    ) -> tuple[TemplateCache, dict[str, list[dict[str, Any]]]]:
        """Build a geometry-masked candidate cache without publishing it."""
        base = self._template_caches.get(workpiece_id)
        if base is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        calibrator = calibrator or self.geometry_calibrator
        if calibrator is None:
            raise ValueError("geometry calibrator is not configured")
        raw_global = getattr(base, "raw_global_vectors", None) or base.global_vectors
        raw_local = getattr(base, "raw_local_features", None) or base.local_features
        candidate_profile = deepcopy(profile)
        candidate_globals: dict[str, np.ndarray] = {}
        candidate_locals: dict[str, list[dict[str, Any]]] = {}
        report: dict[str, list[dict[str, Any]]] = {}

        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            direction = self._geometry_direction(profile, label)
            candidate_direction = self._geometry_direction(candidate_profile, label)
            if direction is None or direction.get("anchor") is None or not direction.get("rules"):
                candidate_globals[label] = raw_global[label]
                candidate_locals[label] = raw_local[label]
                report[label] = [{"status": "not_configured", "index": index}
                                 for index in range(len(paths))]
                continue
            fits: list[tuple[np.ndarray, dict[str, Any]]] = []
            fills: list[list[int]] = []
            label_report: list[dict[str, Any]] = []
            for index, path in enumerate(paths):
                image = _read_image(Path(path))
                fit = calibrator.fit(image, direction)
                fits.append((image, fit))
                item = {"index": index, "status": fit.get("status", "low_confidence")}
                if fit.get("status") == "active":
                    mask = fit["ignore_mask"]
                    fills.append(direction.get("fill_bgr") or self._neutral_fill(image, mask))
                    item.update({
                        "ignored_ratio": float(fit.get("ignored_ratio", np.count_nonzero(mask) / mask.size)),
                    })
                else:
                    item["reason_code"] = fit.get("reason_code", "boundary_not_found")
                label_report.append(item)
            fill_bgr = direction.get("fill_bgr")
            if fill_bgr is None:
                fill_bgr = [int(round(value)) for value in np.median(np.asarray(fills), axis=0)] if fills else [0, 0, 0]
            if isinstance(candidate_direction, dict):
                candidate_direction["fill_bgr"] = list(fill_bgr)
            embeddings: list[np.ndarray] = []
            features: list[dict[str, Any]] = []
            for index, ((image, fit), item) in enumerate(zip(fits, label_report), start=0):
                if fit.get("status") != "active":
                    embeddings.append(raw_global[label][index])
                    features.append(raw_local[label][index])
                    before = self._feature_keypoint_count(raw_local[label][index])
                    item.update({"keypoints_before": before, "keypoints_after": before, "remaining_ratio": 1.0})
                else:
                    mask = fit["ignore_mask"]
                    masked = apply_ignore_mask(image, mask, fill_bgr)
                    embeddings.append(self._global_embedding(masked))
                    extracted = self._extract_local(masked)
                    before = self._feature_keypoint_count(extracted)
                    filtered = filter_features_by_mask(extracted, mask)
                    after = self._feature_keypoint_count(filtered)
                    features.append(filtered)
                    item.update({"keypoints_before": before, "keypoints_after": after,
                                 "remaining_ratio": float(after / before) if before else 1.0})
                if progress_callback is not None:
                    progress_callback(label, index + 1, len(paths))
            candidate_globals[label] = np.stack(embeddings).astype(np.float32)
            candidate_locals[label] = features
            report[label] = label_report
        return TemplateCache(
            global_vectors=candidate_globals,
            local_features=candidate_locals,
            raw_global_vectors={label: vectors.copy() for label, vectors in raw_global.items()},
            raw_local_features=raw_local,
            geometry_profile=candidate_profile,
            geometry_profile_revision=profile.get("profile_revision"),
            geometry_template_report=report,
            ignored_regions={},
        ), report

    def leave_one_out_report(self, record: Any, cache: TemplateCache) -> dict[str, Any]:
        """Score each template against the other templates without re-extraction.

        This is intentionally a small validation pass over the already-built
        candidate cache.  It never calls Paddle, ALIKED, or LightGlue feature
        extraction; only the cached vectors/features are compared.  A side
        with one template cannot provide a leave-one-out same-side reference,
        so that sample is reported as skipped instead of being called a
        regression.
        """
        wrong = 0
        evaluated = 0
        skipped = 0
        vectors = cache.global_vectors
        features = cache.local_features
        for label, paths in (("front", record.front_images), ("back", record.back_images)):
            side_vectors = vectors.get(label)
            side_features = features.get(label)
            if side_vectors is None or not isinstance(side_features, list):
                skipped += len(paths)
                continue
            for index, path in enumerate(paths):
                if index >= len(side_features) or len(side_features) <= 1:
                    skipped += 1
                    continue
                try:
                    image = read_color_image(Path(path))
                    if image is None:
                        skipped += 1
                        continue
                    query_vector = np.asarray(side_vectors[index], dtype=np.float32)
                    global_scores: dict[str, float] = {}
                    local_scores: dict[str, float] = {}
                    for candidate_label, candidate_vectors in vectors.items():
                        candidate_array = np.asarray(candidate_vectors, dtype=np.float32)
                        if candidate_label == label:
                            candidate_array = np.delete(candidate_array, index, axis=0)
                        if candidate_array.size == 0:
                            continue
                        global_scores[candidate_label] = float(np.max(candidate_array @ query_vector))

                        candidate_features = features.get(candidate_label, [])
                        if candidate_label == label:
                            candidate_features = [
                                item for item_index, item in enumerate(candidate_features)
                                if item_index != index
                            ]
                        local_scores[candidate_label] = self._score_local(
                            side_features[index], candidate_features, image.shape[:2]
                        ) if candidate_features else 0.0
                    if len(global_scores) < 2:
                        skipped += 1
                        continue
                    fused = self._fuse_scores(global_scores, local_scores, time.perf_counter())
                    evaluated += 1
                    if fused.get("label") != label:
                        wrong += 1
                except (ImageUnreadableError, AttributeError, IndexError, KeyError, TypeError, ValueError):
                    skipped += 1
        return {
            "status": "completed",
            "correct_to_wrong": int(wrong),
            "evaluated": int(evaluated),
            "skipped": int(skipped),
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
    ) -> tuple[TemplateCache, dict[str, list[dict[str, float | int]]]]:
        """Build a filtered candidate cache without publishing it."""
        cache = self._template_caches.get(workpiece_id)
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
            import torch

            # ALIKED returns inference tensors. LightGlue must run in the same
            # inference context, otherwise PyTorch rejects its Linear layers
            # for attempting to save inference tensors for backward.
            with torch.inference_mode():
                source_image = _read_image(Path(source_path))
                target_image = _read_image(Path(target_path))
                source_features = self._extract_features(source_image, self.extractor, self.device, roi_ratio=ROI_RATIO)
                target_features = self._extract_features(target_image, self.extractor, self.device, roi_ratio=ROI_RATIO)
                match_data = self.matcher({"image0": source_features, "image1": target_features})
                matches = match_data["matches"][0].detach().cpu().numpy()
                if len(matches) < 4:
                    return None
                source_points = source_features["keypoints"][0][matches[:, 0]].detach().cpu().numpy()
                target_points = target_features["keypoints"][0][matches[:, 1]].detach().cpu().numpy()
                return project_region(region, source_points, target_points)
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

    def _rank_local_candidates(
        self,
        global_vectors: Mapping[str, np.ndarray],
        query_embeddings: Mapping[str, np.ndarray],
        local_features: Mapping[str, Sequence[dict[str, Any]]],
    ) -> dict[str, list[int]]:
        rankings = {}
        for label in ("front", "back"):
            vectors = np.asarray(global_vectors.get(label), dtype=np.float32)
            candidates = local_features.get(label)
            if vectors.ndim != 2 or candidates is None:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: missing vectors or local features"
                )
            if vectors.shape[0] == 0 or len(candidates) == 0:
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: at least one local template is required"
                )
            if vectors.shape[0] != len(candidates):
                raise OrientationClassifierError(
                    f"template cache alignment error for {label}: "
                    f"{vectors.shape[0]} global vectors != {len(candidates)} local templates"
                )
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

    def _score_local(self, query_features: dict[str, Any], candidates: Sequence[dict[str, Any]],
                     image_shape: tuple[int, int]) -> float:
        if not candidates:
            return 0.0
        return max(
            float(self._score_feature_pair(
                _move_tensors(query_features, self.device),
                _move_tensors(features, self.device),
                image_shape,
                self.matcher,
            )["score"])
            for features in candidates
        )

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

    def _predict_baseline(self, image: np.ndarray, cache: TemplateCache,
                          started: float, geometry_mask: dict[str, Any] | None = None) -> dict[str, object]:
        raw_globals = getattr(cache, "raw_global_vectors", None) or cache.global_vectors
        raw_locals = getattr(cache, "raw_local_features", None) or cache.local_features
        global_prediction, global_scores, _ = classify_embedding(self._global_embedding(image), raw_globals)
        del global_prediction
        with self._inference_lock:
            query_features = self._extract_features(image, self.extractor, self.device, roi_ratio=ROI_RATIO)
        local_scores = {label: self._score_local(query_features, candidates, image.shape[:2])
                        for label, candidates in raw_locals.items()}
        return self._fuse_scores(global_scores, local_scores, started, geometry_mask=geometry_mask)

    @staticmethod
    def _geometry_report(fit: dict[str, Any]) -> dict[str, Any]:
        result = {key: value for key, value in fit.items() if key != "ignore_mask"}
        result["rules"] = [
            {key: value for key, value in rule.items() if key != "ignore_mask"}
            for rule in fit.get("rules", [])
        ]
        return result

    def _predict_geometry(self, image: np.ndarray, cache: TemplateCache,
                          started: float) -> dict[str, object]:
        profile = cache.geometry_profile or {}
        calibrator = self.geometry_calibrator
        if calibrator is None:
            return self._predict_baseline(image, cache, started, {
                "status": "unavailable", "needs_review": True, "profile_revision": cache.geometry_profile_revision,
            })
        query_features: dict[str, dict[str, Any]] = {}
        query_embeddings: dict[str, np.ndarray] = {}
        reports: dict[str, Any] = {}
        raw_globals = getattr(cache, "raw_global_vectors", None) or cache.global_vectors
        raw_locals = getattr(cache, "raw_local_features", None) or cache.local_features
        for label in ("front", "back"):
            direction = self._geometry_direction(profile, label)
            if direction is None or direction.get("anchor") is None or not direction.get("rules"):
                query_features[label] = self._extract_local(image)
                query_embeddings[label] = self._global_embedding(image)
                reports[label] = {"status": "not_configured"}
                continue
            fit = calibrator.fit(image, direction)
            reports[label] = self._geometry_report(fit)
            if fit.get("status") != "active":
                geometry_mask = {
                    "status": fit.get("status", "low_confidence"),
                    "needs_review": True,
                    "profile_revision": cache.geometry_profile_revision,
                    "directions": reports,
                }
                return self._predict_baseline(image, TemplateCache(
                    global_vectors=raw_globals,
                    local_features=raw_locals,
                    raw_global_vectors=raw_globals,
                    raw_local_features=raw_locals,
                ), started, geometry_mask)
            mask = fit["ignore_mask"]
            fill = direction.get("fill_bgr") or self._neutral_fill(image, mask)
            masked = apply_ignore_mask(image, mask, fill)
            query_embeddings[label] = self._global_embedding(masked)
            query_features[label] = filter_features_by_mask(self._extract_local(masked), mask)
        global_scores = {
            label: float(np.max(cache.global_vectors[label] @ query_embeddings[label]))
            for label in ("front", "back")
        }
        local_scores = {
            label: self._score_local(query_features[label], cache.local_features[label], image.shape[:2])
            for label in ("front", "back")
        }
        geometry_mask = {
            "status": "active",
            "needs_review": False,
            "profile_revision": cache.geometry_profile_revision,
            "directions": reports,
            "ignored_ratio": {
                label: float(reports[label].get("ignored_ratio", 0.0)) for label in ("front", "back")
            },
        }
        return self._fuse_scores(global_scores, local_scores, started, geometry_mask=geometry_mask)

    def predict(self, workpiece_id: str, image_path: Path) -> dict[str, object]:
        started = time.perf_counter()
        cache = self._template_caches.get(workpiece_id)
        if cache is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        image = _read_image(Path(image_path))
        if getattr(cache, "geometry_profile", None) is not None:
            return self._predict_geometry(image, cache, started)
        return self._predict_baseline(image, cache, started)
