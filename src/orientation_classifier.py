"""Reusable PP-ShiTuV2 and ALIKED/LightGlue orientation classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from src.image_io import read_color_image
from src.shitu_baseline import classify_embedding


ROI_RATIO = 1.0
MAX_NUM_KEYPOINTS = 512
GLOBAL_MARGIN_THRESHOLD = 0.05
LOCAL_MIN_SCORE = 4.0
LOCAL_MIN_MARGIN = 0.5
LOCAL_OVERRIDE_MARGIN = 3.0
TEMPLATE_COUNT = 5


class OrientationClassifierError(RuntimeError):
    """Base error for service-facing classifier failures."""


class WorkpieceNotFoundError(OrientationClassifierError):
    """Raised when prediction is requested for an unknown workpiece."""


class ImageUnreadableError(OrientationClassifierError):
    """Raised when OpenCV cannot decode an input image."""


@dataclass(frozen=True)
class TemplateCache:
    """CPU-resident features for one workpiece."""

    global_vectors: dict[str, np.ndarray]
    local_features: dict[str, list[dict[str, Any]]]


def _move_tensors(value: Any, device: Any) -> Any:
    """Move nested Torch tensors while leaving test doubles and scalars unchanged."""
    try:
        import torch
    except ImportError:
        return value
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
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
    ) -> None:
        self.global_predictor = global_predictor
        self.extractor = extractor
        self.matcher = matcher
        self.device = device
        self._extract_features = extract_features_fn or _default_extract_features
        self._score_feature_pair = score_feature_pair_fn or _default_score_feature_pair
        self._template_caches: dict[str, TemplateCache] = {}

    @classmethod
    def load(cls, project_root: Path, model_dir: Path) -> "OrientationClassifier":
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
        return cls(global_predictor, extractor, matcher, device)

    def _global_embedding(self, image: np.ndarray) -> np.ndarray:
        embedding = self.global_predictor.predict([image[:, :, ::-1]])[0]
        return np.asarray(embedding, dtype=np.float32)

    def _extract_local(self, image: np.ndarray) -> dict:
        features = self._extract_features(image, self.extractor, self.device, roi_ratio=ROI_RATIO)
        return _to_cpu(features)

    def build_template_cache(
        self, front_paths: Sequence[Path], back_paths: Sequence[Path]
    ) -> TemplateCache:
        if len(front_paths) != TEMPLATE_COUNT or len(back_paths) != TEMPLATE_COUNT:
            raise ValueError("Each orientation requires exactly five templates")
        global_vectors: dict[str, np.ndarray] = {}
        local_features: dict[str, list[dict[str, Any]]] = {}
        for label, paths in (("front", front_paths), ("back", back_paths)):
            embeddings = []
            features = []
            for path in paths:
                image = _read_image(Path(path))
                embeddings.append(self._global_embedding(image))
                features.append(self._extract_local(image))
            global_vectors[label] = np.stack(embeddings).astype(np.float32)
            local_features[label] = features
        return TemplateCache(global_vectors=global_vectors, local_features=local_features)

    def set_template_cache(self, workpiece_id: str, cache: TemplateCache) -> None:
        self._template_caches[workpiece_id] = cache

    def remove_template_cache(self, workpiece_id: str) -> None:
        self._template_caches.pop(workpiece_id, None)

    def predict(self, workpiece_id: str, image_path: Path) -> dict[str, object]:
        started = time.perf_counter()
        cache = self._template_caches.get(workpiece_id)
        if cache is None:
            raise WorkpieceNotFoundError(f"Unknown workpiece: {workpiece_id}")
        image = _read_image(Path(image_path))

        global_prediction, global_scores, global_margin = classify_embedding(
            self._global_embedding(image), cache.global_vectors
        )
        query_features = self._extract_features(image, self.extractor, self.device, roi_ratio=ROI_RATIO)
        local_scores: dict[str, float] = {}
        for label, candidates in cache.local_features.items():
            gpu_candidates = [_move_tensors(features, self.device) for features in candidates]
            local_scores[label] = max(
                float(
                    self._score_feature_pair(
                        query_features,
                        features,
                        image.shape[:2],
                        self.matcher,
                    )["score"]
                )
                for features in gpu_candidates
            )
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
        return {
            "label": label,
            "global_prediction": global_prediction,
            "global_scores": {key: float(value) for key, value in global_scores.items()},
            "global_margin": float(global_margin),
            "local_prediction": local_prediction,
            "local_scores": local_scores,
            "local_margin": float(local_margin),
            "decision_source": decision_source,
            "needs_review": needs_review,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
