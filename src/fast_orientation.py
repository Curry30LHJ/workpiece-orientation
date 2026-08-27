"""Fast three-slot orientation cache building and Ridge-only inference."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

from src.fast_geometry import (
    FAST_GEOMETRY_FORMAT_VERSION,
    CompiledFastGeometry,
    FastGeometryProcessor,
    FastGeometryVariants,
    compile_fast_geometry,
)
from src.fast_ridge import RidgeHead, fit_ridge_head, pack_embeddings, predict_ridge


FAST_RUNTIME_FORMAT_VERSION = 1
FAST_FEATURE_LAYOUT = ("raw", "front_masked", "back_masked")
FAST_ROTATION_DEGREES = tuple(range(30, 360, 30))
FAST_AUGMENT_BELOW_PER_SIDE = 20
FAST_BUILD_IMAGE_BATCH = 32
_CACHE_ERROR = "FAST_CACHE_REVISION_MISMATCH"


@dataclass(frozen=True)
class FastRuntimeCache:
    format_version: int
    cache_revision: str
    library_revision: int
    geometry_profile_revision: int | None
    model_fingerprint: str
    template_signature: dict[str, Any]
    template_counts: dict[str, int]
    feature_layout: tuple[str, str, str]
    compiled_geometry: CompiledFastGeometry
    ridge_head: RidgeHead
    training_summary: dict[str, Any]


@dataclass(frozen=True)
class _TrainingSample:
    label: str
    source_id: int
    original: bool
    images: tuple[np.ndarray, np.ndarray, np.ndarray]


def _cache_error(message: str) -> ValueError:
    return ValueError(f"{_CACHE_ERROR}: {message}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return sha256(encoded).hexdigest()


def _image_digest(image: np.ndarray) -> str:
    normalized = np.ascontiguousarray(image)
    hasher = sha256()
    hasher.update(str(normalized.shape).encode("ascii"))
    hasher.update(normalized.dtype.str.encode("ascii"))
    hasher.update(normalized.tobytes())
    return hasher.hexdigest()


def _validate_image(image: Any, path: Path) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError(f"template image is unreadable or invalid: {path}")
    if not np.isfinite(image).all():
        raise ValueError(f"template image must be finite: {path}")
    return image


def _compiled_signature(compiled: CompiledFastGeometry) -> dict[str, Any]:
    return {
        "format_version": compiled.format_version,
        "profile_revision": compiled.profile_revision,
        "directions": compiled.directions,
    }


def _rotation_center(variants: FastGeometryVariants, image: np.ndarray) -> tuple[float, float]:
    """Use a fitted rule/anchor center when published, otherwise image center."""
    for direction in variants.directions.values():
        if not isinstance(direction, Mapping):
            continue
        candidates = list(direction.get("rules", ()))
        candidates.append(direction.get("fitted_shape"))
        candidates.append(direction.get("anchor"))
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            shape = candidate.get("fitted_shape", candidate)
            if not isinstance(shape, Mapping):
                continue
            x, y = shape.get("cx"), shape.get("cy")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and np.isfinite((x, y)).all():
                return float(x), float(y)
    height, width = image.shape[:2]
    return width / 2.0, height / 2.0


def _rotate_slots(images: tuple[np.ndarray, np.ndarray, np.ndarray], center: tuple[float, float], degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = images[0].shape[:2]
    matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    return tuple(
        cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        for image in images
    )  # type: ignore[return-value]


class FastOrientationEngine:
    def __init__(
        self,
        embed_batch: Callable[[Sequence[np.ndarray]], list[np.ndarray]],
        geometry: FastGeometryProcessor,
        *,
        image_reader: Callable[[Path], np.ndarray],
    ) -> None:
        self.embed_batch = embed_batch
        self.geometry = geometry
        self.image_reader = image_reader

    def build_cache(
        self,
        front_paths: Sequence[Path],
        back_paths: Sequence[Path],
        *,
        geometry_profile: Mapping[str, Any] | None,
        library_revision: int,
        model_fingerprint: str,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> FastRuntimeCache:
        started = perf_counter()
        if type(library_revision) is not int or library_revision < 0:
            raise ValueError("library_revision must be a non-negative integer")
        if not isinstance(model_fingerprint, str) or not model_fingerprint.strip():
            raise ValueError("model_fingerprint must be a non-empty string")
        paths_by_label = {"front": tuple(Path(path) for path in front_paths), "back": tuple(Path(path) for path in back_paths)}
        if not paths_by_label["front"] or not paths_by_label["back"]:
            raise ValueError("Each orientation requires at least one template")
        compiled = compile_fast_geometry(geometry_profile)
        originals: dict[str, list[tuple[int, np.ndarray, str]]] = {"front": [], "back": []}
        hashes: set[str] = set()
        signature: dict[str, Any] = {"front": [], "back": []}
        for label, paths in paths_by_label.items():
            for source_id, path in enumerate(paths):
                try:
                    image = _validate_image(self.image_reader(path), path)
                except Exception as exc:
                    if isinstance(exc, ValueError) and "finite" in str(exc):
                        raise
                    raise ValueError(f"template image is unreadable: {path}") from exc
                digest = _image_digest(image)
                if digest in hashes:
                    raise ValueError(f"duplicate template image: {path}")
                hashes.add(digest)
                originals[label].append((source_id, image, digest))
                signature[label].append({"path": str(path), "sha256": digest})
        signature["combined_sha256"] = _digest({"front": signature["front"], "back": signature["back"]})

        total_originals = sum(len(items) for items in originals.values())
        samples: dict[str, list[_TrainingSample]] = {"front": [], "back": []}
        geometry_review_counts = {"front": 0, "back": 0}
        completed = 0
        for label in ("front", "back"):
            for source_id, image, _ in originals[label]:
                variants = self.geometry.build_variants(image, compiled)
                if variants.needs_review:
                    geometry_review_counts[label] += 1
                samples[label].append(_TrainingSample(label, source_id, True, variants.images))
                completed += 1
                self._progress(progress_callback, "fast_originals", completed, total_originals, "templates")

                if len(originals[label]) < FAST_AUGMENT_BELOW_PER_SIDE:
                    center = _rotation_center(variants, image)
                    for degree in FAST_ROTATION_DEGREES:
                        samples[label].append(_TrainingSample(label, source_id, False, _rotate_slots(variants.images, center, degree)))

        augmented_counts = {
            label: len(samples[label]) - len(originals[label])
            for label in ("front", "back")
        }
        total_augmented = sum(augmented_counts.values())
        augmented_done = 0
        if total_augmented == 0:
            self._progress(progress_callback, "fast_augmentation", 0, 0, "augmented_samples")
        else:
            for label in ("front", "back"):
                for _ in range(augmented_counts[label]):
                    augmented_done += 1
                    self._progress(progress_callback, "fast_augmentation", augmented_done, total_augmented, "augmented_samples")

        features = self._embed_samples(samples)
        self._progress(progress_callback, "fast_ridge", 0, 1, "ridge_head")
        head = fit_ridge_head(
            np.stack(features["front"]),
            np.stack(features["back"]),
            front_source_ids=[sample.source_id for sample in samples["front"]],
            back_source_ids=[sample.source_id for sample in samples["back"]],
            front_original_rows=[sample.original for sample in samples["front"]],
            back_original_rows=[sample.original for sample in samples["back"]],
        )
        self._progress(progress_callback, "fast_ridge", 1, 1, "ridge_head")
        template_counts = {label: len(originals[label]) for label in ("front", "back")}
        summary = {
            "original_samples": total_originals,
            "augmented_samples": augmented_counts,
            "rotation_degrees": list(FAST_ROTATION_DEGREES),
            "geometry_review_counts": geometry_review_counts,
            "selected_regularization": head.regularization,
            "review_threshold": head.review_threshold,
            "validation_status": head.training_summary["validation_status"],
            "ridge": head.training_summary,
            "build_ms": (perf_counter() - started) * 1000.0,
        }
        revision_payload = {
            "runtime_format_version": FAST_RUNTIME_FORMAT_VERSION,
            "library_revision": library_revision,
            "model_fingerprint": model_fingerprint,
            "template_signature": signature,
            "compiled_geometry": _compiled_signature(compiled),
            "feature_layout": FAST_FEATURE_LAYOUT,
        }
        return FastRuntimeCache(
            FAST_RUNTIME_FORMAT_VERSION,
            _digest(revision_payload),
            library_revision,
            compiled.profile_revision,
            model_fingerprint,
            signature,
            template_counts,
            FAST_FEATURE_LAYOUT,
            compiled,
            head,
            summary,
        )

    def predict(self, image: np.ndarray, cache: FastRuntimeCache) -> dict[str, object]:
        started = perf_counter()
        self._validate_cache(cache)
        variants = self.geometry.build_variants(image, cache.compiled_geometry)
        batch_started = perf_counter()
        embeddings = self.embed_batch(variants.images)
        global_batch_ms = (perf_counter() - batch_started) * 1000.0
        if not isinstance(embeddings, Sequence) or len(embeddings) != len(FAST_FEATURE_LAYOUT):
            raise ValueError("embed_batch must return one embedding for each fast feature slot")
        feature = pack_embeddings(embeddings)
        head_started = perf_counter()
        decision = predict_ridge(cache.ridge_head, feature)
        linear_head_ms = (perf_counter() - head_started) * 1000.0
        codes = list(variants.review_reasons)
        if decision.needs_review:
            codes.append("FAST_CLASSIFIER_LOW_MARGIN")
        if cache.ridge_head.training_summary.get("validation_status") != "validated":
            codes.append("FAST_MODE_NOT_VALIDATED")
        codes = list(dict.fromkeys(codes))
        timings = dict(variants.timings_ms)
        timings.update({"global_batch": global_batch_ms, "linear_head": linear_head_ms})
        timings["total"] = (perf_counter() - started) * 1000.0
        review_reason = "；".join({
            "FAST_GEOMETRY_LOW_CONFIDENCE": "几何拟合置信度不足",
            "FAST_CLASSIFIER_LOW_MARGIN": "分类边界裕量不足",
            "FAST_MODE_NOT_VALIDATED": "当前模板数量不足，模型尚未完成验证",
        }[code] for code in codes)
        return {
            "label": decision.label,
            "needs_review": bool(codes),
            "review_reason_codes": codes,
            "review_reason": review_reason,
            "inference_engine": "fast_geometry",
            "decision_source": "fast_ridge",
            "fast_cache_revision": cache.cache_revision,
            "geometry_status": variants.status,
            "directions": variants.directions,
            "decision_margin": decision.margin,
            "elapsed_ms": timings["total"],
            "timings_ms": timings,
        }

    @staticmethod
    def _progress(callback: Callable[[dict[str, Any]], None] | None, phase: str, completed: int, total: int, unit: str) -> None:
        if callback is not None:
            callback({"phase": phase, "completed": completed, "total": total, "unit": unit})

    def _embed_samples(self, samples: Mapping[str, Sequence[_TrainingSample]]) -> dict[str, list[np.ndarray]]:
        ordered = [sample for label in ("front", "back") for sample in samples[label]]
        images = [image for sample in ordered for image in sample.images]
        embeddings: list[np.ndarray] = []
        for start in range(0, len(images), FAST_BUILD_IMAGE_BATCH):
            chunk = images[start:start + FAST_BUILD_IMAGE_BATCH]
            returned = self.embed_batch(chunk)
            if not isinstance(returned, Sequence) or len(returned) != len(chunk):
                raise ValueError("embed_batch returned an unexpected batch size")
            embeddings.extend(returned)
        packed = [pack_embeddings(embeddings[index:index + 3]) for index in range(0, len(embeddings), 3)]
        front_count = len(samples["front"])
        return {"front": packed[:front_count], "back": packed[front_count:]}

    @staticmethod
    def _validate_cache(cache: FastRuntimeCache) -> None:
        if not isinstance(cache, FastRuntimeCache):
            raise _cache_error("cache type is invalid")
        if cache.format_version != FAST_RUNTIME_FORMAT_VERSION:
            raise _cache_error("runtime format is invalid")
        if not isinstance(cache.cache_revision, str) or not cache.cache_revision:
            raise _cache_error("cache revision is invalid")
        if type(cache.library_revision) is not int or cache.library_revision < 0:
            raise _cache_error("library revision is invalid")
        if not isinstance(cache.model_fingerprint, str) or not cache.model_fingerprint.strip():
            raise _cache_error("model fingerprint is invalid")
        if cache.feature_layout != FAST_FEATURE_LAYOUT or not isinstance(cache.template_signature, dict):
            raise _cache_error("feature layout or template signature is invalid")
        if not isinstance(cache.compiled_geometry, CompiledFastGeometry):
            raise _cache_error("compiled geometry is invalid")
        if cache.geometry_profile_revision != cache.compiled_geometry.profile_revision:
            raise _cache_error("geometry profile revision is invalid")
        head = cache.ridge_head
        if not isinstance(head, RidgeHead) or head.feature_dim <= 0:
            raise _cache_error("ridge head is invalid")
        values = np.asarray(head.weights, dtype=np.float64)
        if values.ndim != 1 or values.size != head.feature_dim or not np.isfinite(values).all():
            raise _cache_error("ridge weights are invalid")
        if not all(np.isfinite(value) for value in (head.bias, head.regularization, head.review_threshold)):
            raise _cache_error("ridge values are invalid")
        if head.regularization <= 0 or head.review_threshold < 0:
            raise _cache_error("ridge values are invalid")
        if cache.compiled_geometry.format_version != FAST_GEOMETRY_FORMAT_VERSION:
            raise _cache_error("geometry format is invalid")
