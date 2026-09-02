"""Fast three-slot orientation cache building and Ridge-only inference."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import logging
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
_FEATURE_ERROR = "FAST_FEATURE_INVALID"
LOGGER = logging.getLogger(__name__)


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


def _feature_error(message: str) -> ValueError:
    return ValueError(f"{_FEATURE_ERROR}: {message}")


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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_template_signature(signature: Any, counts: Mapping[str, int]) -> None:
    if not isinstance(signature, dict) or set(signature) != {"front", "back", "combined_sha256"}:
        raise _cache_error("template signature is invalid")
    for label in ("front", "back"):
        entries = signature[label]
        if not isinstance(entries, list) or len(entries) != counts[label]:
            raise _cache_error("template signature count is invalid")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise _cache_error("template signature entry is invalid")
            if not isinstance(entry["path"], str) or not entry["path"] or not _is_sha256(entry["sha256"]):
                raise _cache_error("template signature digest is invalid")
    combined = _digest({"front": signature["front"], "back": signature["back"]})
    if signature["combined_sha256"] != combined:
        raise _cache_error("template signature digest is invalid")


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


def _cache_revision_payload(
    library_revision: int,
    model_fingerprint: str,
    template_signature: Mapping[str, Any],
    compiled_geometry: CompiledFastGeometry,
) -> dict[str, Any]:
    return {
        "runtime_format_version": FAST_RUNTIME_FORMAT_VERSION,
        "library_revision": library_revision,
        "model_fingerprint": model_fingerprint,
        "template_signature": template_signature,
        "compiled_geometry": _compiled_signature(compiled_geometry),
        "feature_layout": FAST_FEATURE_LAYOUT,
    }


def _geometry_context_shape(image: np.ndarray, geometry: FastGeometryProcessor) -> tuple[int, int]:
    height, width = image.shape[:2]
    max_side = getattr(geometry, "max_side", max(height, width))
    if not isinstance(max_side, (int, float)) or not np.isfinite(max_side) or max_side <= 0:
        return height, width
    scale = min(1.0, float(max_side) / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _rotation_center(
    label: str,
    variants: FastGeometryVariants,
    image: np.ndarray,
    context_shape: tuple[int, int],
) -> tuple[float, float]:
    """Use this label's fitted rule center, mapped from context to source pixels."""
    direction = variants.directions.get(label)
    rules = direction.get("rules", ()) if isinstance(direction, Mapping) else ()
    if isinstance(rules, Sequence) and not isinstance(rules, (str, bytes, bytearray)):
        for rule in rules:
            if not isinstance(rule, Mapping) or rule.get("status") != "active":
                continue
            shape = rule.get("fitted_shape")
            if not isinstance(shape, Mapping):
                continue
            x, y = shape.get("cx"), shape.get("cy")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and np.isfinite((x, y)).all():
                source_height, source_width = image.shape[:2]
                context_height, context_width = context_shape
                return float(x) * source_width / context_width, float(y) * source_height / context_height
    height, width = image.shape[:2]
    return width / 2.0, height / 2.0


def _rotate_slots(images: tuple[np.ndarray, np.ndarray, np.ndarray], center: tuple[float, float], degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = images[0].shape[:2]
    matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    return tuple(
        cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        for image in images
    )  # type: ignore[return-value]


def _pack_three_embeddings(
    embeddings: Sequence[np.ndarray],
    *,
    expected_slot_dim: int | None = None,
) -> tuple[np.ndarray, int]:
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes, bytearray)) or len(embeddings) != 3:
        raise _feature_error("exactly three embeddings are required")
    vectors: list[np.ndarray] = []
    for embedding in embeddings:
        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise _feature_error("embedding is not numeric") from exc
        if vector.size == 0 or not np.isfinite(vector).all():
            raise _feature_error("embedding must be finite and non-empty")
        vectors.append(vector)
    slot_dim = vectors[0].size
    if any(vector.size != slot_dim for vector in vectors[1:]):
        raise _feature_error("embedding slot dimensions must match")
    if expected_slot_dim is not None and slot_dim != expected_slot_dim:
        raise _feature_error("embedding slot dimension does not match cache")
    try:
        return pack_embeddings(vectors), slot_dim
    except ValueError as exc:
        raise _feature_error("embedding packing failed") from exc


class FastOrientationEngine:
    def __init__(
        self,
        embed_batch: Callable[[Sequence[np.ndarray]], list[np.ndarray]],
        geometry: FastGeometryProcessor,
        *,
        image_reader: Callable[[Path], np.ndarray],
        deduplicate_identical_slots: bool = False,
    ) -> None:
        self.embed_batch = embed_batch
        self.geometry = geometry
        self.image_reader = image_reader
        self.deduplicate_identical_slots = deduplicate_identical_slots

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
        augmentation_totals = {
            label: len(originals[label]) * len(FAST_ROTATION_DEGREES)
            if len(originals[label]) < FAST_AUGMENT_BELOW_PER_SIDE else 0
            for label in ("front", "back")
        }
        total_augmented = sum(augmentation_totals.values())
        augmented_done = 0
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
                    center = _rotation_center(label, variants, image, _geometry_context_shape(image, self.geometry))
                    for degree in FAST_ROTATION_DEGREES:
                        samples[label].append(_TrainingSample(label, source_id, False, _rotate_slots(variants.images, center, degree)))
                        augmented_done += 1
                        self._progress(progress_callback, "fast_augmentation", augmented_done, total_augmented, "augmented_samples")

        augmented_counts = {
            label: len(samples[label]) - len(originals[label])
            for label in ("front", "back")
        }
        # A zero-total progress frame is not a progress unit: the Qt client
        # quite correctly rejects it as malformed.  Omit the phase entirely
        # when both sides are above the few-shot augmentation threshold.

        features = self._embed_samples(
            samples,
            progress_callback=progress_callback,
            emit_progress=total_originals > 2,
        )
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
        return FastRuntimeCache(
            FAST_RUNTIME_FORMAT_VERSION,
            _digest(_cache_revision_payload(library_revision, model_fingerprint, signature, compiled)),
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
        embeddings, unique_count, model_calls = self._embed_query_slots_with_stats(
            variants.images,
            slot_names=("raw", "front", "back"),
        )
        global_batch_ms = (perf_counter() - batch_started) * 1000.0
        feature, _ = _pack_three_embeddings(
            embeddings,
            expected_slot_dim=cache.ridge_head.feature_dim // len(FAST_FEATURE_LAYOUT),
        )
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
        timings.update({
            "global_batch": global_batch_ms,
            "global_input_slots": 3,
            "global_unique_slots": unique_count,
            "global_model_calls": model_calls,
            "linear_head": linear_head_ms,
        })
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

    def _embed_query_slots(self, images: Sequence[np.ndarray]) -> tuple[list[np.ndarray], int]:
        """Embed query slots while preserving the historical two-value API."""
        embeddings, unique_count, _ = self._embed_query_slots_with_stats(images)
        return embeddings, unique_count

    def _embed_query_slots_with_stats(
        self,
        images: Sequence[np.ndarray],
        *,
        slot_names: Sequence[str] | None = None,
    ) -> tuple[list[np.ndarray], int, int]:
        """Embed query slots and return (embeddings, unique_slots, model_calls)."""
        calls = 0
        if slot_names is None and len(images) == 3:
            slot_names = ("raw", "front", "back")

        def embed(batch: Sequence[np.ndarray]):
            nonlocal calls
            calls += 1
            return self.embed_batch(batch)

        if not self.deduplicate_identical_slots:
            return embed(images), len(images), calls

        unique_images: list[np.ndarray] = []
        unique_names: list[str] = []
        slot_to_unique: list[int] = []
        for slot_index, image in enumerate(images):
            input_name = (
                str(slot_names[slot_index])
                if slot_names is not None and slot_index < len(slot_names)
                else f"slot-{slot_index}"
            )
            matching_index = None
            for unique_index, unique_image in enumerate(unique_images):
                unique_name = unique_names[unique_index]
                image_shape = unique_shape = None
                image_dtype = unique_dtype = None
                image_nbytes = unique_nbytes = None
                try:
                    image_shape = image.shape
                    unique_shape = unique_image.shape
                    image_dtype = image.dtype
                    unique_dtype = unique_image.dtype
                    image_nbytes = image.nbytes
                    unique_nbytes = unique_image.nbytes
                    is_match = (
                        image_shape == unique_shape
                        and image_dtype == unique_dtype
                        and image_nbytes == unique_nbytes
                        and np.array_equal(image, unique_image)
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "CPU slot dedup comparison failed for input slot %d (%s) against unique slot %d (%s); "
                        "shape=%r/%r dtype=%r/%r nbytes=%r/%r reason=%s: %s: %s",
                        slot_index,
                        input_name,
                        unique_index,
                        unique_name,
                        image_shape,
                        unique_shape,
                        image_dtype,
                        unique_dtype,
                        image_nbytes,
                        unique_nbytes,
                        "slot metadata or pixel comparison raised",
                        type(exc).__name__,
                        exc,
                    )
                    return embed(images), len(images), calls
                if is_match:
                    matching_index = unique_index
                    break
            if matching_index is None:
                matching_index = len(unique_images)
                unique_images.append(image)
                unique_names.append(input_name)
            slot_to_unique.append(matching_index)

        returned = embed(unique_images)
        if not isinstance(returned, Sequence) or len(returned) != len(unique_images):
            raise _feature_error("embedder returned an unexpected batch size")
        return [returned[index] for index in slot_to_unique], len(unique_images), calls

    @staticmethod
    def _progress(callback: Callable[[dict[str, Any]], None] | None, phase: str, completed: int, total: int, unit: str) -> None:
        if callback is not None:
            callback({"phase": phase, "completed": completed, "total": total, "unit": unit})

    def _embed_samples(
        self,
        samples: Mapping[str, Sequence[_TrainingSample]],
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        emit_progress: bool = False,
    ) -> dict[str, list[np.ndarray]]:
        ordered = [sample for label in ("front", "back") for sample in samples[label]]
        images = [image for sample in ordered for image in sample.images]
        embeddings: list[np.ndarray] = []
        batch_count = (len(images) + FAST_BUILD_IMAGE_BATCH - 1) // FAST_BUILD_IMAGE_BATCH
        if emit_progress and batch_count > 0:
            self._progress(progress_callback, "fast_embedding", 0, batch_count, "embedding_batches")
        for batch_index, start in enumerate(range(0, len(images), FAST_BUILD_IMAGE_BATCH), start=1):
            chunk = images[start:start + FAST_BUILD_IMAGE_BATCH]
            returned = self.embed_batch(chunk)
            if not isinstance(returned, Sequence) or len(returned) != len(chunk):
                raise _feature_error("embedder returned an unexpected batch size")
            embeddings.extend(returned)
            if emit_progress:
                self._progress(
                    progress_callback,
                    "fast_embedding",
                    batch_index,
                    batch_count,
                    "embedding_batches",
                )
        packed: list[np.ndarray] = []
        expected_slot_dim: int | None = None
        for index in range(0, len(embeddings), len(FAST_FEATURE_LAYOUT)):
            feature, slot_dim = _pack_three_embeddings(
                embeddings[index:index + len(FAST_FEATURE_LAYOUT)],
                expected_slot_dim=expected_slot_dim,
            )
            expected_slot_dim = slot_dim
            packed.append(feature)
        front_count = len(samples["front"])
        return {"front": packed[:front_count], "back": packed[front_count:]}

    @staticmethod
    def _validate_cache(cache: FastRuntimeCache) -> None:
        if not isinstance(cache, FastRuntimeCache):
            raise _cache_error("cache type is invalid")
        if cache.format_version != FAST_RUNTIME_FORMAT_VERSION:
            raise _cache_error("runtime format is invalid")
        if not isinstance(cache.cache_revision, str) or len(cache.cache_revision) != 64 or not _is_sha256(cache.cache_revision):
            raise _cache_error("cache revision is invalid")
        if type(cache.library_revision) is not int or cache.library_revision < 0:
            raise _cache_error("library revision is invalid")
        if not isinstance(cache.model_fingerprint, str) or not cache.model_fingerprint.strip():
            raise _cache_error("model fingerprint is invalid")
        if cache.feature_layout != FAST_FEATURE_LAYOUT:
            raise _cache_error("feature layout or template signature is invalid")
        if not isinstance(cache.compiled_geometry, CompiledFastGeometry):
            raise _cache_error("compiled geometry is invalid")
        if cache.geometry_profile_revision != cache.compiled_geometry.profile_revision:
            raise _cache_error("geometry profile revision is invalid")
        if not isinstance(cache.template_counts, dict) or set(cache.template_counts) != {"front", "back"}:
            raise _cache_error("template counts are invalid")
        if any(type(cache.template_counts[label]) is not int or cache.template_counts[label] <= 0 for label in ("front", "back")):
            raise _cache_error("template counts are invalid")
        _validate_template_signature(cache.template_signature, cache.template_counts)
        head = cache.ridge_head
        if not isinstance(head, RidgeHead) or head.feature_dim <= 0 or head.feature_dim % len(FAST_FEATURE_LAYOUT) != 0:
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
        expected_revision = _digest(_cache_revision_payload(
            cache.library_revision,
            cache.model_fingerprint,
            cache.template_signature,
            cache.compiled_geometry,
        ))
        if cache.cache_revision != expected_revision:
            raise _cache_error("cache revision does not match cache content")
