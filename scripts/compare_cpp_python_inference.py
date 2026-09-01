"""Compare native PP-ShiTu recognition output with the Python baseline."""

from copy import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import numpy as np


DECISION_FIELDS = (
    "label",
    "needs_review",
    "global_label",
    "geometry_applied",
    "front_score",
    "back_score",
    "margin",
)
_EXACT_DECISION_FIELDS = DECISION_FIELDS[:4]
_SCORE_DECISION_FIELDS = DECISION_FIELDS[4:]
_REQUIRED_NATIVE_TIMING_STAGES = (
    "decode",
    "preprocess",
    "inference",
    "normalize",
    "total",
)


@dataclass(frozen=True)
class EmbeddingRun:
    """Ordered embeddings and measurements from one inference configuration."""

    embeddings: np.ndarray
    ordered_images: tuple[str, ...]
    timings_ms: dict[str, object]
    model_fingerprint: str

    def __post_init__(self) -> None:
        embeddings = np.asarray(self.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
            raise ValueError("embeddings must be a non-empty 2-D matrix")
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        if len(self.ordered_images) != embeddings.shape[0]:
            raise ValueError("ordered image count must match embedding rows")
        if not isinstance(self.model_fingerprint, str) or not self.model_fingerprint:
            raise ValueError("model_fingerprint must be a non-empty string")
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "ordered_images", tuple(str(item) for item in self.ordered_images))
        object.__setattr__(self, "timings_ms", dict(self.timings_ms))


def read_f32_matrix(path: Path, rows: int, columns: int) -> np.ndarray:
    """Read an exact row-major float32 matrix from *path*."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    values = np.fromfile(Path(path), dtype=np.float32)
    expected = rows * columns
    if values.size != expected:
        raise ValueError(
            f"expected {expected} float32 values, found {values.size}"
        )
    return values.reshape(rows, columns)


def compare_embeddings(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    min_cosine: float = 0.9999,
    max_abs_error: float = 0.001,
) -> dict[str, object]:
    """Return row-wise cosine and element-wise error parity metrics."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if (
        reference.shape != candidate.shape
        or reference.ndim != 2
        or reference.shape[0] == 0
        or reference.shape[1] == 0
    ):
        raise ValueError(
            "embedding matrices must be non-empty and have identical 2-D shapes"
        )
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("embedding matrices must contain only finite values")

    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    if np.any(reference_norm == 0.0) or np.any(candidate_norm == 0.0):
        raise ValueError("embedding rows must have non-zero norm")

    cosines = np.sum(reference * candidate, axis=1) / (
        reference_norm * candidate_norm
    )
    minimum_cosine = float(np.min(cosines))
    maximum_absolute_error = float(np.max(np.abs(reference - candidate)))
    return {
        "rows": int(reference.shape[0]),
        "columns": int(reference.shape[1]),
        "minimum_cosine": minimum_cosine,
        "maximum_absolute_error": maximum_absolute_error,
        "minimum_cosine_required": float(min_cosine),
        "maximum_absolute_error_allowed": float(max_abs_error),
        "passed": (
            minimum_cosine >= min_cosine
            and maximum_absolute_error <= max_abs_error
        ),
    }


def project_pipeline_p95(
    *,
    python_pipeline_p95_ms: float,
    python_feature_p95_ms: float,
    native_feature_p95_ms: float,
) -> float:
    """Replace the measured Python feature stage with its native equivalent."""
    values = (
        python_pipeline_p95_ms,
        python_feature_p95_ms,
        native_feature_p95_ms,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("latencies must be finite and non-negative")
    if python_feature_p95_ms > python_pipeline_p95_ms:
        raise ValueError("Python feature P95 cannot exceed pipeline P95")
    return (
        float(python_pipeline_p95_ms)
        - float(python_feature_p95_ms)
        + float(native_feature_p95_ms)
    )


def evaluate_integration_gate(
    *,
    embedding_metrics: Mapping[str, object],
    labels_match: bool,
    review_decisions_match: bool,
    python_feature_p95_ms: float,
    native_feature_p95_ms: float,
    python_pipeline_p95_ms: float,
) -> dict[str, object]:
    """Evaluate the approved correctness and 15% performance gates."""
    if python_feature_p95_ms <= 0.0 or python_pipeline_p95_ms <= 0.0:
        raise ValueError("Python latency baselines must be positive")

    projected_pipeline_p95_ms = project_pipeline_p95(
        python_pipeline_p95_ms=python_pipeline_p95_ms,
        python_feature_p95_ms=python_feature_p95_ms,
        native_feature_p95_ms=native_feature_p95_ms,
    )
    native_feature_improvement_ratio = (
        python_feature_p95_ms - native_feature_p95_ms
    ) / python_feature_p95_ms
    projected_pipeline_improvement_ratio = (
        python_pipeline_p95_ms - projected_pipeline_p95_ms
    ) / python_pipeline_p95_ms
    embedding_gate_passed = bool(embedding_metrics.get("passed", False))
    feature_gate_passed = native_feature_improvement_ratio >= 0.15
    projected_pipeline_gate_passed = projected_pipeline_improvement_ratio >= 0.15

    return {
        "embedding_metrics": dict(embedding_metrics),
        "labels_match": bool(labels_match),
        "review_decisions_match": bool(review_decisions_match),
        "python_feature_p95_ms": float(python_feature_p95_ms),
        "native_feature_p95_ms": float(native_feature_p95_ms),
        "python_pipeline_p95_ms": float(python_pipeline_p95_ms),
        "projected_pipeline_p95_ms": projected_pipeline_p95_ms,
        "native_feature_improvement_ratio": native_feature_improvement_ratio,
        "projected_pipeline_improvement_ratio": (
            projected_pipeline_improvement_ratio
        ),
        "embedding_gate_passed": embedding_gate_passed,
        "feature_gate_passed": feature_gate_passed,
        "projected_pipeline_gate_passed": projected_pipeline_gate_passed,
        "passed": (
            embedding_gate_passed
            and bool(labels_match)
            and bool(review_decisions_match)
            and feature_gate_passed
            and projected_pipeline_gate_passed
        ),
    }


def _timing_summary(samples: Sequence[float]) -> dict[str, object]:
    values = np.asarray(list(samples), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("timing samples must be finite, non-negative, and non-empty")
    return {
        "samples": [float(value) for value in values],
        "p50": float(np.quantile(values, 0.50, method="linear")),
        "p95": float(np.quantile(values, 0.95, method="linear")),
        "p99": float(np.quantile(values, 0.99, method="linear")),
        "max": float(np.max(values)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"unable to hash input image: {path}") from exc
    return digest.hexdigest()


def _validate_image_paths(image_paths: Sequence[Path | str]) -> tuple[Path, ...]:
    paths = tuple(Path(path) for path in image_paths)
    if not paths:
        raise ValueError("at least one image is required")
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"IMAGE_UNREADABLE: {path}")
    return paths


def _load_yaml(path: Path) -> Mapping[str, object]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the PaddleClas configuration") from exc
    try:
        with Path(path).open("r", encoding="utf-8-sig") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RuntimeError(f"unable to read PaddleClas configuration: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("PaddleClas configuration must be a mapping")
    return payload


def _parse_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{name} must be numeric")
    if isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "/" in text:
            parts = text.split("/")
            if len(parts) != 2:
                raise RuntimeError(f"{name} must be numeric")
            try:
                numerator, denominator = (float(part.strip()) for part in parts)
                parsed = numerator / denominator
            except (ValueError, ZeroDivisionError):
                raise RuntimeError(f"{name} must be numeric") from None
        else:
            try:
                parsed = float(text)
            except ValueError:
                raise RuntimeError(f"{name} must be numeric") from None
    else:
        raise RuntimeError(f"{name} must be numeric")
    if not np.isfinite(parsed):
        raise RuntimeError(f"{name} must be finite")
    return parsed


def _sequence_of_numbers(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or len(value) != length:
        raise RuntimeError(f"{name} must contain {length} values")
    return tuple(_parse_number(item, name) for item in value)


@dataclass(frozen=True)
class _PreprocessSpec:
    width: int
    height: int
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _read_preprocess_spec(config_path: Path) -> _PreprocessSpec:
    payload = _load_yaml(config_path)
    section = payload.get("RecPreProcess")
    operations = section.get("transform_ops") if isinstance(section, Mapping) else None
    if not isinstance(operations, Sequence):
        raise RuntimeError("RecPreProcess.transform_ops is missing")
    width, height = 224, 224
    scale = 1.0 / 255.0
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        resize = operation.get("ResizeImage")
        if isinstance(resize, Mapping) and "size" in resize:
            size = resize["size"]
            parsed_size = _sequence_of_numbers(size, "ResizeImage.size", 2)
            width, height = (int(item) for item in parsed_size)
            if width <= 0 or height <= 0 or any(item != int(item) for item in parsed_size):
                raise RuntimeError("ResizeImage.size must contain positive integers")
        normalize = operation.get("NormalizeImage")
        if isinstance(normalize, Mapping):
            if "scale" in normalize:
                scale = _parse_number(normalize["scale"], "NormalizeImage.scale")
            if "mean" in normalize:
                mean = _sequence_of_numbers(normalize["mean"], "NormalizeImage.mean", 3)
            if "std" in normalize:
                std = _sequence_of_numbers(normalize["std"], "NormalizeImage.std", 3)
    if scale <= 0.0 or any(value <= 0.0 for value in std):
        raise RuntimeError("normalization scale and standard deviations must be positive")
    return _PreprocessSpec(width, height, scale, mean, std)


def _format_float(value: float) -> str:
    return format(float(value), ".10g")


def _set_config_value(container: object, key: str, value: object) -> None:
    if isinstance(container, Mapping):
        try:
            container[key] = value  # type: ignore[index]
            return
        except (TypeError, AttributeError):
            pass
    try:
        setattr(container, key, value)
    except (TypeError, AttributeError) as exc:
        raise RuntimeError(f"unable to set PaddleClas configuration field: {key}") from exc


def _get_global_config(config: object) -> object:
    if isinstance(config, Mapping):
        try:
            return config["Global"]  # type: ignore[index]
        except KeyError as exc:
            raise RuntimeError("PaddleClas configuration has no Global section") from exc
    section = getattr(config, "Global", None)
    if section is None:
        raise RuntimeError("PaddleClas configuration has no Global section")
    return section


def _read_rgb_images(paths: Sequence[Path]) -> tuple[list[np.ndarray], float]:
    from src.image_io import read_color_image

    started = time.perf_counter()
    rgb_images: list[np.ndarray] = []
    for path in paths:
        image = read_color_image(path)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"IMAGE_UNREADABLE: {path}")
        rgb_images.append(np.ascontiguousarray(image[:, :, ::-1]))
    elapsed = (time.perf_counter() - started) * 1000.0
    return rgb_images, elapsed


def _validate_embedding_batch(value: object, rows: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise RuntimeError("Python predictor returned an unexpected embedding shape")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Python predictor returned non-finite embeddings")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0.0) or not np.isfinite(norms).all():
        raise RuntimeError("Python predictor returned zero-norm embeddings")
    return matrix


def _close_predictor(predictor: object) -> None:
    for target in (predictor, getattr(predictor, "predictor", None)):
        if target is None:
            continue
        for name in ("close", "destroy", "shutdown"):
            method = getattr(target, name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                return


def run_python_embeddings(
    project_root: Path,
    model_dir: Path,
    config_path: Path,
    image_paths: Sequence[Path],
    *,
    warmup: int,
    iterations: int,
    threads: int,
) -> EmbeddingRun:
    """Run the repository's current PaddleClas ``RecPredictor`` on ordered images."""
    if warmup < 0 or iterations <= 0 or threads <= 0:
        raise ValueError("warmup must be non-negative; iterations and threads must be positive")
    paths = _validate_image_paths(image_paths)
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_dir}")
    if not config_path.is_file():
        raise RuntimeError(f"configuration file does not exist: {config_path}")
    model_fingerprint = _model_fingerprint(model_dir)

    root = Path(project_root)
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)
    try:
        import paddle
        from paddleclas.deploy.python.predict_rec import RecPredictor
        from paddleclas.deploy.utils import config as paddle_config
        from src.paddleclas_inference_compat import (
            create_rec_predictor,
            install_optional_sklearn_stubs,
            prepare_paddle_model_path,
        )

        install_optional_sklearn_stubs()
        paddle.set_device("cpu")
        config = paddle_config.get_config(str(config_path), show=False)
        global_config = _get_global_config(config)
        paddle_model_dir = prepare_paddle_model_path(model_dir)
        _set_config_value(global_config, "rec_inference_model_dir", str(paddle_model_dir))
        _set_config_value(global_config, "use_gpu", False)
        _set_config_value(global_config, "enable_mkldnn", True)
        _set_config_value(global_config, "cpu_num_threads", int(threads))
        _set_config_value(global_config, "enable_benchmark", False)
        predictor = create_rec_predictor(RecPredictor, config, paddle, paddle_model_dir)
    except Exception as exc:
        raise RuntimeError(f"PYTHON_MODEL_LOAD_FAILED: {exc}") from exc

    decode_samples: list[float] = []
    feature_samples: list[float] = []
    total_samples: list[float] = []
    last: np.ndarray | None = None

    def run_once() -> np.ndarray:
        total_started = time.perf_counter()
        rgb_images, decode_ms = _read_rgb_images(paths)
        feature_started = time.perf_counter()
        try:
            returned = predictor.predict(rgb_images, feature_normalize=True)
        except Exception as exc:
            raise RuntimeError(f"PYTHON_INFERENCE_FAILED: {exc}") from exc
        feature_ms = (time.perf_counter() - feature_started) * 1000.0
        matrix = _validate_embedding_batch(returned, len(paths))
        decode_samples.append(decode_ms)
        feature_samples.append(feature_ms)
        total_samples.append((time.perf_counter() - total_started) * 1000.0)
        return matrix

    try:
        for _ in range(warmup):
            rgb_images, _ = _read_rgb_images(paths)
            try:
                _validate_embedding_batch(predictor.predict(rgb_images, feature_normalize=True), len(paths))
            except Exception as exc:
                raise RuntimeError(f"PYTHON_WARMUP_FAILED: {exc}") from exc
        for _ in range(iterations):
            last = run_once()
    finally:
        _close_predictor(predictor)
    assert last is not None
    fingerprint = model_fingerprint
    timings: dict[str, object] = {
        "decode": _timing_summary(decode_samples),
        "feature": _timing_summary(feature_samples),
        "inference": _timing_summary(feature_samples),
        "total": _timing_summary(total_samples),
        "stage_aggregation": "wall",
        "input_sha256": [_sha256_file(path) for path in paths],
    }
    return EmbeddingRun(last, tuple(str(path) for path in paths), timings, fingerprint)


def _model_fingerprint(model_dir: Path) -> str:
    from src.model_fingerprint import model_directory_sha256

    try:
        return model_directory_sha256(Path(model_dir))
    except OSError as exc:
        raise RuntimeError(f"unable to fingerprint model directory: {model_dir}") from exc


def _native_timing_p95(timings: Mapping[str, object], stage: str) -> float:
    value = timings.get(stage)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"native timings missing stage: {stage}")
    p95 = value.get("p95")
    if isinstance(p95, bool) or not isinstance(p95, (int, float)) or not np.isfinite(p95) or p95 < 0.0:
        raise RuntimeError(f"native timings have invalid p95 for stage: {stage}")
    return float(p95)


def validate_native_report(
    report_path: Path,
    *,
    expected_images: Sequence[Path | str],
    expected_model_fingerprint: str | None = None,
    expected_input_hashes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate the native harness contract before consuming its binary output."""
    try:
        with Path(report_path).open("r", encoding="utf-8-sig") as stream:
            report = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read native report: {report_path}") from exc
    if not isinstance(report, dict):
        raise RuntimeError("native report must be a JSON object")
    if report.get("schema_version") != 1:
        raise RuntimeError("native report schema_version is invalid")
    if report.get("ok") is not True:
        error = report.get("error")
        detail = error.get("message") if isinstance(error, Mapping) else "unknown native error"
        raise RuntimeError(f"native inference failed: {detail}")
    expected = [str(path) for path in expected_images]
    ordered = report.get("ordered_images")
    if ordered != expected:
        raise RuntimeError("native report ordered_images do not match requested order")
    if report.get("input_count") != len(expected):
        raise RuntimeError("native report input_count does not match ordered_images")
    dimension = report.get("feature_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise RuntimeError("native report feature_dimension is invalid")
    timings = report.get("timings_ms")
    if not isinstance(timings, Mapping):
        raise RuntimeError("native report timings_ms is missing")
    for stage in _REQUIRED_NATIVE_TIMING_STAGES:
        value = timings.get(stage)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"native report timing stage is missing: {stage}")
        samples = value.get("samples")
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes, bytearray)) or not samples:
            raise RuntimeError(f"native report timing samples are missing: {stage}")
        for key in ("p50", "p95", "p99", "max"):
            number = value.get(key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or number < 0.0:
                raise RuntimeError(f"native report timing value is invalid: {stage}.{key}")
    if expected_model_fingerprint is not None and "model_fingerprint" in report:
        if report.get("model_fingerprint") != expected_model_fingerprint:
            raise RuntimeError("native report model_fingerprint does not match")
    if expected_input_hashes is not None and "input_sha256" in report:
        if report.get("input_sha256") != list(expected_input_hashes):
            raise RuntimeError("native report input_sha256 does not match")
    return report


def run_native_embeddings(
    executable: Path,
    model_dir: Path,
    config_path: Path,
    image_paths: Sequence[Path],
    output_dir: Path,
    *,
    warmup: int,
    iterations: int,
    workers: int,
    batch_size: int,
    threads: int,
) -> EmbeddingRun:
    """Invoke the isolated native harness once and validate its artifacts."""
    if warmup < 0 or iterations <= 0 or workers <= 0 or batch_size <= 0 or threads <= 0:
        raise ValueError("warmup must be non-negative; iterations, workers, batch_size, and threads must be positive")
    executable = Path(executable)
    if not executable.is_file():
        raise RuntimeError(f"native executable does not exist: {executable}")
    model_dir = Path(model_dir)
    config_path = Path(config_path)
    if not model_dir.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_dir}")
    if not config_path.is_file():
        raise RuntimeError(f"configuration file does not exist: {config_path}")
    paths = _validate_image_paths(image_paths)
    spec = _read_preprocess_spec(config_path)
    input_hashes = [_sha256_file(path) for path in paths]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    image_list = output_dir / f"images-{token}.json"
    report_path = output_dir / f"native-{token}.json"
    embeddings_path = output_dir / f"embeddings-{token}.f32"
    image_list.write_text(json.dumps([str(path) for path in paths], ensure_ascii=False), encoding="utf-8")
    command = [
        str(executable),
        "--model-dir", str(model_dir),
        "--image-list", str(image_list),
        "--report", str(report_path),
        "--dump-embeddings", str(embeddings_path),
        "--workers", str(workers),
        "--batch-size", str(batch_size),
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--threads", str(threads),
        "--input-width", str(spec.width),
        "--input-height", str(spec.height),
        "--scale", _format_float(spec.scale),
        "--mean-rgb", ",".join(_format_float(value) for value in spec.mean),
        "--std-rgb", ",".join(_format_float(value) for value in spec.std),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native process failed").strip()
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                error = report.get("error") if isinstance(report, Mapping) else None
                if isinstance(error, Mapping) and error.get("message"):
                    detail = f"{error.get('code', 'NATIVE_ERROR')}: {error['message']} ({detail})"
            except (OSError, json.JSONDecodeError):
                pass
        raise RuntimeError(f"native inference failed with exit code {completed.returncode}: {detail}")
    report = validate_native_report(
        report_path,
        expected_images=paths,
        expected_input_hashes=input_hashes,
    )
    dimension = int(report["feature_dimension"])
    if not embeddings_path.is_file():
        raise RuntimeError("native embedding dump is missing")
    embeddings = read_f32_matrix(embeddings_path, len(paths), dimension)
    timings = dict(report["timings_ms"])
    timings.update({
        "stage_aggregation": report.get("stage_aggregation", "unknown"),
        "input_sha256": input_hashes,
        "native_stdout": completed.stdout,
        "native_stderr": completed.stderr,
        "report_path": str(report_path),
        "embedding_path": str(embeddings_path),
    })
    return EmbeddingRun(
        embeddings,
        tuple(str(path) for path in paths),
        timings,
        _model_fingerprint(model_dir),
    )


def _decision_field(result: Mapping[str, object], field: str) -> object:
    if field == "label":
        return result.get("label")
    if field == "needs_review":
        return bool(result.get("needs_review"))
    if field == "global_label":
        return result.get("global_label", result.get("global_prediction", result.get("label")))
    if field == "geometry_applied":
        if "geometry_applied" in result:
            return bool(result["geometry_applied"])
        status = result.get("geometry_status")
        return status == "active" if status is not None else None
    if field in ("front_score", "back_score"):
        direct = result.get(field)
        if direct is not None:
            return direct
        scores = result.get("global_scores")
        if isinstance(scores, Mapping):
            return scores.get(field.removesuffix("_score"))
        return None
    if field == "margin":
        return result.get("margin", result.get("global_margin", result.get("decision_margin")))
    return result.get(field)


def compare_decisions(
    reference: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    *,
    score_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Compare observable orientation decisions without comparing timing noise."""
    if score_tolerance < 0.0 or not np.isfinite(score_tolerance):
        raise ValueError("score_tolerance must be finite and non-negative")
    mismatches: list[dict[str, object]] = []
    if len(reference) != len(candidate):
        mismatches.append({"index": None, "field": "count", "reference": len(reference), "candidate": len(candidate)})
    for index, (left, right) in enumerate(zip(reference, candidate)):
        for field in _EXACT_DECISION_FIELDS:
            left_value = _decision_field(left, field)
            right_value = _decision_field(right, field)
            if left_value != right_value:
                mismatches.append({"index": index, "field": field, "reference": left_value, "candidate": right_value})
        for field in _SCORE_DECISION_FIELDS:
            left_value = _decision_field(left, field)
            right_value = _decision_field(right, field)
            if left_value is None and right_value is None:
                continue
            try:
                left_number = float(left_value)
                right_number = float(right_value)
            except (TypeError, ValueError):
                mismatches.append({"index": index, "field": field, "reference": left_value, "candidate": right_value})
                continue
            if not np.isfinite(left_number) or not np.isfinite(right_number) or abs(left_number - right_number) > score_tolerance:
                mismatches.append({
                    "index": index,
                    "field": field,
                    "reference": left_value,
                    "candidate": right_value,
                    "absolute_error": abs(left_number - right_number),
                })
    return {
        "count": min(len(reference), len(candidate)),
        "fields": list(DECISION_FIELDS),
        "score_tolerance": float(score_tolerance),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _slot_digest(image: np.ndarray) -> str:
    normalized = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(normalized.shape).encode("ascii"))
    digest.update(normalized.dtype.str.encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def replay_fast_decisions(
    classifier: object,
    cache: object,
    image_paths: Sequence[Path],
    native_slot_embeddings: Mapping[str, np.ndarray],
    *,
    library_revision: int,
) -> dict[str, object]:
    """Replay the fast head with native slot vectors while retaining Python geometry."""
    if library_revision < 0:
        raise ValueError("library_revision must be non-negative")
    engine = getattr(classifier, "fast_engine", None)
    runtime = getattr(cache, "fast_runtime", None)
    if engine is None or runtime is None:
        raise RuntimeError("fast runtime cache and engine are required for decision replay")
    paths = _validate_image_paths(image_paths)
    try:
        from src.fast_orientation import FastOrientationEngine
    except ImportError as exc:
        raise RuntimeError("FastOrientationEngine is unavailable") from exc

    def lookup(images: Sequence[np.ndarray]) -> list[np.ndarray]:
        returned: list[np.ndarray] = []
        for image in images:
            digest = _slot_digest(image)
            vector = native_slot_embeddings.get(digest)
            if vector is None:
                raise RuntimeError(f"native slot embedding is missing: {digest}")
            normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
            if normalized.size == 0 or not np.isfinite(normalized).all():
                raise RuntimeError(f"native slot embedding is invalid: {digest}")
            returned.append(normalized.copy())
        return returned

    native_engine = FastOrientationEngine(
        lookup,
        engine.geometry,
        image_reader=engine.image_reader,
        deduplicate_identical_slots=False,
    )
    python_results: list[Mapping[str, object]] = []
    native_results: list[Mapping[str, object]] = []
    for path in paths:
        python_result = classifier.predict_fast_with_cache(
            cache, path, library_revision=library_revision
        )
        native_classifier = copy(classifier)
        native_classifier.fast_engine = native_engine
        native_result = native_classifier.predict_fast_with_cache(
            cache, path, library_revision=library_revision
        )
        python_results.append(python_result)
        native_results.append(native_result)
    parity = compare_decisions(python_results, native_results)
    return {
        "passed": bool(parity["passed"]),
        "count": len(paths),
        "ordered_images": [str(path) for path in paths],
        "python_results": [dict(item) for item in python_results],
        "native_results": [dict(item) for item in native_results],
        "mismatches": parity["mismatches"],
    }


def _timing_p95(run: EmbeddingRun, stage: str) -> float:
    value = run.timings_ms.get(stage)
    if not isinstance(value, Mapping) or "p95" not in value:
        raise RuntimeError(f"embedding run timing stage is missing: {stage}")
    p95 = value["p95"]
    if isinstance(p95, bool) or not isinstance(p95, (int, float)) or not np.isfinite(p95) or p95 < 0.0:
        raise RuntimeError(f"embedding run timing p95 is invalid: {stage}")
    return float(p95)


def build_comparison_report(
    python_run: EmbeddingRun,
    native_run: EmbeddingRun,
    *,
    labels_match: bool = True,
    review_decisions_match: bool = True,
    python_pipeline_p95_ms: float | None = None,
    decision_parity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble the evidence JSON consumed by the integration decision."""
    if python_run.ordered_images != native_run.ordered_images:
        raise RuntimeError("Python and native ordered_images do not match")
    embedding_metrics = compare_embeddings(python_run.embeddings, native_run.embeddings)
    python_feature_p95_ms = _timing_p95(python_run, "total")
    native_feature_p95_ms = _timing_p95(native_run, "total")
    result: dict[str, object] = {
        "schema_version": 1,
        "ordered_images": list(python_run.ordered_images),
        "input_sha256": list(python_run.timings_ms.get("input_sha256", [])),
        "python_model_fingerprint": python_run.model_fingerprint,
        "native_model_fingerprint": native_run.model_fingerprint,
        "embedding_metrics": embedding_metrics,
        "labels_match": bool(labels_match),
        "review_decisions_match": bool(review_decisions_match),
        "python_timings_ms": dict(python_run.timings_ms),
        "native_timings_ms": dict(native_run.timings_ms),
    }
    if decision_parity is not None:
        result["decision_parity"] = dict(decision_parity)
        result["labels_match"] = bool(decision_parity.get("labels_match", labels_match))
        result["review_decisions_match"] = bool(
            decision_parity.get("review_decisions_match", review_decisions_match)
        )
    if python_pipeline_p95_ms is not None:
        gate = evaluate_integration_gate(
            embedding_metrics=embedding_metrics,
            labels_match=bool(result["labels_match"]),
            review_decisions_match=bool(result["review_decisions_match"]),
            python_feature_p95_ms=python_feature_p95_ms,
            native_feature_p95_ms=native_feature_p95_ms,
            python_pipeline_p95_ms=float(python_pipeline_p95_ms),
        )
        result["gate"] = gate
    else:
        result["gate"] = {
            "embedding_gate_passed": bool(embedding_metrics["passed"]),
            "labels_match": bool(result["labels_match"]),
            "review_decisions_match": bool(result["review_decisions_match"]),
            "passed": bool(
                embedding_metrics["passed"]
                and result["labels_match"]
                and result["review_decisions_match"]
            ),
        }
    return result
