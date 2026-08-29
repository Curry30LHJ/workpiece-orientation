"""Import-only compatibility for PaddleClas inference without scikit-learn."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import types
from typing import Any
from pathlib import Path


_PADDLE_MODEL_TEMP_PATHS: set[Path] = set()


def _is_ascii_path(path: Path | str) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _windows_short_path(path: Path) -> str | None:
    """Return an ASCII 8.3 alias when Windows exposes one for ``path``."""

    if os.name != "nt":
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetShortPathNameW(
            str(path), buffer, len(buffer)
        )
        if length <= 0 or length >= len(buffer):
            return None
        value = buffer.value
        return value if _is_ascii_path(value) else None
    except (AttributeError, OSError):
        return None


def _ascii_temp_parent() -> Path | None:
    candidates = [
        Path(tempfile.gettempdir()),
        Path(os.environ.get("TEMP", "")) if os.environ.get("TEMP") else None,
        Path(os.environ.get("TMP", "")) if os.environ.get("TMP") else None,
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Temp",
        Path(os.environ.get("SystemDrive", "C:")) / "Temp",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            if not candidate.is_dir():
                continue
            alias = _windows_short_path(candidate)
            selected = Path(alias) if alias else candidate
            key = str(selected).casefold()
            if key in seen or not _is_ascii_path(selected):
                continue
            seen.add(key)
            return selected
        except OSError:
            continue
    return None


def _copy_model_to_ascii_path(source: Path) -> Path:
    parent = _ascii_temp_parent()
    if parent is None:
        raise RuntimeError(
            "Paddle model path contains non-ASCII characters and no ASCII temporary directory is available"
        )
    destination = Path(tempfile.mkdtemp(prefix="workpiece-model-", dir=str(parent)))
    if not _is_ascii_path(destination):
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError("temporary Paddle model path is not ASCII")
    try:
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            target = destination / relative
            if item.is_symlink():
                raise RuntimeError(f"model directory contains a symlink: {relative}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif item.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    _PADDLE_MODEL_TEMP_PATHS.add(destination)
    return destination


def prepare_paddle_model_path(model_dir: Path) -> Path:
    """Provide Paddle with a path it can open on Windows.

    Paddle's inference ``Config`` in the shipped legacy runtime still passes
    model filenames through a narrow C++ API.  A package extracted under a
    Chinese (or otherwise non-ASCII) directory can therefore fail before the
    model is loaded.  Prefer an 8.3 alias; when unavailable, copy the model to
    an ASCII temporary directory and retain it until process exit.
    """

    source = Path(model_dir)
    if os.name != "nt" or _is_ascii_path(source) or not source.is_dir():
        return source
    short = _windows_short_path(source)
    if short:
        return Path(short)
    return _copy_model_to_ascii_path(source)


def cleanup_paddle_model_paths() -> None:
    for path in list(_PADDLE_MODEL_TEMP_PATHS):
        shutil.rmtree(path, ignore_errors=True)
        _PADDLE_MODEL_TEMP_PATHS.discard(path)


atexit.register(cleanup_paddle_model_paths)


def uses_legacy_model_format(model_dir: Path) -> bool:
    """Return whether a PP-ShiTu directory uses Paddle's two-file format."""
    root = Path(model_dir)
    return (
        not (root / "inference.json").is_file()
        and (root / "inference.pdmodel").is_file()
        and (root / "inference.pdiparams").is_file()
    )


def create_rec_predictor(rec_predictor: Any, config: Any, paddle: Any, model_dir: Path) -> Any:
    """Construct RecPredictor while supporting legacy two-file model exports."""
    if not uses_legacy_model_format(model_dir):
        return rec_predictor(config)
    original_version = getattr(paddle, "__version__", None)
    global_config = getattr(config, "Global", None)
    if global_config is None and isinstance(config, dict):
        global_config = config.get("Global")
    previous_ir_optim = None
    try:
        # PaddleClas chooses Config(model_dir, 'inference') for Paddle >=2.6,
        # which requires inference.json.  A temporary 2.5 marker selects its
        # legacy Config(model_file, params_file) branch without touching files.
        paddle.__version__ = "2.5.0"
        if global_config is not None:
            if isinstance(global_config, dict):
                previous_ir_optim = global_config.get("ir_optim")
                global_config["ir_optim"] = False
            else:
                previous_ir_optim = getattr(global_config, "ir_optim", None)
                global_config.ir_optim = False
        return rec_predictor(config)
    finally:
        if global_config is not None and previous_ir_optim is not None:
            if isinstance(global_config, dict):
                global_config["ir_optim"] = previous_ir_optim
            else:
                global_config.ir_optim = previous_ir_optim
        if original_version is not None:
            paddle.__version__ = original_version


def install_optional_sklearn_stubs() -> None:
    """Register PaddleClas' metric names only when sklearn is unavailable.

    PaddleClas imports these symbols during ``RecPredictor`` construction, but
    fast inference never executes them.  Keeping the names as failing stubs
    avoids shipping the optional scikit-learn runtime in the frozen backend.
    """
    try:
        import sklearn.metrics  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        def unavailable(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("scikit-learn metrics are unavailable in the inference bundle")

        sklearn = types.ModuleType("sklearn")
        sklearn.__path__ = []  # type: ignore[attr-defined]
        metrics = types.ModuleType("sklearn.metrics")
        preprocessing = types.ModuleType("sklearn.preprocessing")
        for name in (
            "hamming_loss",
            "accuracy_score",
            "multilabel_confusion_matrix",
            "precision_recall_fscore_support",
            "average_precision_score",
        ):
            setattr(metrics, name, unavailable)
        preprocessing.binarize = unavailable
        sklearn.metrics = metrics  # type: ignore[attr-defined]
        sklearn.preprocessing = preprocessing  # type: ignore[attr-defined]
        sys.modules.setdefault("sklearn", sklearn)
        sys.modules.setdefault("sklearn.metrics", metrics)
        sys.modules.setdefault("sklearn.preprocessing", preprocessing)

    # PaddleClas' package initializer also imports ``faiss`` for its optional
    # gallery/search predictor.  Fast PP-ShiTu inference never constructs that
    # predictor, so an empty module is sufficient to keep the import lazy.
    try:
        import faiss  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        faiss_stub = types.ModuleType("faiss")

        def missing_faiss_attribute(name: str) -> Any:
            raise RuntimeError(f"faiss is unavailable in the inference bundle: {name}")

        faiss_stub.__getattr__ = missing_faiss_attribute  # type: ignore[attr-defined]
        sys.modules.setdefault("faiss", faiss_stub)
