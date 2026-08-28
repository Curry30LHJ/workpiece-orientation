"""Import-only compatibility for PaddleClas inference without scikit-learn."""

from __future__ import annotations

import sys
import types
from typing import Any
from pathlib import Path


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
    previous_mkldnn = None
    try:
        # PaddleClas chooses Config(model_dir, 'inference') for Paddle >=2.6,
        # which requires inference.json.  A temporary 2.5 marker selects its
        # legacy Config(model_file, params_file) branch without touching files.
        paddle.__version__ = "2.5.0"
        if global_config is not None:
            if isinstance(global_config, dict):
                previous_mkldnn = global_config.get("enable_mkldnn")
                global_config["enable_mkldnn"] = False
            else:
                previous_mkldnn = getattr(global_config, "enable_mkldnn", None)
                global_config.enable_mkldnn = False
        return rec_predictor(config)
    finally:
        if global_config is not None and previous_mkldnn is not None:
            if isinstance(global_config, dict):
                global_config["enable_mkldnn"] = previous_mkldnn
            else:
                global_config.enable_mkldnn = previous_mkldnn
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
