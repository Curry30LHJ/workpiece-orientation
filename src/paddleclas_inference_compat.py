"""Import-only compatibility for PaddleClas inference without scikit-learn."""

from __future__ import annotations

import sys
import types
from typing import Any


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
