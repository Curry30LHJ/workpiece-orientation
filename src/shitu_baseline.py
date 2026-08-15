"""Small, deterministic utilities for PP-ShiTuV2 template matching."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}


def split_labels(data_dir: Path, template_count: int, seed: int):
    """Return independently shuffled template and held-out paths for labels 0 and 1."""
    templates = {}
    tests = {}
    for label in ("0", "1"):
        label_dir = data_dir / label
        if not label_dir.is_dir():
            raise ValueError(f"Missing label directory: {label_dir}")
        paths = sorted(path for path in label_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if len(paths) <= template_count:
            raise ValueError(f"Label {label} needs more than {template_count} images; found {len(paths)}")
        random.Random(seed).shuffle(paths)
        templates[label] = paths[:template_count]
        tests[label] = paths[template_count:]
    return templates, tests


def classify_embedding(query: np.ndarray, templates: dict[str, np.ndarray]):
    """Classify an L2-normalized vector using the highest cosine template score."""
    scores = {label: float(np.max(embeddings @ query)) for label, embeddings in templates.items()}
    ordered = sorted(scores, key=scores.get, reverse=True)
    return ordered[0], scores, scores[ordered[0]] - scores[ordered[1]]


def build_report(rows: list[dict], low_confidence_margin: float):
    correct = sum(row["actual"] == row["predicted"] for row in rows)
    confusion = {}
    for row in rows:
        key = f"{row['actual']}->{row['predicted']}"
        confusion[key] = confusion.get(key, 0) + 1
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "low_confidence": sum(row["margin"] < low_confidence_margin for row in rows),
        "confusion_matrix": confusion,
    }
