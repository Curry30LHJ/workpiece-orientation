from pathlib import Path

import numpy as np

from src.shitu_baseline import classify_embedding, split_labels


def test_split_labels_selects_five_templates_per_label(tmp_path: Path):
    for label in ("0", "1"):
        label_dir = tmp_path / label
        label_dir.mkdir()
        for index in range(7):
            (label_dir / f"{index}.png").write_bytes(b"image")

    templates, tests = split_labels(tmp_path, template_count=5, seed=7)

    assert {label: len(paths) for label, paths in templates.items()} == {"0": 5, "1": 5}
    assert {label: len(paths) for label, paths in tests.items()} == {"0": 2, "1": 2}
    assert not set(templates["0"]) & set(tests["0"])


def test_split_labels_supports_asymmetric_front_and_back_counts(tmp_path: Path):
    for label in ("0", "1"):
        label_dir = tmp_path / label
        label_dir.mkdir()
        for index in range(8):
            (label_dir / f"{index}.png").write_bytes(b"image")

    templates, tests = split_labels(
        tmp_path, template_count=2, back_template_count=5, seed=7
    )

    assert {label: len(paths) for label, paths in templates.items()} == {"0": 2, "1": 5}
    assert {label: len(paths) for label, paths in tests.items()} == {"0": 6, "1": 3}


def test_classify_embedding_returns_best_label_and_margin():
    templates = {
        "0": np.array([[1.0, 0.0], [0.98, 0.02]], dtype=np.float32),
        "1": np.array([[0.0, 1.0], [0.02, 0.98]], dtype=np.float32),
    }

    predicted, scores, margin = classify_embedding(
        np.array([0.99, 0.01], dtype=np.float32), templates)

    assert predicted == "0"
    assert scores["0"] > scores["1"]
    assert margin == scores["0"] - scores["1"]
