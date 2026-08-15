"""Evaluate PP-ShiTuV2 retrieval with ALIKED + LightGlue local overrides."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aliked_lightglue_matcher import build_models, extract_features, score_feature_pair
from src.evaluate_aliked_lightglue import MAX_NUM_KEYPOINTS, ROI_RATIO, decide_label
from src.shitu_baseline import classify_embedding, split_labels


SEED = 20260813
TEMPLATE_COUNT = 5
GLOBAL_MARGIN_THRESHOLD = 0.05
LOCAL_MARGIN_THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
LOCAL_MIN_SCORE = 4.0


def fuse_decision(
    global_prediction: str,
    global_margin: float,
    local_prediction: str,
    local_margin: float,
    local_margin_threshold: float,
    global_margin_threshold: float,
) -> str:
    """Override an ambiguous global result only with decisive local evidence."""
    if (
        global_margin <= global_margin_threshold
        and local_prediction != "uncertain"
        and local_margin >= local_margin_threshold
    ):
        return local_prediction
    return global_prediction


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _global_predictor():
    from paddleclas.deploy.python.predict_rec import RecPredictor
    from paddleclas.deploy.utils import config as paddle_config

    config_path = ROOT / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml"
    model_dir = ROOT / "third_party" / "models" / "shiru_rec" / "general_PPLCNetV2_base_pretrained_v1.0_infer"
    config = paddle_config.get_config(str(config_path), show=False)
    config.Global.rec_inference_model_dir = str(model_dir)
    config.Global.use_gpu = True
    config.Global.enable_mkldnn = False
    config.Global.enable_benchmark = False
    config.Global.gpu_mem = 1024
    return RecPredictor(config)


def _embedding(predictor, image: np.ndarray) -> np.ndarray:
    return np.asarray(predictor.predict([image[:, :, ::-1]])[0], dtype=np.float32)


def _summary(rows: list[dict], prediction_key: str) -> dict:
    correct = sum(row["actual"] == row[prediction_key] for row in rows)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "confusion_matrix": dict(Counter(f"{row['actual']}->{row[prediction_key]}" for row in rows)),
    }


def evaluate_dataset(data_dir: Path, global_predictor, extractor, matcher, device) -> dict:
    """Evaluate global, local, and confidence-gated fusion on one fixed split."""
    template_paths, held_out = split_labels(data_dir, TEMPLATE_COUNT, SEED)
    global_vectors = {}
    local_templates = {}
    for label, paths in template_paths.items():
        images = [(path, _read_image(path)) for path in paths]
        global_vectors[label] = np.stack([_embedding(global_predictor, image) for _, image in images])
        local_templates[label] = [
            (path, extract_features(image, extractor, device, ROI_RATIO)) for path, image in images
        ]

    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            image = _read_image(path)
            global_prediction, global_scores, global_margin = classify_embedding(
                _embedding(global_predictor, image), global_vectors
            )
            query_features = extract_features(image, extractor, device, ROI_RATIO)
            roi_shape = (round(image.shape[0] * ROI_RATIO), round(image.shape[1] * ROI_RATIO))
            local_scores = {}
            for label, candidates in local_templates.items():
                local_scores[label] = max(
                    score_feature_pair(query_features, features, roi_shape, matcher)["score"]
                    for _, features in candidates
                )
            local_prediction, local_margin = decide_label(local_scores, LOCAL_MIN_SCORE, 0.5)
            row = {
                "file": path.name,
                "actual": actual,
                "global_prediction": global_prediction,
                "global_margin": float(global_margin),
                "global_scores": global_scores,
                "local_prediction": local_prediction,
                "local_margin": float(local_margin),
                "local_scores": local_scores,
            }
            for threshold in LOCAL_MARGIN_THRESHOLDS:
                row[f"fusion_{threshold:g}"] = fuse_decision(
                    global_prediction,
                    float(global_margin),
                    local_prediction,
                    float(local_margin),
                    threshold,
                    GLOBAL_MARGIN_THRESHOLD,
                )
            rows.append(row)

    report = {
        "dataset": data_dir.name,
        "settings": {
            "seed": SEED,
            "template_count": TEMPLATE_COUNT,
            "global_margin_threshold": GLOBAL_MARGIN_THRESHOLD,
            "local_margin_thresholds": LOCAL_MARGIN_THRESHOLDS,
            "local_min_score": LOCAL_MIN_SCORE,
            "roi_ratio": ROI_RATIO,
            "max_num_keypoints": MAX_NUM_KEYPOINTS,
            "note": "Threshold sweep is exploratory and must be validated on a separate registration set before deployment.",
        },
        "templates": {label: [path.name for path in paths] for label, paths in template_paths.items()},
        "global": _summary(rows, "global_prediction"),
        "local": _summary(rows, "local_prediction"),
        "fusion": {str(threshold): _summary(rows, f"fusion_{threshold:g}") for threshold in LOCAL_MARGIN_THRESHOLDS},
        "rows": rows,
    }
    output_dir = ROOT / "reports" / "global_local_fusion" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    global_predictor = _global_predictor()
    extractor, matcher, device = build_models(MAX_NUM_KEYPOINTS)
    for data_dir in args.datasets:
        report = evaluate_dataset(data_dir, global_predictor, extractor, matcher, device)
        print(report["dataset"], report["global"], report["fusion"])


if __name__ == "__main__":
    main()
