"""Evaluate PP-ShiTuV2 few-shot orientation matching on one part folder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from shitu_baseline import build_report, classify_embedding, split_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--template-count", type=int, default=5)
    parser.add_argument("--low-confidence-margin", type=float, default=0.05)
    return parser.parse_args()


def make_predictor(config_path: Path, model_dir: Path):
    project_root = Path(__file__).resolve().parents[1]
    paddleclas_root = project_root / "third_party" / "PaddleClas"
    sys.path.insert(0, str(paddleclas_root))
    from paddleclas.deploy.python.predict_rec import RecPredictor
    from paddleclas.deploy.utils import config as paddle_config

    config = paddle_config.get_config(str(config_path), show=False)
    config.Global.rec_inference_model_dir = str(model_dir)
    config.Global.use_gpu = True
    config.Global.enable_mkldnn = False
    config.Global.enable_benchmark = False
    config.Global.gpu_mem = 1024
    return RecPredictor(config)


def embed(predictor, path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return predictor.predict([image[:, :, ::-1]])[0]


def main() -> None:
    args = parse_args()
    if not (args.model_dir / "inference.pdmodel").is_file():
        raise FileNotFoundError(f"Missing inference.pdmodel in {args.model_dir}")
    templates, test_paths = split_labels(args.data_dir, args.template_count, args.seed)
    predictor = make_predictor(args.config, args.model_dir)
    template_embeddings = {
        label: np.stack([embed(predictor, path) for path in paths])
        for label, paths in templates.items()
    }
    rows = []
    for actual, paths in test_paths.items():
        for path in paths:
            predicted, scores, margin = classify_embedding(embed(predictor, path), template_embeddings)
            rows.append({
                "file": str(path), "actual": actual, "predicted": predicted,
                "correct": actual == predicted, "score_0": scores["0"],
                "score_1": scores["1"], "margin": margin,
                "low_confidence": margin < args.low_confidence_margin,
            })
    report = build_report(rows, args.low_confidence_margin)
    report["seed"] = args.seed
    report["template_count"] = args.template_count
    report["low_confidence_margin"] = args.low_confidence_margin
    report["templates"] = {label: [str(path) for path in paths] for label, paths in templates.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"baseline-{args.seed}"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accuracy={report['accuracy']:.4f} correct={report['correct']}/{report['total']} low_confidence={report['low_confidence']} csv={csv_path} json={json_path}")


if __name__ == "__main__":
    main()
