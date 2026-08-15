"""Print a five-template PP-ShiTuV2 evaluation for a selected dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from print_shitu_baseline import DATA_DIR, MODEL_DIR, CONFIG_PATH, MARGIN, SEED, TEMPLATE_COUNT, embed
from shitu_baseline import build_report, classify_embedding, split_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=[path.name for path in (ROOT / "data").iterdir() if path.is_dir()])
    args = parser.parse_args()
    from paddleclas.deploy.python.predict_rec import RecPredictor
    from paddleclas.deploy.utils import config as paddle_config

    config = paddle_config.get_config(str(CONFIG_PATH), show=False)
    config.Global.rec_inference_model_dir = str(MODEL_DIR)
    config.Global.use_gpu = True
    config.Global.enable_mkldnn = False
    config.Global.enable_benchmark = False
    config.Global.gpu_mem = 1024
    predictor = RecPredictor(config)
    templates, held_out = split_labels(ROOT / "data" / args.dataset, TEMPLATE_COUNT, SEED)
    vectors = {label: __import__("numpy").stack([embed(predictor, path) for path in paths]) for label, paths in templates.items()}
    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            predicted, scores, margin = classify_embedding(embed(predictor, path), vectors)
            rows.append({"file": path.name, "actual": actual, "predicted": predicted, "margin": margin})
    print(args.dataset, build_report(rows, MARGIN))


if __name__ == "__main__":
    main()
