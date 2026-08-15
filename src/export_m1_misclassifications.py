"""Export the held-out M1 misclassifications from the fixed five-template split."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from print_shitu_baseline import CONFIG_PATH, MODEL_DIR, SEED, TEMPLATE_COUNT, embed
from shitu_baseline import classify_embedding, split_labels


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "1_M1"
OUTPUT_DIR = ROOT / "reports" / "1_M1_misclassifications"


def main() -> None:
    from paddleclas.deploy.python.predict_rec import RecPredictor
    from paddleclas.deploy.utils import config as paddle_config

    config = paddle_config.get_config(str(CONFIG_PATH), show=False)
    config.Global.rec_inference_model_dir = str(MODEL_DIR)
    config.Global.use_gpu = True
    config.Global.enable_mkldnn = False
    config.Global.enable_benchmark = False
    config.Global.gpu_mem = 1024
    predictor = RecPredictor(config)
    templates, held_out = split_labels(DATA_DIR, TEMPLATE_COUNT, SEED)
    embeddings = {
        label: np.stack([embed(predictor, path) for path in paths])
        for label, paths in templates.items()
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            predicted, scores, margin = classify_embedding(embed(predictor, path), embeddings)
            if actual == predicted:
                continue
            target = OUTPUT_DIR / f"actual-{actual}_predicted-{predicted}_{path.name}"
            shutil.copy2(path, target)
            rows.append({"source_file": str(path), "exported_file": target.name, "actual": actual, "predicted": predicted, "score_0": scores["0"], "score_1": scores["1"], "margin": margin})
    with (OUTPUT_DIR / "misclassifications.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_file", "exported_file", "actual", "predicted", "score_0", "score_1", "margin"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported={len(rows)} directory={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
