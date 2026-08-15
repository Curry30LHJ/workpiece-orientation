"""Print PP-ShiTuV2 five-template held-out evaluation without writing files."""

from pathlib import Path

import cv2
import numpy as np

from shitu_baseline import build_report, classify_embedding, split_labels


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "1_M7"
MODEL_DIR = ROOT / "third_party" / "models" / "shiru_rec" / "general_PPLCNetV2_base_pretrained_v1.0_infer"
CONFIG_PATH = ROOT / "third_party" / "PaddleClas" / "deploy" / "configs" / "inference_general.yaml"
SEED = 20260813
TEMPLATE_COUNT = 5
MARGIN = 0.05


def embed(predictor, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    return predictor.predict([image[:, :, ::-1]])[0]


def main():
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
    vectors = {label: np.stack([embed(predictor, path) for path in paths]) for label, paths in templates.items()}
    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            predicted, scores, margin = classify_embedding(embed(predictor, path), vectors)
            rows.append({"file": path.name, "actual": actual, "predicted": predicted, "margin": margin})
    report = build_report(rows, MARGIN)
    print("REPORT", report)
    print("TEMPLATES", {label: [path.name for path in paths] for label, paths in templates.items()})
    for row in rows:
        if row["actual"] != row["predicted"] or row["margin"] < MARGIN:
            print("REVIEW", row)


if __name__ == "__main__":
    main()
