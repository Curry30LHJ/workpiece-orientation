"""Benchmark ALIKED + LightGlue front/back template matching."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aliked_lightglue_matcher import build_models, extract_features, score_feature_pair
from src.shitu_baseline import split_labels


SEED = 20260813
TEMPLATE_COUNT = 5
ROI_RATIO = 0.95
MAX_NUM_KEYPOINTS = 1024
MIN_SCORE = 4.0
MIN_MARGIN = 0.5


def decide_label(scores: dict[str, float], min_score: float, min_margin: float) -> tuple[str, float]:
    """Return the strongest side when its local geometric evidence is clear."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    margin = scores[ordered[0]] - scores[ordered[1]]
    if scores[ordered[0]] < min_score or margin < min_margin:
        return "uncertain", margin
    return ordered[0], margin


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def _summary(rows: list[dict]) -> dict:
    forced = [row for row in rows if row["predicted"] != "uncertain"]
    correct = sum(row["actual"] == row["predicted"] for row in rows)
    forced_correct = sum(row["actual"] == row["predicted"] for row in forced)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "forced_total": len(forced),
        "forced_correct": forced_correct,
        "forced_accuracy": forced_correct / len(forced) if forced else 0.0,
        "uncertain": len(rows) - len(forced),
        "confusion_matrix": dict(Counter(f"{row['actual']}->{row['predicted']}" for row in rows)),
    }


def evaluate_dataset(data_dir: Path, extractor, matcher, device) -> dict:
    """Run the deterministic 5+5 benchmark for one dataset."""
    template_paths, held_out = split_labels(data_dir, TEMPLATE_COUNT, SEED)
    templates = {
        label: [
            (path, extract_features(_read_image(path), extractor, device, ROI_RATIO))
            for path in paths
        ]
        for label, paths in template_paths.items()
    }
    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            image = _read_image(path)
            query_features = extract_features(image, extractor, device, ROI_RATIO)
            roi_shape = (round(image.shape[0] * ROI_RATIO), round(image.shape[1] * ROI_RATIO))
            scores = {}
            best = {}
            for label, candidates in templates.items():
                best_path = None
                best_score = None
                for template_path, template_features in candidates:
                    candidate = score_feature_pair(query_features, template_features, roi_shape, matcher)
                    if best_score is None or candidate["score"] > best_score["score"]:
                        best_path, best_score = template_path, candidate
                scores[label] = best_score["score"]
                best[label] = {"template": best_path.name, **best_score}
            predicted, margin = decide_label(scores, MIN_SCORE, MIN_MARGIN)
            rows.append({
                "file": path.name,
                "actual": actual,
                "predicted": predicted,
                "margin": margin,
                "scores": scores,
                "best_templates": best,
            })
    report = {
        "dataset": data_dir.name,
        "settings": {
            "seed": SEED,
            "template_count": TEMPLATE_COUNT,
            "roi_ratio": ROI_RATIO,
            "max_num_keypoints": MAX_NUM_KEYPOINTS,
            "min_score": MIN_SCORE,
            "min_margin": MIN_MARGIN,
        },
        "templates": {label: [path.name for path in paths] for label, paths in template_paths.items()},
        "summary": _summary(rows),
        "rows": rows,
    }
    output_dir = ROOT / "reports" / "aliked_lightglue" / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    extractor, matcher, device = build_models(MAX_NUM_KEYPOINTS)
    for data_dir in args.datasets:
        report = evaluate_dataset(data_dir, extractor, matcher, device)
        print(report["dataset"], report["summary"])


if __name__ == "__main__":
    main()
