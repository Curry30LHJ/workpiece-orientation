"""Evaluate center-ROI SIFT matching on front/back template folders."""

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

from src.local_sift_matcher import extract_center_roi, score_pair
from src.shitu_baseline import split_labels


SEED = 20260813
TEMPLATE_COUNT = 5
ROI_RATIO = 0.8
MIN_SCORE = 4.0
MIN_MARGIN = 0.5
MAX_DIMENSION = 800


def decide_label(scores: dict[str, float], min_score: float, min_margin: float) -> tuple[str, float]:
    """Return the clear winning label or uncertain when evidence is weak."""
    ordered = sorted(scores, key=scores.get, reverse=True)
    margin = scores[ordered[0]] - scores[ordered[1]]
    if scores[ordered[0]] < min_score or margin < min_margin:
        return "uncertain", margin
    return ordered[0], margin


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, MAX_DIMENSION / max(height, width))
    return cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1.0 else image


def _inlier_match_image(query: np.ndarray, template: np.ndarray) -> np.ndarray:
    query_roi = extract_center_roi(query, ROI_RATIO)
    template_roi = extract_center_roi(template, ROI_RATIO)
    sift = cv2.SIFT_create()
    query_keypoints, query_descriptors = sift.detectAndCompute(cv2.cvtColor(query_roi, cv2.COLOR_BGR2GRAY), None)
    template_keypoints, template_descriptors = sift.detectAndCompute(cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY), None)
    if query_descriptors is None or template_descriptors is None:
        return np.hstack((query_roi, template_roi))
    raw_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_descriptors, template_descriptors, k=2)
    good_matches = [first for pair in raw_matches if len(pair) == 2 for first, second in [pair]
                    if first.distance < 0.75 * second.distance]
    if len(good_matches) < 4:
        return cv2.drawMatches(query_roi, query_keypoints, template_roi, template_keypoints, good_matches, None)
    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    template_points = np.float32([template_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(query_points, template_points, cv2.RANSAC, 5.0)
    selected = [match for match, keep in zip(good_matches, mask.ravel()) if keep] if mask is not None else []
    return cv2.drawMatches(query_roi, query_keypoints, template_roi, template_keypoints, selected, None)


def _score_labels(query: np.ndarray, templates: dict[str, list[tuple[Path, np.ndarray]]]):
    scores = {}
    winners = {}
    for label, candidates in templates.items():
        best_path, best_image = max(candidates, key=lambda item: score_pair(query, item[1], ROI_RATIO)["score"])
        result = score_pair(query, best_image, ROI_RATIO)
        scores[label] = result["score"]
        winners[label] = {"path": best_path, **result}
    return scores, winners


def _report(rows: list[dict]) -> dict:
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


def evaluate_dataset(data_dir: Path) -> dict:
    """Evaluate one dataset with the fixed 5+5 template split and write diagnostics."""
    templates_paths, held_out = split_labels(data_dir, TEMPLATE_COUNT, SEED)
    templates = {label: [(path, _read_image(path)) for path in paths] for label, paths in templates_paths.items()}
    output_dir = ROOT / "reports" / "local_sift" / data_dir.name
    matches_dir = output_dir / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for actual, paths in held_out.items():
        for path in paths:
            query = _read_image(path)
            scores, winners = _score_labels(query, templates)
            predicted, margin = decide_label(scores, MIN_SCORE, MIN_MARGIN)
            winner_label = max(scores, key=scores.get)
            row = {
                "file": path.name,
                "actual": actual,
                "predicted": predicted,
                "margin": margin,
                "scores": scores,
                "best_templates": {label: winners[label]["path"].name for label in winners},
                "best_details": {label: {key: value for key, value in winners[label].items() if key != "path"} for label in winners},
            }
            rows.append(row)
            if data_dir.name == "1_M1" and (predicted == "uncertain" or predicted != actual):
                image = _inlier_match_image(query, winners[winner_label]["path"] and _read_image(winners[winner_label]["path"]))
                filename = f"actual-{actual}_predicted-{predicted}_{path.stem}.png"
                cv2.imwrite(str(matches_dir / filename), image)
    report = {
        "dataset": data_dir.name,
        "settings": {
            "seed": SEED, "template_count": TEMPLATE_COUNT, "roi_ratio": ROI_RATIO,
            "min_score": MIN_SCORE, "min_margin": MIN_MARGIN, "max_dimension": MAX_DIMENSION,
        },
        "templates": {label: [path.name for path in paths] for label, paths in templates_paths.items()},
        "summary": _report(rows),
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    for data_dir in args.datasets:
        report = evaluate_dataset(data_dir)
        print(report["dataset"], report["summary"])


if __name__ == "__main__":
    main()
