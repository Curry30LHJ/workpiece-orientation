"""Run full-image 512-keypoint soft-center fusion evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.evaluate_global_local_fusion as fusion
from src.soft_center_matcher import score_feature_pair_soft


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    fusion.score_feature_pair = score_feature_pair_soft
    fusion.ROI_RATIO = 1.0
    fusion.MAX_NUM_KEYPOINTS = 512
    predictor = fusion._global_predictor()
    extractor, matcher, device = fusion.build_models(fusion.MAX_NUM_KEYPOINTS)
    for data_dir in args.datasets:
        report = fusion.evaluate_dataset(data_dir, predictor, extractor, matcher, device)
        print(report["dataset"], report["global"], report["fusion"])


if __name__ == "__main__":
    main()
