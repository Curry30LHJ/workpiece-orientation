"""ALIKED + LightGlue local matching with RANSAC geometric verification."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from lightglue import ALIKED, LightGlue
from lightglue.utils import numpy_image_to_torch

from src.local_sift_matcher import extract_center_roi


def score_correspondences(
    query_points: np.ndarray, template_points: np.ndarray, image_shape: tuple[int, int]
) -> dict[str, float]:
    """Return RANSAC inlier and coverage scores for paired 2D points."""
    empty = {"matches": 0.0, "inliers": 0.0, "coverage": 0.0, "score": 0.0}
    if len(query_points) < 4 or len(template_points) < 4:
        return empty
    _, mask = cv2.findHomography(
        query_points.reshape(-1, 1, 2), template_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0
    )
    if mask is None:
        return {**empty, "matches": float(len(query_points))}
    inlier_points = query_points[mask.ravel().astype(bool)]
    coverage = 0.0
    if len(inlier_points) >= 3:
        coverage = cv2.contourArea(cv2.convexHull(inlier_points.reshape(-1, 1, 2)))
        coverage /= float(image_shape[0] * image_shape[1])
    return {
        "matches": float(len(query_points)),
        "inliers": float(len(inlier_points)),
        "coverage": float(coverage),
        "score": float(len(inlier_points) + coverage),
    }


def build_models(max_num_keypoints: int = 1024):
    """Load ALIKED and its matching LightGlue weights onto the CUDA device."""
    if not torch.cuda.is_available():
        raise RuntimeError("ALIKED + LightGlue evaluation requires a CUDA-capable PyTorch installation")
    device = torch.device("cuda")
    extractor = ALIKED(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="aliked").eval().to(device)
    return extractor, matcher, device


@torch.inference_mode()
def extract_features(image: np.ndarray, extractor, device: torch.device, roi_ratio: float = 0.8) -> dict:
    """Extract ALIKED features from the central ROI of a BGR OpenCV image."""
    roi = extract_center_roi(image, roi_ratio)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = numpy_image_to_torch(rgb).to(device)
    return extractor.extract(tensor)


@torch.inference_mode()
def score_feature_pair(
    query_features: dict, template_features: dict, image_shape: tuple[int, int], matcher
) -> dict[str, float]:
    """Match extracted feature dictionaries and score their geometric consistency."""
    match_data = matcher({"image0": query_features, "image1": template_features})
    matches = match_data["matches"][0].detach().cpu().numpy()
    if len(matches) < 4:
        return {"matches": float(len(matches)), "inliers": 0.0, "coverage": 0.0, "score": 0.0}
    query_points = query_features["keypoints"][0][matches[:, 0]].detach().cpu().numpy()
    template_points = template_features["keypoints"][0][matches[:, 1]].detach().cpu().numpy()
    return score_correspondences(query_points, template_points, image_shape)
