"""Full-image ALIKED matching with a soft center prior."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def score_correspondences_soft(
    query_points: np.ndarray, template_points: np.ndarray, image_shape: tuple[int, int]
) -> dict[str, float]:
    """Score RANSAC inliers while softly down-weighting image-edge points."""
    empty = {"matches": 0.0, "inliers": 0.0, "weighted_inliers": 0.0, "coverage": 0.0, "score": 0.0}
    if len(query_points) < 4 or len(template_points) < 4:
        return empty
    _, mask = cv2.findHomography(
        query_points.reshape(-1, 1, 2), template_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0
    )
    if mask is None:
        return {**empty, "matches": float(len(query_points))}
    inlier_points = query_points[mask.ravel().astype(bool)]
    height, width = image_shape
    center = np.array([(width - 1.0) / 2.0, (height - 1.0) / 2.0], dtype=np.float32)
    radius = max(float(np.linalg.norm(center)), 1.0)
    normalized_distance = np.linalg.norm(inlier_points - center, axis=1) / radius
    weights = 0.35 + 0.65 * np.clip(1.0 - normalized_distance ** 2, 0.0, 1.0)
    weighted_inliers = float(weights.sum())
    coverage = 0.0
    if len(inlier_points) >= 3:
        coverage = cv2.contourArea(cv2.convexHull(inlier_points.reshape(-1, 1, 2)))
        coverage /= float(height * width)
    return {
        "matches": float(len(query_points)),
        "inliers": float(len(inlier_points)),
        "weighted_inliers": weighted_inliers,
        "coverage": float(coverage),
        "score": float(weighted_inliers + coverage),
    }


@torch.inference_mode()
def score_feature_pair_soft(query_features: dict, template_features: dict, image_shape, matcher):
    """Match two full-image feature dictionaries and apply soft-center scoring."""
    match_data = matcher({"image0": query_features, "image1": template_features})
    matches = match_data["matches"][0].detach().cpu().numpy()
    if len(matches) < 4:
        return {"matches": float(len(matches)), "inliers": 0.0, "weighted_inliers": 0.0, "coverage": 0.0, "score": 0.0}
    query_points = query_features["keypoints"][0][matches[:, 0]].detach().cpu().numpy()
    template_points = template_features["keypoints"][0][matches[:, 1]].detach().cpu().numpy()
    return score_correspondences_soft(query_points, template_points, image_shape)
