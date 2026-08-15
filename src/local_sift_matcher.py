"""Local SIFT matching with a center ROI and geometric verification."""

from __future__ import annotations

import cv2
import numpy as np


def extract_center_roi(image: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Return the central `ratio` portion of an image."""
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    height, width = image.shape[:2]
    roi_height = max(1, round(height * ratio))
    roi_width = max(1, round(width * ratio))
    top = (height - roi_height) // 2
    left = (width - roi_width) // 2
    return image[top:top + roi_height, left:left + roi_width]


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def score_pair(query: np.ndarray, template: np.ndarray, ratio: float = 0.8) -> dict[str, float]:
    """Score one image pair by RANSAC-consistent SIFT matches in the center ROI."""
    query_roi = _gray(extract_center_roi(query, ratio))
    template_roi = _gray(extract_center_roi(template, ratio))
    sift = cv2.SIFT_create()
    query_keypoints, query_descriptors = sift.detectAndCompute(query_roi, None)
    template_keypoints, template_descriptors = sift.detectAndCompute(template_roi, None)
    empty = {"good_matches": 0.0, "inliers": 0.0, "coverage": 0.0, "score": 0.0}
    if query_descriptors is None or template_descriptors is None:
        return empty

    raw_matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_descriptors, template_descriptors, k=2)
    good_matches = [first for pair in raw_matches if len(pair) == 2 for first, second in [pair]
                    if first.distance < 0.75 * second.distance]
    if len(good_matches) < 4:
        return {**empty, "good_matches": float(len(good_matches))}

    query_points = np.float32([query_keypoints[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    template_points = np.float32([template_keypoints[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(query_points, template_points, cv2.RANSAC, 5.0)
    if mask is None:
        return {**empty, "good_matches": float(len(good_matches))}

    inlier_points = query_points[mask.ravel().astype(bool)].reshape(-1, 2)
    inliers = len(inlier_points)
    coverage = 0.0
    if inliers >= 3:
        coverage = cv2.contourArea(cv2.convexHull(inlier_points.reshape(-1, 1, 2)))
        coverage /= float(query_roi.shape[0] * query_roi.shape[1])
    return {
        "good_matches": float(len(good_matches)),
        "inliers": float(inliers),
        "coverage": float(coverage),
        "score": float(inliers + coverage),
    }
