import unittest

import numpy as np

from src.aliked_lightglue_matcher import score_correspondences


class AlikedLightGlueMatcherTests(unittest.TestCase):
    def test_score_correspondences_counts_translated_geometric_inliers(self):
        points = np.array([[10, 10], [80, 10], [80, 80], [10, 80]], dtype=np.float32)
        translated = points + np.array([12, -7], dtype=np.float32)

        result = score_correspondences(points, translated, image_shape=(100, 100))

        self.assertEqual(4.0, result["matches"])
        self.assertEqual(4.0, result["inliers"])
        self.assertGreater(result["coverage"], 0.4)

    def test_score_correspondences_returns_zero_without_four_matches(self):
        points = np.array([[10, 10], [80, 10], [80, 80]], dtype=np.float32)

        result = score_correspondences(points, points, image_shape=(100, 100))

        self.assertEqual(0.0, result["inliers"])
        self.assertEqual(0.0, result["score"])


if __name__ == "__main__":
    unittest.main()
