import unittest

import numpy as np

from src.soft_center_matcher import score_correspondences_soft


class SoftCenterMatcherTests(unittest.TestCase):
    def test_center_points_receive_more_weight_than_corner_points(self):
        points = np.array([[145, 145], [155, 145], [145, 155], [290, 290]], dtype=np.float32)

        result = score_correspondences_soft(points, points, image_shape=(300, 300))

        self.assertGreater(result["weighted_inliers"], 3.0)
        self.assertLess(result["weighted_inliers"], result["inliers"])


if __name__ == "__main__":
    unittest.main()
