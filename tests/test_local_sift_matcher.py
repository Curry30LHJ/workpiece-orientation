import unittest

import cv2
import numpy as np

from src.local_sift_matcher import extract_center_roi, score_pair


class LocalSiftMatcherTests(unittest.TestCase):
    def test_extract_center_roi_removes_equal_border(self):
        image = np.zeros((100, 200), dtype=np.uint8)

        roi = extract_center_roi(image, ratio=0.8)

        self.assertEqual((80, 160), roi.shape)

    def test_score_pair_keeps_geometric_inliers_after_translation(self):
        image = np.zeros((256, 256), dtype=np.uint8)
        cv2.putText(image, "M1", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        shifted = cv2.warpAffine(
            image, np.float32([[1, 0, 8], [0, 1, -5]]), (256, 256))

        result = score_pair(image, shifted)

        self.assertGreaterEqual(result["inliers"], 4)
        self.assertGreater(result["score"], 4)


if __name__ == "__main__":
    unittest.main()
