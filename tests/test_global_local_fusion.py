import unittest

from src.evaluate_global_local_fusion import fuse_decision


class GlobalLocalFusionTests(unittest.TestCase):
    def test_fuse_decision_uses_confident_local_result_when_global_is_ambiguous(self):
        result = fuse_decision("0", 0.01, "1", 4.0, local_margin_threshold=2.0, global_margin_threshold=0.05)

        self.assertEqual("1", result)

    def test_fuse_decision_preserves_global_result_when_local_is_uncertain(self):
        result = fuse_decision("0", 0.01, "uncertain", 0.0, local_margin_threshold=2.0, global_margin_threshold=0.05)

        self.assertEqual("0", result)

    def test_fuse_decision_preserves_global_result_when_global_is_confident(self):
        result = fuse_decision("0", 0.2, "1", 9.0, local_margin_threshold=2.0, global_margin_threshold=0.05)

        self.assertEqual("0", result)


if __name__ == "__main__":
    unittest.main()
