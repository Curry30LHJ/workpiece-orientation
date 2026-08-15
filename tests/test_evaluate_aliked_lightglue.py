import unittest

from src.evaluate_aliked_lightglue import decide_label


class AlikedLightGlueEvaluationTests(unittest.TestCase):
    def test_decide_label_rejects_low_evidence_and_near_tie(self):
        self.assertEqual("uncertain", decide_label({"0": 3.0, "1": 2.8}, 4.0, 0.5)[0])
        self.assertEqual("uncertain", decide_label({"0": 12.0, "1": 11.7}, 4.0, 0.5)[0])

    def test_decide_label_returns_clear_winner(self):
        self.assertEqual("1", decide_label({"0": 4.0, "1": 12.0}, 4.0, 0.5)[0])


if __name__ == "__main__":
    unittest.main()
