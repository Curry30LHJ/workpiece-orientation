import unittest

from src.evaluate_local_sift import decide_label


class LocalSiftEvaluationTests(unittest.TestCase):
    def test_decide_label_rejects_low_score_and_near_tie(self):
        self.assertEqual("uncertain", decide_label({"0": 3.0, "1": 2.8}, 4.0, 0.5)[0])
        self.assertEqual("uncertain", decide_label({"0": 8.0, "1": 7.7}, 4.0, 0.5)[0])

    def test_decide_label_keeps_clear_winner(self):
        self.assertEqual("0", decide_label({"0": 8.0, "1": 2.0}, 4.0, 0.5)[0])


if __name__ == "__main__":
    unittest.main()
