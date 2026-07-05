import unittest

from faithgate.calibrate.calibrate import compute_agreement, run_calibration
from faithgate.score.scorer import HeuristicScorer, Judge

JUDGE = Judge(id="h", provider="offline", model="token-overlap")


class AgreementTest(unittest.TestCase):
    def test_perfect_agreement(self):
        results = [
            {"label": 1, "score": 0.9, "abstained": False},
            {"label": 0, "score": 0.1, "abstained": False},
        ]
        m = compute_agreement(results)
        self.assertEqual(m["accuracy"], 1.0)
        self.assertEqual(m["balanced_accuracy"], 1.0)

    def test_abstention_excluded(self):
        results = [
            {"label": 1, "score": None, "abstained": True},
            {"label": 0, "score": 0.2, "abstained": False},
        ]
        m = compute_agreement(results)
        self.assertEqual(m["abstained"], 1)
        self.assertEqual(m["scored"], 1)

    def test_balanced_accuracy_handles_imbalance(self):
        # all-faithful prediction on a skewed set should NOT look great on balanced accuracy
        results = [
            {"label": 1, "score": 0.9, "abstained": False},
            {"label": 1, "score": 0.9, "abstained": False},
            {"label": 0, "score": 0.9, "abstained": False},  # missed
        ]
        m = compute_agreement(results)
        self.assertEqual(m["balanced_accuracy"], 0.5)


class CalibrationRunTest(unittest.IsolatedAsyncioTestCase):
    async def test_heuristic_over_small_goldens(self):
        goldens = [
            {"question": "cap?", "answer": "Paris is the capital of France",
             "contexts": ["The capital of France is Paris"], "label": 1},
            {"question": "x?", "answer": "quokka zeppelin nonsense words",
             "contexts": ["The capital of France is Paris"], "label": 0},
        ]
        m = await run_calibration(HeuristicScorer(JUDGE), JUDGE, goldens)
        self.assertEqual(m["n"], 2)
        self.assertEqual(m["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
