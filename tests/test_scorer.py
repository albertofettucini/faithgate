import math
import unittest

from faithgate.score.scorer import FakeScorer, Judge, Sample, result_from_raw

JUDGE = Judge(id="fake-1", provider="fake", model="fake")
SAMPLE = Sample(question="q", answer="a", contexts=["c"])


class ResultMappingTest(unittest.TestCase):
    def test_normal_score(self):
        r = result_from_raw(0.8, "j")
        self.assertFalse(r.abstained)
        self.assertAlmostEqual(r.score, 0.8)

    def test_nan_abstains_not_zero(self):
        r = result_from_raw(float("nan"), "j")
        self.assertTrue(r.abstained)
        self.assertIsNone(r.score)  # never a fake 0.0

    def test_none_abstains(self):
        r = result_from_raw(None, "j")
        self.assertTrue(r.abstained)
        self.assertIsNone(r.score)


class FakeScorerTest(unittest.IsolatedAsyncioTestCase):
    async def test_scores_value(self):
        r = await FakeScorer(JUDGE, 0.73).ascore(SAMPLE)
        self.assertAlmostEqual(r.score, 0.73)
        self.assertFalse(r.abstained)
        self.assertEqual(r.judge_id, "fake-1")

    async def test_nan_abstains(self):
        r = await FakeScorer(JUDGE, math.nan).ascore(SAMPLE)
        self.assertTrue(r.abstained)
        self.assertIsNone(r.score)


if __name__ == "__main__":
    unittest.main()
