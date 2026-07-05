import json
import os
import tempfile
import unittest

from faithgate.gate.cli import main
from faithgate.score.scorer import Judge, Sample, HeuristicScorer

JUDGE = Judge(id="h", provider="offline", model="token-overlap")


class HeuristicScorerTest(unittest.IsolatedAsyncioTestCase):
    async def test_grounded_scores_high(self):
        r = await HeuristicScorer(JUDGE).ascore(
            Sample("q", "Paris is the capital of France", ["The capital of France is Paris"])
        )
        self.assertFalse(r.abstained)
        self.assertGreater(r.score, 0.6)

    async def test_ungrounded_scores_low(self):
        r = await HeuristicScorer(JUDGE).ascore(
            Sample("q", "purple monkey dishwasher rocket", ["The capital of France is Paris"])
        )
        self.assertLess(r.score, 0.3)

    async def test_empty_answer_abstains(self):
        r = await HeuristicScorer(JUDGE).ascore(Sample("q", "", ["ctx"]))
        self.assertTrue(r.abstained)

    async def test_heuristic_misses_contradiction(self):
        """A DOCUMENTED weakness, asserted on purpose: the heuristic only sees novel tokens,
        so a contradiction built from the context's own words scores HIGH. This is why it is
        never the trusted judge — the README says so, and this test keeps that claim honest."""
        r = await HeuristicScorer(JUDGE).ascore(Sample(
            "how many moons?",
            "Earth has no moon",                     # contradiction, zero novel content words
            ["Earth has one moon"],
        ))
        self.assertGreater(r.score, 0.5)             # wrongly confident — by design limitation

    async def test_unicode_answers_tokenize(self):
        r = await HeuristicScorer(JUDGE).ascore(Sample(
            "başkent?", "Fransa'nın başkenti Paris'tir", ["Fransa'nın başkenti Paris'tir"]
        ))
        self.assertFalse(r.abstained)
        self.assertGreater(r.score, 0.6)


class RunCommandTest(unittest.TestCase):
    def test_run_then_gate_offline(self):
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        suite = os.path.join(d, "suite.jsonl")
        with open(suite, "w", encoding="utf-8") as f:
            f.write(json.dumps({"question": "cap?", "answer": "Paris is the capital",
                                "contexts": ["Paris is the capital of France"]}) + "\n")
            f.write(json.dumps({"question": "made-up?", "answer": "quokka zeppelin tuesday",
                                "contexts": ["Paris is the capital of France"]}) + "\n")

        rc = main(["--db", db_path, "run", "--suite", suite, "--label", "v1", "--judge", "heuristic"])
        self.assertEqual(rc, 0)
        self.assertEqual(main(["--db", db_path, "runs"]), 0)
        self.assertEqual(main(["--db", db_path, "show", "--run", "v1"]), 0)


if __name__ == "__main__":
    unittest.main()
