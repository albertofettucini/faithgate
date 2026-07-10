import json
import unittest

from faithgate.gate.diff import TraceScore, compare, render_json, render_markdown


def ts(key, score, abstained=False):
    return TraceScore(key=key, question=f"q-{key}", score=score, abstained=abstained)


class CompareTest(unittest.TestCase):
    def test_clean_run_passes(self):
        base = [ts("a", 0.90), ts("b", 0.85)]
        head = [ts("a", 0.91), ts("b", 0.86)]
        r = compare(base, head)
        self.assertTrue(r.passed)
        self.assertEqual(r.unchanged, 2)

    def test_drop_is_caught(self):
        base = [ts("a", 0.90), ts("b", 0.88)]
        head = [ts("a", 0.62), ts("b", 0.88)]   # 'a' dropped 0.28
        r = compare(base, head, max_regression=0.05)
        self.assertFalse(r.passed)
        self.assertEqual(len(r.regressions), 1)
        self.assertEqual(r.regressions[0].kind, "dropped")
        self.assertAlmostEqual(r.regressions[0].delta, -0.28)

    def test_below_floor_is_caught_even_if_small_drop(self):
        base = [ts("a", 0.52)]
        head = [ts("a", 0.49)]   # only -0.03 but under the 0.5 floor
        r = compare(base, head, max_regression=0.05, min_score=0.5)
        self.assertFalse(r.passed)
        self.assertEqual(r.regressions[0].kind, "below_floor")

    def test_improvement_does_not_fail(self):
        base = [ts("a", 0.70)]
        head = [ts("a", 0.92)]
        r = compare(base, head)
        self.assertTrue(r.passed)
        self.assertEqual(len(r.improved), 1)

    def test_abstention_not_a_regression_when_others_match(self):
        base = [ts("a", 0.90), ts("b", 0.90)]
        head = [ts("a", None, abstained=True), ts("b", 0.90)]
        r = compare(base, head)
        self.assertTrue(r.passed)              # abstain is not a regression
        self.assertEqual(r.abstained, ["q-a"])
        self.assertEqual(len(r.regressions), 0)

    def test_total_abstention_fails_closed(self):
        # if EVERYTHING abstained, nothing was compared — a broken judge must not look green
        base = [ts("a", 0.90)]
        head = [ts("a", None, abstained=True)]
        r = compare(base, head)
        self.assertFalse(r.passed)
        self.assertEqual(r.verdict, "nothing_compared")

    def test_new_and_missing_tracked_and_zero_match_fails(self):
        base = [ts("a", 0.9)]
        head = [ts("b", 0.9)]
        r = compare(base, head)
        self.assertEqual(r.new_cases, ["q-b"])
        self.assertEqual(r.missing_cases, ["q-a"])
        # fail-closed: nothing was actually compared → the gate must NOT pass vacuously
        self.assertFalse(r.passed)
        self.assertEqual(r.verdict, "nothing_compared")

    def test_floor_applies_to_new_and_abstained_baseline_cases(self):
        # a brand-new case below the floor must gate even without a baseline
        r = compare([ts("a", 0.9)], [ts("a", 0.9), ts("b", 0.10)])
        self.assertFalse(r.passed)
        self.assertEqual(r.regressions[0].kind, "below_floor")
        self.assertIsNone(r.regressions[0].baseline)
        # baseline abstained + head under the floor must also gate
        r2 = compare([ts("a", None, abstained=True)], [ts("a", 0.05)])
        self.assertFalse(r2.passed)

    def test_duplicate_keys_keep_min_and_are_counted(self):
        base = [ts("a", 0.9)]
        head = [ts("a", 0.95), ts("a", 0.20)]  # duplicate key: min (0.20) must win
        r = compare(base, head)
        self.assertEqual(r.duplicates, 1)
        self.assertFalse(r.passed)

    def test_matched_counted(self):
        r = compare([ts("a", 0.9), ts("b", 0.8)], [ts("a", 0.9), ts("b", 0.8)])
        self.assertEqual(r.matched, 2)
        self.assertTrue(r.passed)

    def test_unscored_runs_fail_closed(self):
        # two ingested-but-never-scored runs (zero scores) must NOT pass silently
        r = compare([], [ts("a", 0.9)])
        self.assertFalse(r.passed)
        self.assertEqual(r.verdict, "nothing_compared")
        r2 = compare([], [])
        self.assertFalse(r2.passed)

    def test_baseline_duplicates_keep_max(self):
        # a stray LOW baseline duplicate must not lower the bar: real drop 0.90→0.55 must fail
        r = compare([ts("a", 0.90), ts("a", 0.20)], [ts("a", 0.55)])
        self.assertEqual(r.duplicates, 1)
        self.assertFalse(r.passed)
        self.assertEqual(r.regressions[0].kind, "dropped")

    def test_none_score_without_flag_treated_as_abstained(self):
        # defensive: a NULL score with abstained=0 must never crash or count as a number
        broken = TraceScore(key="a", question="q-a", score=None, abstained=False)
        r = compare([ts("a", 0.9)], [broken])
        self.assertFalse(r.passed)                    # single case → nothing compared
        self.assertEqual(r.abstained, ["q-a"])

    def test_markdown_fail_has_verdict_table_and_policy(self):
        r = compare([ts("a", 0.9)], [ts("a", 0.2)])
        md = render_markdown(r, ["1 baseline case(s) missing"])
        self.assertIn("❌", md)
        self.assertIn("| q-a", md)               # regression table row
        self.assertIn("policy", md)

    def test_markdown_escapes_pipes(self):
        base = [TraceScore(key="k", question="a|b", score=0.9, abstained=False)]
        head = [TraceScore(key="k", question="a|b", score=0.1, abstained=False)]
        md = render_markdown(compare(base, head))
        self.assertIn("a\\|b", md)               # a raw pipe would break the table

    def test_json_roundtrip(self):
        r = compare([ts("a", 0.9)], [ts("a", 0.89)])
        payload = json.loads(render_json(r))
        self.assertTrue(payload["passed"])
        self.assertIn("regressions", payload)
        self.assertEqual(payload["policy_failures"], [])

    def test_exact_threshold_is_consistent(self):
        # float-representation jitter must not flip verdicts at exactly max_regression
        r1 = compare([ts("a", 0.90)], [ts("a", 0.85)], max_regression=0.05)
        r2 = compare([ts("a", 0.85)], [ts("a", 0.80)], max_regression=0.05)
        self.assertEqual(r1.passed, r2.passed)


if __name__ == "__main__":
    unittest.main()
