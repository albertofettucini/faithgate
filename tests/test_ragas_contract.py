"""Contract tests for the RAGAS seam — the exact import surface scorer.py uses lazily.

These run ONLY where ragas is installed (the licenses CI job installs the [claude] extra).
Both audit P0s in this seam (class-name casing, API drift) are the bug class these catch
offline in under a second; "unit tests green" alone proves nothing about the seam the
product actually ships on.
"""
import unittest

try:
    import ragas  # noqa: F401
    HAVE_RAGAS = True
except ImportError:
    HAVE_RAGAS = False


@unittest.skipUnless(HAVE_RAGAS, "ragas not installed (run via the licenses CI job)")
class RagasContractTest(unittest.TestCase):
    def test_import_surface(self):
        from ragas.dataset_schema import SingleTurnSample  # noqa: F401
        from ragas.llms import LangchainLLMWrapper  # noqa: F401
        from ragas.metrics import Faithfulness  # noqa: F401
        from ragas.metrics import FaithfulnesswithHHEM  # noqa: F401  (lowercase w!)

    def test_sample_kwargs(self):
        from ragas.dataset_schema import SingleTurnSample

        s = SingleTurnSample(user_input="q", response="a", retrieved_contexts=["c"])
        self.assertEqual(s.user_input, "q")

    def test_single_turn_ascore_exists(self):
        from ragas.metrics import Faithfulness

        self.assertTrue(hasattr(Faithfulness, "single_turn_ascore"))

    def test_wrapper_supports_temperature_bypass_and_run_config(self):
        # our fix for Sonnet-5's sampling-param rejection depends on these knobs existing.
        # NOTE: ragas exports the wrapper behind a deprecation shim, so we CONSTRUCT with the
        # kwargs (signature introspection sees only the shim) and check they took effect.
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        w = LangchainLLMWrapper(
            object(),
            run_config=RunConfig(timeout=60, max_retries=2, max_wait=5),
            bypass_temperature=True,
        )
        self.assertTrue(w.bypass_temperature)
        self.assertEqual(w.run_config.max_retries, 2)

    def test_our_scorer_builds_metric(self):
        from faithgate.score.scorer import Judge, RagasFaithfulnessScorer

        judge = Judge(id="contract", provider="test", model="stub")
        scorer = RagasFaithfulnessScorer(judge, llm=object())
        metric = scorer._build_metric()
        self.assertTrue(hasattr(metric, "single_turn_ascore"))


if __name__ == "__main__":
    unittest.main()
