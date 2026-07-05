"""The ONE place RAGAS is touched. Everything above this seam sees only plain dataclasses.

We never author metric math — RAGAS computes faithfulness. Our job is the honest framing:
NaN / zero-extracted-statements becomes an explicit ``abstained`` result, never a fake ``0.0``;
each result names the judge that produced it; the judge is swappable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class Sample:
    """A single RAG turn to score."""
    question: str
    answer: str
    contexts: list  # list[str] of retrieved chunks — the ground for faithfulness


@dataclass(frozen=True)
class Judge:
    """Identity + provenance of a judge configuration. Pinned into the run manifest."""
    id: str
    provider: str                       # 'anthropic' | 'fake' | ...
    model: str
    kind: str = "frontier"              # 'frontier' | 'local_verification' | 'heuristic_proxy'
    temperature: Optional[float] = None  # None = model default (Sonnet-5-class rejects non-default sampling)
    seed: Optional[int] = None


@dataclass(frozen=True)
class FaithfulnessResult:
    """What the rest of the system sees. No per-claim list (RAGAS doesn't expose it publicly)."""
    score: Optional[float]              # None when abstained
    judge_id: str
    reason: str
    abstained: bool
    confidence: Optional[float] = None


def result_from_raw(raw: Any, judge_id: str, reason: str = "") -> FaithfulnessResult:
    """Map a raw RAGAS score to a result. NaN/None → abstain (never 0.0, never a regression)."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return FaithfulnessResult(
            score=None,
            judge_id=judge_id,
            reason=reason or "judge could not extract verifiable statements",
            abstained=True,
        )
    return FaithfulnessResult(score=float(raw), judge_id=judge_id, reason=reason, abstained=False)


class FaithfulnessScorer(Protocol):
    judge: Judge
    async def ascore(self, sample: Sample) -> FaithfulnessResult: ...


class FakeScorer:
    """Deterministic scorer for tests/CI — no RAGAS, no network. Pass a float, or NaN to abstain."""

    def __init__(self, judge: Judge, value: Any) -> None:
        self.judge = judge
        self._value = value

    async def ascore(self, sample: Sample) -> FaithfulnessResult:
        return result_from_raw(self._value, self.judge.id, reason="fake scorer")


class HeuristicScorer:
    """Offline, no-API-key grounding proxy — fraction of the answer's content words found in the
    contexts. This is NOT a real judge (it can't tell paraphrase from contradiction); it exists so
    the whole pipeline runs with zero key/network, and it is always labeled as the offline stand-in.
    """

    _STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "at", "it", "its",
             "and", "or", "that", "this", "as", "by", "for", "with", "about"}

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    @staticmethod
    def _tokens(text: str) -> set:
        import re
        return set(re.findall(r"[^\W_]+", (text or "").lower(), re.UNICODE))

    async def ascore(self, sample: Sample) -> FaithfulnessResult:
        answer_words = self._tokens(sample.answer) - self._STOP
        context_words: set = set()
        for chunk in sample.contexts:
            context_words |= self._tokens(chunk)
        if not answer_words:
            return result_from_raw(float("nan"), self.judge.id, "no content words to check")
        grounded = len(answer_words & context_words) / len(answer_words)
        return result_from_raw(round(grounded, 2), self.judge.id, "offline grounding proxy (not a real judge)")


class RagasFaithfulnessScorer:
    """Embeds RAGAS. ``ragas`` is imported lazily so importing this module never requires it.

    Default path: ``Faithfulness`` (decomposition + NLI on the frontier judge).
    ``local_verification`` path: ``FaithfulnessWithHHEM`` — HHEM does the NLI on-device, but the
    LLM still does claim extraction, so this is NOT 'fully local'. It needs ``faithgate[local]``.
    """

    def __init__(self, judge: Judge, llm: Any, *, use_hhem: bool = False, device: str = "cpu") -> None:
        self.judge = judge
        self._llm = llm
        self._use_hhem = use_hhem
        self._device = device
        self._metric = None

    def _build_metric(self):
        import warnings

        # ragas 0.4.x emits DeprecationWarnings for the pinned legacy import surface we use
        # knowingly (see pyproject + PLAN "Changed during build") — keep user output clean.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas($|\.)")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"faithgate($|\.)")
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        # bypass_temperature: ragas injects temperature=0.01 per call unless bypassed, and
        # Sonnet-5-class models reject non-default sampling params — the audit's temperature
        # P0 through the back door. RunConfig: ragas defaults retry EVERY exception 10× with
        # up to 60s waits, turning any persistent error into a silent multi-minute hang.
        wrapped = LangchainLLMWrapper(
            self._llm,
            run_config=RunConfig(timeout=60, max_retries=2, max_wait=5),
            bypass_temperature=True,
        )
        if self._use_hhem:
            _HHEM_MSG = "local-verification mode needs the HHEM extra — install `faithgate[local]`."
            try:
                # real ragas class name is lowercase-w FaithfulnesswithHHEM
                from ragas.metrics import FaithfulnesswithHHEM
            except ImportError as exc:
                raise ImportError(_HHEM_MSG) from exc
            try:
                # torch/transformers load lazily HERE (first instantiation also downloads
                # the HHEM weights from HuggingFace)
                return FaithfulnesswithHHEM(llm=wrapped, device=self._device)
            except ImportError as exc:
                raise ImportError(_HHEM_MSG) from exc
        from ragas.metrics import Faithfulness

        return Faithfulness(llm=wrapped)

    async def ascore(self, sample: Sample) -> FaithfulnessResult:
        from ragas.dataset_schema import SingleTurnSample

        if self._metric is None:
            self._metric = self._build_metric()
        ragas_sample = SingleTurnSample(
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=list(sample.contexts),
        )
        raw = await self._metric.single_turn_ascore(ragas_sample)
        return result_from_raw(raw, self.judge.id)


def build_scorer(judge: Judge, llm: Any, *, use_hhem: bool = False, device: str = "cpu") -> FaithfulnessScorer:
    """Factory for the real RAGAS-backed scorer. Tests construct FakeScorer directly."""
    return RagasFaithfulnessScorer(judge, llm, use_hhem=use_hhem, device=device)
