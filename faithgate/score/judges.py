"""Judge construction. The default is Claude via the user's OWN ``ANTHROPIC_API_KEY``.

Three named judges:
  * ``claude``      — the trusted default. Frontier decomposition + NLI. Needs the key.
  * ``claude-local``— local-verification: HHEM does the NLI on-device, Claude still decomposes.
                      Needs the key AND the ``faithgate[local]`` extra. NOT 'fully offline'.
  * ``heuristic``   — offline grounding proxy, no key, no network. A stand-in, never trusted.
"""
from __future__ import annotations

import os

from .scorer import HeuristicScorer, Judge, RagasFaithfulnessScorer

DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"


def build_claude_llm(model: str):
    """Construct a LangChain Claude client from the environment key (never hardcoded).

    No temperature is passed: Sonnet-5-class models reject non-default sampling params, so we
    use the model default and record temperature=None in provenance. max_tokens is explicit so
    judge behavior doesn't depend on library-version defaults.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "the Claude judge needs the claude extra — install `faithgate[claude]`. "
            "Or use `--judge heuristic` to run offline without any install."
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "set ANTHROPIC_API_KEY to use the Claude judge (export it in your shell; never "
            "hardcode it in a file). Or use `--judge heuristic` to run offline without a key."
        )
    # bounded per request: a wedged HTTP call must surface as an error, never an endless hang
    return ChatAnthropic(model=model, max_tokens=4096, timeout=60, max_retries=2)


def make_scorer(name: str = "claude", *, model: str = DEFAULT_CLAUDE_MODEL, use_hhem: bool = False):
    """Return (scorer, judge) for a named judge."""
    if name == "heuristic":
        judge = Judge(id="heuristic", provider="offline", model="token-overlap", kind="heuristic_proxy")
        return HeuristicScorer(judge), judge

    if name in ("claude", "claude-local"):
        use_hhem = use_hhem or name == "claude-local"
        judge = Judge(
            id=model,
            provider="anthropic",
            model=model,
            kind="local_verification" if use_hhem else "frontier",
        )
        llm = build_claude_llm(model)
        return RagasFaithfulnessScorer(judge, llm, use_hhem=use_hhem), judge

    raise ValueError(f"unknown judge {name!r} (expected: claude, claude-local, heuristic)")
