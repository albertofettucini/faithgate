"""Judge calibration — measure how often the judge agrees with human labels.

The honesty core: instead of trusting the judge blindly, we run it over a small hand-labeled
golden set and publish its agreement-with-humans. Abstentions are excluded from the math (an
honest "I couldn't score this" is neither right nor wrong).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..score.scorer import Sample

DEFAULT_GOLDENS = Path(__file__).with_name("goldens") / "faithfulness.jsonl"


def load_goldens(path) -> list:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _aggregate(results: list, threshold: float) -> dict:
    tp = tn = fp = fn = abstained = 0
    for r in results:
        if r["abstained"] or r["score"] is None:
            abstained += 1
            continue
        pred = 1 if r["score"] >= threshold else 0
        label = r["label"]
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    scored = tp + tn + fp + fn
    accuracy = (tp + tn) / scored if scored else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0   # faithful correctly kept
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # unfaithful correctly caught
    return {
        "n": len(results), "scored": scored, "abstained": abstained,
        "accuracy": accuracy, "balanced_accuracy": (sensitivity + specificity) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": threshold,
    }


def compute_agreement(results: list, threshold: float = 0.5) -> dict:
    """results: {label:0|1, score:float|None, abstained:bool, category?:str}. label 1 = faithful.

    When results span multiple categories (suite strata), a per-category breakdown is included —
    that's how a judge's blind spots become visible instead of averaged away."""
    metrics = _aggregate(results, threshold)
    by_cat: dict = {}
    for r in results:
        by_cat.setdefault(r.get("category", "clean"), []).append(r)
    metrics["categories"] = (
        {cat: _aggregate(rs, threshold) for cat, rs in sorted(by_cat.items())}
        if len(by_cat) > 1 else {}
    )
    return metrics


PER_SAMPLE_TIMEOUT = 120  # seconds — a stuck judge call becomes a visible error, not a hang


async def run_calibration(scorer, judge, goldens: list, threshold: float = 0.5,
                          progress: bool = True) -> dict:
    import asyncio
    import sys

    results = []
    errors = []
    for i, g in enumerate(goldens, 1):
        try:
            r = await asyncio.wait_for(
                scorer.ascore(Sample(g["question"], g["answer"], g.get("contexts", []))),
                timeout=PER_SAMPLE_TIMEOUT,
            )
            results.append({"label": int(g["label"]), "score": r.score, "abstained": r.abstained,
                            "category": g.get("category", "clean")})
        except Exception as exc:  # same policy as the worker: an error is an abstention, not a crash
            name = "TimeoutError (judge call exceeded %ss)" % PER_SAMPLE_TIMEOUT \
                if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
            errors.append(f"{name}: {exc}")
            results.append({"label": int(g["label"]), "score": None, "abstained": True,
                            "category": g.get("category", "clean")})
        if progress:
            print(f"\r  [{i}/{len(goldens)}] judged · {len(errors)} errored", end="",
                  file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    metrics = compute_agreement(results, threshold)
    metrics["errors"] = errors
    return metrics


def render_report(judge, m: dict) -> str:
    pct = lambda x: f"{100 * x:.0f}%"
    lines = [
        f"calibration — judge: {judge.id}",
        f"  examples: {m['n']}  (scored {m['scored']}, abstained {m['abstained']})",
        f"  agreement with humans: {pct(m['accuracy'])} accuracy · {pct(m['balanced_accuracy'])} balanced"
        f"  (n={m['n']} — directional, not precise)",
        f"  faithful kept: {m['tp']}/{m['tp'] + m['fn']}   unfaithful caught: {m['tn']}/{m['tn'] + m['fp']}",
    ]
    if m.get("categories"):
        lines.append("  by category:")
        for cat, cm in m["categories"].items():
            caught = f"{cm['tn']}/{cm['tn'] + cm['fp']}" if (cm['tn'] + cm['fp']) else "–"
            lines.append(f"    {cat:<24} {pct(cm['balanced_accuracy'])} balanced · "
                         f"unfaithful caught: {caught} · n={cm['n']}")
    errors = m.get("errors") or []
    if errors:
        # a broken judge must be loud, never laundered into quiet abstention — but a single
        # hiccup among many scored cases is an abstention note, not a broken judge
        if m["scored"] == 0:
            lines.append(f"  ✗ {len(errors)} case(s) errored — the judge is not working.")
        else:
            lines.append(f"  ⚠ {len(errors)} case(s) errored — treated as abstentions.")
        lines.append(f"    first error: {errors[0][:300]}")
    return "\n".join(lines)
