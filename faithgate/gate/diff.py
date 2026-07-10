"""The regression gate — compare two runs and decide pass/fail.

A run is one version of the app under test. We match the same test case across two runs by
content key (not row id), so we compare like with like. A test case "regresses" if its score
dropped more than ``max_regression`` OR fell below ``min_score`` (the floor applies to every
non-abstained head score, even for cases with no baseline). Abstained cases are excluded from
the math and reported separately — an honest judge declining is not a regression.

Fail-closed semantics: if ZERO cases match while the baseline has cases, the gate FAILS with a
"nothing compared" verdict instead of passing vacuously — a renamed suite must never turn CI
green. New/missing/duplicate cases are always visible in the report.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from ..store import db

_EPS = 1e-9  # float guard: exactly-at-threshold behaves the same for every score pair


@dataclass(frozen=True)
class TraceScore:
    """One scored test case in a run."""
    key: str
    question: str
    score: Optional[float]   # None when abstained
    abstained: bool
    trace_id: str = ""


@dataclass(frozen=True)
class Change:
    question: str
    key: str
    baseline: Optional[float]   # None when the case has no scored baseline
    head: float
    delta: Optional[float]
    kind: str  # 'dropped' | 'below_floor' | 'improved'


@dataclass
class DiffResult:
    regressions: list = field(default_factory=list)   # Change (dropped / below_floor)
    improved: list = field(default_factory=list)      # Change (improved)
    unchanged: int = 0
    matched: int = 0                                  # cases scored on BOTH sides
    abstained: list = field(default_factory=list)     # questions excluded from the math
    new_cases: list = field(default_factory=list)     # in head, not in baseline
    missing_cases: list = field(default_factory=list) # in baseline, not in head
    duplicates: int = 0                               # duplicate content keys collapsed (min kept)
    verdict: str = ""                                 # 'ok' | 'nothing_compared'
    passed: bool = True


def _is_abstained(t) -> bool:
    """None/NaN scores are abstentions no matter what the flag says — never comparable numbers."""
    return t.abstained or t.score is None or (isinstance(t.score, float) and math.isnan(t.score))


def _by_key(traces: list, keep: str = "min") -> "tuple[dict, int]":
    """Index by content key, counting duplicate keys so the report can surface the collapse.

    Head duplicates keep the LOWEST score (strict on the new version); baseline duplicates keep
    the HIGHEST (a stray low baseline entry must not quietly lower the bar a regression is
    measured against)."""
    out: dict = {}
    dupes = 0
    sentinel = float("inf") if keep == "min" else float("-inf")
    for t in traces:
        prev = out.get(t.key)
        if prev is None:
            out[t.key] = t
            continue
        dupes += 1
        prev_score = sentinel if _is_abstained(prev) else prev.score
        this_score = sentinel if _is_abstained(t) else t.score
        if (keep == "min" and this_score < prev_score) or (keep == "max" and this_score > prev_score):
            out[t.key] = t
    return out, dupes


def compare(
    baseline: list,
    head: list,
    *,
    max_regression: float = 0.05,
    min_score: float = 0.5,
) -> DiffResult:
    by_base, dupes_b = _by_key(baseline, keep="max")
    by_head, dupes_h = _by_key(head, keep="min")

    result = DiffResult()
    result.duplicates = dupes_b + dupes_h
    result.new_cases = [by_head[k].question for k in by_head if k not in by_base]
    result.missing_cases = [by_base[k].question for k in by_base if k not in by_head]

    for key, ht in by_head.items():
        bt = by_base.get(key)
        if _is_abstained(ht):
            result.abstained.append(ht.question)
            continue
        # the floor guards EVERY scored head case — new, matched, or abstained-baseline
        if ht.score < min_score - _EPS:
            base_val = bt.score if (bt and not _is_abstained(bt)) else None
            delta = (ht.score - base_val) if base_val is not None else None
            result.regressions.append(Change(ht.question, key, base_val, ht.score, delta, "below_floor"))
            if bt is not None and not _is_abstained(bt):
                result.matched += 1
            continue
        if bt is None:
            continue  # new case above the floor: reported in new_cases, not gated on delta
        if _is_abstained(bt):
            result.abstained.append(ht.question)
            continue
        result.matched += 1
        delta = ht.score - bt.score
        if delta < -max_regression - _EPS:
            result.regressions.append(Change(ht.question, key, bt.score, ht.score, delta, "dropped"))
        elif delta > max_regression + _EPS:
            result.improved.append(Change(ht.question, key, bt.score, ht.score, delta, "improved"))
        else:
            result.unchanged += 1

    if result.matched == 0 and not result.regressions:
        # nothing was actually compared (empty/unscored runs included) — refuse to bless it
        result.verdict = "nothing_compared"
        result.passed = False
    else:
        result.verdict = "ok"
        result.passed = len(result.regressions) == 0
    return result


def load_run_scores(conn: sqlite3.Connection, run_id: str) -> list:
    """Read one run's faithfulness scores as TraceScore objects (the gate's authoritative read)."""
    rows = db.fetchall(conn, """
        SELECT t.id AS trace_id, t.context_hash AS key, t.input_query AS question,
               s.score AS score, s.abstained AS abstained
        FROM eval_score s
        JOIN trace t ON t.id = s.trace_id
        WHERE s.run_id = ? AND s.metric = 'faithfulness'
    """, (run_id,))
    return [
        TraceScore(key=r["key"], question=r["question"], score=r["score"],
                   abstained=bool(r["abstained"]), trace_id=r["trace_id"])
        for r in rows
    ]


def render_markdown(result: DiffResult, policy_failures: list = ()) -> str:
    """PR-comment-ready markdown: verdict, counts, and a per-case regression table."""
    verdict = "✅ **PASS**" if result.passed else "❌ **FAIL**"
    esc = lambda s: str(s).replace("|", "\\|")
    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    lines = [
        f"### FaithGate regression gate: {verdict}",
        "",
        f"`matched {result.matched}` · `regressed {len(result.regressions)}` · "
        f"`improved {len(result.improved)}` · `unchanged {result.unchanged}` · "
        f"`abstained {len(result.abstained)}` · `new {len(result.new_cases)}` · "
        f"`missing {len(result.missing_cases)}`",
    ]
    if result.verdict == "nothing_compared":
        lines += ["", "⚠️ **Nothing compared** — no case in the new run matches the baseline."]
    if result.regressions:
        lines += ["", "| case | baseline | new | Δ |", "|---|---|---|---|"]
        for c in result.regressions:
            delta = "—" if c.delta is None else f"{c.delta:+.2f}"
            tag = " *(below floor)*" if c.kind == "below_floor" else ""
            lines.append(f"| {esc(c.question)}{tag} | {fmt(c.baseline)} | {fmt(c.head)} | {delta} |")
    if result.improved:
        lines += ["", "<details><summary>Improved (" + str(len(result.improved)) + ")</summary>", ""]
        for c in result.improved:
            lines.append(f"- {esc(c.question)}  {fmt(c.baseline)} → {fmt(c.head)}")
        lines += ["", "</details>"]
    for label, items in (("Abstained", result.abstained), ("Missing", result.missing_cases),
                         ("New", result.new_cases)):
        if items:
            shown = ", ".join(esc(q) for q in items[:5])
            more = f" … +{len(items) - 5}" if len(items) > 5 else ""
            lines.append(f"- **{label} ({len(items)})**: {shown}{more}")
    for p in policy_failures:
        lines.append(f"- ✗ **policy**: {p}")
    lines += ["", "<sub>FaithGate — local faithfulness regression gate · scores measured against "
              "sources, judge honesty measured against humans</sub>"]
    return "\n".join(lines)


def render_json(result: DiffResult, policy_failures: list = ()) -> str:
    """Machine-readable verdict for scripting."""
    import json
    from dataclasses import asdict

    payload = asdict(result)
    payload["policy_failures"] = list(policy_failures)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _list_some(lines: list, items: list, label: str, cap: int = 5) -> None:
    lines.append(f"  {label} ({len(items)}):")
    for q in items[:cap]:
        lines.append(f"    — {q}")
    if len(items) > cap:
        lines.append(f"    … and {len(items) - cap} more")


def render_text(result: DiffResult) -> str:
    """A human-readable gate report (also the seed of the CI PR comment)."""
    lines = []
    verdict = "PASS ✅" if result.passed else "FAIL ❌"
    lines.append(f"faithgate regression gate: {verdict}")
    lines.append("")
    lines.append(
        f"  matched {result.matched} · {len(result.regressions)} regressed · "
        f"{len(result.improved)} improved · {result.unchanged} unchanged · "
        f"{len(result.abstained)} abstained · {len(result.new_cases)} new · "
        f"{len(result.missing_cases)} missing"
        + (f" · {result.duplicates} duplicate keys (min kept)" if result.duplicates else "")
    )
    if result.verdict == "nothing_compared":
        lines.append("")
        lines.append("  ⚠ NOTHING COMPARED — no case in the new run matches the baseline.")
        lines.append("  A renamed or rewritten suite must not pass silently; check the suites.")
    if result.regressions:
        lines.append("")
        lines.append("  Regressions:")
        for c in result.regressions:
            if c.kind == "below_floor":
                base = f"{c.baseline:.2f} → " if c.baseline is not None else ""
                lines.append(f"    ❌ {c.question}  {base}{c.head:.2f}  (below floor)")
            else:
                lines.append(f"    ❌ {c.question}  {c.baseline:.2f} → {c.head:.2f}  ({c.delta:+.2f})")
    if result.improved:
        lines.append("")
        lines.append("  Improved:")
        for c in result.improved:
            lines.append(f"    ✅ {c.question}  {c.baseline:.2f} → {c.head:.2f}  ({c.delta:+.2f})")
    if result.abstained:
        lines.append("")
        _list_some(lines, result.abstained, "Abstained (judge declined — not counted)")
    if result.missing_cases:
        lines.append("")
        _list_some(lines, result.missing_cases, "Missing (in baseline, absent from new run)")
    if result.new_cases:
        lines.append("")
        _list_some(lines, result.new_cases, "New (no baseline yet)")
    return "\n".join(lines)
