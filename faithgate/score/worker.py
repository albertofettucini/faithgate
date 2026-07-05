"""Scoring loop — take captured-but-unscored traces and score them.

Scoring is decoupled from capture, so scoring a trace never slows the app under test.
``score_pending`` is the core used both by the live background worker and by the CLI's
``run`` command. It records the judge once (provenance) and writes one score per trace,
mapping abstention through to the store (never a fake 0.0).
"""
from __future__ import annotations

import json
import sqlite3

from ..store import db
from .scorer import FaithfulnessScorer, Judge, Sample


def _record_judge(conn: sqlite3.Connection, judge: Judge) -> str:
    judge_run_id = db.new_id()
    db.insert(conn, "judge_run", {
        "id": judge_run_id,
        "judge_type": judge.kind,
        "judge_model": judge.model,
        "judge_provider": judge.provider,
        "decomposer_model": None if judge.kind == "heuristic_proxy" else judge.model,
        "is_default": 1 if judge.kind == "frontier" else 0,
        "temperature": judge.temperature,
        "seed": judge.seed,
    })
    return judge_run_id


async def score_pending(
    conn: sqlite3.Connection,
    scorer: FaithfulnessScorer,
    judge: Judge,
    *,
    run_id: str | None = None,
    retry_errors: bool = False,
) -> "tuple[int, int]":
    """Score pending traces (optionally scoped to one run; optionally re-scoring errored ones).

    Returns (scored, errored) — callers must surface errors, never report them as successes:
    a fully broken judge must not look like a green run.
    """
    statuses = "('pending','error')" if retry_errors else "('pending')"
    sql = f"SELECT * FROM trace WHERE status IN {statuses}"
    params: tuple = ()
    if run_id is not None:
        sql += " AND run_id=?"
        params = (run_id,)

    import asyncio

    traces = db.fetchall(conn, sql, params)
    if not traces:
        return 0, 0
    judge_run_id = _record_judge(conn, judge)
    errored = 0
    per_call_timeout = 180  # a wedged judge call becomes a recorded error, never a silent hang
    for trace in traces:
        sample = Sample(
            question=trace["input_query"],
            answer=trace["output_response"],
            contexts=json.loads(trace["context_json"]),
        )
        try:
            result = await asyncio.wait_for(scorer.ascore(sample), timeout=per_call_timeout)
        except Exception as exc:  # a judge/network failure is an abstention, not a regression
            errored += 1
            db.insert(conn, "eval_score", {
                "id": db.new_id(), "trace_id": trace["id"], "run_id": trace["run_id"],
                "judge_run_id": judge_run_id, "metric": "faithfulness",
                "score": None, "abstained": 1, "reason": f"scorer error: {exc}",
            })
            conn.execute("UPDATE trace SET status='error' WHERE id=?", (trace["id"],))
            conn.commit()
            continue

        db.insert(conn, "eval_score", {
            "id": db.new_id(), "trace_id": trace["id"], "run_id": trace["run_id"],
            "judge_run_id": judge_run_id, "metric": "faithfulness",
            "score": result.score,
            "passed": None,  # the gate decides pass/fail later, against a baseline
            "reason": result.reason,
            "abstained": 1 if result.abstained else 0,
            "confidence": result.confidence,
        })
        conn.execute("UPDATE trace SET status='scored' WHERE id=?", (trace["id"],))
        conn.commit()

    return len(traces) - errored, errored
