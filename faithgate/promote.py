"""Promote captured failures into reusable regression test cases — the flywheel.

A promoted case is (question + contexts [+ a stable identity]): the bad ANSWER itself is not
promoted — it stays reachable as history through ``origin_trace_id``. When the suite is replayed,
the user's app produces a fresh answer and THAT gets scored. Nothing is ever promoted silently:
the CLI always shows what it is about to do and requires confirmation (or an explicit --yes).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

from .keys import content_key
from .store import db

DEFAULT_DATASET = "regressions"


@dataclass(frozen=True)
class Candidate:
    trace_id: str
    question: str
    contexts: list
    score: Optional[float]
    run_label: Optional[str]


@dataclass(frozen=True)
class PromoteResult:
    status: str                  # 'promoted' | 'skipped_duplicate'
    item_id: Optional[str]       # new dataset_item id (or the existing duplicate's id)
    question: str
    score: Optional[float] = None


def find_candidates(conn: sqlite3.Connection, *, run_id: str = None,
                    below: float = 0.5, limit: int = 50) -> list:
    """Scored (non-abstained) traces under the threshold, not yet promoted, worst first."""
    sql = """
        SELECT t.id AS trace_id, t.input_query AS question, t.context_json AS context_json,
               MIN(s.score) AS score, r.label AS run_label
        FROM eval_score s
        JOIN trace t ON t.id = s.trace_id
        LEFT JOIN run r ON r.id = t.run_id
        WHERE s.metric = 'faithfulness' AND s.abstained = 0 AND s.score < ?
          AND NOT EXISTS (SELECT 1 FROM dataset_item di WHERE di.origin_trace_id = t.id)
    """
    params = [below]
    if run_id is not None:
        sql += " AND t.run_id = ?"
        params.append(run_id)
    sql += " GROUP BY t.id ORDER BY score ASC LIMIT ?"
    params.append(limit)
    rows = db.fetchall(conn, sql, params)
    return [Candidate(r["trace_id"], r["question"], json.loads(r["context_json"]),
                      r["score"], r["run_label"]) for r in rows]


def resolve_trace(conn: sqlite3.Connection, ref: str):
    """Accept a full trace id or a unique prefix. Returns (trace_id, error_message)."""
    rows = db.fetchall(conn, "SELECT id FROM trace WHERE id LIKE ? LIMIT 5", (ref + "%",))
    if not rows:
        return None, f"trace not found: {ref!r} (see `faithgate candidates`)"
    if len(rows) > 1:
        ids = ", ".join(r["id"][:12] + "…" for r in rows)
        return None, f"prefix {ref!r} is ambiguous ({ids}) — use more characters"
    return rows[0]["id"], None


def _dataset_id(conn: sqlite3.Connection, name: str) -> str:
    rows = db.fetchall(conn, "SELECT id FROM dataset WHERE name = ?", (name,))
    if rows:
        return rows[0]["id"]
    dataset_id = db.new_id()
    db.insert(conn, "dataset", {"id": dataset_id, "name": name, "kind": "promoted",
                                "description": "promoted from captured failures"})
    return dataset_id


def promote_trace(conn: sqlite3.Connection, trace_id: str, *,
                  dataset_name: str = DEFAULT_DATASET,
                  allow_duplicate: bool = False) -> PromoteResult:
    trace = db.fetchall(conn, "SELECT * FROM trace WHERE id = ?", (trace_id,))[0]
    contexts = json.loads(trace["context_json"])
    new_key = content_key(trace["input_query"], contexts)

    score_rows = db.fetchall(conn, """
        SELECT score FROM eval_score
        WHERE trace_id = ? AND metric = 'faithfulness' AND abstained = 0
        ORDER BY id DESC LIMIT 1
    """, (trace_id,))
    score = score_rows[0]["score"] if score_rows else None

    dataset_id = _dataset_id(conn, dataset_name)

    if not allow_duplicate:
        for item in db.fetchall(
                conn, "SELECT id, input_query, context_json FROM dataset_item WHERE dataset_id = ?",
                (dataset_id,)):
            if content_key(item["input_query"], json.loads(item["context_json"])) == new_key:
                return PromoteResult("skipped_duplicate", item["id"], trace["input_query"], score)

    item_id = db.new_id()
    db.insert(conn, "dataset_item", {
        "id": item_id,
        "dataset_id": dataset_id,
        "input_query": trace["input_query"],
        "context_json": trace["context_json"],
        "origin_trace_id": trace_id,   # provenance: score/run/judge all reachable via this join
    })
    return PromoteResult("promoted", item_id, trace["input_query"], score)


def list_datasets(conn: sqlite3.Connection) -> list:
    return db.fetchall(conn, """
        SELECT d.name, d.kind, COUNT(di.id) AS items
        FROM dataset d LEFT JOIN dataset_item di ON di.dataset_id = d.id
        GROUP BY d.id ORDER BY d.name
    """)


def export_probes(conn: sqlite3.Connection, dataset_name: str):
    """Yield probe dicts (id + question + contexts, NO answer) for the user's app to answer.
    The exported id becomes the case's stable identity when the answered suite is run."""
    rows = db.fetchall(conn, """
        SELECT di.id, di.input_query, di.context_json
        FROM dataset_item di JOIN dataset d ON d.id = di.dataset_id
        WHERE d.name = ? ORDER BY di.id
    """, (dataset_name,))
    for r in rows:
        yield {"id": r["id"], "question": r["input_query"],
               "contexts": json.loads(r["context_json"])}
