"""The simple capture path — record one RAG turn into the store (status 'pending').

This is the low-friction alternative to OpenTelemetry ingestion: an app can call
``capture(...)`` directly. Both paths write the same ``trace`` row, so everything
downstream (scoring, diff, panel) is identical regardless of how the trace arrived.
"""
from __future__ import annotations

import json
import sqlite3

from ..keys import content_key
from ..store import db


def capture(
    conn: sqlite3.Connection,
    run_id: str,
    question: str,
    answer: str,
    contexts: list,
    *,
    source: str = "decorator",
    otel_trace_id: str | None = None,
    prompt_version_id: str | None = None,
    case_id: str | None = None,
) -> str:
    """Store one (question, answer, retrieved-contexts) turn. Returns the new trace id.

    ``prompt_version_id`` stamps which version of the app produced this turn (see
    ``faithgate.keys.version_key``) — the join key for future replay/drift/CI layers.
    ``case_id`` (optional) pins the case's identity for the diff: without it, cases are matched
    by content, so rewording a question mints a new case; with it, the reworded case keeps its
    baseline across versions.
    """
    trace_id = db.new_id()
    db.insert(conn, "trace", {
        "id": trace_id,
        "run_id": run_id,
        "input_query": question,
        "output_response": answer,
        "context_json": json.dumps(contexts, ensure_ascii=False),
        "context_hash": f"id:{case_id}" if case_id else content_key(question, contexts),
        "prompt_version_id": prompt_version_id,
        "source": source,
        "otel_trace_id": otel_trace_id,
        "status": "pending",
    })
    return trace_id
