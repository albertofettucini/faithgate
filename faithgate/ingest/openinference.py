"""OpenInference / OpenTelemetry span adapter — the ONE place experimental semconv is read.

Existing OpenInference/OpenLLMetry instrumentation already captures LLM + retrieval spans across
providers; we just ingest them. This module maps a span's (still-experimental) GenAI attributes
into our stable internal shape (question, answer, contexts). If the spec renames a key, only this
file changes — and the bundled `self_test()` fails loudly at startup instead of silently dropping
every trace.
"""
from __future__ import annotations

import json
import re

_DOC_RE = re.compile(r"^retrieval\.documents\.(\d+)\.document\.content$")


def _prompt_version(attrs: dict):
    """Version of the system under test, if the app stamped it on the span."""
    v = attrs.get("faithgate.prompt_version") or attrs.get("prompt.version")
    if v:
        return str(v)
    meta = attrs.get("metadata")  # OpenInference convention: JSON-string metadata attribute
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            return None
    if isinstance(meta, dict):
        v = meta.get("prompt_version") or meta.get("prompt_version_id")
        return str(v) if v else None
    return None


def _first_message(attrs: dict, prefix: str):
    return attrs.get(f"{prefix}.0.message.content")


def _retrieved_contexts(attrs: dict) -> list:
    docs = []
    for key, value in attrs.items():
        m = _DOC_RE.match(key)
        if m:
            docs.append((int(m.group(1)), value))
    return [v for _, v in sorted(docs)]


def span_to_turn(span: dict):
    """Extract (question, answer, contexts) from an OpenInference span dict, or None if it isn't one."""
    attrs = span.get("attributes") or {}
    question = attrs.get("input.value") or _first_message(attrs, "llm.input_messages")
    answer = attrs.get("output.value") or _first_message(attrs, "llm.output_messages")
    if not (question and answer):
        return None
    return {
        "question": str(question),
        "answer": str(answer),
        "contexts": _retrieved_contexts(attrs),
        "otel_trace_id": span.get("trace_id"),
        "prompt_version_id": _prompt_version(attrs),
    }


def capture_spans(conn, run_id: str, spans: list) -> int:
    """Turn OpenInference spans into captured traces. Returns how many were captured."""
    from .decorator import capture

    n = 0
    for span in spans:
        turn = span_to_turn(span)
        if turn:
            capture(conn, run_id, turn["question"], turn["answer"], turn["contexts"],
                    source="otel", otel_trace_id=turn.get("otel_trace_id"),
                    prompt_version_id=turn.get("prompt_version_id"))
            n += 1
    return n


FIXTURE_SPAN = {
    "trace_id": "fixture-0001",
    "attributes": {
        "openinference.span.kind": "CHAIN",
        "input.value": "What is the capital of France?",
        "output.value": "Paris is the capital of France.",
        "retrieval.documents.0.document.content": "France's capital is Paris.",
        "retrieval.documents.1.document.content": "Paris is in northern France.",
        "metadata": "{\"prompt_version\": \"v7\"}",
    },
}


def self_test() -> bool:
    """Assert the adapter still extracts a full turn from a known-good span. Run at startup."""
    turn = span_to_turn(FIXTURE_SPAN)
    assert turn and turn["question"] and turn["answer"] and turn["contexts"], \
        "OpenInference adapter failed on the fixture span — the semconv keys may have changed."
    return True
