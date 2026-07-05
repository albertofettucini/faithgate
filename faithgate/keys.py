"""Content key — the stable identity of a test case across runs.

We diff two versions by matching the SAME question+context, never by row id. Normalizing
whitespace and case means trivial edits don't accidentally look like a different test case.
"""
from __future__ import annotations

import hashlib
import json


def version_key(prompt_template: str, model: str, params: dict = None) -> str:
    """Deterministic short id for a system-under-test version: hash of prompt template + model
    + key params. Lets apps stamp captured traffic without inventing version names."""
    blob = json.dumps({"t": prompt_template, "m": model, "p": params or {}},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def content_key(question: str, contexts: list) -> str:
    norm_q = " ".join(question.lower().split())
    norm_ctx = "\n".join(" ".join(str(c).lower().split()) for c in contexts)
    return hashlib.sha256((norm_q + "\x00" + norm_ctx).encode("utf-8")).hexdigest()
