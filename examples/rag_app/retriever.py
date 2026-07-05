"""A deliberately tiny keyword retriever — enough to be a real RAG app, small enough to read."""
from __future__ import annotations

import glob
import os
import re


def load_docs(docs_dir: str) -> dict:
    docs = {}
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.md"))):
        with open(path, encoding="utf-8") as fh:
            docs[os.path.basename(path)] = fh.read().strip()
    return docs


def _tokens(text: str) -> set:
    out = set()
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        out.add(w)
        if len(w) > 3 and w.endswith("s"):
            out.add(w[:-1])  # crude singular so 'moons' matches 'moon'
    return out


def retrieve(docs: dict, query: str, k: int = 1) -> list:
    q = _tokens(query)

    def score(item):
        name, text = item
        return len(q & (_tokens(text) | _tokens(name.replace(".md", ""))))

    ranked = sorted(docs.items(), key=score, reverse=True)
    return [text for _, text in ranked[:k]]
