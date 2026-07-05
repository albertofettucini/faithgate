"""A tiny RAG app — the thing faithgate watches.

It retrieves from docs/ and answers each question, emitting a suite JSONL that `faithgate run`
can score. If ANTHROPIC_API_KEY is set it answers with Claude (a real RAG app); otherwise it
falls back to a trivial extractive answer so the demo runs offline.

    python examples/rag_app/app.py > suite.jsonl
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from retriever import load_docs, retrieve  # noqa: E402

QUESTIONS = [
    "Which is the smallest planet?",
    "How many moons does Earth have?",
    "Which is the largest planet?",
    "Which planet is the hottest?",
    "What are Saturn's rings made of?",
    "Which planet has the fastest winds?",
    "When was Pluto reclassified as a dwarf planet?",
    "At what altitude does the ISS orbit?",
]


def answer(question: str, contexts: list) -> str:
    context = "\n".join(contexts)
    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        # no temperature: Sonnet-5-class models reject non-default sampling params
        llm = ChatAnthropic(model="claude-sonnet-5", max_tokens=1024)
        prompt = (
            "Answer the question using ONLY the context. Be concise.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return llm.invoke(prompt).content.strip()
    # offline fallback: first sentence of the top retrieved context
    return contexts[0].split(". ")[0].strip() + "." if contexts else ""


def main() -> int:
    docs = load_docs(os.path.join(os.path.dirname(__file__), "docs"))
    for question in QUESTIONS:
        contexts = retrieve(docs, question, k=1)
        print(json.dumps(
            {"question": question, "answer": answer(question, contexts), "contexts": contexts},
            ensure_ascii=False,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
