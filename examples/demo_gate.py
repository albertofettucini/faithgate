"""End-to-end demo of the regression gate — runs with NO API key and NO scripted numbers.

It scores two REAL suites of the demo RAG app with the offline heuristic judge:
  * suite_baseline.jsonl  — grounded answers
  * suite_regressed.jsonl — the same suite where three answers started hallucinating

The heuristic judge genuinely catches the planted hallucinations, the gate goes RED, and the
process exits 1 — exactly what the CI job asserts on every push.

    PYTHONPATH=. python3 examples/demo_gate.py            # self-contained, temp db
    PYTHONPATH=. python3 examples/demo_gate.py my.db      # seed a persistent db to explore
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

from faithgate.gate.diff import compare, load_run_scores, render_text
from faithgate.ingest.decorator import capture
from faithgate.score.judges import make_scorer
from faithgate.score.worker import score_pending
from faithgate.store import db

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "rag_app", "suite_baseline.jsonl")
REGRESSED = os.path.join(HERE, "rag_app", "suite_regressed.jsonl")


async def make_run(conn, label: str, suite_path: str) -> str:
    scorer, judge = make_scorer("heuristic")
    run_id = db.new_id()
    db.insert(conn, "run", {"id": run_id, "label": label, "status": "scoring"})
    with open(suite_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            capture(conn, run_id, item["question"], item["answer"], item.get("contexts", []),
                    prompt_version_id=label)
    await score_pending(conn, scorer, judge, run_id=run_id)
    conn.execute("UPDATE run SET status='complete' WHERE id=?", (run_id,))
    conn.commit()
    return run_id


async def main(db_path: str = None) -> int:
    conn = db.connect(db_path or tempfile.mktemp(suffix=".db"))
    db.init_db(conn)

    base_run = await make_run(conn, "prompt v1 (baseline)", BASELINE)
    head_run = await make_run(conn, "prompt v2 (regressed)", REGRESSED)

    if db_path:
        print(f"Seeded demo runs into {db_path}")
        print(f'  Try:  faithgate runs --db {db_path}')
        print(f'        faithgate gate --db {db_path} --base "prompt v1 (baseline)" --head "prompt v2 (regressed)"')
        return 0

    result = compare(load_run_scores(conn, base_run), load_run_scores(conn, head_run))
    print()
    print(render_text(result))
    print()
    print(f"CI exit code would be: {0 if result.passed else 1}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    raise SystemExit(asyncio.run(main(_path)))
