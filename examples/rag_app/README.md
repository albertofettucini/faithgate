# Demo RAG app

A tiny retrieval-augmented app used as the thing faithgate watches. It retrieves from `docs/`
and answers three questions.

- `app.py` — retrieves + answers, emitting a suite JSONL. Uses Claude when `ANTHROPIC_API_KEY`
  is set; otherwise falls back to a trivial offline answer.
- `suite_baseline.jsonl`, `suite_candidate.jsonl` — pre-generated answer suites used by the
  `eval-gate` CI workflow so it runs deterministically with no API key.

## Try it

```bash
# score two versions and gate them (offline, no key)
PYTHONPATH=. python -m faithgate.gate.cli --db demo.db run \
  --suite examples/rag_app/suite_baseline.jsonl  --label baseline  --judge heuristic
PYTHONPATH=. python -m faithgate.gate.cli --db demo.db run \
  --suite examples/rag_app/suite_candidate.jsonl --label candidate --judge heuristic
PYTHONPATH=. python -m faithgate.gate.cli --db demo.db gate --base baseline --head candidate
```

To see the gate go **red**, edit one answer in `suite_candidate.jsonl` into something the context
does not support, then re-run the three commands.
