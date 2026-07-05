# faithgate

> **You changed a prompt. Did any answer quietly start making things up?**
>
> faithgate scores every answer against its sources, diffs versions, and **fails CI on
> regression** — with a judge whose trustworthiness is *measured* (85%, n=40), never assumed.
> Zero infra, fully local, **pytest for prompts.**

![tests](https://github.com/albertofettucini/faithgate/actions/workflows/tests.yml/badge.svg)
![eval-gate](https://github.com/albertofettucini/faithgate/actions/workflows/eval-gate.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## What it is

It captures your app's question → answer → retrieved-context turns, scores each for **faithfulness**
(is the answer supported by the context it was given?), and **fails the build when a new version
scores worse than the last**. Everything runs locally — traces never leave your machine.

**Zero telemetry.** faithgate itself sends nothing anywhere: no analytics, no phoning home, no
account. The only network traffic is the API call to the judge **you** configured (plus a one-time
HuggingFace model download if you opt into the `[local]` HHEM extra).

## See it catch a regression (60 seconds, offline, no API key)

```bash
git clone https://github.com/albertofettucini/faithgate && cd faithgate

PYTHONPATH=. python3 examples/demo_gate.py
```

```
faithgate regression gate: FAIL ❌

  matched 12 · 3 regressed · 0 improved · 9 unchanged · 0 abstained · 0 new · 0 missing

  Regressions:
    ❌ How many moons does Earth have?  1.00 → 0.29  (below floor)
    ❌ When was Pluto reclassified as a dwarf planet?  1.00 → 0.12  (below floor)
    ❌ At what altitude does the ISS orbit?  0.90 → 0.20  (below floor)

CI exit code would be: 1
```

That's a real end-to-end run: a demo RAG suite where three answers started hallucinating, scored by
the offline judge, caught by the gate. No scripted numbers.

Drive it yourself:

```bash
PYTHONPATH=. python3 -m faithgate.gate.cli --db demo.db run \
  --suite examples/rag_app/suite_baseline.jsonl  --label baseline  --judge heuristic
PYTHONPATH=. python3 -m faithgate.gate.cli --db demo.db run \
  --suite examples/rag_app/suite_regressed.jsonl --label regressed --judge heuristic
PYTHONPATH=. python3 -m faithgate.gate.cli --db demo.db gate --base baseline --head regressed
PYTHONPATH=. python3 -m faithgate.gate.cli --db demo.db up    # browse it → http://127.0.0.1:7654
```

The **base install is stdlib-only**: capture, the offline judge, the gate, and the web panel need
zero dependencies. The real judge lives in an extra (below).

## Why it exists (the honest version)

The "simple + local + eval-first" lane is already well served:

| Tool | Reality |
|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | MIT, one-command, local, eval-first. Acquired by OpenAI (2026). |
| [Arize Phoenix](https://github.com/Arize-ai/phoenix) | `pip install` + SQLite, local, ships judge templates. Core is Elastic-2.0 (source-available). |
| [DeepEval](https://github.com/confident-ai/deepeval) | Apache-2.0, `pip install` + Ollama, 50+ metrics. |

**Why not just promptfoo?** You probably should use promptfoo — faithgate is not a replacement
for your eval stack, it's a **complement** that does exactly one thing: fail-closed blocking of
*faithfulness* regressions, version-to-version, with honesty guarantees about the judge that
produced every number. Keep your existing suites and tools; faithgate runs beside them as the
narrow tripwire for "did my app quietly start making things up." And the roadmap inverts the one
thing hand-written suites can't do: promptfoo's golden sets are authored by you — faithgate's v2
mines them **from your production traffic** (the schema already makes promotion two INSERTs).

faithgate does **not** try to out-feature anyone. It targets the narrow seam:

1. **A regression *gate*, not a dashboard** — run the suite on every change, diff the scores, fail
   the build. Fail-closed: zero matched cases or total abstention can never turn CI green.
2. **Radical honesty about the judge** — a frontier judge (Claude) by default; a *precisely labeled*
   local mode; a measured judge-vs-human agreement number; abstention instead of fake zeros; and a
   judge swap between runs is **flagged and blocked**, never mistaken for a model regression.
3. **Truly zero-infra** — the base tool is Python stdlib only. No Postgres, no ClickHouse, no
   Docker, no Node, no web framework. The one metric library (RAGAS) is an opt-in extra.

## Real scoring with Claude

The trusted default judge is Claude, using **your own** API key (never hardcoded, `.env` is
gitignored):

```bash
# needs Python 3.10+ — on stock macOS (3.9) the easiest path is uv:
#   brew install uv && uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[claude]"            # or: python3 -m pip install --upgrade pip && pip install -e ".[claude]"
export ANTHROPIC_API_KEY=sk-ant-...      # your own key, read from the environment only
faithgate run --suite my_suite.jsonl --label v2 --judge claude
```

A suite is JSONL, one turn per line (`id` is optional — it pins the case's identity so a reworded
question keeps its baseline instead of being treated as a new case):

```json
{"id": "capital-q", "question": "...", "answer": "...", "contexts": ["retrieved chunk 1", "..."]}
```

## Honest judging

faithgate never presents a score as more certain than it is.

- **Frontier judge by default.** Small local models are weak judges — an 8B model scores ~61% on
  faithfulness benchmarks. So Claude is the trusted default; everything else is a labeled trade-off.
- **`local-verification` is labeled precisely.** `--judge claude-local` runs the entailment check
  on-device (Vectara HHEM, via `faithgate[local]`) but still uses Claude for claim extraction. It
  is **not** "fully offline", and the tool says so. Its cost is measured, not guessed: on-device
  verification trades ~8 points of balanced agreement vs the full Claude judge (see the table).
- **The offline judge is deliberately distrusted.** `--judge heuristic` is a token-overlap proxy so
  the whole pipeline runs keyless. Its measured weakness — it cannot see contradictions built from
  the context's own words — is **asserted by a unit test** (`test_heuristic_misses_contradiction`),
  not hidden in a footnote.
- **Abstention, not fake zeros.** When the judge can't score, the result is `abstained` — excluded
  from the gate math and reported separately. If *everything* abstained or errored, the gate and the
  CLI **fail loudly** instead of laundering a broken judge into a green run.
- **Judge changes are flagged.** Every run records a manifest (judge id/model/kind, RAGAS version,
  runner version, suite hash). `gate` refuses to compare runs whose judges differ (exit 3) unless
  you pass `--allow-judge-change`.
- **Measured agreement, sample size shown.** `faithgate calibrate` runs the judge over a 40-example
  hand-labeled golden set (faithful paraphrases, partial support, date/entity swaps, negations,
  unsupported additions):

  | judge | agreement (balanced) | faithful kept | unfaithful caught |
  |---|---|---|---|
  | `claude` (default, claude-sonnet-5) | **85%** *(n=40 — directional)* | 20/20 | 14/20 |
  | `claude-local` (HHEM verify on-device) | 77% *(n=40 — directional)* | 19/20 | 12/20 |
  | `heuristic` (offline proxy) | 68% *(n=40 — directional)* | 18/20 | 9/20 |

  Even the trusted judge is not an oracle — 85% agreement is consistent with published
  frontier-judge benchmarks, and that's exactly why the number is measured and shown instead
  of assumed. Re-measure against your own key anytime: `faithgate calibrate --judge claude`.

## In CI

`.github/workflows/eval-gate.yml` runs **two** offline, deterministic jobs on every push/PR:

- **gate** — baseline vs candidate must stay green (the normal PR gate).
- **proves-detection** — baseline vs a suite with *planted hallucinations* must go **red**; the job
  inverts the exit code. If the gate ever loses its ability to catch a known regression, **this job
  fails the build.** A green badge here isn't decoration — it's continuously re-proven detection.

Both jobs post the score-diff table to the Actions run summary.

## How it works

```
your app ──spans──► POST /v1/spans ─┐        (or: faithgate.capture(...) / faithgate run --suite)
                                     ├─► SQLite (WAL) ──► scoring (RAGAS or offline proxy)
     ingest adapter (one file,       │        │                    │
     isolates experimental semconv)  │        ▼                    ▼
                                     │   web panel      version-to-version diff (by content key)
                                     └────────┴────────────────────┴──► gate: pass / fail / not-comparable
```

- **Capture:** ingest OpenInference-shaped JSON spans, call `faithgate.capture()` directly, or feed
  a suite file. Every trace can carry a `prompt_version_id` (see `faithgate.keys.version_key`) — the
  join key for the replay/drift layers on the [roadmap](ROADMAP.md).
- **Score:** [RAGAS](https://github.com/explodinggradients/ragas) computes faithfulness — embedded,
  not re-implemented. A CI contract-test job imports the exact RAGAS surface we use, so an
  incompatible release breaks loudly.
- **Store:** SQLite + WAL, one file, zero server.
- **Gate:** runs are matched by content (never row id); new, missing, duplicate, and abstained cases
  are always visible in the report.

## Design notes (the non-obvious parts)

Full rationale — locked decisions, fail-closed semantics, schema pre-wiring, deliberate descopes —
lives in [DESIGN.md](DESIGN.md). Highlights:

- **We show the score, not a faked per-claim breakdown.** RAGAS only exposes a scalar publicly;
  scraping its internals would be a fragile reimplementation. v1 shows the score + a one-line
  reason and stays honest. (The per-claim path returns via RAGAS's newer API — see ROADMAP v3.)
- **"Local" can't mean "no LLM."** RAGAS's HHEM variant still needs an LLM to *extract* claims — so
  faithgate ships `local-verification` (HHEM verifies, Claude extracts) and refuses to advertise a
  keyless trusted mode it can't honestly deliver.
- **Fail-closed gate semantics.** The gate's verdict is a function of (matched, regressed, abstained,
  new, missing) — not just "any regressions?". A renamed suite, a broken judge, or 100% abstention
  fails the gate instead of passing vacuously.
- **Case identity is content-based by default.** Rewording a question mints a new case (only the
  score floor guards it, not the delta). If your suite evolves, give cases a stable `id` — that
  pins identity across rewordings.

## Commands

| Command | What it does |
|---|---|
| `faithgate run --suite S --label L [--judge]` | score a suite of answers into a named run |
| `faithgate gate --base A --head B [--allow-judge-change]` | compare two runs; exit non-zero on regression |
| `faithgate runs` | list captured runs |
| `faithgate show --run R` | show a run's scored cases |
| `faithgate score [--judge] [--run] [--retry-errors]` | score pending traces; optionally re-score errored ones |
| `faithgate calibrate [--judge]` | judge agreement with the human-labeled golden set |
| `faithgate up` | start the local web panel |

Judges: `claude` (default, needs key + `[claude]` extra) · `claude-local` (HHEM, `[local]` extra) ·
`heuristic` (offline, zero deps).
Exit codes: `0` ok · `1` gate failed (regression / nothing compared) · `2` usage or input error ·
`3` judge changed between runs.

## Roadmap

See [ROADMAP.md](ROADMAP.md) — v2 auto-eval generation (bad production answers become regression
tests), v2.5 CI gates over the mined dataset, v3 prescriptive feedback + failure clustering, then
replay / drift / cost-quality reports. The v1 schema already carries the join keys they need.

## License

MIT — see [LICENSE](LICENSE).
