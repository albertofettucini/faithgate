# FaithGate — design rationale

Why the tool is shaped the way it is: the locked decisions, the traps they avoid, and what was
deliberately cut. Companion to [README.md](README.md) (what it does) and [ROADMAP.md](ROADMAP.md)
(what comes next).

## The one principle

**Never present a number as more certain than it is.** Every design call below is this principle
applied somewhere: to the judge, to the gate verdict, to the docs, to the dependency list.

## Positioning

The "simple + local + eval-first" lane is well served (promptfoo, Arize Phoenix, DeepEval — see the
README's comparison). FaithGate deliberately does NOT compete there. It is a *complement* with one
job: **fail-closed blocking of faithfulness regressions**, version-to-version, with measured honesty
about the judge that produced every number.

## Locked decisions

| # | Decision | Why |
|---|---|---|
| D1 | No per-claim breakdown in v1 | RAGAS publicly returns only a scalar; scraping its internals for per-claim verdicts would be a fragile reimplementation that breaks on minor releases. The panel shows the score + a one-line reason and stays honest. (RAGAS's newer collections API exposes `.reason` — the v3 path.) |
| D2 | "Local" mode = `local-verification`, precisely labeled | RAGAS's HHEM variant still requires an LLM for claim *extraction* — HHEM only replaces the entailment check. A "fully offline trusted judge" cannot honestly be delivered, so it isn't advertised. |
| D3 | Frontier judge (Claude) as the trusted default | Laptop-size local models score ~61% on faithfulness benchmarks — near coin-flip on hard cases. The offline heuristic exists so the pipeline runs keyless, but it is deliberately distrusted and its blindness is asserted by a unit test. |
| D4 | Capture = span ingest + `capture()` helper + suite files | No custom proxy (single point of failure in the user's request path) and no per-provider SDK (unmaintainable). The ingest adapter is one file, isolating the still-experimental GenAI semconv; a startup self-test fails loudly if extraction breaks. |
| D5 | SQLite + WAL, single file | Single-writer is a non-issue for a single-user tool. Analytical layers (DuckDB etc.) were cut when plain SQL proved sufficient — see "Changed during build". |
| D6 | Embed RAGAS; never re-implement metric math | The metric is battle-tested and citable. This project's value is the harness around it: the gate, the diffing, the calibration, the honesty surfaces. A CI contract-test job imports the exact RAGAS surface used, so an incompatible release breaks loudly. |
| D7 | Abstention is a distinct state | NaN / zero-extracted-statements / judge errors are never a `0.0` and never a regression — but if *everything* abstained, the gate fails closed instead of blessing an unjudged run. |
| D8 | Manifests + judge-change guard | Every run pins judge id/model/kind, RAGAS version, runner version, suite hash. The gate refuses to compare runs whose judges differ (exit 3): a judge swap must never masquerade as a model regression. Corrupted manifests fail closed — corruption is not absence. |
| D9 | Content-keyed diffing | Runs are matched by `SHA256(normalized question + contexts)` (or an explicit per-case `id`), never by row id — so the gate distinguishes "same case, score moved" from "different case". |
| D10 | Zero-dependency base install | Capture, the offline judge, the gate, and the stdlib web panel run with no third-party packages. RAGAS + the Claude client live in the `[claude]` extra; torch/HHEM in `[local]`. |
| D11 | Version identity at capture time | Every trace can carry a `prompt_version_id` (see `faithgate.keys.version_key`) — the join key for the replay/drift layers on the roadmap. System-under-test identity is kept separate from judge identity: merging them would destroy the "did my app change or did my judge change?" distinction. |

## Gate semantics (fail-closed)

The verdict is a function of *(matched, regressed, abstained, new, missing, duplicates)* — not just
"any regressions?":

- **Zero matched cases → FAIL** ("nothing compared"). A renamed suite, an unscored run, or 100%
  abstention can never turn CI green.
- **The score floor guards every scored case** — new cases and abstained-baseline cases included.
- **Duplicates:** the head keeps the *lowest* duplicate score (strict on the new version); the
  baseline keeps the *highest* (a stray low baseline entry must not quietly lower the bar).
- **Policy knobs** close the quiet channels: `--fail-on-missing` (deleted cases), `--max-abstained`
  (targeted abstention). Exit codes: 0 ok · 1 fail · 2 usage/input error · 3 judge changed.
- Known, documented limitation: without an explicit `id`, rewording a question mints a new case —
  inherent to content-keyed matching; the `id` field is the escape hatch.

## Data model (score-centric, forward-wired)

Seven tables: `run`, `trace`, `span`, `judge_run`, `eval_score`, `dataset`, `dataset_item`.
`eval_score` is the hub linking a captured response to its number, grounding context, and judge
provenance. Three forward-looking choices cost nothing now and unblock the roadmap without
migrations:

- `eval_score.metric` is a column (second metric = INSERT, not ALTER).
- A captured response and a curated test case share row shape; `dataset_item.origin_trace_id` ↔
  `trace.source_dataset_item_id` pre-wire "promote a bad production answer into a regression test"
  as two INSERTs (roadmap v2).
- Contexts live on the trace (not only inside raw span attributes), so replay and promotion never
  re-parse experimental span formats. Raw spans are retained verbatim as churn insurance.

## The calibration harness

The judge is an AI too, so it takes an exam: a 40-example hand-labeled golden set, stratified across
failure types (faithful paraphrases, partial support, date/entity swaps, negations, unsupported
additions). `faithgate calibrate` reports agreement-with-humans per judge, with n shown, and its
error surface is loud: per-sample timeouts, live progress, first-error printed, exit 2 when nothing
was actually judged. Measured results live in the README and are re-measurable by anyone with a key.

## Changed during build (deliberate descopes)

Recorded because pretending the original plan was built would violate the one principle:

- **Panel:** stdlib `http.server` instead of a web framework — this is what makes the base install
  dependency-free.
- **DuckDB analytics layer:** cut; plain SQL over SQLite is sufficient at this scale.
- **Async scoring worker:** deferred; `run` scores synchronously after capture, ingested spans are
  scored via `faithgate score`.
- **OTLP wire format:** deferred; `/v1/spans` accepts OpenInference-shaped JSON and is documented as
  such, not as OTLP compatibility.
- **Per-call hardening added after real-world testing:** RAGAS injects `temperature=0.01` per call
  unless bypassed (Sonnet-5-class models reject non-default sampling params) and retries every
  exception 10× with up to 60s waits — turning any persistent error into a multi-minute silent
  hang. FaithGate bypasses the injection, caps retries, bounds every judge call with a timeout, and
  surfaces the first real error instead.

## What proves the claims

Every capability sentence in the README maps to a test or CI job. The flagship: the `eval-gate`
workflow's **proves-detection** job scores a suite with planted hallucinations and *asserts the
gate fails* by inverting the exit code — if the gate ever loses its ability to catch a known
regression, CI itself goes red.
