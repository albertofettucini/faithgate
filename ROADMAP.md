# FaithGate roadmap — post-v1 layers

All layers feed on the same v1 foundation: **trace + score + version identity stored together**
(`trace.prompt_version_id` at capture time, `run.*`/`manifest_json` for suite runs, raw span
retention in `span.attributes_json`). None of this is v1 scope; nothing here may widen v1.

## v2 — auto-eval generation (the star)
Low-scoring captured responses get promoted into a reusable, versioned test dataset. Production
failures become regression tests.
*Already pre-wired in v1:* `dataset.kind` enum, `dataset_item.origin_trace_id` ↔
`trace.source_dataset_item_id`, contexts stored on the trace — promotion is two INSERTs.

## v2.5 — CI quality gate over the mined dataset
The auto-generated dataset runs on every prompt change; a quality drop below threshold blocks
the PR. The twist vs promptfoo: **the golden set is mined from production automatically, not
hand-written.** (v1 already ships the gate mechanics; this layer swaps in the mined suite via
`run --suite dataset:<name>`.)

## v3 — prescriptive feedback + failure clustering
Not just a score: "bad *because X*, try Y", plus auto-grouping bad responses into named failure
modes ("hallucinated dates", "ignored provided context").
*Known tension (flagged, not redesigned):* v1 deliberately dropped per-claim reasoning because
RAGAS's legacy API only returns a scalar publicly. The ragas 0.4 `collections` API returns a
`MetricResult` with `.reason` — migrating the scorer seam (one file) unlocks the raw material
for clustering. Reserved column: `eval_score.claims_json`.

## v3+ — replay ("time machine")
Re-run captured production traffic against a new prompt/model offline; diff scores before
deploying. *Enabled by v1:* contexts live on the trace (generation-only replay with identical
retrieval) and verbatim spans retain full input messages (full replay). Join key:
`prompt_version_id`.

## v3+ — drift detection
Continuous scoring exposes "no code change, but quality dropped" when a provider silently
updates a model. *Requires:* stable judge across time — exactly what the run manifest +
judge-change guard enforce. Passive report.

## v3+ — cost-quality map
Replay captured traffic on a cheaper model, compare scores: "X% of your queries hold quality at
10× lower cost." **Passive report, NOT active routing.**

## Standing constraints
- Every layer stays inside the brand: one metric done honestly, zero-infra, labeled judges,
  abstention over fake numbers.
- Drift/cost-map are the closest to the saturated observability-platform lane faithgate
  deliberately exited — they ship as *reports over data you already have*, never as dashboards,
  alerting, or routing.
- Any v1 decision that would block a layer here gets flagged and decided explicitly — no silent
  redesigns in either direction.
