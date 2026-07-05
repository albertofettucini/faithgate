-- faithgate system-of-record. SQLite + WAL. Applied idempotently at startup (no migrations in v1).
-- Score-centric: eval_score is the hub linking a captured response to its number, its grounding
-- context, and its judge provenance. A few columns are reserved NOW so v2 (promote a bad response
-- into a reusable dataset) and v3 (prescriptive feedback) are pure additive writes, never a rewrite.

-- one gate execution = one prompt/model version under test
CREATE TABLE IF NOT EXISTS run (
  id              TEXT PRIMARY KEY,                 -- ULID
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  label           TEXT,                             -- e.g. git sha or 'prompt v7'
  prompt_version  TEXT,
  model_under_test TEXT,                            -- the APP's model, NOT the judge
  runner_version  TEXT,
  dataset_id      TEXT REFERENCES dataset(id),      -- suite under test; NULL for ad-hoc capture
  manifest_json   TEXT,                             -- FROZEN: judge model/seed/temp, ragas ver, hhem rev, dataset hash, threshold
  git_sha         TEXT,
  status          TEXT NOT NULL DEFAULT 'capturing' -- capturing|scoring|complete|failed
);

-- one captured request/response unit (the thing scored, and the v2 promotion source)
CREATE TABLE IF NOT EXISTS trace (
  id              TEXT PRIMARY KEY,                 -- ULID
  run_id          TEXT REFERENCES run(id),          -- nullable: capture can land before a run claims it
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  otel_trace_id   TEXT,                             -- join-back to spans
  input_query     TEXT NOT NULL,                    -- the question
  output_response TEXT NOT NULL,                    -- the answer (shown in panel; promoted in v2)
  context_json    TEXT NOT NULL,                    -- retrieved chunks (RAGAS-required; v2 rebuilds a test case from this)
  context_hash    TEXT,                             -- diff match key: content hash, or 'id:<case_id>' when the suite pins identity
  prompt_version_id TEXT,                           -- version of the SYSTEM UNDER TEST at capture time (join key
                                                    -- for replay/drift/CI layers; run-level fields cover CLI suites,
                                                    -- this covers continuous ingest where versions can mix)
  source          TEXT NOT NULL DEFAULT 'otel',     -- otel|decorator
  source_dataset_item_id TEXT REFERENCES dataset_item(id), -- set only when replaying a dataset item (v2 link); NULL in v1
  status          TEXT NOT NULL DEFAULT 'pending'   -- pending|scored|error
);
CREATE INDEX IF NOT EXISTS idx_trace_run    ON trace(run_id);
CREATE INDEX IF NOT EXISTS idx_trace_otel   ON trace(otel_trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_status ON trace(status);

-- raw OTel/OpenInference spans (lossless capture; only the adapter reads attributes_json)
CREATE TABLE IF NOT EXISTS span (
  id              TEXT PRIMARY KEY,                 -- ULID
  trace_id        TEXT NOT NULL REFERENCES trace(id),
  otel_span_id    TEXT,
  parent_span_id  TEXT,
  name            TEXT,
  span_kind       TEXT,                             -- normalized enum LLM|RETRIEVER|EMBEDDING|CHAIN (NOT raw semconv)
  start_time      TEXT,
  end_time        TEXT,
  attributes_json TEXT,                             -- verbatim attribute bag
  schema_adapter_version TEXT                        -- which adapter parsed this; bump on semconv change
);
CREATE INDEX IF NOT EXISTS idx_span_trace ON span(trace_id);

-- judge provenance (honest-computation made queryable)
CREATE TABLE IF NOT EXISTS judge_run (
  id              TEXT PRIMARY KEY,                 -- ULID
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  judge_type      TEXT NOT NULL,                    -- frontier | local_verification
  judge_model     TEXT,                             -- e.g. 'claude-sonnet-4-6' (and HHEM rev for the NLI step)
  judge_provider  TEXT,
  decomposer_model TEXT,                            -- the LLM doing claim extraction (frontier even in local_verification)
  is_default      INTEGER NOT NULL DEFAULT 0,       -- the trusted headline judge? (local-8B decompose NEVER is_default=1)
  ragas_metric_impl TEXT,                           -- 'Faithfulness' | 'FaithfulnessWithHHEM'
  temperature     REAL,
  seed            INTEGER,
  prompt_template_hash TEXT,
  golden_agreement REAL                             -- measured human-agreement for THIS config; NULL until calibrated
);

-- THE HUB: one faithfulness score for one trace under one run
CREATE TABLE IF NOT EXISTS eval_score (
  id              TEXT PRIMARY KEY,                 -- ULID
  trace_id        TEXT NOT NULL REFERENCES trace(id),
  run_id          TEXT REFERENCES run(id),          -- denormalized for fast diff/rollup
  judge_run_id    TEXT REFERENCES judge_run(id),
  metric          TEXT NOT NULL DEFAULT 'faithfulness', -- a COLUMN so a 2nd metric is an INSERT, not an ALTER
  score           REAL,                             -- 0..1 ; NULL when abstained
  passed          INTEGER,                          -- stored, not recomputed (survives a threshold change)
  reason          TEXT,                             -- RAGAS single-line reason (panel) — NOT per-claim
  claims_json     TEXT,                             -- RESERVED for v3; NULL in all of v1, no v1 writer
  abstained       INTEGER NOT NULL DEFAULT 0,       -- NaN/empty/error -> 1 ; honesty surface (never a fake 0.0)
  confidence      REAL,
  created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_score_trace ON eval_score(trace_id);
CREATE INDEX IF NOT EXISTS idx_score_run   ON eval_score(run_id, metric);

-- v1 holds ONLY the golden set; the shape IS the v2 promotion target
CREATE TABLE IF NOT EXISTS dataset (
  id              TEXT PRIMARY KEY,                 -- ULID
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  name            TEXT,
  kind            TEXT NOT NULL DEFAULT 'golden',   -- v1 writes only 'golden'; enum existing is the v2 unblocker (golden|promoted|regression)
  description     TEXT
);
CREATE TABLE IF NOT EXISTS dataset_item (
  id              TEXT PRIMARY KEY,                 -- ULID
  dataset_id      TEXT NOT NULL REFERENCES dataset(id),
  created_at      TEXT NOT NULL DEFAULT (datetime('now')),
  input_query     TEXT NOT NULL,
  context_json    TEXT NOT NULL,
  reference_answer TEXT,                            -- golden items have a human reference; promoted-from-bad items may not
  human_label     TEXT,                             -- for golden-set agreement scoring (e.g. '1'/'0')
  origin_trace_id TEXT REFERENCES trace(id)         -- v1: always NULL ; v2: 'promote to dataset' sets it to the bad trace
);
CREATE INDEX IF NOT EXISTS idx_item_dataset ON dataset_item(dataset_id);
