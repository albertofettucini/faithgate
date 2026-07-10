# Ideas — recorded, deliberately not built

Scope locks first, so the list can't creep: FaithGate stays **one metric done well** (no
relevancy/coherence/etc. — that lane belongs to DeepEval), **no panel expansion**, and every
shipped capability sentence keeps code/test backing. Unflattering numbers get published.

- **Judge cascade** — deterministic substring pre-filter → HHEM NLI → LLM judge only on
  escalation. All three stages already exist as separate modes; the feature is chaining them with
  per-stage provenance. Source: a practitioner's production setup.
- **Agent side-effect assertions** — for prompts that drive coding agents: diff an
  expected-vs-actual touched-files manifest ("the output looked fine, the diff didn't").
- **pytest plugin** — after score caching lands (without it, per-PR frontier-judge tests are
  slow, costly, and flaky).
- **Failure-mining policy** — promote-everything vs filtered promotion is an open research
  question; the answer may come from community usage.
- **Judge diversity** — reduce same-family correlation (e.g. Claude judging Claude) by mixing
  judge families in verification; see the relative-deltas note in the README.
