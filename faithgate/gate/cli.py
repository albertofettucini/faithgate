"""faithgate command-line interface.

    faithgate runs                       list captured runs
    faithgate run   --suite S --label L  score a suite of answers into a run
    faithgate gate  --base A --head B    compare two runs; exit non-zero on regression
    faithgate show  --run R              show one run's scored traces
    faithgate score [--retry-errors]     score pending (ingested-but-unscored) traces
    faithgate calibrate                  judge agreement with the human-labeled golden set
    faithgate candidates / promote       turn captured failures into regression test cases
    faithgate datasets / export          list datasets / export probes for your app
    faithgate up                         start the local web panel

Every command accepts --db before OR after the subcommand (defaults to ~/.faithgate/faithgate.db).

Exit codes: 0 ok · 1 gate failed (regression / nothing compared) · 2 usage or input error ·
3 judge changed between runs (scores not comparable; pass --allow-judge-change to override).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3

from .. import __version__
from ..gate.diff import compare, load_run_scores, render_json, render_markdown, render_text
from ..store import db

DEFAULT_DB = os.path.expanduser("~/.faithgate/faithgate.db")

_JUDGE_KEYS = ("judge_id", "judge_model", "judge_kind")  # manifest keys that define comparability


def _resolve_run(conn: sqlite3.Connection, ref: str):
    """Accept a run id or a run label; newest wins on a label tie (ULIDs sort by time)."""
    rows = db.fetchall(
        conn,
        "SELECT id FROM run WHERE id = ? OR label = ? ORDER BY id DESC LIMIT 1",
        (ref, ref),
    )
    return rows[0]["id"] if rows else None


def _ragas_version():
    try:
        from importlib.metadata import version
        return version("ragas")
    except Exception:
        return None


def _judge_manifest(judge) -> dict:
    return {
        "judge_id": judge.id,
        "judge_model": judge.model,
        "judge_provider": judge.provider,
        "judge_kind": judge.kind,
        "temperature": judge.temperature,
        "seed": judge.seed,
        "ragas_version": _ragas_version(),
        "runner_version": __version__,
    }


def _load_manifest(conn: sqlite3.Connection, run_id: str):
    """Returns (manifest_or_None, status) — status ∈ ok|missing|corrupt. Corruption is NOT the
    same as absence: a mangled manifest must not silently defeat the judge-change guard."""
    rows = db.fetchall(conn, "SELECT manifest_json FROM run WHERE id = ?", (run_id,))
    if rows and rows[0]["manifest_json"]:
        try:
            return json.loads(rows[0]["manifest_json"]), "ok"
        except json.JSONDecodeError:
            return None, "corrupt"
    return None, "missing"


def _cmd_runs(conn: sqlite3.Connection) -> int:
    rows = db.fetchall(conn, """
        SELECT r.id, r.label, r.status, r.created_at, COUNT(t.id) AS n
        FROM run r LEFT JOIN trace t ON t.run_id = r.id
        GROUP BY r.id ORDER BY r.id
    """)
    if not rows:
        print("No runs yet.")
        return 0
    print(f"{'CREATED':<20}  {'STATUS':<10}  {'CASES':>5}  LABEL")
    for r in rows:
        print(f"{r['created_at']:<20}  {r['status']:<10}  {r['n']:>5}  {r['label'] or '(unnamed)'}")
    return 0


def _read_suite(path: str):
    """Parse and validate a suite JSONL fully BEFORE any row is written — a broken suite must
    not leave a half-created run behind. Returns (items, error_message)."""
    items = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    return None, f"{path}:{lineno}: invalid JSON — {exc}"
                if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                    return None, (f"{path}:{lineno}: each line must be a JSON object "
                                  "with 'question' and 'answer' keys")
                items.append(item)
    except OSError as exc:
        return None, f"cannot read suite: {exc}"
    if not items:
        return None, f"{path}: suite is empty"
    return items, None


def _cmd_run(conn: sqlite3.Connection, args) -> int:
    import asyncio

    from ..ingest.decorator import capture
    from ..score.judges import make_scorer
    from ..score.worker import score_pending

    try:
        scorer, judge = make_scorer(args.judge)
    except (RuntimeError, ImportError, ValueError) as exc:
        print(exc)
        return 2

    items, err = _read_suite(args.suite)
    if err:
        print(err)
        return 2

    with open(args.suite, "rb") as fh:
        suite_sha = hashlib.sha256(fh.read()).hexdigest()[:12]
    manifest = _judge_manifest(judge)
    manifest["suite_sha"] = suite_sha

    run_id = db.new_id()
    db.insert(conn, "run", {
        "id": run_id, "label": args.label, "status": "scoring",
        "manifest_json": json.dumps(manifest), "runner_version": __version__,
    })
    try:
        for item in items:
            capture(conn, run_id, item["question"], item["answer"], item.get("contexts", []),
                    prompt_version_id=item.get("prompt_version_id") or args.label,
                    case_id=item.get("id"))
        scored, errored = asyncio.run(score_pending(conn, scorer, judge, run_id=run_id))
    except BaseException:
        conn.execute("UPDATE run SET status='failed' WHERE id=?", (run_id,))
        conn.commit()
        raise
    all_failed = errored > 0 and scored == 0
    conn.execute("UPDATE run SET status=? WHERE id=?",
                 ("failed" if all_failed else "complete", run_id))
    conn.commit()
    print(f"captured {len(items)}, scored {scored}, errored {errored} "
          f"into run '{args.label}'  (judge: {judge.id})")
    if errored:
        print(f"⚠ {errored} case(s) errored — inspect with `faithgate show --run \"{args.label}\"`,"
              " then `faithgate score --retry-errors`.")
    if all_failed:
        print("✗ every case errored — the judge is not working; refusing to report success.")
        return 2
    return 0


def _cmd_gate(conn: sqlite3.Connection, args) -> int:
    base_id = _resolve_run(conn, args.base)
    head_id = _resolve_run(conn, args.head)
    if base_id is None or head_id is None:
        missing = args.base if base_id is None else args.head
        print(f"Run not found: {missing!r}. See `faithgate runs`.")
        return 2

    # honesty guard: a judge swap must never masquerade as a model regression
    base_m, base_st = _load_manifest(conn, base_id)
    head_m, head_st = _load_manifest(conn, head_id)
    if "corrupt" in (base_st, head_st):
        # fail closed: corruption ≠ absence — an unreadable manifest must not defeat the guard
        print("⚠ MANIFEST UNREADABLE — judge comparability unknown (corrupted manifest_json).")
        if not args.allow_judge_change:
            print("  Fix the run row, or pass --allow-judge-change to compare anyway.")
            return 3
        print("  (--allow-judge-change given — proceeding anyway)")
    elif base_m and head_m:
        changed = [k for k in _JUDGE_KEYS if base_m.get(k) != head_m.get(k)]
        if changed:
            print("⚠ JUDGE CHANGED between runs — scores are NOT comparable:")
            for k in changed:
                print(f"    {k}: {base_m.get(k)!r} → {head_m.get(k)!r}")
            if not args.allow_judge_change:
                print("  Re-run both suites with the same judge, or pass --allow-judge-change.")
                return 3
            print("  (--allow-judge-change given — proceeding anyway)")
    elif base_st == "missing" and head_st == "missing":
        print("note: neither run has a manifest — judge comparability unknown.")
    else:
        print("note: one run has no manifest — judge comparability unknown.")

    head_scores = load_run_scores(conn, head_id)
    result = compare(
        load_run_scores(conn, base_id),
        head_scores,
        max_regression=args.max_regression,
        min_score=args.min_score,
    )

    # policy knobs closing the quiet channels: deleted cases and targeted abstention
    policy_failures = []
    if args.fail_on_missing and result.missing_cases:
        policy_failures.append(
            f"{len(result.missing_cases)} baseline case(s) missing from the new run (--fail-on-missing)")
    if args.max_abstained is not None and len(result.abstained) > args.max_abstained:
        policy_failures.append(
            f"{len(result.abstained)} abstained case(s) exceeds --max-abstained {args.max_abstained}")
    if policy_failures:
        result.passed = False

    # write the verdict back so history survives threshold changes (PLAN D13)
    regressed_ids = {c.key for c in result.regressions}
    for t in head_scores:
        if t.abstained or t.score is None or not t.trace_id:
            continue
        db_passed = 0 if t.key in regressed_ids else 1
        conn.execute(
            "UPDATE eval_score SET passed=? WHERE trace_id=? AND run_id=? AND metric='faithfulness'",
            (db_passed, t.trace_id, head_id),
        )
    if head_m is not None:
        head_m["gate"] = {"base": args.base, "max_regression": args.max_regression,
                          "min_score": args.min_score, "passed": result.passed}
        conn.execute("UPDATE run SET manifest_json=? WHERE id=?", (json.dumps(head_m), head_id))
    conn.commit()

    if args.format == "markdown":
        print(render_markdown(result, policy_failures))
    elif args.format == "json":
        print(render_json(result, policy_failures))
    else:
        print(render_text(result))
        for p in policy_failures:
            print(f"  ✗ policy: {p}")
    return 0 if result.passed else 1


def _cmd_show(conn: sqlite3.Connection, args) -> int:
    run_id = _resolve_run(conn, args.run)
    if run_id is None:
        print(f"Run not found: {args.run!r}. See `faithgate runs`.")
        return 2
    rows = db.fetchall(conn, """
        SELECT t.input_query AS q, s.score AS score, s.abstained AS abstained, s.reason AS reason
        FROM trace t LEFT JOIN eval_score s ON s.trace_id = t.id
        WHERE t.run_id = ? ORDER BY t.id
    """, (run_id,))
    for r in rows:
        if r["abstained"]:
            mark = "  ?  "
        elif r["score"] is None:
            mark = "  ·  "
        else:
            mark = f" {r['score']:.2f}"
        print(f"{mark}  {r['q']}")
        if r["reason"]:
            print(f"        ↳ {r['reason']}")
    return 0


def _cmd_score(conn: sqlite3.Connection, args) -> int:
    import asyncio

    from ..score.judges import make_scorer
    from ..score.worker import score_pending

    try:
        scorer, judge = make_scorer(args.judge)
    except (RuntimeError, ImportError, ValueError) as exc:
        print(exc)
        return 2
    run_id = None
    if args.run:
        run_id = _resolve_run(conn, args.run)
        if run_id is None:
            print(f"Run not found: {args.run!r}. See `faithgate runs`.")
            return 2
    scored, errored = asyncio.run(
        score_pending(conn, scorer, judge, run_id=run_id, retry_errors=args.retry_errors)
    )
    print(f"scored {scored}, errored {errored}  (judge: {judge.id})")
    if errored and scored == 0:
        print("✗ every case errored — the judge is not working.")
        return 2
    return 0


def _short(text: str, width: int = 58) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def _cmd_candidates(conn: sqlite3.Connection, args) -> int:
    from ..promote import find_candidates

    run_id = None
    if args.run:
        run_id = _resolve_run(conn, args.run)
        if run_id is None:
            print(f"Run not found: {args.run!r}. See `faithgate runs`.")
            return 2
    candidates = find_candidates(conn, run_id=run_id, below=args.below, limit=args.limit)
    if not candidates:
        print(f"No unpromoted failures below {args.below}.")
        return 0
    print(f"{'TRACE':<15} {'SCORE':>5}  {'RUN':<16} QUESTION")
    for c in candidates:
        label = _short(c.run_label or "(ad-hoc)", 16)
        print(f"{c.trace_id[:14]:<15} {c.score:>5.2f}  {label:<16} {_short(c.question)}")
    print(f"\npromote one:  faithgate promote {candidates[0].trace_id[:14]}")
    print(f"promote all:  faithgate promote --below {args.below}" +
          (f" --run \"{args.run}\"" if args.run else ""))
    return 0


def _cmd_promote(conn: sqlite3.Connection, args) -> int:
    import sys

    from ..promote import find_candidates, promote_trace, resolve_trace

    def report(result):
        if result.status == "promoted":
            origin = f" (origin score {result.score:.2f})" if result.score is not None else ""
            print(f"✅ promoted → {args.to} [{result.item_id[:8]}]{origin}: {_short(result.question)}")
        else:
            print(f"↷ already in {args.to} [{result.item_id[:8]}]: {_short(result.question)}")

    if args.trace:  # single, explicit
        trace_id, err = resolve_trace(conn, args.trace)
        if err:
            print(err)
            return 2
        report(promote_trace(conn, trace_id, dataset_name=args.to,
                             allow_duplicate=args.allow_duplicate))
        return 0

    # bulk: show the table, then ONE confirmation — never silent
    run_id = None
    if args.run:
        run_id = _resolve_run(conn, args.run)
        if run_id is None:
            print(f"Run not found: {args.run!r}. See `faithgate runs`.")
            return 2
    candidates = find_candidates(conn, run_id=run_id, below=args.below, limit=args.limit)
    if not candidates:
        print(f"No unpromoted failures below {args.below}.")
        return 0
    print(f"{'TRACE':<15} {'SCORE':>5}  QUESTION")
    for c in candidates:
        print(f"{c.trace_id[:14]:<15} {c.score:>5.2f}  {_short(c.question)}")
    if not args.yes:
        if not sys.stdin.isatty():
            print(f"\n{len(candidates)} case(s) listed — re-run with --yes to promote non-interactively.")
            return 2
        answer = input(f"\nPromote these {len(candidates)} case(s) into '{args.to}'? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("cancelled — nothing promoted.")
            return 0
    promoted = skipped = 0
    for c in candidates:
        result = promote_trace(conn, c.trace_id, dataset_name=args.to,
                               allow_duplicate=args.allow_duplicate)
        report(result)
        promoted += result.status == "promoted"
        skipped += result.status != "promoted"
    print(f"\ndone: {promoted} promoted · {skipped} skipped (duplicates)")
    return 0


def _cmd_datasets(conn: sqlite3.Connection) -> int:
    from ..promote import list_datasets

    rows = list_datasets(conn)
    if not rows:
        print("No datasets yet — `faithgate promote` creates one from captured failures.")
        return 0
    print(f"{'NAME':<20} {'KIND':<10} ITEMS")
    for r in rows:
        print(f"{r['name']:<20} {r['kind']:<10} {r['items']}")
    return 0


def _cmd_export(conn: sqlite3.Connection, args) -> int:
    from ..promote import export_probes

    count = 0
    for probe in export_probes(conn, args.dataset):
        print(json.dumps(probe, ensure_ascii=False))
        count += 1
    if count == 0:
        print(f"dataset {args.dataset!r} is empty or does not exist "
              "(see `faithgate datasets`)", file=__import__("sys").stderr)
        return 2
    return 0


def _cmd_calibrate(args) -> int:
    import asyncio

    from ..calibrate.calibrate import DEFAULT_GOLDENS, load_goldens, render_report, run_calibration
    from ..score.judges import make_scorer

    try:
        scorer, judge = make_scorer(args.judge)
    except (RuntimeError, ImportError, ValueError) as exc:
        print(exc)
        return 2

    goldens = load_goldens(args.goldens or DEFAULT_GOLDENS)
    metrics = asyncio.run(run_calibration(scorer, judge, goldens))
    print(render_report(judge, metrics))
    if metrics["scored"] == 0 and metrics["abstained"] > 0:
        return 2  # nothing was actually judged — do not report a usable-looking result
    return 0


_INIT_SUITE = """\
{"id": "example-1", "question": "What is the capital of France?", "answer": "Paris is the capital of France.", "contexts": ["France is a country in Western Europe. Its capital and largest city is Paris."]}
{"id": "example-2", "question": "When was the Eiffel Tower completed?", "answer": "It was completed in 1889.", "contexts": ["Construction of the Eiffel Tower finished in 1889 for the World's Fair."]}
{"id": "example-3", "question": "REPLACE these examples with real questions from YOUR app", "answer": "Each line: a question, your app's answer, and the retrieved context it was given.", "contexts": ["Give every case a stable id so reworded questions keep their baseline."]}
"""

_INIT_WORKFLOW = """\
name: faithgate

on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # TODO: replace this placeholder with YOUR app answering the questions in
      # evals/baseline.jsonl and writing evals/candidate.jsonl (same ids, fresh answers).
      # Until then, candidate = baseline, so the gate stays green.
      - name: Generate candidate answers (placeholder)
        run: cp evals/baseline.jsonl evals/candidate.jsonl

      - uses: albertofettucini/faithgate@main
        with:
          baseline-suite: evals/baseline.jsonl
          candidate-suite: evals/candidate.jsonl
          judge: heuristic
          # for the trusted Claude judge instead:
          #   judge: claude
          #   anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
"""


def _cmd_init(args) -> int:
    suite_path = os.path.join(args.dir, "evals", "baseline.jsonl")
    workflow_path = os.path.join(args.dir, ".github", "workflows", "faithgate.yml")

    existing = [p for p in (suite_path, workflow_path) if os.path.exists(p)]
    if existing:
        print("refusing to overwrite existing files:")
        for p in existing:
            print(f"  {p}")
        return 2

    os.makedirs(os.path.dirname(suite_path), exist_ok=True)
    os.makedirs(os.path.dirname(workflow_path), exist_ok=True)
    with open(suite_path, "w", encoding="utf-8") as fh:
        fh.write(_INIT_SUITE)
    with open(workflow_path, "w", encoding="utf-8") as fh:
        fh.write(_INIT_WORKFLOW)

    print("created:")
    print(f"  {suite_path}")
    print(f"  {workflow_path}")
    print("")
    print("next steps:")
    print("  1. Replace the example cases in evals/baseline.jsonl with real turns from your app.")
    print("  2. Edit the TODO step in the workflow: your app answers the same questions into")
    print("     evals/candidate.jsonl on every PR.")
    print("  3. Commit both files — the gate posts its verdict on your next PR.")
    return 0


def _cmd_up(args) -> int:
    from ..panel.server import serve

    serve(args.db, host=args.host, port=args.port)
    return 0


def main(argv: list = None) -> int:
    # --db is accepted both before and after the subcommand: the top-level default applies,
    # and a subcommand-level --db (SUPPRESS default) overrides only when actually given.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default=argparse.SUPPRESS, help="path to the local store")

    parser = argparse.ArgumentParser(prog="faithgate", description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=DEFAULT_DB, help="path to the local store")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("runs", help="list captured runs", parents=[common])

    g = sub.add_parser("gate", help="compare two runs; exit non-zero on regression", parents=[common])
    g.add_argument("--base", required=True, help="baseline run id or label")
    g.add_argument("--head", required=True, help="new run id or label")
    g.add_argument("--max-regression", type=float, default=0.05, help="max allowed score drop")
    g.add_argument("--min-score", type=float, default=0.5, help="hard floor any case must clear")
    g.add_argument("--allow-judge-change", action="store_true",
                   help="compare anyway when the judge differs between runs")
    g.add_argument("--fail-on-missing", action="store_true",
                   help="fail when baseline cases are missing from the new run")
    g.add_argument("--max-abstained", type=int, default=None,
                   help="fail when more than N cases abstained (guards targeted abstention)")
    g.add_argument("--format", default="text", choices=["text", "markdown", "json"],
                   help="report format (markdown is PR-comment ready)")

    s = sub.add_parser("show", help="show a run's scored traces", parents=[common])
    s.add_argument("--run", required=True, help="run id or label")

    r = sub.add_parser("run", help="score a suite of answers into a run", parents=[common])
    r.add_argument("--suite", required=True, help="JSONL file: {question, answer, contexts:[...]} per line")
    r.add_argument("--label", required=True, help="name for this run/version")
    r.add_argument("--judge", default="claude", choices=["claude", "claude-local", "heuristic"],
                   help="claude (default, needs key) | claude-local (HHEM) | heuristic (offline, no key)")

    sc = sub.add_parser("score", help="score any pending (captured-but-unscored) traces", parents=[common])
    sc.add_argument("--judge", default="claude", choices=["claude", "claude-local", "heuristic"])
    sc.add_argument("--run", default=None, help="limit to one run id or label")
    sc.add_argument("--retry-errors", action="store_true",
                    help="also re-score traces that previously errored")

    c = sub.add_parser("calibrate", help="measure judge agreement with a human-labeled golden set",
                       parents=[common])
    c.add_argument("--judge", default="claude", choices=["claude", "claude-local", "heuristic"])
    c.add_argument("--goldens", default=None, help="labeled JSONL (defaults to the built-in set)")

    u = sub.add_parser("up", help="start the local web panel", parents=[common])
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=7654)

    i = sub.add_parser("init", help="scaffold a starter suite + CI workflow into a project")
    i.add_argument("--dir", default=".", help="target project directory")

    ca = sub.add_parser("candidates", help="list captured failures eligible for promotion",
                        parents=[common])
    ca.add_argument("--run", default=None, help="limit to one run id or label")
    ca.add_argument("--below", type=float, default=0.5, help="score threshold")
    ca.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("promote", help="turn a captured failure into a regression test case",
                       parents=[common])
    p.add_argument("trace", nargs="?", default=None, help="trace id (or unique prefix); omit for bulk")
    p.add_argument("--to", default="regressions", help="target dataset name")
    p.add_argument("--run", default=None, help="bulk: limit to one run id or label")
    p.add_argument("--below", type=float, default=0.5, help="bulk: score threshold")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--yes", action="store_true", help="bulk: skip the confirmation prompt")
    p.add_argument("--allow-duplicate", action="store_true",
                   help="promote even when an identical case already exists")

    sub.add_parser("datasets", help="list datasets", parents=[common])

    e = sub.add_parser("export", help="export a dataset as probe JSONL for your app to answer",
                       parents=[common])
    e.add_argument("dataset", help="dataset name (see `faithgate datasets`)")

    args = parser.parse_args(argv)
    if args.cmd == "up":
        return _cmd_up(args)
    if args.cmd == "init":
        return _cmd_init(args)
    if args.cmd == "calibrate":
        return _cmd_calibrate(args)

    try:
        conn = db.connect(args.db)
        db.init_db(conn)
    except OSError as exc:
        print(f"cannot open the store at {args.db!r}: {exc}")
        return 2
    if args.cmd == "runs":
        return _cmd_runs(conn)
    if args.cmd == "gate":
        return _cmd_gate(conn, args)
    if args.cmd == "show":
        return _cmd_show(conn, args)
    if args.cmd == "run":
        return _cmd_run(conn, args)
    if args.cmd == "score":
        return _cmd_score(conn, args)
    if args.cmd == "candidates":
        return _cmd_candidates(conn, args)
    if args.cmd == "promote":
        return _cmd_promote(conn, args)
    if args.cmd == "datasets":
        return _cmd_datasets(conn)
    if args.cmd == "export":
        return _cmd_export(conn, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
