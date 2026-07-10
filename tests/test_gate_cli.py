import json
import os
import tempfile
import unittest

from faithgate.gate.cli import main
from faithgate.score.scorer import FakeScorer, Judge, Sample
from faithgate.score.worker import score_pending
from faithgate.store import db

SUITE = [
    {"question": "cap?", "answer": "Paris is the capital",
     "contexts": ["Paris is the capital of France"]},
    {"question": "moons?", "answer": "Earth has one moon",
     "contexts": ["Earth has one moon"]},
]


def write_suite(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")


class DbFlagPositionTest(unittest.TestCase):
    def test_db_accepted_before_and_after_subcommand(self):
        d = tempfile.mkdtemp()
        a, b = os.path.join(d, "a.db"), os.path.join(d, "b.db")
        self.assertEqual(main(["--db", a, "runs"]), 0)       # before
        self.assertEqual(main(["runs", "--db", b]), 0)       # after
        self.assertTrue(os.path.exists(a) and os.path.exists(b))


class RunValidationTest(unittest.TestCase):
    def test_invalid_suite_leaves_no_run_row(self):
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        bad = os.path.join(d, "bad.jsonl")
        with open(bad, "w") as f:
            f.write('{"question": "q"}\n')  # missing 'answer'
        rc = main(["--db", db_path, "run", "--suite", bad, "--label", "x", "--judge", "heuristic"])
        self.assertEqual(rc, 2)
        conn = db.connect(db_path)
        db.init_db(conn)
        self.assertEqual(len(db.fetchall(conn, "SELECT id FROM run")), 0)

    def test_non_dict_json_lines_exit_2(self):
        # valid JSON that isn't an object (5, null, true) must be a friendly exit 2, not a traceback
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        for payload in ("5", "null", "true", '"just a string"'):
            bad = os.path.join(d, "scalar.jsonl")
            with open(bad, "w") as f:
                f.write(payload + "\n")
            rc = main(["--db", db_path, "run", "--suite", bad, "--label", "x", "--judge", "heuristic"])
            self.assertEqual(rc, 2, payload)

    def test_all_errored_run_exits_2(self):
        from unittest.mock import patch

        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        suite = os.path.join(d, "s.jsonl")
        write_suite(suite, SUITE)
        judge = Judge(id="boom", provider="fake", model="boom")
        with patch("faithgate.score.judges.make_scorer", return_value=(_BoomScorer(judge), judge)):
            rc = main(["--db", db_path, "run", "--suite", suite, "--label", "x", "--judge", "heuristic"])
        self.assertEqual(rc, 2)  # a fully broken judge must not look like a green run
        conn = db.connect(db_path)
        db.init_db(conn)
        self.assertEqual(db.fetchall(conn, "SELECT status FROM run")[0]["status"], "failed")

    def test_public_capture_reexport(self):
        import faithgate

        self.assertTrue(callable(faithgate.capture))  # README documents faithgate.capture(...)

    def test_case_id_pins_identity_across_rewording(self):
        from faithgate.gate.diff import compare, load_run_scores

        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        v1 = os.path.join(d, "v1.jsonl")
        v2 = os.path.join(d, "v2.jsonl")
        # same case id, reworded question in v2 — must still MATCH (not become a 'new' case)
        write_suite(v1, [{"id": "capital", "question": "What is the capital of France?",
                          "answer": "Paris is the capital", "contexts": ["Paris is the capital of France"]}])
        write_suite(v2, [{"id": "capital", "question": "Which city is France's capital?",
                          "answer": "Paris is the capital", "contexts": ["Paris is the capital of France"]}])
        main(["--db", db_path, "run", "--suite", v1, "--label", "v1", "--judge", "heuristic"])
        main(["--db", db_path, "run", "--suite", v2, "--label", "v2", "--judge", "heuristic"])
        conn = db.connect(db_path)
        db.init_db(conn)
        runs = {r["label"]: r["id"] for r in db.fetchall(conn, "SELECT id, label FROM run")}
        result = compare(load_run_scores(conn, runs["v1"]), load_run_scores(conn, runs["v2"]))
        self.assertEqual(result.matched, 1)
        self.assertEqual(result.new_cases, [])

    def test_default_db_dir_is_created(self):
        # regression test for the first-run crash: parent dir must be auto-created
        d = tempfile.mkdtemp()
        nested = os.path.join(d, "does", "not", "exist", "t.db")
        conn = db.connect(nested)
        db.init_db(conn)
        self.assertTrue(os.path.exists(nested))


class JudgeChangeGuardTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.db_path = os.path.join(self.d, "t.db")
        suite = os.path.join(self.d, "suite.jsonl")
        write_suite(suite, SUITE)
        main(["--db", self.db_path, "run", "--suite", suite, "--label", "base", "--judge", "heuristic"])
        main(["--db", self.db_path, "run", "--suite", suite, "--label", "head", "--judge", "heuristic"])

    def _tamper_head_judge(self):
        conn = db.connect(self.db_path)
        db.init_db(conn)
        row = db.fetchall(conn, "SELECT id, manifest_json FROM run WHERE label='head'")[0]
        m = json.loads(row["manifest_json"])
        m["judge_model"] = "claude-sonnet-5"
        m["judge_id"] = "claude-sonnet-5"
        conn.execute("UPDATE run SET manifest_json=? WHERE id=?", (json.dumps(m), row["id"]))
        conn.commit()

    def test_same_judge_gates_normally(self):
        rc = main(["--db", self.db_path, "gate", "--base", "base", "--head", "head"])
        self.assertEqual(rc, 0)

    def test_judge_change_blocks_with_exit_3(self):
        self._tamper_head_judge()
        rc = main(["--db", self.db_path, "gate", "--base", "base", "--head", "head"])
        self.assertEqual(rc, 3)

    def test_allow_judge_change_overrides(self):
        self._tamper_head_judge()
        rc = main(["--db", self.db_path, "gate", "--base", "base", "--head", "head",
                   "--allow-judge-change"])
        self.assertEqual(rc, 0)

    def test_gate_writes_passed_back(self):
        main(["--db", self.db_path, "gate", "--base", "base", "--head", "head"])
        conn = db.connect(self.db_path)
        db.init_db(conn)
        rows = db.fetchall(conn, """
            SELECT s.passed FROM eval_score s JOIN run r ON r.id = s.run_id
            WHERE r.label='head'
        """)
        self.assertTrue(rows and all(r["passed"] == 1 for r in rows))


class InitTest(unittest.TestCase):
    def test_scaffolds_two_files_and_refuses_overwrite(self):
        d = tempfile.mkdtemp()
        self.assertEqual(main(["init", "--dir", d]), 0)
        suite = os.path.join(d, "evals", "baseline.jsonl")
        wf = os.path.join(d, ".github", "workflows", "faithgate.yml")
        self.assertTrue(os.path.exists(suite) and os.path.exists(wf))
        with open(wf) as f:
            content = f.read()
        self.assertIn("albertofettucini/faithgate@", content)
        self.assertIn("candidate-suite", content)
        with open(suite) as f:
            first = json.loads(f.readline())
        self.assertIn("id", first)
        # never overwrite an existing setup
        self.assertEqual(main(["init", "--dir", d]), 2)

    def test_gate_formats_run_clean(self):
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")
        suite = os.path.join(d, "s.jsonl")
        write_suite(suite, SUITE)
        main(["--db", db_path, "run", "--suite", suite, "--label", "a", "--judge", "heuristic"])
        main(["--db", db_path, "run", "--suite", suite, "--label", "b", "--judge", "heuristic"])
        self.assertEqual(main(["--db", db_path, "gate", "--base", "a", "--head", "b",
                               "--format", "markdown"]), 0)
        self.assertEqual(main(["--db", db_path, "gate", "--base", "a", "--head", "b",
                               "--format", "json"]), 0)


class _BoomScorer:
    def __init__(self, judge):
        self.judge = judge

    async def ascore(self, sample):
        raise RuntimeError("judge exploded")


class WorkerErrorSurfacingTest(unittest.IsolatedAsyncioTestCase):
    async def test_errors_counted_and_retryable(self):
        conn = db.connect(os.path.join(tempfile.mkdtemp(), "t.db"))
        db.init_db(conn)
        judge = Judge(id="f", provider="fake", model="fake")
        run_id = db.new_id()
        db.insert(conn, "run", {"id": run_id, "label": "r", "status": "scoring"})
        from faithgate.ingest.decorator import capture
        capture(conn, run_id, "q", "a", ["c"])

        scored, errored = await score_pending(conn, _BoomScorer(judge), judge, run_id=run_id)
        self.assertEqual((scored, errored), (0, 1))  # a broken judge must not look green

        # errored traces are not stuck: retry_errors re-scores them
        scored2, errored2 = await score_pending(
            conn, FakeScorer(judge, 0.9), judge, run_id=run_id, retry_errors=True
        )
        self.assertEqual((scored2, errored2), (1, 0))


if __name__ == "__main__":
    unittest.main()
