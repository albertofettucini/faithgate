import json
import os
import tempfile
import unittest

from faithgate.gate.cli import main
from faithgate.ingest.decorator import capture
from faithgate.promote import (export_probes, find_candidates, list_datasets,
                               promote_trace, resolve_trace)
from faithgate.store import db


def make_store():
    conn = db.connect(os.path.join(tempfile.mkdtemp(), "t.db"))
    db.init_db(conn)
    return conn


def add_scored_trace(conn, run_id, question, score, abstained=0, answer="ans", contexts=None):
    trace_id = capture(conn, run_id, question, answer, contexts or ["ctx"])
    db.insert(conn, "eval_score", {
        "id": db.new_id(), "trace_id": trace_id, "run_id": run_id,
        "metric": "faithfulness", "score": None if abstained else score,
        "abstained": abstained,
    })
    conn.execute("UPDATE trace SET status='scored' WHERE id=?", (trace_id,))
    conn.commit()
    return trace_id


class CandidatesTest(unittest.TestCase):
    def setUp(self):
        self.conn = make_store()
        self.run_id = db.new_id()
        db.insert(self.conn, "run", {"id": self.run_id, "label": "r1", "status": "complete"})

    def test_filters_threshold_abstained_and_promoted(self):
        bad = add_scored_trace(self.conn, self.run_id, "bad q", 0.2)
        add_scored_trace(self.conn, self.run_id, "good q", 0.9)
        add_scored_trace(self.conn, self.run_id, "abstained q", None, abstained=1)
        already = add_scored_trace(self.conn, self.run_id, "already q", 0.1,
                                   contexts=["other ctx"])
        promote_trace(self.conn, already)

        got = find_candidates(self.conn, below=0.5)
        self.assertEqual([c.trace_id for c in got], [bad])   # worst-first, others excluded
        self.assertEqual(got[0].run_label, "r1")


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.conn = make_store()
        self.run_id = db.new_id()
        db.insert(self.conn, "run", {"id": self.run_id, "label": "r1", "status": "complete"})
        self.trace_id = add_scored_trace(self.conn, self.run_id, "kaç ay var?", 0.29)

    def test_promote_creates_dataset_and_provenance(self):
        result = promote_trace(self.conn, self.trace_id)
        self.assertEqual(result.status, "promoted")
        self.assertAlmostEqual(result.score, 0.29)
        # provenance is one join away: item → origin trace → score/run
        row = db.fetchall(self.conn, """
            SELECT d.name, d.kind, di.origin_trace_id, s.score
            FROM dataset_item di
            JOIN dataset d ON d.id = di.dataset_id
            JOIN eval_score s ON s.trace_id = di.origin_trace_id
        """)[0]
        self.assertEqual((row["name"], row["kind"]), ("regressions", "promoted"))
        self.assertEqual(row["origin_trace_id"], self.trace_id)
        self.assertAlmostEqual(row["score"], 0.29)

    def test_duplicate_skipped_unless_allowed(self):
        first = promote_trace(self.conn, self.trace_id)
        again = promote_trace(self.conn, self.trace_id)
        self.assertEqual(again.status, "skipped_duplicate")
        self.assertEqual(again.item_id, first.item_id)
        forced = promote_trace(self.conn, self.trace_id, allow_duplicate=True)
        self.assertEqual(forced.status, "promoted")

    def test_prefix_resolution(self):
        full, err = resolve_trace(self.conn, self.trace_id[:8])
        self.assertIsNone(err)
        self.assertEqual(full, self.trace_id)
        _, err = resolve_trace(self.conn, "ZZZZ")
        self.assertIn("not found", err)


class FlywheelLoopTest(unittest.TestCase):
    """The whole point: a real failure becomes a case the gate recognises on the next run."""

    def test_promote_export_answer_run_gate(self):
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "t.db")

        conn = db.connect(db_path)
        db.init_db(conn)
        run_id = db.new_id()
        db.insert(conn, "run", {"id": run_id, "label": "prod", "status": "complete"})
        add_scored_trace(conn, run_id, "How many moons does Earth have?", 0.2,
                         answer="Earth has two moons", contexts=["Earth has one moon"])

        # bulk promote non-interactively
        rc = main(["--db", db_path, "promote", "--below", "0.5", "--yes"])
        self.assertEqual(rc, 0)

        # export probes → the app answers them (well, this time) → two suite runs
        conn = db.connect(db_path)
        db.init_db(conn)
        probes = list(export_probes(conn, "regressions"))
        self.assertEqual(len(probes), 1)
        self.assertTrue(probes[0]["id"])

        suite = os.path.join(d, "answered.jsonl")
        with open(suite, "w", encoding="utf-8") as f:
            f.write(json.dumps({**probes[0], "answer": "Earth has one moon"}) + "\n")
        main(["--db", db_path, "run", "--suite", suite, "--label", "v1", "--judge", "heuristic"])
        main(["--db", db_path, "run", "--suite", suite, "--label", "v2", "--judge", "heuristic"])

        # the promoted case's id pins identity → the gate matches it across runs
        rc = main(["--db", db_path, "gate", "--base", "v1", "--head", "v2"])
        self.assertEqual(rc, 0)


class BulkNonInteractiveTest(unittest.TestCase):
    def test_bulk_without_yes_refuses_when_not_a_tty(self):
        conn = make_store()
        run_id = db.new_id()
        db.insert(conn, "run", {"id": run_id, "label": "r", "status": "complete"})
        add_scored_trace(conn, run_id, "q", 0.1)
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]
        rc = main(["--db", db_path, "promote", "--below", "0.5"])   # no --yes, stdin not a tty
        self.assertEqual(rc, 2)  # never silent, never hanging


if __name__ == "__main__":
    unittest.main()
