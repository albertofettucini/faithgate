import os
import sqlite3
import tempfile
import unittest

from faithgate.store import db


class StoreRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.conn = db.connect(os.path.join(self.dir, "t.db"))
        db.init_db(self.conn)

    def tearDown(self):
        self.conn.close()

    def test_schema_creates_all_tables(self):
        rows = db.fetchall(self.conn, "SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in rows}
        self.assertEqual(
            {"run", "trace", "span", "judge_run", "eval_score", "dataset", "dataset_item"},
            names,
        )

    def test_run_trace_score_roundtrip(self):
        run_id = db.new_id()
        db.insert(self.conn, "run", {"id": run_id, "label": "prompt v1", "status": "complete"})

        trace_id = db.new_id()
        db.insert(self.conn, "trace", {
            "id": trace_id, "run_id": run_id,
            "input_query": "What is the capital of France?",
            "output_response": "Paris.",
            "context_json": '["France is a country in Europe; its capital is Paris."]',
            "status": "scored",
        })

        db.insert(self.conn, "eval_score", {
            "id": db.new_id(), "trace_id": trace_id, "run_id": run_id,
            "metric": "faithfulness", "score": 0.91, "passed": 1,
            "reason": "fully grounded", "abstained": 0,
        })

        rows = db.fetchall(
            self.conn,
            "SELECT score, abstained, metric FROM eval_score WHERE trace_id=?",
            (trace_id,),
        )
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["score"], 0.91)
        self.assertEqual(rows[0]["abstained"], 0)
        self.assertEqual(rows[0]["metric"], "faithfulness")

    def test_foreign_keys_enforced(self):
        # eval_score.trace_id references a non-existent trace -> FK violation
        with self.assertRaises(sqlite3.IntegrityError):
            db.insert(self.conn, "eval_score", {
                "id": db.new_id(), "trace_id": "does-not-exist",
                "metric": "faithfulness", "score": 0.5, "abstained": 0,
            })

    def test_ulid_shape(self):
        ulid = db.new_id()
        self.assertEqual(len(ulid), 26)
        self.assertNotEqual(db.new_id(), db.new_id())


if __name__ == "__main__":
    unittest.main()
