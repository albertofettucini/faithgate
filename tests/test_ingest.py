import os
import tempfile
import unittest

from faithgate.ingest.openinference import FIXTURE_SPAN, capture_spans, self_test, span_to_turn
from faithgate.store import db


class SpanAdapterTest(unittest.TestCase):
    def test_extracts_full_turn(self):
        turn = span_to_turn(FIXTURE_SPAN)
        self.assertEqual(turn["question"], "What is the capital of France?")
        self.assertEqual(turn["answer"], "Paris is the capital of France.")
        self.assertEqual(len(turn["contexts"]), 2)
        self.assertEqual(turn["contexts"][0], "France's capital is Paris.")

    def test_non_llm_span_returns_none(self):
        self.assertIsNone(span_to_turn({"attributes": {"openinference.span.kind": "RETRIEVER"}}))

    def test_contexts_sorted_by_index(self):
        span = {"attributes": {
            "input.value": "q", "output.value": "a",
            "retrieval.documents.1.document.content": "second",
            "retrieval.documents.0.document.content": "first",
        }}
        self.assertEqual(span_to_turn(span)["contexts"], ["first", "second"])

    def test_self_test_passes(self):
        self.assertTrue(self_test())

    def test_version_extracted_from_metadata(self):
        self.assertEqual(span_to_turn(FIXTURE_SPAN)["prompt_version_id"], "v7")

    def test_version_flat_attr_and_absent(self):
        span = {"attributes": {"input.value": "q", "output.value": "a",
                               "faithgate.prompt_version": "abc123"}}
        self.assertEqual(span_to_turn(span)["prompt_version_id"], "abc123")
        bare = {"attributes": {"input.value": "q", "output.value": "a"}}
        self.assertIsNone(span_to_turn(bare)["prompt_version_id"])


class CaptureSpansTest(unittest.TestCase):
    def test_captures_into_store(self):
        conn = db.connect(os.path.join(tempfile.mkdtemp(), "t.db"))
        db.init_db(conn)
        run_id = db.new_id()
        db.insert(conn, "run", {"id": run_id, "label": "ingest", "status": "capturing"})
        n = capture_spans(conn, run_id, [FIXTURE_SPAN])
        self.assertEqual(n, 1)
        rows = db.fetchall(
            conn,
            "SELECT input_query, source, status, prompt_version_id FROM trace WHERE run_id=?",
            (run_id,),
        )
        self.assertEqual(rows[0]["source"], "otel")
        self.assertEqual(rows[0]["status"], "pending")
        self.assertEqual(rows[0]["prompt_version_id"], "v7")


class VersionKeyTest(unittest.TestCase):
    def test_deterministic_and_sensitive(self):
        from faithgate.keys import version_key

        a = version_key("Answer using context: {q}", "claude-sonnet-5", {"temp": 0})
        b = version_key("Answer using context: {q}", "claude-sonnet-5", {"temp": 0})
        c = version_key("Answer using context: {q}", "claude-haiku-4-5", {"temp": 0})
        self.assertEqual(a, b)          # same inputs → same id
        self.assertNotEqual(a, c)       # model change → different id
        self.assertEqual(len(a), 12)


if __name__ == "__main__":
    unittest.main()
