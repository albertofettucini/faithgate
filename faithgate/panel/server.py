"""The local web panel — a read-only view of captured runs and their scores.

Built on Python's stdlib ``http.server`` (no FastAPI/Flask/Jinja). The whole tool stays
install-light: the only non-stdlib dependency is the metric engine (RAGAS), pulled in lazily
only when you actually score. The panel renders plain HTML; a fresh SQLite connection is opened
per request so it's thread-safe under ThreadingHTTPServer.
"""
from __future__ import annotations

import html
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from ..store import db

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { padding: 20px 28px; border-bottom: 1px solid #21262d; }
header .name { font-weight: 700; font-size: 18px; }
header .name span { color: #3fb950; }
header .tag { color: #8b949e; font-size: 13px; margin-top: 2px; }
main { padding: 24px 28px; max-width: 920px; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; color: #8b949e; font-weight: 500; font-size: 12px;
     text-transform: uppercase; letter-spacing: .04em; padding: 8px 10px; }
td { padding: 10px; border-top: 1px solid #21262d; vertical-align: top; }
.badge { display: inline-block; min-width: 46px; text-align: center; padding: 2px 8px;
         border-radius: 6px; font-weight: 600; font-variant-numeric: tabular-nums; }
.good { background: #12361f; color: #3fb950; }
.mid  { background: #3a2d0c; color: #d29922; }
.bad  { background: #3c1618; color: #f85149; }
.abst { background: #21262d; color: #8b949e; }
.q { font-weight: 500; }
.ans { color: #8b949e; font-size: 13px; margin-top: 3px; }
.reason { color: #6e7681; font-size: 12px; margin-top: 3px; }
.muted { color: #8b949e; }
.back { font-size: 13px; }
"""


def _badge(score, abstained) -> str:
    if abstained or score is None:
        return '<span class="badge abst">—</span>'
    cls = "good" if score >= 0.8 else "mid" if score >= 0.5 else "bad"
    return f'<span class="badge {cls}">{score:.2f}</span>'


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)} · FaithGate</title><style>{_CSS}</style></head><body>"
        "<header><div class='name'>Faith<span>Gate</span></div>"
        "<div class='tag'>local faithfulness regression gate</div></header>"
        f"<main>{body}</main></body></html>"
    )


def render_runs(conn: sqlite3.Connection) -> str:
    rows = db.fetchall(conn, """
        SELECT r.id, r.label, r.status, r.created_at,
               (SELECT COUNT(*) FROM trace t WHERE t.run_id = r.id) AS n,
               AVG(CASE WHEN s.abstained = 0 THEN s.score END) AS mean
        FROM run r LEFT JOIN eval_score s ON s.run_id = r.id
        GROUP BY r.id ORDER BY r.id DESC
    """)
    if not rows:
        return _page("Runs", "<p class='muted'>No runs captured yet.</p>")
    body = ["<table><tr><th>Mean</th><th>Run</th><th>Cases</th><th>Status</th><th>When</th></tr>"]
    for r in rows:
        mean = _badge(r["mean"], r["mean"] is None)
        label = html.escape(r["label"] or "(unnamed)")
        body.append(
            f"<tr><td>{mean}</td>"
            f"<td><a href='/run/{r['id']}'>{label}</a></td>"
            f"<td>{r['n']}</td><td class='muted'>{r['status']}</td>"
            f"<td class='muted'>{r['created_at']}</td></tr>"
        )
    body.append("</table>")
    return _page("Runs", "".join(body))


def render_run(conn: sqlite3.Connection, run_ref: str) -> str:
    run = db.fetchall(conn, "SELECT id, label FROM run WHERE id = ? OR label = ? LIMIT 1", (run_ref, run_ref))
    if not run:
        return _page("Not found", "<p class='muted'>Run not found.</p><p class='back'><a href='/'>← all runs</a></p>")
    run_id, label = run[0]["id"], run[0]["label"]
    rows = db.fetchall(conn, """
        SELECT t.input_query AS q, t.output_response AS ans,
               s.score AS score, s.abstained AS abstained, s.reason AS reason
        FROM trace t LEFT JOIN eval_score s ON s.trace_id = t.id
        WHERE t.run_id = ? ORDER BY t.created_at
    """, (run_id,))
    body = [f"<p class='back'><a href='/'>← all runs</a></p><h2>{html.escape(label or '(unnamed)')}</h2><table>"]
    for r in rows:
        reason = f"<div class='reason'>{html.escape(r['reason'])}</div>" if r["reason"] else ""
        body.append(
            f"<tr><td>{_badge(r['score'], r['abstained'])}</td>"
            f"<td><div class='q'>{html.escape(r['q'])}</div>"
            f"<div class='ans'>{html.escape(r['ans'] or '')}</div>{reason}</td></tr>"
        )
    body.append("</table>")
    return _page(label or "Run", "".join(body))


class _Handler(BaseHTTPRequestHandler):
    db_path = None

    def _send(self, code: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/favicon.ico":
            self._send(204, "")
            return
        conn = db.connect(self.db_path)
        db.init_db(conn)
        try:
            if path == "/":
                self._send(200, render_runs(conn))
            elif path.startswith("/run/"):
                self._send(200, render_run(conn, unquote(path[len("/run/"):])))
            else:
                self._send(404, _page("404", "<p class='muted'>Not found.</p>"))
        finally:
            conn.close()

    def do_POST(self) -> None:
        import json

        if urlparse(self.path).path != "/v1/spans":
            self._send(404, '{"error":"not found"}', "application/json")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, '{"error":"invalid json"}', "application/json")
            return
        spans = payload.get("spans", []) if isinstance(payload, dict) else payload
        label = payload.get("run_label", "ingested") if isinstance(payload, dict) else "ingested"

        conn = db.connect(self.db_path)
        db.init_db(conn)
        try:
            from ..ingest.openinference import capture_spans

            rows = db.fetchall(conn, "SELECT id FROM run WHERE label=? ORDER BY created_at DESC LIMIT 1", (label,))
            run_id = rows[0]["id"] if rows else db.new_id()
            if not rows:
                db.insert(conn, "run", {"id": run_id, "label": label, "status": "capturing"})
            captured = capture_spans(conn, run_id, spans)
            self._send(200, json.dumps({"captured": captured, "run": label}), "application/json")
        finally:
            conn.close()

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


def serve(db_path: str, host: str = "127.0.0.1", port: int = 7654) -> None:
    _Handler.db_path = db_path
    from ..ingest.openinference import self_test

    try:
        self_test()
    except AssertionError as exc:
        # fail LOUDLY: a broken adapter would silently drop every ingested trace
        raise SystemExit(f"faithgate: startup self-test failed — {exc}")
    server = ThreadingHTTPServer((host, port), _Handler)
    print(f"faithgate panel → http://{host}:{port}   (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        server.shutdown()
