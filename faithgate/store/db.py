"""SQLite (WAL) system-of-record. Raw sqlite3 — no ORM, no migrations.

The whole storage layer is intentionally small enough to read in one sitting:
connect → init_db → insert/fetchall, plus a dependency-free ULID for primary keys.
"""
from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCHEMA = Path(__file__).with_name("schema.sql")

# Crockford base32 — ULIDs are lexicographically sortable by creation time and need no coordination.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id() -> str:
    """A ULID: 48-bit millisecond timestamp + 80 bits of randomness, Crockford base32 (26 chars)."""
    value = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def connect(path: str | os.PathLike) -> sqlite3.Connection:
    """Open the store in WAL mode. Single-writer is a non-issue for a one-user tool."""
    path = str(path)
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        path = str(Path(path).expanduser())
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply schema.sql idempotently (every statement is CREATE ... IF NOT EXISTS)."""
    conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
    conn.commit()


def insert(conn: sqlite3.Connection, table: str, row: Mapping[str, Any]) -> None:
    """Insert one row. Column names come from code (never user input), so the f-string is safe here."""
    cols = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({placeholders})", tuple(row.values()))
    conn.commit()


def fetchall(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, tuple(params)).fetchall()
