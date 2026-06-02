"""Crash-resilient WS outbox.

Replaces the in-memory ring buffer with a SQLite-backed durable queue. The
contract:

  - On ``_emit``: ``record(exp_id, seq, kind, payload_json)`` is called BEFORE
    the frame goes on the wire. If the process crashes between record and
    send, the row is still on disk and will be replayed on reconnect.
  - On ``Ack`` from the server: ``ack(exp_id, last_seq)`` deletes every row
    for that experiment with seq <= last_seq.
  - On reconnect: ``pending()`` yields every unacked row in (experiment_id,
    seq) order; the client resends them before processing new requests.
  - ``gc()`` drops rows older than 24 h regardless of ack status — defensive
    cleanup so a stalled experiment can't grow the DB unbounded.

Concurrency: a single sqlite3.Connection per ``Outbox`` instance with
``check_same_thread=False`` and an explicit ``threading.Lock`` around each
transaction. WAL journal mode + synchronous=NORMAL gives durable inserts at
roughly 1 ms each on local disk, which is well under the rate the WS can
ship them.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, Tuple

logger = logging.getLogger(__name__)

GC_DEFAULT_AGE_S = 86400.0  # 24 h


class Outbox:
    """SQLite-backed unacked-event queue. Thread-safe."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we batch inside the lock
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_exp_seq
                    ON events(experiment_id, seq);
                CREATE INDEX IF NOT EXISTS idx_events_created
                    ON events(created_at);
                """
            )

    def record(self, experiment_id: str, seq: int, kind: str, payload_json: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(experiment_id, seq, kind, payload_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (experiment_id, int(seq), kind, payload_json, time.time()),
            )

    def ack(self, experiment_id: str, last_seq: int) -> int:
        """Delete every row for ``experiment_id`` whose seq <= last_seq.

        Returns the count of rows removed.
        """
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE experiment_id = ? AND seq <= ?",
                (experiment_id, int(last_seq)),
            )
            return cur.rowcount or 0

    def pending(self) -> list[Tuple[str, int, str, str]]:
        """Snapshot of every unacked row, ordered by (experiment_id, seq).

        Materialized as a list so the caller can iterate without holding the
        DB lock — important because the caller will await network sends.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT experiment_id, seq, kind, payload_json FROM events"
                " ORDER BY experiment_id, seq"
            )
            return cur.fetchall()

    def pending_count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM events")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def gc(self, older_than_seconds: float = GC_DEFAULT_AGE_S) -> int:
        cutoff = time.time() - float(older_than_seconds)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM events WHERE created_at < ?",
                (cutoff,),
            )
            removed = cur.rowcount or 0
        if removed:
            logger.info(f"outbox.gc: removed {removed} events older than {older_than_seconds}s")
        return removed

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # Iterator protocol — convenience for tests
    def __iter__(self) -> Iterable[Tuple[str, int, str, str]]:
        return iter(self.pending())
