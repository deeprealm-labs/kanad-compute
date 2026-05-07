"""SQLite outbox unit tests.

Crash-resilient delivery semantics:
  - record / pending / ack round-trip
  - ack deletes only rows for that experiment with seq <= last_seq
  - gc drops rows older than the cutoff regardless of ack
  - thread-safety: many threads recording concurrently land all rows
  - reopening the file recovers pending rows (durability)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from kanad_compute.outbox import Outbox


def test_record_and_pending_roundtrip(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    o.record("exp-A", 1, "Log", '{"a":1}')
    o.record("exp-A", 2, "Progress", '{"i":2}')
    o.record("exp-B", 1, "Log", '{"b":1}')

    rows = o.pending()
    assert [(e, s, k) for e, s, k, _ in rows] == [
        ("exp-A", 1, "Log"),
        ("exp-A", 2, "Progress"),
        ("exp-B", 1, "Log"),
    ]


def test_ack_deletes_up_to_seq_for_experiment(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    for s in (1, 2, 3, 4, 5):
        o.record("exp-A", s, "Log", "x")
    for s in (1, 2):
        o.record("exp-B", s, "Log", "y")

    removed = o.ack("exp-A", 3)
    assert removed == 3

    remaining = [(e, s) for e, s, _, _ in o.pending()]
    assert remaining == [("exp-A", 4), ("exp-A", 5), ("exp-B", 1), ("exp-B", 2)]


def test_ack_no_op_when_no_match(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    o.record("exp-A", 5, "Log", "x")
    assert o.ack("exp-A", 3) == 0
    assert o.pending_count() == 1
    assert o.ack("exp-MISSING", 99) == 0


def test_gc_drops_old_rows(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    o.record("exp-A", 1, "Log", "x")
    # Backdate the row
    with o._lock:
        o._conn.execute("UPDATE events SET created_at = ? WHERE seq = 1", (time.time() - 10000,))
    o.record("exp-A", 2, "Log", "y")  # fresh

    removed = o.gc(older_than_seconds=3600.0)
    assert removed == 1
    remaining = [s for _, s, _, _ in o.pending()]
    assert remaining == [2]


def test_durability_across_reopen(tmp_path: Path):
    db = tmp_path / "ob.db"
    o1 = Outbox(db)
    o1.record("exp-A", 1, "Log", '{"hello":"world"}')
    o1.record("exp-A", 2, "FinalResult", '{"energy":-1.85}')
    o1.close()

    o2 = Outbox(db)
    rows = o2.pending()
    assert len(rows) == 2
    assert rows[0][3] == '{"hello":"world"}'
    assert rows[1][2] == "FinalResult"


def test_concurrent_record_from_many_threads(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    N = 8
    PER = 25

    def worker(tid: int):
        for s in range(PER):
            o.record(f"exp-{tid}", s, "Log", f"t{tid}-s{s}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert o.pending_count() == N * PER


def test_pending_ordered_by_experiment_then_seq(tmp_path: Path):
    o = Outbox(tmp_path / "ob.db")
    # Record out of order to verify ORDER BY
    o.record("exp-B", 2, "Log", "x")
    o.record("exp-A", 3, "Log", "x")
    o.record("exp-A", 1, "Log", "x")
    o.record("exp-B", 1, "Log", "x")
    o.record("exp-A", 2, "Log", "x")

    rows = [(e, s) for e, s, _, _ in o.pending()]
    assert rows == [("exp-A", 1), ("exp-A", 2), ("exp-A", 3), ("exp-B", 1), ("exp-B", 2)]
