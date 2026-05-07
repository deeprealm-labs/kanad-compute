"""Per-experiment seq.json persistence + outbox-backed Ack trimming.

The seq.json file stores the highest acked seq per experiment so that on
reconnect ``Hello.last_ack_seq`` lets the server skip already-delivered
events without scanning the durable outbox.
"""

from __future__ import annotations

import json
from pathlib import Path

from kanad_compute.protocol import Ack
from kanad_compute.ws_client import (
    ComputeWSClient,
    _ExperimentBuffer,
)


def _client(tmpdir: str) -> ComputeWSClient:
    return ComputeWSClient(
        kanad_url="http://localhost",
        config={"state_dir": tmpdir, "api_key": "k", "node_id": "test-node"},
    )


def test_save_and_load_seq_state_roundtrip(tmp_path: Path):
    a = _client(str(tmp_path))
    a._buffers["exp-A"] = _ExperimentBuffer(next_seq=8, last_ack_seq=7)
    a._buffers["exp-B"] = _ExperimentBuffer(next_seq=4, last_ack_seq=3)
    a._save_seq_state()

    seq_file = tmp_path / "seq.json"
    assert seq_file.exists()
    on_disk = json.loads(seq_file.read_text())
    assert on_disk == {"exp-A": 7, "exp-B": 3}

    b = _client(str(tmp_path))
    assert b._buffers["exp-A"].last_ack_seq == 7
    assert b._buffers["exp-A"].next_seq == 8
    assert b._buffers["exp-B"].last_ack_seq == 3


def test_save_skips_unacked_experiments(tmp_path: Path):
    c = _client(str(tmp_path))
    c._buffers["fresh"] = _ExperimentBuffer(next_seq=1, last_ack_seq=0)
    c._buffers["acked"] = _ExperimentBuffer(next_seq=5, last_ack_seq=4)
    c._save_seq_state()
    on_disk = json.loads((tmp_path / "seq.json").read_text())
    assert "fresh" not in on_disk
    assert on_disk == {"acked": 4}


def test_handle_ack_trims_outbox_and_advances_pointer(tmp_path: Path):
    c = _client(str(tmp_path))
    c._buffers.setdefault("e1", _ExperimentBuffer())
    for s in (1, 2, 3, 4, 5):
        c._outbox.record("e1", s, "Log", '{"frame": ' + str(s) + "}")
    c._handle_ack(Ack(experiment_id="e1", last_seq=3))
    remaining = [(seq, kind) for _, seq, kind, _ in c._outbox.pending()]
    assert remaining == [(4, "Log"), (5, "Log")]
    assert c._buffers["e1"].last_ack_seq == 3


def test_handle_ack_idempotent_when_behind(tmp_path: Path):
    c = _client(str(tmp_path))
    c._buffers.setdefault("e1", _ExperimentBuffer(next_seq=12, last_ack_seq=9))
    for s in (10, 11):
        c._outbox.record("e1", s, "Log", "x")
    c._handle_ack(Ack(experiment_id="e1", last_seq=5))  # behind
    assert c._buffers["e1"].last_ack_seq == 9  # unchanged
    remaining = [seq for _, seq, _, _ in c._outbox.pending()]
    assert remaining == [10, 11]  # nothing trimmed


def test_jitter_is_within_twenty_percent_band():
    samples = [ComputeWSClient._jitter(10) for _ in range(2000)]
    assert min(samples) >= 8.0 - 1e-9
    assert max(samples) <= 12.0 + 1e-9


def test_state_dir_creation_idempotent(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "state"
    c = _client(str(nested))
    assert nested.exists()
    c._save_seq_state()
