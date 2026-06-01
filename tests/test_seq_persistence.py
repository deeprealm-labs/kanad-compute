"""Per-experiment seq.json persistence + ring-buffer behaviour."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kanad_compute.protocol import Ack
from kanad_compute.ws_client import (
    RING_MAX,
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


def test_handle_ack_trims_unacked_and_advances_pointer(tmp_path: Path):
    c = _client(str(tmp_path))
    buf = c._buffers.setdefault("e1", _ExperimentBuffer())
    for s in (1, 2, 3, 4, 5):
        buf.unacked.append((s, "Log", "frame"))
    c._handle_ack(Ack(experiment_id="e1", last_seq=3))
    assert [s for s, _, _ in buf.unacked] == [4, 5]
    assert buf.last_ack_seq == 3


def test_handle_ack_idempotent_when_behind(tmp_path: Path):
    c = _client(str(tmp_path))
    buf = c._buffers.setdefault("e1", _ExperimentBuffer(next_seq=10, last_ack_seq=9))
    for s in (10, 11):
        buf.unacked.append((s, "Log", "x"))
    c._handle_ack(Ack(experiment_id="e1", last_seq=5))  # behind
    assert buf.last_ack_seq == 9  # unchanged
    assert [s for s, _, _ in buf.unacked] == [10, 11]  # nothing trimmed


def test_evict_old_drops_oldest_non_terminal(tmp_path: Path):
    buf = _ExperimentBuffer()
    buf.unacked.append((1, "Log", "a"))
    buf.unacked.append((2, "FinalResult", "b"))
    buf.unacked.append((3, "Log", "c"))
    ComputeWSClient._evict_old(buf)
    kinds = [k for _, k, _ in buf.unacked]
    assert kinds == ["FinalResult", "Log"]


def test_evict_old_preserves_terminal_only_buffer(tmp_path: Path):
    buf = _ExperimentBuffer()
    buf.unacked.append((1, "FinalResult", "a"))
    buf.unacked.append((2, "Error", "b"))
    ComputeWSClient._evict_old(buf)
    assert len(buf.unacked) == 2  # nothing evicted


def test_jitter_is_within_twenty_percent_band():
    samples = [ComputeWSClient._jitter(10) for _ in range(2000)]
    assert min(samples) >= 8.0 - 1e-9
    assert max(samples) <= 12.0 + 1e-9


def test_state_dir_creation_idempotent(tmp_path: Path):
    nested = tmp_path / "deep" / "nested" / "state"
    c = _client(str(nested))
    assert nested.exists()
    # Saving without any populated state should not error
    c._save_seq_state()
