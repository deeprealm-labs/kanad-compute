"""Throttle + thread-safety unit tests for _make_progress_cb.

The progress callback runs on a worker thread and bridges to the asyncio
loop via run_coroutine_threadsafe. These tests verify:

  - First emit always passes (no last_ts to compare against)
  - Calls within PROGRESS_MIN_INTERVAL_MS without significant energy change
    are dropped at the source
  - Significant energy improvements bypass the time throttle
  - The bridge actually delivers payloads to ``_emit`` from a non-loop thread
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from kanad_compute.ws_client import (
    PROGRESS_ENERGY_DELTA,
    PROGRESS_MIN_INTERVAL_MS,
    ComputeWSClient,
)


@pytest.fixture
def loop_and_thread():
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)


@pytest.fixture
def client(tmp_path: Path) -> ComputeWSClient:
    return ComputeWSClient(
        kanad_url="http://127.0.0.1:0",
        config={"state_dir": str(tmp_path), "api_key": "x", "node_id": "n"},
    )


def _stub_emit(client: ComputeWSClient):
    """Replace _emit with a recording async stub. Returns the list it appends to."""
    received: list[tuple[str, str, dict]] = []

    async def fake_emit(experiment_id: str, kind: str, payload: dict):
        received.append((experiment_id, kind, payload))

    client._emit = fake_emit  # type: ignore[assignment]
    return received


def _wait_for(predicate, timeout: float = 1.0, step: float = 0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


def test_first_emit_always_passes(client, loop_and_thread):
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-1", loop_and_thread)

    cb(iteration=0, energy=-1.0)

    assert _wait_for(lambda: len(received) == 1)
    assert received[0][1] == "Progress"
    assert received[0][2]["iteration"] == 0
    assert received[0][2]["energy"] == -1.0


def test_throttle_drops_close_emits(client, loop_and_thread):
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-2", loop_and_thread)

    # First emit always lands. Subsequent fast calls within the interval
    # without significant energy change must be dropped.
    cb(iteration=0, energy=-1.0)
    for i in range(1, 100):
        cb(iteration=i, energy=-1.0 + 1e-7 * i)  # delta well below threshold

    # Give the loop a moment to process whatever did make it through
    time.sleep(0.02)
    assert len(received) <= 2, f"expected ≤2 frames, got {len(received)}"


def test_throttle_passes_after_interval(client, loop_and_thread):
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-3", loop_and_thread)

    cb(iteration=0, energy=-1.0)
    assert _wait_for(lambda: len(received) == 1)

    time.sleep((PROGRESS_MIN_INTERVAL_MS + 50) / 1000.0)
    cb(iteration=1, energy=-1.0 + 1e-7)  # no significant energy change

    assert _wait_for(lambda: len(received) == 2)
    assert received[1][2]["iteration"] == 1


def test_significant_energy_delta_bypasses_throttle(client, loop_and_thread):
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-4", loop_and_thread)

    cb(iteration=0, energy=-1.0)
    assert _wait_for(lambda: len(received) == 1)

    # Same instant; energy improvement well above the delta threshold.
    cb(iteration=1, energy=-1.0 - PROGRESS_ENERGY_DELTA * 100)

    assert _wait_for(lambda: len(received) == 2)
    assert received[1][2]["iteration"] == 1


def test_progress_runs_on_loop_from_worker_thread(client, loop_and_thread):
    """The callback fires from a worker thread and must reach _emit on the
    asyncio loop via run_coroutine_threadsafe."""
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-5", loop_and_thread)

    fired = threading.Event()

    def worker():
        cb(iteration=0, energy=-2.5, message="from worker")
        fired.set()

    threading.Thread(target=worker, daemon=True).start()
    assert fired.wait(timeout=1.0)
    assert _wait_for(lambda: len(received) == 1)
    assert received[0][2]["message"] == "from worker"


def test_state_tracks_last_iteration_for_flush(client, loop_and_thread):
    """Even when the throttle drops emits, the flush state tracks the latest
    iteration count and energy so the terminal flush in _handle_experiment
    can deliver the converged frame."""
    received = _stub_emit(client)
    cb = client._make_progress_cb("exp-6", loop_and_thread)

    cb(iteration=0, energy=-1.0)
    for i in range(1, 50):
        cb(iteration=i, energy=-1.0 - 1e-7 * i)  # all dropped

    state = cb._state  # type: ignore[attr-defined]
    assert state["last_iteration"] == 49
    assert state["last_energy"] is not None
