"""End-to-end WS client smoke test.

Boots an in-process FastAPI server with a minimal /api/compute/connect
endpoint that mimics the real one (Hello → Registered → ExperimentRequest →
expects ExperimentEvent stream). Drives a real ``ComputeWSClient`` against
it and asserts the dispatch path produces a FinalResult.

Worker.run_calculation is monkey-patched to a stub so the test stays fast
and doesn't pull pyscf / qiskit at import time.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import kanad_compute.ws_client as ws_client_mod
from kanad_compute.protocol import (
    PROTOCOL_VERSION,
    ExperimentEvent,
    ExperimentRequest,
    Hello,
    Registered,
    parse_server_message,
)
from kanad_compute.ws_client import ComputeWSClient


# ── Test fixture: minimal /api/compute/connect server ────────────────────────


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _build_app(received_events: list[dict], dispatched: asyncio.Event) -> FastAPI:
    """Server that registers any node, dispatches one experiment, collects events."""
    app = FastAPI()

    @app.websocket("/api/compute/connect")
    async def connect(ws: WebSocket):
        await ws.accept()

        # Wait for Hello
        first = await ws.receive_json()
        assert first["type"] == "Hello"

        # Send Registered
        await ws.send_json(
            Registered(
                node_id=first["node_id"],
                session_id="sess-test",
                protocol_version=PROTOCOL_VERSION,
            ).model_dump(mode="json")
        )

        # Dispatch one ExperimentRequest as soon as we're registered
        from kanad_compute.protocol import Atom, MoleculeSpec, SolverSpec
        req = ExperimentRequest(
            experiment_id="exp-smoke",
            user_id="u1",
            molecule=MoleculeSpec(
                atoms=[Atom(symbol="H", position=[0, 0, 0]),
                       Atom(symbol="H", position=[0, 0, 0.74])],
                basis="sto-3g",
            ),
            solver=SolverSpec(type="vqe", max_iterations=5),
            backend="kanad_compute",
        )
        await ws.send_json(req.model_dump(mode="json"))
        dispatched.set()

        try:
            while True:
                raw = await ws.receive_text()
                received_events.append(json.loads(raw))
        except WebSocketDisconnect:
            return

    return app


class _BoundServer:
    """Run uvicorn in a worker thread, exposing started/stopped sync."""

    def __init__(self, app: FastAPI, port: int):
        cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(cfg)
        self.port = port
        self._thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        # Wait for uvicorn to be ready
        import time as _time
        for _ in range(100):
            if self.server.started:
                return
            _time.sleep(0.05)
        raise RuntimeError("uvicorn did not start in time")

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=5)


# ── The actual test ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_to_finalresult_roundtrip(tmp_path: Path, monkeypatch):
    # Stub worker.run_calculation so we don't drag pyscf/qiskit into the test.
    def fake_run_calculation(job, gpu_enabled=False, *, cancel_check=None):
        return {
            "status": "completed",
            "energy": -1.85,
            "hf_energy": -1.83,
            "fci_energy": -1.86,
            "n_evaluations": 12,
            "converged": True,
            "wall_time_ms": 42,
        }

    monkeypatch.setattr(ws_client_mod, "run_calculation", fake_run_calculation)
    # sysinfo can be heavy on import too — stub
    monkeypatch.setattr(ws_client_mod, "get_system_info", lambda gpu=False: {"os": "test"})

    received: list[dict] = []
    dispatched = asyncio.Event()
    app = _build_app(received, dispatched)
    port = _free_port()
    server = _BoundServer(app, port)
    server.start()
    try:
        client = ComputeWSClient(
            kanad_url=f"http://127.0.0.1:{port}",
            config={
                "state_dir": str(tmp_path),
                "api_key": "tk",
                "node_id": "node-smoke",
            },
        )

        # Run one connection iteration in a task; cancel after the FinalResult arrives.
        task = asyncio.create_task(client._connect_once())

        # Wait until the server has either dispatched OR the client errored.
        for _ in range(100):
            await asyncio.sleep(0.1)
            if any(
                ev.get("kind") == "FinalResult" or ev.get("kind") == "Error"
                for ev in received
            ):
                break

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        kinds = [ev.get("kind") for ev in received if ev.get("type") == "ExperimentEvent"]
        assert "Log" in kinds, f"expected Log, got events: {received}"
        assert "FinalResult" in kinds, f"expected FinalResult, got events: {received}"

        # Final payload integrity
        finals = [
            ev for ev in received
            if ev.get("type") == "ExperimentEvent" and ev.get("kind") == "FinalResult"
        ]
        assert len(finals) == 1
        assert finals[0]["payload"]["energy"] == -1.85
        assert finals[0]["payload"]["actual_backend"] == "kanad_compute"
    finally:
        server.stop()
