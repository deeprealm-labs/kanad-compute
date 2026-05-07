"""Outbound WebSocket client — replaces remote_worker.py polling loop.

Connects to wss://kanad.xyz/api/compute/connect and:
  1. Sends Hello with node identity + system info.
  2. Waits for Registered.
  3. Receives ExperimentRequest, runs it via worker.run_calculation in a
     thread, and streams ExperimentEvents (Log → FinalResult/Error) back.
  4. Pong-replies to server Pings; sends own Pings to keep NATs alive.

Uses the `websockets` library (sync API in a worker thread). Reconnect with
exponential backoff on transport errors.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .protocol import (
    PROTOCOL_VERSION,
    CancelExperiment,
    ExperimentEvent,
    ExperimentRequest,
    Hello,
    Ping,
    Pong,
    Registered,
    is_compatible,
    parse_server_message,
)
from .sysinfo import get_system_info
from .worker import run_calculation

logger = logging.getLogger(__name__)

PING_INTERVAL_S = 15
RECONNECT_MIN_S = 1
RECONNECT_MAX_S = 30


class ComputeWSClient:
    def __init__(self, kanad_url: str, config: dict):
        self.kanad_url = kanad_url.rstrip("/")
        self.config = config
        self.api_key = config.get("api_key", "")
        self.node_id = config.get("node_id", "unknown")
        self.gpu_enabled = bool(config.get("gpu_enabled", False))

        self._executor = ThreadPoolExecutor(
            max_workers=config.get("max_workers", 2),
            thread_name_prefix="kanad-job",
        )
        self._cancelled: set[str] = set()
        self._seq_by_exp: dict[str, int] = {}
        self._send_lock = asyncio.Lock()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    # ── public entry ────────────────────────────────────────────────────────

    def run_forever_sync(self) -> None:
        """Blocking entry point — runs the asyncio loop in this thread."""
        asyncio.run(self._run_forever())

    async def _run_forever(self) -> None:
        backoff = RECONNECT_MIN_S
        while True:
            try:
                await self._connect_once()
                backoff = RECONNECT_MIN_S
            except (ConnectionClosed, OSError) as e:
                logger.info(f"WS disconnected: {e}; reconnecting in {backoff}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_S)
            except Exception as e:
                logger.exception(f"WS unexpected error: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_S)

    # ── one connection lifetime ─────────────────────────────────────────────

    async def _connect_once(self) -> None:
        ws_url = self._ws_url()
        logger.info(f"Connecting to {ws_url}")

        async with websockets.connect(
            ws_url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            ping_interval=None,  # we manage pings explicitly
            max_size=4 * 1024 * 1024,
        ) as ws:
            self._ws = ws

            # Hello
            sys_info = get_system_info(self.gpu_enabled)
            hello = Hello(
                node_id=self.node_id,
                system_info=sys_info,
                vault={
                    "ibm": bool(self.config.get("ibm_api_token")),
                    "ionq": bool(self.config.get("ionq_api_key")),
                },
            )
            await ws.send(hello.model_dump_json())

            registered_raw = await asyncio.wait_for(ws.recv(), timeout=10)
            registered = parse_server_message(json.loads(registered_raw))
            if not isinstance(registered, Registered):
                raise RuntimeError(f"expected Registered, got {type(registered).__name__}")
            if not is_compatible(registered.protocol_version):
                raise RuntimeError(
                    f"server protocol_version={registered.protocol_version} "
                    f"incompatible with client {PROTOCOL_VERSION}"
                )
            logger.info(
                f"Registered: session={registered.session_id[:8]} "
                f"server_proto={registered.protocol_version}"
            )

            pinger = asyncio.create_task(self._ping_loop(ws))
            try:
                async for raw in ws:
                    try:
                        msg = parse_server_message(json.loads(raw))
                    except (ValueError, json.JSONDecodeError) as e:
                        logger.warning(f"Bad server message: {e}")
                        continue

                    if isinstance(msg, ExperimentRequest):
                        asyncio.create_task(self._handle_experiment(msg))
                    elif isinstance(msg, Ping):
                        await self._send(Pong(ts_ms=msg.ts_ms))
                    elif isinstance(msg, CancelExperiment):
                        self._cancelled.add(msg.experiment_id)
                        logger.info(f"Cancel requested for {msg.experiment_id[:8]}")
                    # Ack / Pong — handled in PR3 (outbox); ignored here
            finally:
                pinger.cancel()
                self._ws = None

    # ── experiment dispatch ────────────────────────────────────────────────

    async def _handle_experiment(self, req: ExperimentRequest) -> None:
        logger.info(
            f"Experiment {req.experiment_id[:8]} received: "
            f"{req.solver.type} on {req.backend}"
        )

        await self._emit(req.experiment_id, "Log", {
            "level": "info",
            "message": f"Starting {req.solver.type} on {req.backend}",
        })

        # Build the dict shape worker.run_calculation expects.
        # MoleculeSpec.atoms is list[Atom]; flatten to {symbol, position} dicts.
        atoms = [{"symbol": a.symbol, "position": a.position} for a in req.molecule.atoms]
        creds = req.backend_credentials or {}
        job_record = {
            "job_id": req.experiment_id,
            "atoms": atoms,
            "basis": req.molecule.basis,
            "charge": req.molecule.charge,
            "multiplicity": req.molecule.multiplicity,
            "solver": req.solver.type,
            "backend": req.backend if req.backend != "kanad_compute" else "statevector",
            "max_iterations": req.solver.max_iterations,
            "max_excitations": req.solver.max_excitations,
            "ansatz_type": req.solver.ansatz_type,
            "optimizer": req.solver.optimizer,
            "mapper_type": req.solver.mapper_type,
            "ibm_api_token": creds.get("ibm_api_token"),
            "ibm_crn": creds.get("ibm_crn"),
            "ionq_api_key": creds.get("ionq_api_key"),
        }

        loop = asyncio.get_running_loop()
        t0 = time.time()
        try:
            result = await loop.run_in_executor(
                self._executor, run_calculation, job_record, self.gpu_enabled
            )
        except Exception as e:
            await self._emit(req.experiment_id, "Error", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            return

        if req.experiment_id in self._cancelled:
            self._cancelled.discard(req.experiment_id)
            await self._emit(req.experiment_id, "Error", {"message": "Cancelled by user", "code": "cancelled"})
            return

        if result.get("status") == "failed":
            await self._emit(req.experiment_id, "Error", {
                "message": result.get("error_message", "Calculation failed"),
                "traceback": result.get("traceback"),
            })
            return

        wall_ms = int((time.time() - t0) * 1000)
        await self._emit(req.experiment_id, "FinalResult", {
            "energy": result.get("energy"),
            "hf_energy": result.get("hf_energy"),
            "fci_energy": result.get("fci_energy"),
            "error_mha": result.get("error_mha"),
            "n_evaluations": result.get("n_evaluations"),
            "converged": result.get("converged"),
            "convergence_history": result.get("convergence_history"),
            "wall_time_ms": result.get("wall_time_ms") or wall_ms,
            "actual_backend": "kanad_compute",
        })

    # ── outbound helpers ───────────────────────────────────────────────────

    async def _emit(self, experiment_id: str, kind: str, payload: dict[str, Any]) -> None:
        seq = self._seq_by_exp.get(experiment_id, 0) + 1
        self._seq_by_exp[experiment_id] = seq
        ev = ExperimentEvent(
            experiment_id=experiment_id,
            seq=seq,
            ts_ms=int(time.time() * 1000),
            kind=kind,  # type: ignore[arg-type]
            payload=payload,
        )
        await self._send(ev)

    async def _send(self, msg: Any) -> None:
        ws = self._ws
        if not ws:
            return
        async with self._send_lock:
            try:
                await ws.send(msg.model_dump_json())
            except ConnectionClosed:
                pass

    async def _ping_loop(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL_S)
                await self._send(Ping(ts_ms=int(time.time() * 1000)))
        except asyncio.CancelledError:
            return

    def _ws_url(self) -> str:
        # http(s)://… → ws(s)://…
        u = self.kanad_url
        if u.startswith("https://"):
            return "wss://" + u[len("https://"):] + "/api/compute/connect"
        if u.startswith("http://"):
            return "ws://" + u[len("http://"):] + "/api/compute/connect"
        return u + "/api/compute/connect"


def start_ws_client(kanad_url: str, config: dict) -> threading.Thread:
    """Spawn the WS client on a background thread. Returns the thread handle."""
    client = ComputeWSClient(kanad_url, config)
    t = threading.Thread(target=client.run_forever_sync, daemon=True, name="kanad-ws")
    t.start()
    return t
