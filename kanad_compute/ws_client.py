"""Outbound WebSocket client — replaces remote_worker.py polling loop.

Connects to wss://kanad.xyz/api/compute/connect and:
  1. Sends Hello with node identity + system info + last_ack_seq for resume.
  2. Waits for Registered. Verifies protocol version compatibility.
  3. Replays any unacked events left over from a prior connection.
  4. Receives ExperimentRequest, runs it via worker.run_calculation in a
     thread, and streams ExperimentEvents (Log → FinalResult/Error) back.
  5. Honours server Acks to drop outbox rows and persist last_ack_seq.
  6. Pong-replies to server Pings; sends own Pings to keep NATs alive.

Reliability sketch:
  - Every event is persisted to a SQLite ``Outbox`` BEFORE the frame goes on
    the wire. If the process crashes between record and send, the row is
    still on disk and is replayed on reconnect.
  - ``last_ack_seq`` is also persisted to ``~/.kanad-compute/state/seq.json``
    (atomic tmpfile + rename) so Hello can populate it without scanning the
    whole outbox at startup.
  - Reconnect uses exponential backoff (1s → 30s) with ±20% jitter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .outbox import Outbox
from .protocol import (
    PROTOCOL_VERSION,
    Ack,
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


def _kanad_core_version() -> Optional[str]:
    """Best-effort lookup of the kanad-core package version.

    Cloud uses this to surface compatibility info on the admin status page
    and to refuse jobs that need a newer solver schema. Returns None if
    kanad-core isn't installed (e.g. tests stubbing the worker).
    """
    try:
        import kanad
        return getattr(kanad, "__version__", None)
    except Exception:
        return None


def _vault_status(config: dict) -> dict[str, bool]:
    """Hello.vault: prefer real keyring presence, fall back to config dict.

    Phase 2.4 stores creds in the OS keyring. Older configs that pass tokens
    via init args still work — config presence counts for the same logical
    backend.
    """
    status: dict[str, bool] = {}
    try:
        from .vault import Vault
        status = Vault().status()
    except Exception as e:
        logger.debug(f"vault status unavailable: {e}")
    # Config-based fallback merges in OR: a config-provided IBM token is as
    # good as a vaulted one for "do you have IBM creds at all" purposes.
    if config.get("ibm_api_token"):
        status["ibm"] = True
    if config.get("ionq_api_key"):
        status["ionq"] = True
    if config.get("bluequbit_api_key"):
        status["bluequbit"] = True
    return status

# Progress event throttling. Solver callbacks can fire 100s of times per second
# on small molecules; we drop emits that arrive within the time window unless
# the energy has improved by at least the delta threshold.
PROGRESS_MIN_INTERVAL_MS = 100
PROGRESS_ENERGY_DELTA = 1e-4   # Hartree


class _ExperimentBuffer:
    """Per-experiment in-memory counters.

    Durable unacked-event storage lives in the SQLite ``Outbox``. This class
    only tracks fast in-memory state:

    next_seq:      next seq to assign on _emit (monotonic, per-experiment)
    last_ack_seq:  highest seq the server has acked (mirrored to seq.json so
                   Hello.last_ack_seq can be sent without scanning the outbox)
    """
    __slots__ = ("next_seq", "last_ack_seq")

    def __init__(self, next_seq: int = 1, last_ack_seq: int = 0):
        self.next_seq = next_seq
        self.last_ack_seq = last_ack_seq


class ComputeWSClient:
    def __init__(self, kanad_url: str, config: dict):
        self.kanad_url = kanad_url.rstrip("/")
        self.config = config
        # Prefer the device-auth JWT (Phase 2.3) from the vault, fall back to
        # the legacy ``kanad_compute_key`` in config. Server's ``_authenticate``
        # accepts either, so this is a transparent migration.
        device_token = ""
        try:
            from .vault import Vault
            device_token = Vault().get("kanad_access_token") or ""
        except Exception:
            pass
        # Config override beats vault if the user explicitly sets a token in
        # config; otherwise vault wins when present, else the legacy api_key.
        self.api_key = (
            config.get("kanad_access_token")
            or device_token
            or config.get("api_key", "")
        )
        self.node_id = config.get("node_id", "unknown")
        self.gpu_enabled = bool(config.get("gpu_enabled", False))

        self._executor = ThreadPoolExecutor(
            max_workers=config.get("max_workers", 2),
            thread_name_prefix="kanad-job",
        )
        # cancel_check closures read this set. threading.Event isn't quite
        # right because we want set-membership lookup from many threads.
        self._cancelled: set[str] = set()
        self._buffers: dict[str, _ExperimentBuffer] = {}
        self._send_lock = asyncio.Lock()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

        # State persistence: ~/.kanad-compute/state/{seq.json, outbox.db}
        # State dir is overridable for tests.
        state_dir = config.get("state_dir") or (Path.home() / ".kanad-compute" / "state")
        self._state_dir = Path(state_dir)
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create state dir {self._state_dir}: {e}")
        self._seq_path = self._state_dir / "seq.json"
        self._outbox = Outbox(self._state_dir / "outbox.db")
        self._outbox.gc()  # opportunistic 24h cleanup at startup
        self._load_seq_state()

    # ── persistence ────────────────────────────────────────────────────────

    def _load_seq_state(self) -> None:
        if not self._seq_path.exists():
            return
        try:
            data = json.loads(self._seq_path.read_text())
            for exp_id, last_ack in data.items():
                self._buffers[exp_id] = _ExperimentBuffer(
                    next_seq=int(last_ack) + 1,
                    last_ack_seq=int(last_ack),
                )
            logger.info(f"Loaded {len(data)} experiment seq states from {self._seq_path}")
        except Exception as e:
            logger.warning(f"Failed to load seq state from {self._seq_path}: {e}")

    def _save_seq_state(self) -> None:
        try:
            data = {
                exp_id: buf.last_ack_seq
                for exp_id, buf in self._buffers.items()
                if buf.last_ack_seq > 0
            }
            tmp = self._seq_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data))
            os.replace(tmp, self._seq_path)
        except Exception as e:
            logger.warning(f"Failed to persist seq state: {e}")

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
                delay = self._jitter(backoff)
                logger.info(f"WS disconnected: {e}; reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, RECONNECT_MAX_S)
            except Exception as e:
                delay = self._jitter(backoff)
                logger.exception(f"WS unexpected error: {e}; reconnecting in {delay:.1f}s")
                await asyncio.sleep(delay)
                backoff = min(backoff * 2, RECONNECT_MAX_S)

    @staticmethod
    def _jitter(seconds: float) -> float:
        # ±20% jitter — avoids thundering-herd reconnects after a server blip.
        return seconds * (0.8 + 0.4 * random.random())

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

            # Hello — last_ack_seq lets the server skip already-delivered events
            sys_info = get_system_info(self.gpu_enabled)
            hello = Hello(
                node_id=self.node_id,
                system_info=sys_info,
                kanad_core_version=_kanad_core_version(),
                vault=_vault_status(self.config),
                last_ack_seq={
                    exp_id: buf.last_ack_seq
                    for exp_id, buf in self._buffers.items()
                    if buf.last_ack_seq > 0
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

            # Replay any unacked events from prior connections. The server
            # dedupes against last_seq, so duplicates from the seq.json hint
            # are harmless.
            await self._replay_unacked()

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
                    elif isinstance(msg, Ack):
                        self._handle_ack(msg)
                    elif isinstance(msg, Ping):
                        await self._send_raw(Pong(ts_ms=msg.ts_ms))
                    elif isinstance(msg, CancelExperiment):
                        self._cancelled.add(msg.experiment_id)
                        logger.info(f"Cancel requested for {msg.experiment_id[:8]}")
                    # Pong — handled implicitly (server-initiated; our Pings get Pongs back as Ack)
            finally:
                pinger.cancel()
                self._ws = None

    async def _replay_unacked(self) -> None:
        replayed = 0
        for exp_id, _seq, _kind, frame in self._outbox.pending():
            await self._send_text(frame)
            replayed += 1
        if replayed:
            logger.info(f"Replayed {replayed} unacked events on reconnect")

    def _handle_ack(self, ack: Ack) -> None:
        buf = self._buffers.get(ack.experiment_id)
        if buf is None:
            return
        self._outbox.ack(ack.experiment_id, ack.last_seq)
        if ack.last_seq > buf.last_ack_seq:
            buf.last_ack_seq = ack.last_seq
            self._save_seq_state()

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
        # Credentials: prefer the local vault (Phase 2.4) over the wire-provided
        # ones (Phase 1 fallback). Cloud will eventually stop sending creds at
        # all; until then a vaulted secret wins.
        wire_creds = req.backend_credentials or {}
        creds = dict(wire_creds)
        try:
            from .vault import Vault
            for key, val in Vault().all().items():
                if val:
                    creds[key] = val
        except Exception as e:
            logger.debug(f"vault read failed during dispatch: {e}")
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
        cancel_check = self._make_cancel_check(req.experiment_id)
        progress_cb = self._make_progress_cb(req.experiment_id, loop)
        try:
            result = await loop.run_in_executor(
                self._executor,
                lambda: run_calculation(
                    job_record, self.gpu_enabled,
                    cancel_check=cancel_check, progress_cb=progress_cb,
                ),
            )
        except TypeError:
            # worker.run_calculation may not yet accept progress_cb kwarg
            # (older bundled wheels). Fall back through the legacy signatures.
            try:
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: run_calculation(
                        job_record, self.gpu_enabled, cancel_check=cancel_check,
                    ),
                )
            except TypeError:
                try:
                    result = await loop.run_in_executor(
                        self._executor, run_calculation, job_record, self.gpu_enabled
                    )
                except Exception as e:
                    await self._emit(req.experiment_id, "Error", {
                        "message": str(e), "traceback": traceback.format_exc(),
                    })
                    return
            except Exception as e:
                await self._emit(req.experiment_id, "Error", {
                    "message": str(e), "traceback": traceback.format_exc(),
                })
                return
        except Exception as e:
            await self._emit(req.experiment_id, "Error", {
                "message": str(e),
                "traceback": traceback.format_exc(),
            })
            return

        if result.get("status") == "cancelled" or req.experiment_id in self._cancelled:
            self._cancelled.discard(req.experiment_id)
            await self._emit(req.experiment_id, "Error", {
                "message": result.get("error_message") or "Cancelled by user",
                "code": "cancelled",
            })
            return

        if result.get("status") == "failed":
            await self._emit(req.experiment_id, "Error", {
                "message": result.get("error_message", "Calculation failed"),
                "traceback": result.get("traceback"),
            })
            return

        # Final-iteration flush: ensure the last progress frame the throttle
        # may have dropped is actually delivered before FinalResult.
        flush_state = getattr(progress_cb, "_state", None)
        if flush_state is not None and flush_state["last_iteration"] >= 0:
            payload: dict[str, Any] = {"iteration": flush_state["last_iteration"]}
            if flush_state["last_energy"] is not None:
                payload["energy"] = flush_state["last_energy"]
            await self._emit(req.experiment_id, "Progress", payload)

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

    def _make_cancel_check(self, experiment_id: str):
        cancelled = self._cancelled
        return lambda: experiment_id in cancelled

    def _make_progress_cb(self, experiment_id: str, loop: asyncio.AbstractEventLoop):
        """Return a thread-safe callable for solvers to report progress.

        The closure is invoked from the worker thread (where solvers run);
        each call submits a coroutine to ``loop`` via run_coroutine_threadsafe.

        Throttling rules:
          - First call: always emits.
          - Subsequent calls within PROGRESS_MIN_INTERVAL_MS: dropped, UNLESS
            energy has improved by at least PROGRESS_ENERGY_DELTA.
        """
        state = {"last_ts": 0.0, "last_energy": None, "last_iteration": -1}

        def cb(**kwargs: Any) -> None:
            iteration = kwargs.get("iteration", state["last_iteration"] + 1)
            energy = kwargs.get("energy")
            message = kwargs.get("message")
            gradient_norm = kwargs.get("gradient_norm")

            now = time.time()
            is_first = state["last_ts"] == 0.0
            elapsed_ms = (now - state["last_ts"]) * 1000.0
            time_pass = elapsed_ms >= PROGRESS_MIN_INTERVAL_MS
            energy_pass = (
                energy is not None
                and state["last_energy"] is not None
                and abs(float(energy) - float(state["last_energy"])) >= PROGRESS_ENERGY_DELTA
            )

            # Always update last_iteration so flush at end has the true terminal value
            state["last_iteration"] = int(iteration)
            if energy is not None:
                state["last_energy"] = float(energy)

            if not (is_first or time_pass or energy_pass):
                return

            state["last_ts"] = now

            payload: dict[str, Any] = {"iteration": int(iteration)}
            if energy is not None:
                payload["energy"] = float(energy)
            if gradient_norm is not None:
                payload["gradient_norm"] = float(gradient_norm)
            if message is not None:
                payload["message"] = str(message)

            try:
                asyncio.run_coroutine_threadsafe(
                    self._emit(experiment_id, "Progress", payload), loop
                )
            except Exception as e:
                logger.debug(f"progress emit dropped for {experiment_id[:8]}: {e}")

        cb._state = state  # type: ignore[attr-defined]
        return cb

    # ── outbound helpers ───────────────────────────────────────────────────

    async def _emit(self, experiment_id: str, kind: str, payload: dict[str, Any]) -> None:
        buf = self._buffers.setdefault(experiment_id, _ExperimentBuffer())
        seq = buf.next_seq
        buf.next_seq = seq + 1

        ev = ExperimentEvent(
            experiment_id=experiment_id,
            seq=seq,
            ts_ms=int(time.time() * 1000),
            kind=kind,  # type: ignore[arg-type]
            payload=payload,
        )
        frame = ev.model_dump_json()

        # Persist BEFORE send. If we crash between record and send, the row
        # is on disk and will be replayed on the next connect.
        self._outbox.record(experiment_id, seq, kind, frame)
        await self._send_text(frame)

    async def _send_raw(self, msg: Any) -> None:
        await self._send_text(msg.model_dump_json())

    async def _send_text(self, text: str) -> None:
        ws = self._ws
        if not ws:
            return
        async with self._send_lock:
            try:
                await ws.send(text)
            except ConnectionClosed:
                pass

    async def _ping_loop(self, ws) -> None:
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL_S)
                await self._send_raw(Ping(ts_ms=int(time.time() * 1000)))
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
