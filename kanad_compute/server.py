"""Kanad Compute Server — FastAPI app that accepts computation jobs."""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

SESSION_FILE = Path.home() / ".kanad-compute" / "session.json"

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

from .config import load_config
from .worker import run_calculation
from .sysinfo import get_system_info

logger = logging.getLogger(__name__)

# ─────────── Models ───────────

class JobSubmit(BaseModel):
    atoms: list[dict]
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0           # 2S = n_alpha - n_beta (open-shell support)
    solver: str = "physics_vqe"
    backend: str = "statevector"
    max_iterations: int = 100
    max_excitations: int = 5
    ansatz_type: str = "hardware_efficient"
    ibm_api_token: Optional[str] = None
    ibm_crn: Optional[str] = None
    ibm_backend_name: Optional[str] = None   # QPU the user picked (None = auto least-busy)
    ionq_api_key: Optional[str] = None
    custom_solver: Optional[dict] = None   # Workshop custom solver packet {base_type, config, name}
    kind: str = "energy"        # workflow: energy (single-point) | dynamics | materials
    config: Optional[dict] = None   # workflow-specific config (MD params, lattice, ...)


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    result: Optional[dict] = None
    progress: Optional[dict] = None   # live {iteration, energy, max_iterations, phase}


# ─────────── App ───────────

def create_app(config: Optional[dict] = None) -> FastAPI:
    if config is None:
        config = load_config()

    # Surface the worker's step-by-step progress ([SQD] preparing/sampling/... lines)
    # in the node's own log. Its INFO logs were dropped: no handler on the
    # kanad_compute logger + Python's lastResort handler only emits WARNING+.
    import logging as _logging
    import sys as _sys
    _kc = _logging.getLogger("kanad_compute")
    _kc.setLevel(_logging.INFO)
    if not any(isinstance(h, _logging.StreamHandler) for h in _kc.handlers):
        _h = _logging.StreamHandler(_sys.stdout)
        _h.setFormatter(_logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        _kc.addHandler(_h)
    _kc.propagate = False  # its own handler writes to stdout; don't double-log via root

    app = FastAPI(
        title="Kanad Compute",
        version="0.1.0",
        description="Local quantum chemistry compute server for Kanad platform",
    )

    allowed_origins = [
        "https://kanad.xyz",
        "https://kanad-app.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "https://kanad-api-640826962316.us-central1.run.app",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

    # Request size limit (1 MB)
    MAX_BODY = 1_048_576

    class SizeLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            cl = request.headers.get("content-length")
            if cl and int(cl) > MAX_BODY:
                from starlette.responses import JSONResponse
                return JSONResponse({"detail": "Request too large"}, status_code=413)
            return await call_next(request)

    app.add_middleware(SizeLimitMiddleware)

    # State
    jobs: dict[str, dict] = {}
    cancel_flags: set[str] = set()   # job_ids the user asked to cancel (checked by the solver)
    executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 2))
    api_key = config.get("api_key", "")

    def _verify_key(authorization: str = Header(default="")):
        token = authorization.removeprefix("Bearer ").strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ─── Endpoints ───

    @app.get("/health")
    async def health(request: Request):
        return {
            "status": "healthy",
            "service": "kanad-compute",
            "version": "0.1.0",
            "node_id": config.get("node_id"),
            "public_url": getattr(request.app.state, "public_url", None),
        }

    @app.get("/info")
    async def info(authorization: str = Header(default="")):
        _verify_key(authorization)
        sys_info = get_system_info(config.get("gpu_enabled", True))
        sys_info["node_id"] = config.get("node_id")
        sys_info["gpu_device"] = config.get("gpu_device", "auto")
        sys_info["max_qubits"] = config.get("max_qubits", 33)
        sys_info["max_workers"] = config.get("max_workers", 2)
        sys_info["active_jobs"] = sum(
            1 for j in jobs.values() if j["status"] in ("pending", "running")
        )
        sys_info["has_ibm"] = bool(config.get("ibm_api_token"))
        sys_info["has_ionq"] = bool(config.get("ionq_api_key"))
        return sys_info

    @app.post("/jobs", response_model=JobStatus)
    async def submit_job(job: JobSubmit, authorization: str = Header(default="")):
        _verify_key(authorization)

        # Check qubit limit — electronic-structure jobs only. Materials is a periodic HF
        # workflow (not qubit-bounded), and a supercell legitimately has many atoms, so the
        # 2*len(atoms) qubit cap must not apply to it.
        if job.kind != "materials":
            n_qubits = 2 * len(job.atoms)
            max_q = config.get("max_qubits", 33)
            if n_qubits > max_q:
                raise HTTPException(
                    status_code=400,
                    detail=f"Molecule requires {n_qubits} qubits, server limit is {max_q}",
                )

        # Check capacity
        active = sum(1 for j in jobs.values() if j["status"] in ("pending", "running"))
        if active >= config.get("max_workers", 2):
            raise HTTPException(status_code=429, detail="Server at capacity, try later")

        job_id = str(uuid.uuid4())
        job_data = job.model_dump()

        # Inject server-side credentials if user didn't provide them
        if not job_data.get("ibm_api_token") and config.get("ibm_api_token"):
            job_data["ibm_api_token"] = config["ibm_api_token"]
        if not job_data.get("ibm_crn") and config.get("ibm_crn"):
            job_data["ibm_crn"] = config["ibm_crn"]
        if not job_data.get("ionq_api_key") and config.get("ionq_api_key"):
            job_data["ionq_api_key"] = config["ionq_api_key"]

        jobs[job_id] = {"status": "pending", "result": None, "submitted_at": time.time(), "progress": None}

        # Run in background thread
        loop = asyncio.get_event_loop()
        gpu = config.get("gpu_enabled", False)
        gpu_device = config.get("gpu_device", "auto")

        def _execute():
            jobs[job_id]["status"] = "running"
            # Live progress: the solver streams per-iteration / per-phase updates here
            # (read by the app via GET /jobs/{id}.progress while it polls).
            def _progress(info):
                try:
                    jobs[job_id]["progress"] = info
                except Exception:
                    pass
            # Cooperative cancellation: the solver's callback checks this each step and
            # raises, so the worker actually stops (frees GPU/QPU) — not just a status flip.
            result = run_calculation(job_data, gpu_enabled=gpu, gpu_device=gpu_device,
                                     progress_cb=_progress,
                                     cancel_check=lambda: job_id in cancel_flags)
            if job_id in cancel_flags or result.get("status") == "cancelled":
                jobs[job_id]["status"] = "cancelled"
                jobs[job_id]["result"] = result
                cancel_flags.discard(job_id)
            else:
                jobs[job_id]["status"] = result.get("status", "completed")
                jobs[job_id]["result"] = result

        executor.submit(_execute)

        return JobStatus(job_id=job_id, status="pending")

    @app.get("/jobs/{job_id}", response_model=JobStatus)
    async def get_job(job_id: str, authorization: str = Header(default="")):
        _verify_key(authorization)
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = jobs[job_id]
        return JobStatus(job_id=job_id, status=j["status"], result=j.get("result"),
                         progress=j.get("progress"))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, authorization: str = Header(default="")):
        _verify_key(authorization)
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        # Flag for cooperative cancellation (the running solver checks it and stops);
        # mark cancelled now so a poll sees it immediately. A QPU sampling job already
        # submitted to IBM can't be interrupted mid-shot — cancellation takes effect at
        # the next phase/iteration boundary.
        cancel_flags.add(job_id)
        if jobs[job_id]["status"] in ("pending", "running"):
            jobs[job_id]["status"] = "cancelled"
        return {"status": "cancelled", "job_id": job_id}

    @app.get("/jobs")
    async def list_jobs(authorization: str = Header(default="")):
        _verify_key(authorization)
        return [
            {"job_id": jid, "status": j["status"]}
            for jid, j in sorted(jobs.items(), key=lambda x: x[1].get("submitted_at", 0), reverse=True)
        ][:20]

    # ─── Session (pushed by kanad-app over SSH at pairing / refresh) ───
    # kanad-app authenticates the user, then pushes the user's identity, plan, and
    # experiment history here so the node TUI can show a logged-in session. The
    # node never authenticates the user itself.
    # Secret session fields that must stay in memory only, never written to disk.
    _SESSION_SECRET_KEYS = ("token", "jwt", "access_token", "ibm_api_token", "ionq_api_key")

    @app.post("/session")
    async def push_session(session: dict, authorization: str = Header(default="")):
        _verify_key(authorization)
        # Full session (incl. the short-lived token) lives in memory for this process.
        app.state.session = session
        # Persist ONLY non-secret display fields so a TUI restart can show identity
        # without ever leaving a credential on disk. The token is re-pushed on refresh.
        try:
            safe = {k: v for k, v in session.items() if k not in _SESSION_SECRET_KEYS}
            SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
            SESSION_FILE.write_text(json.dumps(safe, indent=2))
            SESSION_FILE.chmod(0o600)
        except Exception as e:  # persistence is best-effort; TUI can read app.state
            logger.warning("Could not persist session.json: %s", e)
        who = session.get("user_email") or session.get("email") or "user"
        logger.info("Session pushed for %s (plan=%s)", who, session.get("plan"))
        return {"ok": True, "user": who}

    @app.get("/session")
    async def get_session(authorization: str = Header(default="")):
        _verify_key(authorization)
        sess = getattr(app.state, "session", None)
        if sess is None and SESSION_FILE.exists():
            try:
                sess = json.loads(SESSION_FILE.read_text())
            except Exception:
                sess = None
        return sess or {}

    return app
