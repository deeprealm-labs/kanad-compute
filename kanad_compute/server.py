"""Kanad Compute Server — FastAPI app that accepts computation jobs."""

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

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
    solver: str = "physics_vqe"
    backend: str = "statevector"
    max_iterations: int = 100
    max_excitations: int = 5
    ansatz_type: str = "hardware_efficient"
    ibm_api_token: Optional[str] = None
    ibm_crn: Optional[str] = None
    ionq_api_key: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, running, completed, failed
    result: Optional[dict] = None


# ─────────── App ───────────

def create_app(config: Optional[dict] = None) -> FastAPI:
    if config is None:
        config = load_config()

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
    executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 2))
    api_key = config.get("api_key", "")

    def _verify_key(authorization: str = Header(default="")):
        token = authorization.removeprefix("Bearer ").strip()
        if token != api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    # ─── Endpoints ───

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "service": "kanad-compute",
            "version": "0.1.0",
            "node_id": config.get("node_id"),
        }

    @app.get("/info")
    async def info(authorization: str = Header(default="")):
        _verify_key(authorization)
        sys_info = get_system_info(config.get("gpu_enabled", False))
        sys_info["node_id"] = config.get("node_id")
        sys_info["max_qubits"] = config.get("max_qubits", 20)
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

        # Check qubit limit
        n_qubits = 2 * len(job.atoms)
        max_q = config.get("max_qubits", 20)
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
        if not job_data.get("ionq_api_key") and config.get("ionq_api_key"):
            job_data["ionq_api_key"] = config["ionq_api_key"]

        jobs[job_id] = {"status": "pending", "result": None, "submitted_at": time.time()}

        # Run in background thread
        loop = asyncio.get_event_loop()
        gpu = config.get("gpu_enabled", False)

        def _execute():
            jobs[job_id]["status"] = "running"
            result = run_calculation(job_data, gpu_enabled=gpu)
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
        return JobStatus(job_id=job_id, status=j["status"], result=j.get("result"))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, authorization: str = Header(default="")):
        _verify_key(authorization)
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        jobs[job_id]["status"] = "cancelled"
        return {"status": "cancelled", "job_id": job_id}

    @app.get("/jobs")
    async def list_jobs(authorization: str = Header(default="")):
        _verify_key(authorization)
        return [
            {"job_id": jid, "status": j["status"]}
            for jid, j in sorted(jobs.items(), key=lambda x: x[1].get("submitted_at", 0), reverse=True)
        ][:20]

    return app
