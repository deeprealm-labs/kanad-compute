"""Remote worker — polls Kanad platform for jobs and executes them locally.

This is the outbound connection model:
  kanad-compute → connects TO kanad.xyz (no port forwarding needed)
  kanad-compute → polls for pending jobs
  kanad-compute → runs jobs locally
  kanad-compute → pushes results back
"""

import time
import logging
import httpx

from .worker import run_calculation
from .sysinfo import get_system_info

logger = logging.getLogger(__name__)

POLL_INTERVAL = 2  # seconds
HEARTBEAT_INTERVAL = 30  # seconds


def start_worker(kanad_url: str, config: dict, status: dict = None):
    """Main worker loop — runs in a background thread.

    ``status`` (optional) is a shared dict the loop updates for a live TUI:
    connected, polls, active, recent (list of {id,name,solver,status,energy})."""
    if status is None:
        status = {}
    status.setdefault("recent", [])
    api_key = config.get("api_key", "")
    node_id = config.get("node_id", "unknown")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Register with Kanad platform
    try:
        sys_info = get_system_info(config.get("gpu_enabled", False))
        resp = httpx.post(
            f"{kanad_url}/api/compute/register",
            json={"node_id": node_id, "system_info": sys_info},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info(f"Registered with Kanad platform at {kanad_url}")
            status["connected"] = True
        else:
            logger.warning(f"Registration failed ({resp.status_code}): {resp.text[:200]}")
            status["connected"] = False
            status["last_error"] = f"register {resp.status_code}: {resp.text[:80]}"
    except Exception as e:
        logger.error(f"Failed to connect to Kanad platform: {e}")
        logger.info("Will keep retrying...")
        status["connected"] = False
        status["last_error"] = str(e)[:100]

    # Poll loop — each job runs in a WORKER THREAD so the main loop keeps HEARTBEATING while a
    # long job runs (e.g. an IBM QPU submission that blocks on the IBM queue). Previously the
    # single-threaded loop froze inside _execute_job and the app marked the node offline.
    import threading
    from concurrent.futures import ThreadPoolExecutor

    active = {"n": 0}
    active_lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kc-job")
    last_heartbeat = time.time()

    def _run_and_report(job, entry):
        job_id = job.get("job_id")
        try:
            result = _execute_job(job, config)
            entry["status"] = "completed"
            entry["energy"] = result.get("energy")
            # sampling_backend_used / _requested / _fallback_reason tell the app WHERE an SQD
            # job actually sampled (QPU vs statevector) so a silent fallback is never mistaken
            # for a real QPU run.
            result_payload = {
                "energy": result.get("energy"),
                "hf_energy": result.get("hf_energy"),
                "fci_energy": result.get("fci_energy"),
                "error_mha": result.get("error_mha"),
                "n_evaluations": result.get("n_evaluations"),
                "converged": result.get("converged", True),
                "convergence_history": result.get("convergence_history"),
                "wall_time": result.get("wall_time"),
                "sampling_backend_used": result.get("sampling_backend_used"),
                "sampling_requested": result.get("sampling_requested"),
                "sampling_fallback_reason": result.get("sampling_fallback_reason"),
                "status": "completed",
            }
            try:
                if result.get("sampling_backend_used"):
                    entry["backend"] = result["sampling_backend_used"]
            except Exception:
                pass
            try:
                push_resp = httpx.post(
                    f"{kanad_url}/api/compute/jobs/{job_id}/result",
                    json=result_payload, headers=headers, timeout=10,
                )
                if push_resp.status_code == 200:
                    logger.info(f"Job {job_id[:8]} completed — E = {result.get('energy', '?')}")
                else:
                    logger.warning(f"Failed to push result for {job_id[:8]}: {push_resp.status_code}")
            except Exception as e:
                logger.warning(f"Result push failed for {job_id[:8]}: {e}")
        except Exception as e:
            logger.error(f"Job {job_id[:8]} failed: {e}")
            entry["status"] = "failed"
            status["last_error"] = str(e)[:100]
            try:
                httpx.post(
                    f"{kanad_url}/api/compute/jobs/{job_id}/result",
                    json={"status": "failed", "error_message": str(e)},
                    headers=headers, timeout=10,
                )
            except Exception:
                pass
        finally:
            with active_lock:
                active["n"] = max(0, active["n"] - 1)
            status["active"] = active["n"]

    while True:
        try:
            # Heartbeat — on every tick, INCLUDING while a job executes in the worker thread,
            # so a long QPU job never makes the app think the node went offline.
            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                try:
                    httpx.post(
                        f"{kanad_url}/api/compute/heartbeat",
                        json={"node_id": node_id, "active_jobs": active["n"]},
                        headers=headers, timeout=5,
                    )
                    last_heartbeat = time.time()
                except Exception:
                    pass

            # Only claim new work when idle — the server marks claimed jobs 'running', so we
            # must be ready to run them; one batch at a time keeps GPU memory uncontended.
            with active_lock:
                busy = active["n"] > 0
            if busy:
                time.sleep(POLL_INTERVAL)
                continue

            # Poll for jobs
            try:
                resp = httpx.get(
                    f"{kanad_url}/api/compute/jobs", headers=headers, timeout=10,
                )
                if resp.status_code != 200:
                    time.sleep(POLL_INTERVAL)
                    continue

                status["connected"] = True
                status["polls"] = status.get("polls", 0) + 1
                jobs = resp.json().get("jobs", [])

                if not jobs:
                    time.sleep(POLL_INTERVAL)
                    continue

                for job in jobs:
                    job_id = job.get("job_id")
                    if not job_id:
                        continue
                    with active_lock:
                        active["n"] += 1
                    status["active"] = active["n"]
                    entry = {"id": job_id, "name": job.get("molecule_name") or "?",
                             "solver": job.get("solver_type") or "?", "status": "running", "energy": None}
                    status["recent"].insert(0, entry)
                    del status["recent"][8:]
                    logger.info(f"Picked up job {job_id[:8]} — {job.get('molecule_name', '?')} / {job.get('solver_type', '?')}")
                    executor.submit(_run_and_report, job, entry)
                time.sleep(POLL_INTERVAL)

            except httpx.ConnectError:
                logger.debug("Kanad platform unreachable, retrying...")
                status["connected"] = False
                time.sleep(5)
            except Exception as e:
                logger.debug(f"Poll error: {e}")
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Worker error: {e}")
            time.sleep(5)


def _execute_job(job: dict, config: dict) -> dict:
    """Execute a single job using the existing worker infrastructure."""
    import time as _time

    start = _time.time()

    # Parse atoms string "H 0 0 0; Li 0 0 1.6" into list of dicts
    atoms_raw = job.get("atoms", "")
    atoms = []
    if isinstance(atoms_raw, str) and atoms_raw.strip():
        for part in atoms_raw.split(";"):
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            if len(tokens) >= 4:
                atoms.append({
                    "symbol": tokens[0],
                    "position": [float(tokens[1]), float(tokens[2]), float(tokens[3])],
                })
    elif isinstance(atoms_raw, list):
        atoms = atoms_raw

    if not atoms:
        raise ValueError(f"No atoms parsed from: {atoms_raw!r}")

    # Build job dict in the format run_calculation expects (top-level keys, not nested config).
    # IMPORTANT: pass through backend + cloud credentials + spin/kind/custom — this is what
    # lets an SQD job actually sample on the user's IBM QPU (backend='ibm'). Hardcoding
    # 'statevector' here silently forced every job onto the local simulator.
    job_record = {
        "job_id": job.get("job_id"),
        "kind": job.get("kind", "energy"),
        "atoms": atoms,
        "basis": job.get("basis") or "sto-3g",
        "charge": job.get("charge", 0),
        "spin": job.get("spin", 0),
        "solver": job.get("solver_type", "physics_vqe"),
        "backend": job.get("backend") or "statevector",
        "max_iterations": job.get("max_iterations", 100),
        "max_excitations": job.get("max_excitations", 5),
        "ansatz_type": job.get("ansatz_type", "hardware_efficient"),
        "ibm_api_token": job.get("ibm_api_token"),
        "ibm_crn": job.get("ibm_crn"),
        "ibm_backend_name": job.get("ibm_backend_name"),
        "ionq_api_key": job.get("ionq_api_key"),
        "custom_solver": job.get("custom_solver"),
        # force_qpu: node config (all jobs) OR per-job flag — no silent statevector fallback.
        "force_qpu": bool(config.get("force_qpu")) or bool(job.get("force_qpu")),
    }

    # run_calculation returns the result dict
    result = run_calculation(job_record, gpu_enabled=bool(config.get("gpu_enabled")),
                             gpu_device=config.get("gpu_device", "auto"))

    if result.get("status") == "failed":
        raise RuntimeError(result.get("error_message", "Calculation failed"))

    result["wall_time"] = round(_time.time() - start, 2)
    return result
