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


def start_worker(kanad_url: str, config: dict):
    """Main worker loop — runs in a background thread."""
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
        else:
            logger.warning(f"Registration failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Failed to connect to Kanad platform: {e}")
        logger.info("Will keep retrying...")

    # Poll loop
    active_jobs = 0
    last_heartbeat = time.time()

    while True:
        try:
            # Heartbeat
            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                try:
                    httpx.post(
                        f"{kanad_url}/api/compute/heartbeat",
                        json={"node_id": node_id, "active_jobs": active_jobs},
                        headers=headers,
                        timeout=5,
                    )
                    last_heartbeat = time.time()
                except Exception:
                    pass

            # Poll for jobs
            try:
                resp = httpx.get(
                    f"{kanad_url}/api/compute/jobs",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code != 200:
                    time.sleep(POLL_INTERVAL)
                    continue

                data = resp.json()
                jobs = data.get("jobs", [])

                if not jobs:
                    time.sleep(POLL_INTERVAL)
                    continue

                # Process each job
                for job in jobs:
                    job_id = job.get("job_id")
                    if not job_id:
                        continue

                    active_jobs += 1
                    logger.info(f"Picked up job {job_id[:8]} — {job.get('molecule_name', '?')} / {job.get('solver_type', '?')}")

                    try:
                        result = _execute_job(job, config)

                        # Push result back
                        result_payload = {
                            "energy": result.get("energy"),
                            "hf_energy": result.get("hf_energy"),
                            "fci_energy": result.get("fci_energy"),
                            "error_mha": result.get("error_mha"),
                            "n_evaluations": result.get("n_evaluations"),
                            "converged": result.get("converged", True),
                            "convergence_history": result.get("convergence_history"),
                            "wall_time": result.get("wall_time"),
                            "status": "completed",
                        }

                        push_resp = httpx.post(
                            f"{kanad_url}/api/compute/jobs/{job_id}/result",
                            json=result_payload,
                            headers=headers,
                            timeout=10,
                        )
                        if push_resp.status_code == 200:
                            logger.info(f"Job {job_id[:8]} completed — E = {result.get('energy', '?')}")
                        else:
                            logger.warning(f"Failed to push result for {job_id[:8]}: {push_resp.status_code}")

                    except Exception as e:
                        logger.error(f"Job {job_id[:8]} failed: {e}")
                        try:
                            httpx.post(
                                f"{kanad_url}/api/compute/jobs/{job_id}/result",
                                json={"status": "failed", "error_message": str(e)},
                                headers=headers,
                                timeout=10,
                            )
                        except Exception:
                            pass
                    finally:
                        active_jobs = max(0, active_jobs - 1)

            except httpx.ConnectError:
                logger.debug("Kanad platform unreachable, retrying...")
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

    # Build job dict in the format run_calculation expects (top-level keys, not nested config)
    job_record = {
        "job_id": job.get("job_id"),
        "atoms": atoms,
        "basis": job.get("basis") or "sto-3g",
        "charge": job.get("charge", 0),
        "solver": job.get("solver_type", "physics_vqe"),
        "backend": "statevector",
        "max_iterations": job.get("max_iterations", 100),
        "max_excitations": job.get("max_excitations", 5),
        "ansatz_type": job.get("ansatz_type", "hardware_efficient"),
    }

    # run_calculation returns the result dict
    result = run_calculation(job_record)

    if result.get("status") == "failed":
        raise RuntimeError(result.get("error_message", "Calculation failed"))

    result["wall_time"] = round(_time.time() - start, 2)
    return result
