"""Compute worker — executes Kanad solvers and returns results."""

import logging
import time
import traceback
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _build_bond(atoms: list[dict], basis: str = "sto-3g", charge: int = 0):
    """Build a Kanad bond/molecule from atom list."""
    from kanad import BondFactory

    if len(atoms) == 2:
        a1, a2 = atoms
        import math
        dx = a1["position"][0] - a2["position"][0]
        dy = a1["position"][1] - a2["position"][1]
        dz = a1["position"][2] - a2["position"][2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return BondFactory.create_bond(
            a1["symbol"], a2["symbol"],
            distance=dist, basis=basis, charge=charge,
        )
    else:
        return BondFactory.create_molecule(
            [(a["symbol"], a["position"]) for a in atoms],
            basis=basis, charge=charge,
        )


def _build_pyscf_mol(atoms: list[dict], basis: str = "sto-3g", charge: int = 0):
    """Build a PySCF mol object from atom list."""
    from pyscf import gto
    atom_str = "; ".join(
        f'{a["symbol"]} {a["position"][0]} {a["position"][1]} {a["position"][2]}'
        for a in atoms
    )
    return gto.M(atom=atom_str, basis=basis, charge=charge, verbose=0)


def run_calculation(job: dict, gpu_enabled: bool = False) -> dict:
    """
    Run a Kanad calculation job. Returns result dict.

    job schema:
        atoms: [{symbol, position: [x,y,z]}, ...]
        basis: str
        charge: int
        solver: str (physics_vqe, hardware_vqe, vqe, sqd, hybrid_subspace, ...)
        backend: str (statevector, aer, ibm_quantum, ionq)
        max_iterations: int
        max_excitations: int (for physics_vqe)
        ansatz_type: str
        ibm_api_token: str | None
        ibm_crn: str | None
        ionq_api_key: str | None
    """
    t0 = time.time()
    result: dict[str, Any] = {
        "status": "running",
        "energy": None,
        "hf_energy": None,
        "fci_energy": None,
        "error_mha": None,
        "n_evaluations": None,
        "converged": None,
        "convergence_history": None,
        "wall_time_ms": None,
    }

    try:
        atoms = job["atoms"]
        basis = job.get("basis", "sto-3g")
        charge = job.get("charge", 0)
        solver_type = job.get("solver", "physics_vqe")
        backend_type = job.get("backend", "statevector")
        max_iterations = job.get("max_iterations", 100)
        max_excitations = job.get("max_excitations", 5)
        ansatz_type = job.get("ansatz_type", "hardware_efficient")

        logger.info(f"Running {solver_type} on {len(atoms)} atoms, backend={backend_type}")

        # --- Select backend ---
        backend = "statevector"
        if backend_type == "aer":
            if gpu_enabled:
                try:
                    from qiskit_aer import AerSimulator
                    AerSimulator(method="statevector", device="GPU")
                    backend = "aer"
                    logger.info("Using Aer GPU backend")
                except Exception:
                    backend = "aer"
                    logger.info("Using Aer CPU backend")
            else:
                backend = "aer"

        # --- Run solver ---
        if solver_type in ("physics_vqe", "smart"):
            sol_result = _run_physics_vqe(
                atoms, basis, charge, backend,
                max_iterations, max_excitations,
                ibm_token=job.get("ibm_api_token"),
                ionq_key=job.get("ionq_api_key"),
            )
        elif solver_type == "hardware_vqe":
            sol_result = _run_hardware_vqe(
                atoms, basis, charge, backend,
                ibm_token=job.get("ibm_api_token"),
            )
        elif solver_type == "hybrid_subspace":
            sol_result = _run_hybrid_subspace(
                atoms, basis, charge, backend,
                ibm_token=job.get("ibm_api_token"),
            )
        elif solver_type == "sqd":
            sol_result = _run_sqd(atoms, basis, charge)
        elif solver_type == "krylov_sqd":
            sol_result = _run_krylov_sqd(atoms, basis, charge)
        elif solver_type == "vqe":
            sol_result = _run_vqe(
                atoms, basis, charge, backend,
                max_iterations, ansatz_type,
            )
        elif solver_type == "varqite":
            sol_result = _run_varqite(atoms, basis, charge, ansatz_type)
        elif solver_type == "qeom":
            sol_result = _run_qeom(atoms, basis, charge)
        elif solver_type == "efficient_vqe":
            sol_result = _run_efficient_vqe(atoms, basis, charge, max_iterations)
        elif solver_type == "excited_states":
            sol_result = _run_excited_states(atoms, basis, charge)
        else:
            # Default to physics_vqe
            sol_result = _run_physics_vqe(
                atoms, basis, charge, backend,
                max_iterations, max_excitations,
            )

        result.update(sol_result)
        result["status"] = "completed"

    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        result["status"] = "failed"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["wall_time_ms"] = int((time.time() - t0) * 1000)
    return result


# ─────────── Solver implementations ───────────


def _run_physics_vqe(
    atoms, basis, charge, backend, max_iter, max_exc,
    ibm_token=None, ionq_key=None,
) -> dict:
    from kanad.solvers import PhysicsVQE

    mol = _build_pyscf_mol(atoms, basis, charge)
    kwargs: dict[str, Any] = {"pyscf_mol": mol, "max_excitations": max_exc}

    if backend not in ("statevector", "aer"):
        kwargs["backend"] = backend
    if ibm_token:
        kwargs["ibm_api_token"] = ibm_token
    if ionq_key:
        kwargs["ionq_api_key"] = ionq_key

    solver = PhysicsVQE(**kwargs)
    res = solver.solve()

    history = []
    if hasattr(solver, "_energy_history"):
        history = [
            {"iteration": i, "energy": float(e)}
            for i, e in enumerate(solver._energy_history)
        ]

    return {
        "energy": float(res.energy),
        "hf_energy": float(res.hf_energy) if hasattr(res, "hf_energy") else None,
        "fci_energy": float(res.fci_energy) if hasattr(res, "fci_energy") else None,
        "error_mha": float(res.error_mha) if hasattr(res, "error_mha") else None,
        "n_evaluations": int(res.n_evaluations) if hasattr(res, "n_evaluations") else None,
        "converged": bool(getattr(res, "converged", True)),
        "convergence_history": history or None,
    }


def _run_hardware_vqe(atoms, basis, charge, backend, ibm_token=None) -> dict:
    from kanad.solvers import HardwareVQE

    bond = _build_bond(atoms, basis, charge)
    solver = HardwareVQE(bond=bond, circuit_type="hea")

    if backend == "ibm_quantum" and ibm_token:
        from kanad.backends.ibm import IBMBackend
        ibm = IBMBackend(api_token=ibm_token)
        res = solver.solve_hardware(ibm)
    else:
        res = solver.solve_local()

    return {
        "energy": float(res.energy),
        "error_mha": float(getattr(res, "error_mha", 0)),
        "n_evaluations": int(getattr(res, "n_evaluations", 0)),
        "converged": True,
    }


def _run_hybrid_subspace(atoms, basis, charge, backend, ibm_token=None) -> dict:
    from kanad.solvers import HybridSubspaceVQE

    bond = _build_bond(atoms, basis, charge)
    solver = HybridSubspaceVQE(bond=bond)
    res = solver.solve()

    return {
        "energy": float(res.energy),
        "fci_energy": float(getattr(res, "fci_energy", 0)),
        "error_mha": float(getattr(res, "error_mha", 0)),
        "n_evaluations": int(getattr(res, "n_evaluations", 0)),
        "converged": True,
    }


def _run_sqd(atoms, basis, charge) -> dict:
    from kanad.solvers import SQDSolver
    bond = _build_bond(atoms, basis, charge)
    solver = SQDSolver(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.energy),
        "error_mha": float(getattr(res, "error_mha", 0)),
        "converged": True,
    }


def _run_krylov_sqd(atoms, basis, charge) -> dict:
    from kanad.solvers import KrylovSQDSolver
    bond = _build_bond(atoms, basis, charge)
    solver = KrylovSQDSolver(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.energy),
        "error_mha": float(getattr(res, "error_mha", 0)),
        "converged": True,
    }


def _run_vqe(atoms, basis, charge, backend, max_iter, ansatz_type) -> dict:
    from kanad.solvers import VQESolver
    bond = _build_bond(atoms, basis, charge)
    solver = VQESolver(
        bond, ansatz_type=ansatz_type,
        backend=backend, max_iterations=max_iter,
    )
    res = solver.solve()
    return {
        "energy": float(res["energy"]),
        "n_evaluations": int(res.get("n_evaluations", 0)),
        "converged": bool(res.get("converged", True)),
    }


def _run_varqite(atoms, basis, charge, ansatz_type) -> dict:
    from kanad.solvers import VarQITESolver
    bond = _build_bond(atoms, basis, charge)
    solver = VarQITESolver(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.energy),
        "converged": True,
    }


def _run_qeom(atoms, basis, charge) -> dict:
    from kanad.solvers import qEOMVQE
    bond = _build_bond(atoms, basis, charge)
    solver = qEOMVQE(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.energy),
        "converged": True,
    }


def _run_efficient_vqe(atoms, basis, charge, max_iter) -> dict:
    from kanad.solvers import EfficientVQE
    bond = _build_bond(atoms, basis, charge)
    solver = EfficientVQE(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.energy),
        "error_mha": float(getattr(res, "error_mha", 0)),
        "n_evaluations": int(getattr(res, "n_evaluations", 0)),
        "converged": True,
    }


def _run_excited_states(atoms, basis, charge) -> dict:
    from kanad.solvers import ExcitedStatesSolver
    bond = _build_bond(atoms, basis, charge)
    solver = ExcitedStatesSolver(bond=bond)
    res = solver.solve()
    return {
        "energy": float(res.ground_energy) if hasattr(res, "ground_energy") else float(res.energy),
        "converged": True,
    }
