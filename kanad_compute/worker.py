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
        # Polyatomic: BondFactory.create_molecule ignores positions/charge and
        # fabricates geometry. Build a real Molecule (honors positions+charge+basis);
        # its .hamiltonian works with the solvers via BaseSolver._resolve_system.
        return _build_molecule(atoms, basis, charge)


def _build_molecule(atoms: list[dict], basis: str = "sto-3g", charge: int = 0, spin: int = 0):
    """Build a kanad Molecule from explicit atoms (symbol+position), honoring
    charge, spin (=2S), and basis. Works for ANY atom count (2+) and is the
    correct path for polyatomic + open-shell systems — unlike BondFactory.create_
    molecule, which ignores positions/charge and only generates preset geometries.
    Its .hamiltonian is a MolecularHamiltonian usable by SamplingSQDSolver."""
    import numpy as np
    from kanad.core.molecule import Molecule
    from kanad.core.atom import Atom
    atom_objs = [Atom(a["symbol"], position=np.array(a["position"], dtype=float)) for a in atoms]
    return Molecule(atom_objs, charge=charge, spin=spin, basis=basis)


def _build_pyscf_mol(atoms: list[dict], basis: str = "sto-3g", charge: int = 0):
    """Build a PySCF mol object from atom list."""
    from pyscf import gto
    atom_str = "; ".join(
        f'{a["symbol"]} {a["position"][0]} {a["position"][1]} {a["position"][2]}'
        for a in atoms
    )
    return gto.M(atom=atom_str, basis=basis, charge=charge, verbose=0)


# Partially-filled d/f shells → strong static correlation. AVAS on the valence
# d-manifold is the right active space for these (frontier would try to freeze
# singly-occupied orbitals on open-shell metals and fail).
_TRANSITION_METALS = {
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
}
# Keep the SQD subspace tractable on the GPU: ~10 active orbitals (≈20 qubits)
# → CASCI ~6e4 dets, det_ci in seconds. 12 orbitals (CASCI ~8.5e5) starts to drag.
_AS_MAX_ACTIVE_ORBITALS = 10
# Full systems at/under this qubit count skip the reduction entirely (small molecules
# run full-space, exactly as before — C2/O2/N2/CO etc. are unaffected).
_AS_FULL_QUBIT_BUDGET = 24


def _maybe_reduce_active_space(atoms, basis, charge, spin):
    """Return (active_space_Hamiltonian, info_dict) for large / strongly-correlated
    systems, else (None, None) so the caller uses the full Molecule Hamiltonian.

    A transition-metal dimer/cluster (Cr2, Fe2, …) needs 70+ qubits at STO-3G — far
    past what statevector sampling can hold — yet its chemistry lives in a small
    valence active space. Recipe (validated on MI300X: Cr2 frontier(5,5)/20q
    -2064.28 in 4.7s; Fe2 AVAS-3d/20q -2497.89 in 1.5s):
      - transition metal present → 2nd-order SCF (bare ROHF won't converge Cr2/Fe2)
        + AVAS on the valence d-manifold (keeps the SOMOs active);
      - else (large main-group) → RHF/ROHF + a frontier(k,k) window sized to the budget.
    """
    mol = _build_pyscf_mol(atoms, basis, charge)
    if spin:
        mol.spin = int(spin)
    full_q = 2 * mol.nao_nr()
    if full_q <= _AS_FULL_QUBIT_BUDGET:
        return None, None  # small enough to run the full space

    from pyscf import scf
    from kanad.core.active_space import ActiveSpaceSelector, build_active_space_hamiltonian

    tms = sorted({a["symbol"] for a in atoms if a["symbol"] in _TRANSITION_METALS})
    open_shell = bool(spin)
    base = scf.ROHF(mol) if open_shell else scf.RHF(mol)
    # 2nd-order (Newton) SCF for transition metals — plain ROHF/RHF diverges on Cr2/Fe2.
    mf = base.newton() if tms else base
    mf.max_cycle = 200
    if not tms:
        mf.level_shift = 0.3
    mf.kernel()

    k = _AS_MAX_ACTIVE_ORBITALS // 2
    if open_shell and tms:
        # Open-shell metal: frontier would try to FREEZE the singly-occupied d orbitals
        # (the frozen-core transform requires doubly-occupied frozens → it errors). AVAS
        # on the valence d-manifold keeps the SOMOs active. (Validated: Fe2 → 10 orbitals.)
        labels = [f"{s} {shell}" for s in tms for shell in ("3d", "4d", "5d")]
        sel = ActiveSpaceSelector(mf).avas(labels)
        method = f"AVAS d-manifold ({'+'.join(tms)})"
    elif open_shell:
        # Open-shell main-group: widen the occupied window so the SOMOs land in the
        # active set rather than the frozen core.
        sel = ActiveSpaceSelector(mf).frontier(k + 2, k)
        method = f"frontier({k + 2},{k})"
    else:
        # Closed-shell (incl. Cr2): a frontier(k,k) window around the Fermi level is the
        # valence d-manifold and stays at the target size. (Validated: Cr2 → 10 orbitals.)
        sel = ActiveSpaceSelector(mf).frontier(k, k)
        method = f"frontier({k},{k}) around the Fermi level"

    ham = build_active_space_hamiltonian(mf, sel)
    info = {
        "active_orbitals": int(ham.n_orbitals),
        "active_electrons": int(ham.n_electrons),
        "active_qubits": int(2 * ham.n_orbitals),
        "full_qubits": int(full_q),
        "method": method,
        "scf_converged": bool(mf.converged),
    }
    logger.info("Active-space reduction: %d→%d qubits via %s (%de,%do); SCF converged=%s",
                full_q, info["active_qubits"], method,
                info["active_electrons"], info["active_orbitals"], mf.converged)
    return ham, info


def _resolve_sv_backend(backend_type: str, gpu_device: str = "auto", gpu_enabled: bool = False) -> str:
    """Pick the local statevector engine for the VQE-family solvers. Only an
    explicit 'statevector' request is upgraded to the GPU (rocm-planck) when
    gpu_device is amd/auto and the planck GPU core is present — preserving exact
    statevector semantics. 'aer' keeps its (noisy) shot semantics; cloud backend
    types (ibm/ionq/bluequbit) keep the local 'statevector' fallback and are driven
    by their credentials in the solver. NVIDIA/cudaq statevector is deferred."""
    if backend_type == "statevector":
        gd = (gpu_device or "auto").lower()
        if gd in ("amd", "auto"):
            try:
                import planck
                if getattr(planck, "_GPU_CORE_AVAILABLE", False):
                    return "planck"
            except Exception:
                pass
        return "statevector"
    if backend_type == "aer":
        return "aer"
    return "statevector"


class JobCancelled(Exception):
    """Raised from a solver callback when the user cancels — stops the solve so the
    node frees the GPU/QPU instead of running a discarded job to completion."""


def run_calculation(job: dict, gpu_enabled: bool = False, gpu_device: str = "auto",
                    progress_cb=None, cancel_check=None) -> dict:
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
        ibm_crn = job.get("ibm_crn")  # for QPU sampling passthrough (hybrid SQD)

        logger.info(f"Running {solver_type} on {len(atoms)} atoms, "
                    f"backend={backend_type}, gpu_device={gpu_device}")

        # --- Select the local statevector engine (GPU when available) ---
        backend = _resolve_sv_backend(backend_type, gpu_device, gpu_enabled)
        if backend != backend_type:
            logger.info("Statevector engine: %s -> %s (gpu_device=%s)", backend_type, backend, gpu_device)

        # Live progress: translate a solver's per-evaluation callback
        # (iteration, energy, *params) into the node's progress channel. Phases
        # (sampling/diagonalizing) are emitted directly by the SQD runner.
        _prog_state = {"max_iter": 0}

        def _solver_cb(iteration, energy, *args):
            if cancel_check and cancel_check():
                raise JobCancelled()
            try:
                _prog_state["max_iter"] = max(_prog_state["max_iter"], int(iteration))
            except Exception:
                pass
            if not progress_cb:
                return
            try:
                progress_cb({"iteration": int(iteration), "energy": float(energy),
                             "max_iterations": int(max_iterations), "phase": "optimizing"})
            except Exception:
                pass

        # Throttled live CPU/GPU/VRAM utilization tag for progress lines (rocm-smi is
        # a subprocess, so refresh at most once per ~2s no matter how often we log).
        _res_cache = {"t": 0.0, "summary": ""}

        def _resources():
            now = time.time()
            if now - _res_cache["t"] > 2.0:
                try:
                    from .sysinfo import resource_summary
                    _res_cache["summary"] = resource_summary()
                except Exception:
                    _res_cache["summary"] = ""
                _res_cache["t"] = now
            return _res_cache["summary"]

        def _phase(phase, message=None, **extra):
            if cancel_check and cancel_check():
                raise JobCancelled()
            if not progress_cb:
                return
            # Accept BOTH conventions: _phase("name", "msg", key=val) and the SQD
            # runner's _phase({"phase": "name", "message": "msg", ...}). The dict form
            # was silently nesting under `phase` (message became None), so every
            # pre-IBM progress line was dropped by the app before it reached the log.
            if isinstance(phase, dict):
                payload = dict(phase)
                payload.setdefault("message", message)
            else:
                payload = {"phase": phase, "message": message, **extra}
            # Attach live resource utilization (structured field + inline in the message
            # so it shows in the app log without any app-side change).
            try:
                res = _resources()
                if res:
                    payload["resources"] = res
                    if payload.get("message"):
                        payload["message"] = f"{payload['message']}  ·  {res}"
            except Exception:
                pass
            try:
                progress_cb(payload)
            except Exception:
                pass

        # --- Run solver ---
        # Workflow discriminator: 'energy' (default) is the single-point electronic-structure
        # job the node has always run; 'dynamics'/'materials' route to whole-workflow runners
        # before the per-solver dispatch. Absent/unknown kind → energy (backward compatible).
        _kind = job.get("kind", "energy")
        _custom = job.get("custom_solver")
        if _kind == "dynamics":
            sol_result = _run_dynamics(job, gpu_device, progress_cb=progress_cb, cancel_check=cancel_check)
        elif _kind == "photodynamics":
            sol_result = _run_photodynamics(job, gpu_device, progress_cb=progress_cb, cancel_check=cancel_check)
        elif _kind == "materials":
            sol_result = _run_materials(job, gpu_device, progress_cb=progress_cb, cancel_check=cancel_check)
        elif _custom:
            # Workshop custom solver packet (config-based) — route to its base solver.
            sol_result = _run_custom_solver(
                atoms, basis, charge, _custom.get("base_type"), _custom.get("config"),
                backend, gpu_device, spin=int(job.get("spin", 0) or 0),
                ibm_token=job.get("ibm_api_token"), ibm_crn=ibm_crn, callback=_solver_cb, phase_cb=_phase)
        elif solver_type in ("physics_vqe", "smart"):
            sol_result = _run_physics_vqe(
                atoms, basis, charge, backend,
                max_iterations, max_excitations,
                ibm_token=job.get("ibm_api_token"),
                ionq_key=job.get("ionq_api_key"),
                callback=_solver_cb,
            )
        elif solver_type == "hardware_vqe":
            sol_result = _run_hardware_vqe(
                atoms, basis, charge, backend,
                ibm_token=job.get("ibm_api_token"),
                callback=_solver_cb,
            )
        elif solver_type == "hybrid_subspace":
            sol_result = _run_hybrid_subspace(
                atoms, basis, charge, backend,
                ibm_token=job.get("ibm_api_token"),
            )
        elif solver_type == "sampling_sqd" or (solver_type == "sqd" and backend_type in ("ibm", "bluequbit")):
            # Hybrid SQD: quantum sampling (QPU via user's IBM key, or local
            # statevector) + classical diagonalization via rocm-planck det_ci.
            _samp = backend_type if backend_type in ("ibm", "bluequbit") else "statevector"
            try:
                sol_result = _run_sampling_sqd(
                    atoms, basis, charge, _samp, gpu_device,
                    ibm_token=job.get("ibm_api_token"), ibm_crn=ibm_crn,
                    ibm_backend=job.get("ibm_backend_name"),
                    spin=int(job.get("spin", 0) or 0), phase_cb=_phase,
                )
            except Exception as _samp_err:
                # A cloud sampler (IBM/BlueQubit) can fail on a bad/expired token, no
                # available QPU, queue/network errors, etc. The correlated SQD result does
                # NOT depend on WHERE the state is sampled — so fall back to local
                # statevector sampling (+ GPU det_ci) rather than failing the whole job.
                if _samp != "statevector":
                    logger.warning("SQD %s sampling failed (%s); falling back to statevector sampling",
                                   _samp, _samp_err)
                    _phase("sampling_fallback",
                           f"{_samp.upper()} sampling unavailable ({str(_samp_err)[:90]}) — "
                           "falling back to local statevector sampling")
                    sol_result = _run_sampling_sqd(
                        atoms, basis, charge, "statevector", gpu_device,
                        ibm_token=None, ibm_crn=None, ibm_backend=None,
                        spin=int(job.get("spin", 0) or 0), phase_cb=_phase,
                    )
                else:
                    raise
        elif solver_type == "sqd":
            sol_result = _run_sqd(atoms, basis, charge)
        elif solver_type == "krylov_sqd":
            sol_result = _run_krylov_sqd(atoms, basis, charge)
        elif solver_type == "vqe":
            sol_result = _run_vqe(
                atoms, basis, charge, backend,
                max_iterations, ansatz_type, callback=_solver_cb,
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

        # Authoritative evaluation count for iterative (VQE-family) solvers: the
        # SolverResult exposes `iterations`, not always `n_evaluations`, so the
        # runners often leave it None → the UI showed 0/garbage. Use the count the
        # solver actually streamed through the callback. SQD has no per-iteration
        # loop (max_iter stays 0) → n_evaluations stays None (UI shows determinants).
        if _prog_state["max_iter"] and not sol_result.get("n_evaluations"):
            sol_result["n_evaluations"] = _prog_state["max_iter"]
        result.update(sol_result)
        # Compute error_mha if we have both energy and fci_energy
        if result.get("energy") and result.get("fci_energy") and result.get("error_mha") is None:
            result["error_mha"] = abs(result["energy"] - result["fci_energy"]) * 1000
        result["status"] = "completed"

    except JobCancelled:
        logger.info("Calculation cancelled by user")
        result["status"] = "cancelled"
        result["error_message"] = "Cancelled by user"
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        result["status"] = "failed"
        result["error_message"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["wall_time_ms"] = int((time.time() - t0) * 1000)
    return result


# ─────────── Workflow runners (whole-experiment, not single-point) ───────────


def _run_dynamics(job: dict, gpu_device: str = "auto", progress_cb=None, cancel_check=None) -> dict:
    """Run a whole molecular-dynamics trajectory on this node (one node job, not per-step
    dispatch). Correlated (SQD/VQE) forces use the capability-protocol path — a ForceProvider
    solver whose energy_fn() is finite-differenced for a Pulay-correct gradient — NOT the
    legacy frozen-theta 'sqd' force_method (wrong off-equilibrium). Classical hf/mp2 forces
    use PySCF gradients directly. Frames are streamed via progress_cb and returned (<=50)."""
    from kanad.dynamics import MDSimulator

    atoms = job["atoms"]
    basis = job.get("basis", "sto-3g")
    charge = job.get("charge", 0)
    spin = int(job.get("spin", 0) or 0)
    cfg = job.get("config") or {}

    force_method = str(cfg.get("force_method", "hf")).lower()
    n_steps = int(cfg.get("n_steps", 100))
    timestep = float(cfg.get("timestep", 0.5))
    temperature = float(cfg.get("temperature", 300.0))
    integrator = cfg.get("integrator", "velocity_verlet")
    thermostat = cfg.get("thermostat", "berendsen")
    if thermostat in (None, "none", "None"):
        thermostat = None
    save_frequency = int(cfg.get("save_frequency", 10))
    equilibrate = bool(cfg.get("equilibrate", False))
    n_equil_steps = int(cfg.get("n_equil_steps", 0))

    molecule = _build_molecule(atoms, basis, charge, spin)

    md_kwargs: dict[str, Any] = dict(
        temperature=temperature, timestep=timestep,
        integrator=integrator, thermostat=thermostat,
    )
    force_label = force_method
    if force_method in ("sqd", "sampling_sqd", "solver", "hivqe"):
        from kanad.solvers import SamplingSQDSolver
        _sqd = SamplingSQDSolver(
            molecule.hamiltonian,
            n_samples=int(cfg.get("shots") or 4000),
            backend="statevector", cisd_seed=True, recover_configurations=True,
        )
        md_kwargs["force_method"] = "solver"
        md_kwargs["solver"] = _sqd
        force_label = "sqd (statevector sampling + GPU det_ci)"
    elif force_method in ("vqe", "physics_vqe"):
        from kanad.solvers import PhysicsVQE
        _pv = PhysicsVQE(molecule.hamiltonian, max_excitations=int(cfg.get("max_excitations", 5)))
        md_kwargs["force_method"] = "solver"
        md_kwargs["solver"] = _pv
        force_label = "physics_vqe"
    else:
        md_kwargs["force_method"] = force_method  # 'hf' / 'mp2' — PySCF gradients

    if cfg.get("environment"):
        md_kwargs["environment"] = cfg["environment"]

    if progress_cb:
        try:
            progress_cb({"phase": "dynamics",
                         "message": f"MD start: {n_steps} steps · {integrator} · {force_label} forces"})
        except Exception:
            pass

    md = MDSimulator(molecule, **md_kwargs)

    def _tolist(x):
        try:
            return x.tolist() if hasattr(x, "tolist") else list(x)
        except Exception:
            return x

    frames: list[dict] = []

    def _on_frame(step, total, positions, temperature_, total_energy, ke, pe, forces):
        if cancel_check and cancel_check():
            raise JobCancelled()
        frames.append({
            "step": int(step),
            "positions": _tolist(positions),
            "temperature": float(temperature_) if temperature_ is not None else None,
            "total_energy": float(total_energy) if total_energy is not None else None,
            "kinetic_energy": float(ke) if ke is not None else None,
            "potential_energy": float(pe) if pe is not None else None,
            "forces": _tolist(forces),
        })
        if progress_cb:
            try:
                progress_cb({"phase": "dynamics", "iteration": int(step), "max_iterations": int(total),
                             "energy": float(total_energy) if total_energy is not None else None,
                             "message": f"MD step {step}/{total}" +
                                        (f" · T={temperature_:.0f}K" if temperature_ is not None else "")})
            except Exception:
                pass

    result = md.run(
        n_steps=n_steps, save_frequency=save_frequency,
        equilibrate=equilibrate, n_equil_steps=n_equil_steps, on_frame=_on_frame,
    )

    if len(frames) > 50:
        stride = max(1, len(frames) // 50)
        traj = frames[::stride][:50]
    else:
        traj = frames

    def _f(v):
        try:
            return float(v)
        except Exception:
            return None

    return {
        "kind": "dynamics",
        "trajectory": traj,
        "n_frames_total": len(frames),
        "avg_temperature": _f(getattr(result, "avg_temperature", None)),
        "temperature_std": _f(getattr(result, "temperature_std", None)),
        "avg_total_energy": _f(getattr(result, "avg_total_energy", None)),
        "energy_drift": _f(getattr(result, "energy_drift", None)),
        "n_steps_completed": int(getattr(result, "n_steps_completed", 0) or 0),
        "force_method": force_label,
        "solver_used": f"kanad_compute:md:{force_label}",
    }


def _run_photodynamics(job: dict, gpu_device: str = "auto", progress_cb=None, cancel_check=None) -> dict:
    """Run a laser–matter photodynamics simulation (diatomic, single electronic manifold)
    on this node. Honest scope: the framework has no non-adiabatic couplings yet, so this
    is population dynamics on adiabatic surfaces (coherent control), not true surface-hopping
    photochemistry. use_quantum routes excited states through qEOM-VQE locally."""
    from kanad.dynamics.photodynamics import PhotodynamicsSimulator, LaserField

    atoms = job["atoms"]
    basis = job.get("basis", "sto-3g")
    charge = job.get("charge", 0)
    cfg = job.get("config") or {}
    if len(atoms) != 2:
        raise ValueError("Photodynamics supports diatomic (2-atom) systems only")

    # PhotodynamicsSimulator needs bond.atom_1/atom_2, which only covalent bonds expose —
    # an auto-classified IONIC diatomic (e.g. HF) crashes with "no attribute 'atom_1'".
    # The electronic H0 is identical; bond type only steers ansatz selection, which the
    # photodynamics state diagonalization does not use. So force covalent.
    import numpy as _np
    from kanad import BondFactory
    _d = float(_np.linalg.norm(_np.array(atoms[1]["position"], dtype=float)
                               - _np.array(atoms[0]["position"], dtype=float)))
    bond = BondFactory.create_bond(atoms[0]["symbol"], atoms[1]["symbol"],
                                   bond_type="covalent", distance=_d, charge=charge)

    _env_map = {"continuous": "cw", "cw": "cw", "gaussian": "gaussian", "sin2": "sin2",
                "chirped": "chirped", "sech": "sech", "rectangular": "rectangular"}
    envelope = _env_map.get(str(cfg.get("envelope", "gaussian")).lower(), "gaussian")
    laser = LaserField(
        intensity=float(cfg.get("intensity", 1e12)),
        wavelength=float(cfg.get("wavelength", 200.0)),
        polarization=cfg.get("polarization", [0, 0, 1]),
        pulse_duration=float(cfg.get("pulse_duration", 50.0)),
        envelope=envelope,
    )
    photo_kwargs: dict[str, Any] = {
        "laser_field": laser,
        "n_states": int(cfg.get("n_states", 3)),
        "propagator": cfg.get("propagator", "rk4"),
    }
    use_quantum = bool(cfg.get("use_quantum", False))
    if use_quantum:
        photo_kwargs["use_quantum"] = True

    if progress_cb:
        try:
            progress_cb({"phase": "photodynamics",
                         "message": f"Laser {cfg.get('wavelength', 200)}nm · "
                                    f"{'qEOM-VQE' if use_quantum else 'classical'} · propagating"})
        except Exception:
            pass
    if cancel_check and cancel_check():
        raise JobCancelled()

    sim = PhotodynamicsSimulator(bond, **photo_kwargs)
    total_time = float(cfg.get("total_time", 200.0))
    dt = float(cfg.get("dt", 0.1))
    result = sim.run(total_time=total_time, dt=dt)

    if cancel_check and cancel_check():
        raise JobCancelled()

    total_points = len(result.times)
    step = max(1, total_points // 500) if total_points > 500 else 1
    idx = list(range(0, total_points, step))
    return {
        "kind": "photodynamics",
        "times": [float(result.times[i]) for i in idx],
        "populations": [result.populations[i].tolist() for i in idx],
        "energies": [float(result.energies[i]) for i in idx],
        "field_amplitudes": [float(result.field_amplitudes[i]) for i in idx],
        "excitation_probability": float(result.excitation_probability),
        "final_populations": result.final_population.tolist(),
        "n_steps": total_points,
        "solver_used": f"kanad_compute:photodynamics:{'qeom-vqe' if use_quantum else 'classical'}",
    }


def _materials_bands_dos(xtal_atoms, lattice, kpts, compute_bands, compute_dos) -> dict:
    """Periodic SCF + DOS + band structure + gap for a crystal (runs on the node GPU/CPU).
    Mirrors the app's in-process _compute_bands_dos so results are identical, just offloaded."""
    import numpy as np
    res: dict = {}

    def _set_gap(ph):
        try:
            gap = ph.get_band_gap()
            res["band_gap_ev"] = float(gap["gap"])
            res["band_gap_type"] = gap.get("type")
            res["band_gap_method"] = gap.get("method")
            res["band_gap_metallic"] = bool(gap.get("metallic"))
            res["band_gap_gamma_only"] = bool(gap.get("gamma_only"))
            res["band_gap_caveat"] = gap.get("caveat")
        except Exception as ge:
            res.setdefault("band_gap_error", str(ge))

    try:
        from kanad.core.hamiltonians.periodic_hamiltonian import PeriodicHamiltonian
        ph = PeriodicHamiltonian(xtal_atoms, lattice)
        ph.solve_scf()
        # Gap + DOS both read the SCF-mesh band_energies, so compute them BEFORE
        # compute_band_structure overwrites those with high-symmetry-path energies. The
        # SCF-mesh gap is always available (no kpath needed) — so binaries like NaCl whose
        # get_kpath returns no path still get a gap instead of None.
        if compute_bands:
            _set_gap(ph)
        if compute_dos:
            try:
                from kanad.analysis.dos_calculator import DOSCalculator
                dos_calc = DOSCalculator(ph)
                be_eV = np.asarray(ph.band_energies) * 27.2114
                dos_res = dos_calc.compute_dos(
                    energy_range=(float(be_eV.min()) - 3.0, float(be_eV.max()) + 3.0),
                    n_points=400, sigma=0.15, units='eV')
                res["dos_energies"] = [float(e) for e in np.asarray(dos_res["energies"]).tolist()]
                res["dos_values"] = [float(d) for d in np.asarray(dos_res["dos"]).tolist()]
                if dos_res.get("fermi_energy") is not None:
                    res["dos_fermi_ev"] = float(dos_res["fermi_energy"])
            except Exception as de:
                res["dos_error"] = str(de)
        if compute_bands and kpts is not None:
            bs = ph.compute_band_structure(kpts)
            res["band_structure"] = np.asarray(bs["band_energies"]).tolist()
            res["band_kdistances"] = [float(d) for d in np.asarray(bs["k_distances"]).tolist()]
            if bs.get("labels"):
                res["kpath_labels"] = list(bs["labels"])
                res["kpath_distances"] = [float(d) for d in np.asarray(bs.get("label_positions") or []).tolist()]
            _set_gap(ph)  # refine along the high-symmetry path when one is available
    except Exception as be:
        res["band_error"] = str(be)
    return res


def _run_materials(job: dict, gpu_device: str = "auto", progress_cb=None, cancel_check=None) -> dict:
    """Build a periodic crystal + (optionally) its band structure/DOS/gap on this node.
    Offloads the heavy periodic HF SCF to the node GPU/CPU. Uses the same kanad crystal
    builder + PeriodicHamiltonian the app uses in-process, so the science is identical."""
    import numpy as np
    from kanad.core.io import build_crystal, build_binary_crystal, get_kpath

    cfg = job.get("config") or {}
    element = cfg["element"]
    element_b = cfg.get("element_b")
    lattice_type = cfg.get("lattice_type", "bcc")
    a = float(cfg["lattice_constant"])
    size = tuple(cfg.get("size") or (1, 1, 1))
    n_kpoints = int(cfg.get("n_kpoints", 30))
    compute_bands = bool(cfg.get("compute_bands", False))
    compute_dos = bool(cfg.get("compute_dos", False))

    if progress_cb:
        try:
            progress_cb({"phase": "materials",
                         "message": f"Building {element}{element_b or ''} {lattice_type} (a={a} A)"})
        except Exception:
            pass

    if element_b:
        xtal = build_binary_crystal(element, element_b, lattice_type, a, size=size)
    else:
        xtal = build_crystal(element, lattice_type, a, size=size)

    atoms = [{"symbol": at.symbol, "position": [float(x) for x in at.position]} for at in xtal.atoms]
    lattice = getattr(xtal, "lattice", None)
    lattice_vectors = None
    for attr in ("lattice_vectors", "vectors", "matrix", "a_vectors"):
        if lattice is not None and hasattr(lattice, attr):
            try:
                lattice_vectors = np.asarray(getattr(lattice, attr)).tolist()
                break
            except Exception:
                pass
    try:
        kpts, klabels, kdist = get_kpath(lattice_type, n_points=n_kpoints)
    except Exception:
        kpts, klabels, kdist = None, [], []

    out = {
        "kind": "materials",
        "formula": getattr(xtal, "formula", None),
        "n_atoms": len(atoms),
        "atoms": atoms,
        "lattice_type": lattice_type,
        "lattice_constant": a,
        "lattice_vectors": lattice_vectors,
        "kpath_labels": list(klabels),
        "kpath_distances": [float(d) for d in (kdist or [])],
        "solver_used": "kanad_compute:materials:periodic_hf",
    }
    if (compute_bands or compute_dos) and lattice is not None:
        if cancel_check and cancel_check():
            raise JobCancelled()
        if progress_cb:
            try:
                progress_cb({"phase": "materials", "message": "Periodic HF SCF + bands/DOS on node"})
            except Exception:
                pass
        out.update(_materials_bands_dos(xtal.atoms, lattice, kpts, compute_bands, compute_dos))
    return out


# ─────────── Solver implementations ───────────


def _run_physics_vqe(
    atoms, basis, charge, backend, max_iter, max_exc,
    ibm_token=None, ionq_key=None, callback=None,
) -> dict:
    from kanad.solvers import PhysicsVQE

    mol = _build_pyscf_mol(atoms, basis, charge)
    kwargs: dict[str, Any] = {"pyscf_mol": mol, "max_excitations": max_exc}

    if backend not in ("statevector", "aer"):
        kwargs["backend"] = backend
    # PhysicsVQE takes `cloud_credentials` (a dict), NOT ibm_api_token/ionq_api_key —
    # and only needs them on an actual cloud backend. On the compute node it runs
    # locally (statevector/planck), so creds are irrelevant. Passing them as the wrong
    # kwarg crashed physics_vqe whenever the user had IBM creds set (now sent on every
    # request under BYOS); only SQD genuinely needs the QPU credentials.
    if backend not in ("statevector", "aer", "planck"):
        _creds = {}
        if ibm_token:
            _creds["ibm_api_token"] = ibm_token
        if ionq_key:
            _creds["ionq_api_key"] = ionq_key
        if _creds:
            kwargs["cloud_credentials"] = _creds

    solver = PhysicsVQE(**kwargs)
    # PhysicsVQE.solve() signature varies by framework version — pass a progress
    # callback only if it actually accepts one (older/other solvers don't).
    import inspect as _inspect
    try:
        _accepts_cb = callback is not None and "callback" in _inspect.signature(solver.solve).parameters
    except (TypeError, ValueError):
        _accepts_cb = False
    res = solver.solve(callback=callback) if _accepts_cb else solver.solve()

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


def _run_hardware_vqe(atoms, basis, charge, backend, ibm_token=None, callback=None) -> dict:
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


def _build_correlated_ansatz(ham, n_layers: int = 1, seed: int = 0):
    """A correlated (entangling) seed circuit for QPU SQD sampling, returned as a
    Qiskit circuit. The selected-CI does the variational work, so a random-bound
    HEA suffices to spread amplitude across the dominant determinants (and it
    passes the solver's no-entangling guard). build_circuit() returns kanad's
    QuantumCircuit (get_num_parameters/bind_parameters/to_qiskit, not the Qiskit API)."""
    import numpy as np
    from kanad.core.ansatze import HardwareEfficientAnsatz
    n_qubits = 2 * ham.n_orbitals
    ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_electrons=ham.n_electrons, n_layers=n_layers)
    qc = ansatz.build_circuit()
    nparams = qc.get_num_parameters() if hasattr(qc, "get_num_parameters") else 0
    if nparams > 0:
        rng = np.random.default_rng(seed)
        vals = list(rng.uniform(-0.3, 0.3, nparams))
        try:
            bound = qc.bind_parameters(vals)
            qc = bound if bound is not None else qc
        except Exception:
            pass
    return qc.to_qiskit() if hasattr(qc, "to_qiskit") else qc


def _sqd_ao_1rdm(solver, ham):
    """Full AO-basis correlated 1-RDM from the SQD wavefunction — the genuine quantum
    observable ⟨ψ|a†_p a_q|ψ⟩ — as a plain nested list (JSON-safe), or None on failure.

    Active-space Hamiltonians embed the correlated active block into the full MO space
    (frozen core doubly occupied, virtuals empty) then rotate to AO — handled + trace-
    validated by ActiveHamiltonian.set_quantum_density_matrix (which also does the right
    thing for AVAS/MP2NO rotated orbitals). Full-space Hamiltonians transform the full
    MO 1-RDM directly via the canonical mf.mo_coeff. Validated (LiH full, HF-molecule
    active space): the resulting dipole/charges match FCI, not HF.
    """
    try:
        active_mo_rdm = solver.get_1rdm_active_mo()
        if hasattr(ham, "set_quantum_density_matrix"):
            ham.set_quantum_density_matrix(active_mo_rdm)   # embeds + validates trace = N_elec
            ao = ham._quantum_density_matrix_ao
        else:
            from kanad.core.density.density_storage import mo_to_ao_1rdm
            ao = mo_to_ao_1rdm(active_mo_rdm, ham.mf.mo_coeff)
        return [[float(x) for x in row] for row in ao]
    except Exception as e:
        logger.warning("[SQD] quantum 1-RDM unavailable (%s); the app will fall back to HF for properties", e)
        return None


def _run_sampling_sqd(atoms, basis, charge, sampling_backend, gpu_device,
                      ibm_token=None, ibm_crn=None, ibm_backend=None, n_samples=4000, spin=0,
                      rounds=3, expansion_per_round=24, energy_tol=1e-4, phase_cb=None) -> dict:
    """The flagship hybrid SQD: quantum sampling (QPU/statevector) + classical
    diagonalization on the GPU node via rocm-planck det_ci (gpu_device).
    Uses a kanad Molecule so polyatomic + open-shell (spin) systems work.
    Iterative subspace expansion (rounds/expansion_per_round/energy_tol) grows the
    determinant subspace toward FCI for multireference accuracy.

    Config rationale: cisd_seed already seeds a full CISD subspace (e.g. ~16k dets
    for C2/sto-3g, diagonalized in ~9s on the GPU), so each expansion round only
    needs to graft the *important* higher excitations. expansion_per_round=24 keeps
    that growth bounded (a few rounds converge via energy_tol instead of ballooning
    the subspace toward the full FCI dimension), and rounds=3 caps the worst case.
    An aggressive 100/round flooded the subspace and made strongly-multireference
    cases (C2/O2) run for many minutes for sub-mHa gains."""
    from kanad.solvers.sampling_sqd import SamplingSQDSolver

    # Emit a workflow step to BOTH the node log and the app's live progress/log stream.
    # Distinct phase names give the app a clean timeline (preparing → active_space →
    # ansatz → submitting → queued → sampling → diagonalizing → done) instead of a
    # single opaque "running".
    def _emit(message, phase="sampling", **extra):
        logger.info("[SQD] %s", message)
        if phase_cb:
            try:
                phase_cb({"phase": phase, "message": message, **extra})
            except Exception:
                pass

    _samp_label = {"ibm": "IBM Quantum (QPU)", "bluequbit": "BlueQubit"}.get(sampling_backend, "statevector")
    _emit(f"Preparing system: {len(atoms)} atom(s), basis {basis}, charge {charge}, 2S={spin}", phase="preparing")

    # Large / strongly-correlated systems (transition-metal dimers & clusters) get an
    # automatic valence active-space reduction so the sampled subspace + det_ci stay
    # tractable; small molecules (≤24q full) run the full space unchanged.
    as_ham, as_info = _maybe_reduce_active_space(atoms, basis, charge, spin)
    if as_ham is not None:
        ham = as_ham
        # A valence active space is, by construction, the strongly-correlated core of the
        # system — its CASDI subspace is far denser than a small diatomic's, so the
        # 24/round expansion tuned for C2/O2 balloons it and stalls. Tamer rounds keep
        # these tractable (validated: Cr2/Fe2 (10e,10o) in ~2-5s on the GPU).
        rounds = min(rounds, 2)
        expansion_per_round = min(expansion_per_round, 12)
        _emit(f"Active space: {as_info['full_qubits']}→{as_info['active_qubits']} qubits "
              f"({as_info['active_electrons']}e,{as_info['active_orbitals']}o) via {as_info['method']}"
              f"  [SCF converged={as_info['scf_converged']}]", phase="active_space")
    else:
        mol = _build_molecule(atoms, basis, charge, spin)
        ham = mol.hamiltonian
        _emit(f"Full-space Hamiltonian: {2 * ham.n_orbitals} qubits ({ham.n_electrons}e, {ham.n_orbitals}o)",
              phase="preparing")

    solver = SamplingSQDSolver(
        ham, n_samples=n_samples, backend=sampling_backend,
        recover_configurations=True, ci_backend="pyscf", target_sz=spin / 2.0, random_seed=0,
        # CISD completeness seed: union the sampled subspace with HF + all
        # (N,Sz)-preserving singles & doubles so the diagonalized subspace captures
        # real correlation even when the sample is HF-dominated (robust default).
        cisd_seed=True,
        # ibm_backend_name = the QPU the user picked in the app (None → auto least-busy).
        ibm_api_token=ibm_token, ibm_crn=ibm_crn, ibm_backend_name=ibm_backend, gpu_device=gpu_device,
    )

    # QPU/cloud sampling needs a correlated ansatz (the solver refuses a vacuous
    # HF circuit there). On statevector, cisd_seed already guarantees a correlated
    # subspace, so HF sampling + CISD expansion suffices (no ansatz needed).
    if sampling_backend in ("ibm", "bluequbit"):
        _emit(f"Building correlated ansatz for {2 * ham.n_orbitals}-qubit {_samp_label} sampling", phase="ansatz")
        ansatz_circuit = _build_correlated_ansatz(ham)
    else:
        ansatz_circuit = None

    # Surface the real IBM job id once the QPU job is submitted (the framework fires
    # this from inside the sampler, after transpiling + queue submission).
    def _on_ibm_submit(info):
        _emit(f"IBM job {info.get('job_id')} submitted on {info.get('backend')} "
              f"({info.get('n_qubits')}q · depth {info.get('depth')} · {info.get('n_2q')} 2q-gates) — waiting in QPU queue",
              phase="queued", ibm_job_id=info.get("job_id"), ibm_backend=info.get("backend"))
    try:
        solver._on_ibm_submit = _on_ibm_submit
    except Exception:
        pass

    if sampling_backend == "ibm":
        _emit(f"Submitting {n_samples}-shot sampling job to IBM Quantum "
              f"({ibm_backend or 'auto least-busy QPU'}) — transpiling + queueing", phase="submitting")
    else:
        _emit(f"Sampling {n_samples} shots on {_samp_label}, then GPU det_ci diagonalization", phase="sampling")
    # ITERATIVE subspace expansion (multireference): grow the determinant subspace
    # over rounds — each round rediagonalized on the GPU det_ci — toward FCI.
    # Single-shot SQD under-captures strong static correlation (stretched bonds,
    # diradicals, multireference); this is variational so energy only improves.
    # NB: sampling + configuration-recovery + GPU det_ci all happen inside
    # solve_iterative() as one blocking call; the QPU-submission point is surfaced
    # mid-way via _on_ibm_submit ("queued"). We emit "complete" once it returns —
    # no premature "diagonalizing" line (it would claim sampling was done before the
    # QPU job had even been submitted).
    res = solver.solve_iterative(
        ansatz_circuit=ansatz_circuit,
        max_iterations=int(rounds), expansion_per_round=int(expansion_per_round),
        energy_tol=float(energy_tol),
    ).to_dict()
    # Device the subspace diagonalization (det_ci) ran on: 'amd'/'nvidia' (GPU) or 'cpu'.
    dev = getattr(solver, "_diag_device_used", None) or res.get("device_used") or "cpu"
    # Correlated AO-basis 1-RDM — the genuine quantum observable ⟨ψ|a†_p a_q|ψ⟩ from the
    # SQD wavefunction. Returned so the app builds dipole/atomic-charges/bond-order/energy
    # decomposition from the CORRELATED density, not a throwaway HF SCF. None → app uses HF.
    quantum_1rdm_ao = _sqd_ao_1rdm(solver, ham)
    if quantum_1rdm_ao is not None:
        _emit("Built correlated 1-RDM (AO basis) — property analysis will use quantum observables",
              phase="observables")
    _emit(f"Done: E = {float(res['energy']):.6f} Ha over "
          f"{int(res.get('n_determinants') or len(res.get('determinants') or []) or 0)} determinants "
          f"({int(res.get('iterations_done') or 0)} expansion round(s), det_ci on {dev})", phase="complete")
    # Subspace-expansion energy trace (per round) → the results convergence chart.
    _hist = res.get("energy_history") or ([res["energy"]] if res.get("energy") is not None else [])
    convergence = [{"iteration": i, "energy": float(e)} for i, e in enumerate(_hist)]
    return {
        "energy": float(res["energy"]),
        "fci_energy": res.get("fci_energy"),
        "error_mha": (float(res["error_mha"]) if res.get("error_mha") is not None else None),
        # solve_iterative grows a `determinants` list across expansion rounds but doesn't
        # always restamp n_determinants — fall back to the final subspace size so the UI's
        # "Determinants" metric is populated (not blank) for the hybrid SQD path.
        "n_determinants": int(res.get("n_determinants") or len(res.get("determinants") or []) or 0),
        "n_iterations": int(res.get("iterations_done") or 0),
        "convergence_history": convergence or None,
        "energy_history": [float(e) for e in _hist] or None,
        "converged": True,
        "solver_used": f"sampling_sqd[{dev}]",
        # When a valence active space was applied (TM dimers/clusters), report it so
        # the result is honest about what was actually diagonalized (vs the full system).
        "active_space": as_info,
        # Correlated AO 1-RDM (nested list, ~n_ao²) → the app's analysis uses this
        # quantum density for dipole/charges/bond-order/energy-decomposition.
        "quantum_1rdm_ao": quantum_1rdm_ao,
    }


_VQE_OK_ANSATZE = {'hardware_efficient', 'real_amplitudes', 'efficient_su2', 'two_local'}


def _run_vqe(atoms, basis, charge, backend, max_iter, ansatz_type,
             optimizer=None, mapper_type=None, callback=None) -> dict:
    from kanad.solvers import VQESolver
    bond = _build_bond(atoms, basis, charge)
    # Framework 0.1.2 VQESolver rejects physics_driven/governance/ucc/* — clamp any
    # unsupported type (incl. the schema default 'physics_driven') so the node never crashes.
    if ansatz_type not in _VQE_OK_ANSATZE:
        ansatz_type = 'hardware_efficient'
    kw = dict(ansatz_type=ansatz_type, backend=backend, max_iterations=max_iter)
    if optimizer:
        kw["optimizer"] = optimizer
    if mapper_type:
        kw["mapper_type"] = mapper_type
    if callback:
        kw["callback"] = callback
    solver = VQESolver(bond, **kw)
    res = solver.solve()
    # VQESolver.solve() now returns a SolverResult (unified protocol), not a dict —
    # flatten via to_dict() (also tolerates a legacy dict return).
    d = res.to_dict() if hasattr(res, "to_dict") else res
    return {
        "energy": float(d["energy"]),
        "n_evaluations": int(d.get("n_evaluations", 0) or 0),
        "converged": bool(d.get("converged", True)),
    }


# Workshop custom solver: base_type -> framework solver (mirrors the app's
# custom_solver.BASE_TYPE_TO_SOLVER). Config is config-based (sanitized kwargs),
# never arbitrary code, so the node just runs the mapped solver with the params.
_CUSTOM_BASE_TO_SOLVER = {
    "vqe": "vqe", "physics": "physics_vqe", "subspace": "sampling_sqd",
    "time_evolution": "varqite", "custom": "vqe",
}


def _run_custom_solver(atoms, basis, charge, base_type, config, backend, gpu_device,
                       spin=0, ibm_token=None, ibm_crn=None, callback=None, phase_cb=None) -> dict:
    """Execute a Workshop custom solver on the node by routing its base_type to the
    matching framework runner and applying the (sanitized) custom config."""
    cfg = config or {}
    bt = (base_type or "vqe").lower()
    if bt == "physics":
        r = _run_physics_vqe(atoms, basis, charge, backend,
                             int(cfg.get("max_iterations", 100)), int(cfg.get("max_excitations", 5)),
                             ibm_token=ibm_token, callback=callback)
    elif bt == "subspace":
        r = _run_sampling_sqd(atoms, basis, charge,
                              ("ibm" if backend == "ibm" else "statevector"), gpu_device,
                              ibm_token=ibm_token, ibm_crn=ibm_crn,
                              n_samples=int(cfg.get("n_samples", 4000)), spin=spin, phase_cb=phase_cb)
    elif bt == "time_evolution":
        r = _run_varqite(atoms, basis, charge, cfg.get("ansatz_type", "hardware_efficient"))
    else:  # vqe / custom
        r = _run_vqe(atoms, basis, charge, backend, int(cfg.get("max_iterations", 100)),
                     cfg.get("ansatz_type", "hardware_efficient"),
                     optimizer=cfg.get("optimizer"), mapper_type=cfg.get("mapper_type"), callback=callback)
    r["solver_used"] = "custom:%s[%s]" % (bt, r.get("solver_used") or backend)
    return r


def _run_varqite(atoms, basis, charge, ansatz_type) -> dict:
    from kanad.solvers import VarQITESolver
    bond = _build_bond(atoms, basis, charge)
    solver = VarQITESolver(bond=bond)
    # max_tau=2.0 converges H2 to ~0 mHa in ~50s; the default (10.0) runs the adaptive
    # integrator for minutes. Keep the dtau fine (0.1) — the fixed-step path is unstable.
    res = solver.solve(max_tau=2.0, dtau=0.1)
    return {
        "energy": float(res.energy),
        "converged": bool(getattr(res, "converged", True)),
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
