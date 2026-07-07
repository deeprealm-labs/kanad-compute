"""Solver + ANALYSIS audit (re-test after fixes #2-#5), planck vs statevector.

Runs the unified-protocol solvers on real molecules WITH analysis enabled, and
deeply validates the attached analysis dict (energy decomposition self-consistency,
bond orders, dipole) plus planck<->statevector energy parity and the converged flag.
Also re-checks the specific bug fixes inline (ExcitedStates/H2O no-crash, PhysicsVQE
converged=True at chemical accuracy). This is a solver audit on molecular sims, not a
synthetic-circuit benchmark.

Run:  PYTHONPATH=<parent-of-kanad> python benchmarks/planck_solver_analysis_audit.py
"""
import json
import os
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
OUT = "benchmarks/out/planck_solver_analysis_audit.json"
RESULTS = []


def _save():
    os.makedirs("benchmarks/out", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)


# ---- molecules ---------------------------------------------------------------
def h2():
    from kanad.bonds import BondFactory
    return BondFactory.create_bond("H", "H", distance=0.74)


def lih():
    from kanad.bonds import BondFactory
    return BondFactory.create_bond("Li", "H", distance=1.60)


def h2o():
    from kanad.builder import MolecularBuilder
    return (MolecularBuilder
            .from_atoms([("O", (0, 0, 0.117)), ("H", (0, 0.757, -0.467)),
                         ("H", (0, -0.757, -0.467))])
            .basis("sto-3g").build())


MOLECULES = {"H2": h2, "LiH": lih, "H2O": h2o}


# ---- analysis validation -----------------------------------------------------
def validate_analysis(an):
    """Physical-invariant checks on a BaseSolver analysis dict
    {energy_components, bonding, properties}."""
    c = {}
    if not isinstance(an, dict):
        return {"analysis_present": False}
    c["analysis_present"] = True

    def _real(x):
        return float(np.real(x))

    try:
        ec = an.get("energy_components") or {}
        if ec and ec.get("total") is not None:
            parts = sum(_real(ec.get(k, 0.0)) for k in ("nuclear_repulsion", "one_electron", "two_electron"))
            c["energy_components_self_consistent"] = bool(abs(_real(ec["total"]) - parts) < 1e-3)
    except Exception as e:
        c["energy_components_error"] = f"{type(e).__name__}: {e}"

    try:
        bo = (an.get("bonding") or {}).get("bond_orders")
        if bo is not None:
            vals = list(bo.values()) if isinstance(bo, dict) else np.atleast_1d(bo).tolist()
            arr = np.array([_real(v) for v in vals], dtype=float)
            c["bond_orders_nonneg_finite"] = bool(arr.size == 0 or
                                                   (np.all(arr >= -1e-6) and np.all(np.isfinite(arr))))
    except Exception as e:
        c["bond_orders_error"] = f"{type(e).__name__}: {e}"

    try:
        dm = (an.get("properties") or {}).get("dipole_moment")
        if dm is not None:
            mag = _real(dm) if np.isscalar(dm) or np.ndim(dm) == 0 else float(np.linalg.norm(np.real(dm)))
            c["dipole_nonneg_finite"] = bool(mag >= -1e-9 and np.isfinite(mag))
    except Exception as e:
        c["dipole_error"] = f"{type(e).__name__}: {e}"

    return c


# ---- solver constructors (enable_analysis where supported) -------------------
def _vqe(s, b):
    from kanad.solvers import VQESolver
    return VQESolver(s, backend=b, use_cache=False, enable_analysis=True)


def _physvqe(s, b):
    from kanad.solvers import PhysicsVQE
    return PhysicsVQE(s, backend=b, max_excitations=6)   # no enable_analysis kwarg


def _excited(s, b):
    from kanad.solvers import ExcitedStatesSolver
    return ExcitedStatesSolver(s, method="cis", n_states=3, backend=b)


SOLVERS = [("VQESolver", _vqe), ("PhysicsVQE", _physvqe), ("ExcitedStates", _excited)]


def run_cell(sname, make, mol, backend):
    rec = {"solver": sname, "molecule": mol, "backend": backend}
    t = time.perf_counter()
    try:
        res = make(MOLECULES[mol](), backend)
        out = res.solve()
        d = out.to_dict()
        rec["energy"] = float(out.energy)
        rec["converged"] = d.get("converged")
        rec["hf_energy"] = d.get("hf_energy")
        rec["convergence_warning"] = d.get("convergence_warning")
        rec["method"] = d.get("method")
        rec["analysis_checks"] = validate_analysis(d.get("analysis"))
        rec["n_excited"] = (len(out.states) if getattr(out, "states", None) is not None else None)
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc().splitlines()[-3:]
    rec["t"] = round(time.perf_counter() - t, 2)
    RESULTS.append(rec)
    _save()
    print(f"  [{'OK ' if rec.get('ok') else 'ERR'}] {sname:14s} {mol:4s} {backend:11s} "
          f"E={rec.get('energy')} conv={rec.get('converged')} "
          f"checks={rec.get('analysis_checks')} {rec.get('error','')[:50]}", flush=True)
    return rec


def main():
    for sname, make in SOLVERS:
        print(f"\n=== {sname} ===", flush=True)
        for mol in MOLECULES:
            sv = run_cell(sname, make, mol, "statevector")
            pl = run_cell(sname, make, mol, "planck")
            if sv.get("ok") and pl.get("ok") and sv.get("energy") is not None and pl.get("energy") is not None:
                dE = abs(pl["energy"] - sv["energy"])
                print(f"      -> parity |dE| = {dE:.2e}" + ("  MISMATCH" if dE > 1e-3 else ""), flush=True)
    _save()
    print("\nwrote", OUT)
    print("SOLVER_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
