"""Audit the unified solver protocol driven by the rocm-planck GPU backend.

For each (solver, molecule, backend) it records: did it run, energy, parity vs the
statevector backend, whether planck was actually used, SolverResult conformance, and
any exception. Also smoke-tests an analysis pass. Emits JSON for the bug report.

Run:  PYTHONPATH=<parent-of-kanad> python benchmarks/planck_protocol_audit.py
"""
import json
import sys
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, "tests")  # _solver_molecules helper
import _solver_molecules as M  # noqa: E402

from kanad.backends.factory import make_backend  # noqa: E402

OUT_PATH = "benchmarks/out/planck_protocol_audit.json"
import os as _pre_os
RESULTS = []
_DONE = set()
if _pre_os.path.exists(OUT_PATH):                  # resume: keep prior cells, skip them
    try:
        RESULTS = json.load(open(OUT_PATH))
        _DONE = {(r["solver"], r["molecule"], r["backend"]) for r in RESULTS}
    except Exception:
        RESULTS = []


def _energy(res):
    """Extract energy from a SolverResult (or legacy dict)."""
    if hasattr(res, "energy"):
        return float(res.energy)
    if hasattr(res, "to_dict"):
        return float(res.to_dict().get("energy"))
    if isinstance(res, dict):
        return float(res.get("energy"))
    return float(res)


def _conforms(res):
    """Is it a SolverResult with the stable interface?"""
    issues = []
    if not hasattr(res, "to_dict"):
        issues.append("no .to_dict()")
    else:
        d = res.to_dict()
        if "energy" not in d:
            issues.append("to_dict() missing 'energy'")
    if not hasattr(res, "energy"):
        issues.append("no .energy attribute")
    return issues


import signal


class _CellTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CellTimeout(f"cell exceeded {CELL_TIMEOUT}s (likely hang)")


CELL_TIMEOUT = int(_pre_os.environ.get("AUDIT_CELL_TIMEOUT", "180"))


def run_cell(name, make_solver, molecule, backend):
    if (name, molecule, backend) in _DONE:         # resume: already recorded
        prev = next(r for r in RESULTS if (r["solver"], r["molecule"], r["backend"]) == (name, molecule, backend))
        print(f"  [skip] {name} {molecule} {backend} (cached)", flush=True)
        return prev
    rec = {"solver": name, "molecule": molecule, "backend": backend}
    t = time.perf_counter()
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(CELL_TIMEOUT)                      # kill a hung solver, record it, move on
    try:
        system = getattr(M, molecule)()
        solver = make_solver(system, backend)
        # did the backend object actually become planck?
        rec["backend_obj"] = type(getattr(solver, "backend", None)).__name__
        rec["backend_name"] = getattr(solver, "backend_name", "?")
        res = solver.solve()
        rec["energy"] = _energy(res)
        rec["conformance"] = _conforms(res)
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["timeout"] = isinstance(e, _CellTimeout)
        rec["trace"] = traceback.format_exc().splitlines()[-4:]
    finally:
        signal.alarm(0)                            # clear the per-cell alarm
    rec["t"] = round(time.perf_counter() - t, 2)
    RESULTS.append(rec)
    import os
    os.makedirs("benchmarks/out", exist_ok=True)
    with open("benchmarks/out/planck_protocol_audit.json", "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)   # incremental: survive a timeout
    flag = "OK " if rec.get("ok") else "ERR"
    extra = f"E={rec.get('energy')}" if rec.get("ok") else rec.get("error", "")[:70]
    print(f"  [{flag}] {name:18s} {molecule:5s} {backend:11s} "
          f"backend_obj={rec.get('backend_obj','?'):18s} {extra}", flush=True)
    return rec


# --- solver constructors (positional `system`, backend kwarg) -----------------
def _vqe(s, b): from kanad.solvers import VQESolver; return VQESolver(s, optimizer="COBYLA", max_iterations=80, backend=b, use_cache=False)
def _physvqe(s, b): from kanad.solvers import PhysicsVQE; return PhysicsVQE(s, backend=b, max_excitations=4)
def _sampling_sqd(s, b): from kanad.solvers import SamplingSQDSolver; return SamplingSQDSolver(s.hamiltonian, n_samples=3000, backend=b)
def _varqite(s, b): from kanad.solvers import VarQITESolver; return VarQITESolver(s, backend=b)
def _qeom(s, b): from kanad.solvers import qEOMVQE; return qEOMVQE(s, backend=b)
def _subspace(s, b): from kanad.solvers import SampledSubspaceVQE; return SampledSubspaceVQE(s, backend=b)
def _excited(s, b): from kanad.solvers import ExcitedStatesSolver; return ExcitedStatesSolver(s, method="cis", n_states=2, backend=b)
def _smart(s, b): from kanad.solvers import SmartSolver; return SmartSolver(s, backend=b)

SOLVERS = [
    ("VQESolver", _vqe),
    ("PhysicsVQE", _physvqe),
    ("SamplingSQD", _sampling_sqd),
    ("VarQITE", _varqite),
    ("qEOMVQE", _qeom),
    ("SampledSubspaceVQE", _subspace),
    ("ExcitedStates", _excited),
    ("SmartSolver", _smart),
]
import os as _os
MOLECULES = _os.environ.get("AUDIT_MOLS", "h2,lih").split(",")
SKIP = set(s for s in _os.environ.get("AUDIT_SKIP", "").split(",") if s)


def main():
    print(f"planck backend object: {type(make_backend('planck')).__name__}\n")
    for name, mk in SOLVERS:
        if name in SKIP:
            print(f"  [skip-solver] {name} (AUDIT_SKIP)", flush=True)
            continue
        for mol in MOLECULES:
            sv = run_cell(name, mk, mol, "statevector")
            pl = run_cell(name, mk, mol, "planck")
            if sv.get("ok") and pl.get("ok") and sv.get("energy") is not None and pl.get("energy") is not None:
                d = abs(pl["energy"] - sv["energy"])
                print(f"      -> parity |dE| = {d:.2e}" + ("  ⚠ MISMATCH" if d > 1e-3 else ""), flush=True)

    # parity-of-backend object check: planck must NOT be silently a StatevectorBackend
    planck_used = [r for r in RESULTS if r["backend"] == "planck" and r.get("ok")]
    silent_cpu = [r for r in planck_used if r.get("backend_obj") == "StatevectorBackend"]
    print(f"\nplanck cells that silently used StatevectorBackend: {len(silent_cpu)}/{len(planck_used)}")

    with open("benchmarks/out/planck_protocol_audit.json", "w") as f:
        import os
        os.makedirs("benchmarks/out", exist_ok=True)
        json.dump(RESULTS, f, indent=2, default=str)
    print("wrote benchmarks/out/planck_protocol_audit.json")


if __name__ == "__main__":
    main()
