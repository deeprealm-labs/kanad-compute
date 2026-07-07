"""Photodynamics (quantum, vqe_backend='planck') + analysis-workflow validation on GPU.

Deeply examines returned values: population conservation, energy finiteness/bounds,
RDM/density sanity, analysis-tool outputs. Defensive — captures every failure for the
bug report. Run: PYTHONPATH=<parent-of-kanad> python benchmarks/planck_photodyn_analysis.py
"""
import json
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "tests")
import _solver_molecules as M  # noqa: E402

OUT = {"photodynamics": {}, "analysis": {}}


def section(title):
    print(f"\n### {title} ###", flush=True)


# ----------------------------------------------------------------------------
section("Photodynamics — quantum (qEOM) on planck backend")
try:
    from kanad.dynamics import LaserField, PhotodynamicsSimulator
    from kanad.core.bonds.bond_factory import BondFactory
    bond = BondFactory.create_bond("H", "H", distance=0.74)
    laser = LaserField(intensity=1e12, wavelength=200, polarization=[0, 0, 1],
                       pulse_duration=2.0, envelope="gaussian")
    sim = PhotodynamicsSimulator(bond, laser, n_states=2, propagator="rk4",
                                 use_quantum=True, vqe_backend="planck")
    res = sim.run(total_time=2.0, dt=0.1, save_interval=2)
    pops = np.asarray(getattr(res, "populations", []))
    energies = np.asarray(getattr(res, "energies", []))
    pop_sum = pops.sum(axis=1) if pops.ndim == 2 else np.array([np.nan])
    rec = {
        "ran": True,
        "n_steps_saved": int(pops.shape[0]) if pops.ndim == 2 else 0,
        "population_conservation_maxdev": float(np.max(np.abs(pop_sum - 1.0))) if pops.size else None,
        "energies_finite": bool(np.all(np.isfinite(energies))) if energies.size else None,
        "populations_nonneg": bool(np.all(pops >= -1e-9)) if pops.size else None,
        "result_fields": [a for a in dir(res) if not a.startswith("_")][:20],
    }
    print("  population conservation max|sum-1| =", rec["population_conservation_maxdev"])
    print("  energies finite =", rec["energies_finite"], "| pops>=0 =", rec["populations_nonneg"])
    OUT["photodynamics"] = rec
except Exception as e:
    OUT["photodynamics"] = {"ran": False, "error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc().splitlines()[-5:]}
    print("  ERROR:", OUT["photodynamics"]["error"])


# ----------------------------------------------------------------------------
section("Analysis workflow — on a planck-backed solver result")
try:
    from kanad.solvers import PhysicsVQE
    from kanad.analysis import EnergyAnalyzer, BondingAnalyzer
    sysm = M.lih()
    solver = PhysicsVQE(sysm, backend="planck", max_excitations=6)
    result = solver.solve()
    d = result.to_dict()
    E = float(result.energy)
    rec = {"ran": True, "energy": E, "energy_finite": bool(np.isfinite(E)),
           "to_dict_keys": sorted(d.keys())}

    # energy convergence analysis (if history present)
    ea = EnergyAnalyzer(solver.hamiltonian)
    hist = d.get("energy_history") or getattr(result, "energy_history", None)
    if hist is not None and len(np.atleast_1d(hist)) > 1:
        conv = ea.analyze_convergence(np.asarray(hist, dtype=float))
        rec["convergence_analysis_keys"] = sorted(conv.keys()) if isinstance(conv, dict) else str(type(conv))

    # bonding analysis
    try:
        ba = BondingAnalyzer(solver.hamiltonian)
        bonding = ba.analyze_bonding_type()
        rec["bonding_keys"] = sorted(bonding.keys()) if isinstance(bonding, dict) else str(type(bonding))
        rec["bonding_finite"] = all(
            np.isfinite(v) for v in bonding.values() if isinstance(v, (int, float))
        ) if isinstance(bonding, dict) else None
    except Exception as be:
        rec["bonding_error"] = f"{type(be).__name__}: {be}"

    # RDM / density validation if exposed
    for k in ("density_matrix", "rdm1", "quantum_1rdm", "quantum_rdm1"):
        v = d.get(k)
        if v is not None:
            a = np.asarray(v)
            if a.ndim == 2 and a.shape[0] == a.shape[1]:
                rec[f"{k}_hermitian"] = bool(np.allclose(a, a.conj().T, atol=1e-6))
                rec[f"{k}_trace"] = float(np.trace(a).real)
    print("  energy =", E, "| finite =", rec["energy_finite"])
    print("  bonding_keys =", rec.get("bonding_keys"), "| conv =", rec.get("convergence_analysis_keys"))
    OUT["analysis"] = rec
except Exception as e:
    OUT["analysis"] = {"ran": False, "error": f"{type(e).__name__}: {e}",
                       "trace": traceback.format_exc().splitlines()[-5:]}
    print("  ERROR:", OUT["analysis"]["error"])


import os
os.makedirs("benchmarks/out", exist_ok=True)
with open("benchmarks/out/planck_photodyn_analysis.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
print("\nwrote benchmarks/out/planck_photodyn_analysis.json")
