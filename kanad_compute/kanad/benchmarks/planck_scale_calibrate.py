"""Calibrate the analysis-scaling ladder WITHOUT allocating any statevector.

For each recipe (molecule/basis/active-space) report:
  n_qubits = 2 * n_active_orbitals, n_pauli_terms = len(SparsePauliOp), HF energy,
  and build/JW timings. From the Pauli-term count we estimate the cost of ONE
  full molecular <H> at that size (terms x 2^n bandwidth pass), which is what bounds
  a full solve+analysis. Pure host work — safe to run on a CPU box.

Run:  PYTHONPATH=<parent-of-kanad> python benchmarks/planck_scale_calibrate.py
"""
import json
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def h_chain(n, r=1.0, basis="sto-3g"):
    """Linear H_n chain, spacing r Angstrom. n orbitals (sto-3g) -> 2n qubits."""
    from kanad.builder import MolecularBuilder
    atoms = [("H", (0.0, 0.0, i * r)) for i in range(n)]
    return MolecularBuilder.from_atoms(atoms).basis(basis).build()


def h2o():
    from kanad.builder import MolecularBuilder
    return (MolecularBuilder
            .from_atoms([("O", (0.0, 0.0, 0.117)),
                         ("H", (0.0, 0.757, -0.467)),
                         ("H", (0.0, -0.757, -0.467))])
            .basis("sto-3g").build())


def n2_cas(n_active, r=1.10):
    """N2 / cc-pVDZ, manual active window of n_active orbitals around the frontier."""
    from kanad.builder import MolecularBuilder
    # 28 spatial orbitals total; center the active window on the HOMO region.
    lo = max(0, 7 - n_active // 2)
    active = list(range(lo, lo + n_active))
    frozen = list(range(lo))
    return (MolecularBuilder
            .from_atoms([("N", (0, 0, 0)), ("N", (0, 0, r))])
            .basis("cc-pvdz")
            .active_space("manual", frozen=frozen, active=active)
            .build())


RECIPES = [
    ("H2O/sto-3g", h2o),
    ("H10/sto-3g", lambda: h_chain(10)),
    ("H11/sto-3g", lambda: h_chain(11)),
    ("H12/sto-3g", lambda: h_chain(12)),
    ("H13/sto-3g", lambda: h_chain(13)),
    ("H14/sto-3g", lambda: h_chain(14)),
    ("H15/sto-3g", lambda: h_chain(15)),
    ("H16/sto-3g", lambda: h_chain(16)),
    ("H17/sto-3g", lambda: h_chain(17)),
    ("N2/ccpvdz/cas9", lambda: n2_cas(9)),
    ("N2/ccpvdz/cas11", lambda: n2_cas(11)),
]

BW_TBps = 4.0  # MI300X measured statevector bandwidth, TB/s (from the capacity benchmark)


def main():
    out = []
    for name, ctor in RECIPES:
        rec = {"recipe": name}
        try:
            t0 = time.perf_counter()
            sysm = ctor()
            ham = sysm.hamiltonian
            norb = int(ham.n_orbitals)
            nq = 2 * norb
            rec["n_orbitals"] = norb
            rec["n_qubits"] = nq
            rec["n_electrons"] = int(getattr(ham, "n_electrons", -1))
            rec["hf_energy"] = float(getattr(ham, "hf_energy", float("nan")))
            rec["t_build"] = round(time.perf_counter() - t0, 2)
            # JW Hamiltonian term count (NO 2^n allocation)
            t1 = time.perf_counter()
            spo = ham.to_sparse_hamiltonian("jordan_wigner")
            rec["n_pauli_terms"] = int(len(spo))
            rec["t_sparse"] = round(time.perf_counter() - t1, 2)
            # cost model for ONE full <H>: terms * (2^n complex128 bytes) / bandwidth
            state_bytes = (2 ** nq) * 16
            rec["state_GB_c128"] = round(state_bytes / 1e9, 2)
            rec["state_GB_c64"] = round(state_bytes / 2 / 1e9, 2)
            traffic = rec["n_pauli_terms"] * state_bytes      # bytes touched per <H>
            rec["sec_per_expectation_c128"] = round(traffic / (BW_TBps * 1e12), 2)
            rec["ok"] = True
        except Exception as e:
            import traceback
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc().splitlines()[-3:]
        out.append(rec)
        if rec.get("ok"):
            print(f"{name:18s} q={rec['n_qubits']:>3} terms={rec['n_pauli_terms']:>7} "
                  f"HF={rec['hf_energy']:>12.5f} state(c128)={rec['state_GB_c128']:>8} GB "
                  f"~{rec['sec_per_expectation_c128']:>9}s/<H>  (build {rec['t_build']}s, jw {rec['t_sparse']}s)",
                  flush=True)
        else:
            print(f"{name:18s} ERR {rec['error'][:80]}", flush=True)

    import os
    os.makedirs("benchmarks/out", exist_ok=True)
    with open("benchmarks/out/planck_scale_calibrate.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nwrote benchmarks/out/planck_scale_calibrate.json")


if __name__ == "__main__":
    main()
