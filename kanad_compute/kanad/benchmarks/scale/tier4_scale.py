"""Tier-4 large-scale benchmark: SQD scaling 20->28q + Cr2 + DFT-fails, vs CASCI and vs DFT.

Run on the AMD cluster (235GB, CPU statevector):
    PYTHONPATH=/tmp/kanad-pkg /root/miniconda3/bin/python -m benchmarks.tier4_scale

Each case: builder statevector SQD vs singlet CASCI (quantum reference) AND vs RKS/B3LYP (the DFT contrast).
Prints one TIER4 line per case (machine-parseable) + TIER4_DONE.
"""
from __future__ import annotations
import time
import numpy as np
from pyscf import gto, scf, mcscf, dft

from kanad import MolecularBuilder


def casci_singlet(atom, basis, frozen, active, spin=0):
    mol = gto.M(atom=atom, basis=basis, verbose=0, spin=spin)
    mf = scf.RHF(mol).run(verbose=0)
    n_act_e = mol.nelectron - 2 * len(frozen)
    cas = mcscf.CASCI(mf, ncas=len(active), nelecas=n_act_e)
    cas.fcisolver.conv_tol = 1e-10
    cas.fcisolver.max_cycle = 400
    try:
        cas.fix_spin_(ss=0)
    except Exception:
        pass
    cas.run(verbose=0)
    return float(cas.e_tot)


def dft_energy(atom, basis, xc='b3lyp', spin=0):
    try:
        mol = gto.M(atom=atom, basis=basis, verbose=0, spin=spin)
        mf = dft.RKS(mol); mf.xc = xc; mf.run(verbose=0)
        return float(mf.e_tot)
    except Exception as e:
        return float('nan')


def _atoms(atom_str):
    out = []
    for p in atom_str.split(';'):
        t = p.split()
        out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


# (label, atom, basis, frozen, active, n_samples)
SCALING = [
    ("N2 CAS(10,10) 20q", "N 0 0 0; N 0 0 1.10", "cc-pvdz", [0, 1], list(range(2, 12)), 80000),
    ("N2 CAS(12,12) 24q", "N 0 0 0; N 0 0 1.10", "cc-pvdz", [0, 1], list(range(2, 14)), 100000),
    ("N2 CAS(14,14) 28q", "N 0 0 0; N 0 0 1.10", "cc-pvdz", [],     list(range(14)),     120000),
]

# DFT-fails: stretched N2 (multireference), twisted ethylene (90deg diradical), O3 (multireference)
DFT_FAILS = [
    ("N2 stretched r=2.1 CAS(10,10) 20q", "N 0 0 0; N 0 0 2.10", "cc-pvdz", [0, 1], list(range(2, 12)), 80000),
    ("C2H4 twisted-90 CAS(2,2)->grow", "C 0 0 0; C 0 0 1.33; H 0.92 0 -0.5; H -0.92 0 -0.5; H 0 0.92 1.83; H 0 -0.92 1.83",
     "sto-3g", [0, 1, 2, 3, 4], list(range(5, 11)), 60000),
]


def run_case(label, atom, basis, frozen, active, ns):
    n_q = 2 * len(active)
    try:
        e_cas = casci_singlet(atom, basis, frozen, active)
    except Exception as e:
        e_cas = float('nan')
    e_dft = dft_energy(atom, basis)
    qs = (MolecularBuilder.from_atoms(_atoms(atom)).basis(basis)
          .active_space("manual", frozen=frozen, active=active)
          .solver("sqd", n_samples=ns, max_iterations=5, expansion_per_round=80,
                  energy_tol=1e-6, random_seed=0)
          .build())
    t0 = time.time()
    res = qs.solve()
    dt = time.time() - t0
    e_sqd = res["energy"]
    gap_cas = (e_sqd - e_cas) * 1000.0 if e_cas == e_cas else float('nan')
    print(f"TIER4 | {label} | {n_q}q | CASCI={e_cas:.6f} | SQD={e_sqd:.6f} | "
          f"gap_vs_CASCI={gap_cas:+.4f}mHa | DFT(b3lyp)={e_dft:.6f} | "
          f"dets={res.get('n_determinants')} | t={dt:.0f}s", flush=True)


def main():
    print("=" * 90, flush=True)
    print("TIER 4 — SQD scaling + DFT-fails (vs CASCI and vs B3LYP)", flush=True)
    print("=" * 90, flush=True)
    print("\n--- Scaling 20 -> 28 qubits ---", flush=True)
    for c in SCALING:
        try: run_case(*c)
        except Exception as e: print(f"TIER4 | {c[0]} | FAILED: {type(e).__name__}: {e}", flush=True)
    print("\n--- DFT-fails ---", flush=True)
    for c in DFT_FAILS:
        try: run_case(*c)
        except Exception as e: print(f"TIER4 | {c[0]} | FAILED: {type(e).__name__}: {e}", flush=True)
    print("TIER4_DONE", flush=True)


if __name__ == "__main__":
    main()
