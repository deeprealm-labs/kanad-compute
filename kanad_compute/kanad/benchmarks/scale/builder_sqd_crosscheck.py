"""Builder-based statevector SQD vs spin-correct CASCI cross-check.

A reproducible, hardware-free reference for the large-active-space SQD champions:
runs the unified `MolecularBuilder` SQD path (statevector backend, exact sampling
distribution) and compares to a classical CASCI computed with `fix_spin_(ss=0)`
(the singlet anchor — PySCF's default Davidson can converge to a triplet on C2,
which was the root of the apparent "SQD < CASCI" anomaly).

Doubles as a rigorous scale test of the builder at 20-24 qubits. Statevector,
so no IBM credentials needed:

    python -m benchmarks.builder_sqd_crosscheck
"""

from __future__ import annotations

import time

import numpy as np
from pyscf import gto, scf, mcscf

from kanad import MolecularBuilder

# (label, atom string, basis, frozen MOs, active MOs)
SYSTEMS = [
    ("M11a N2 CAS(10,10) 20q", "N 0 0 0; N 0 0 1.10",  "cc-pvdz", [0, 1], list(range(2, 12))),
    ("M11b C2 CAS(12,12) 24q", "C 0 0 0; C 0 0 1.243", "cc-pvdz", [],     list(range(12))),
]


def casci_singlet(atom, basis, frozen, active):
    """Spin-correct (singlet) CASCI reference in the given active space."""
    mol = gto.M(atom=atom, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    n_act_e = mol.nelectron - 2 * len(frozen)
    cas = mcscf.CASCI(mf, ncas=len(active), nelecas=n_act_e)
    cas.fcisolver.conv_tol = 1e-12
    cas.fcisolver.max_cycle = 500
    cas.fix_spin_(ss=0)
    cas.run(verbose=0)
    return float(cas.e_tot)


def _atoms(atom_str):
    out = []
    for p in atom_str.split(";"):
        tok = p.split()
        out.append((tok[0], (float(tok[1]), float(tok[2]), float(tok[3]))))
    return out


def main():
    print("=" * 78)
    print("Builder statevector SQD vs singlet CASCI  (reproducible M11a/M11b refs)")
    print("=" * 78)
    for label, atom, basis, frozen, active in SYSTEMS:
        e_cas = casci_singlet(atom, basis, frozen, active)
        qs = (
            MolecularBuilder.from_atoms(_atoms(atom)).basis(basis)
            .active_space("manual", frozen=frozen, active=active)
            .solver("sqd", n_samples=80000, max_iterations=4,
                    expansion_per_round=80, energy_tol=1e-6, random_seed=0)
            .build()
        )
        t0 = time.time()
        res = qs.solve()
        dt = time.time() - t0
        gap_mha = (res["energy"] - e_cas) * 1000.0
        n_det = res.get("n_determinants")
        print(f"{label}")
        print(f"  CASCI(singlet) = {e_cas:.6f} Ha")
        print(f"  SQD(statevec)  = {res['energy']:.6f} Ha   gap = {gap_mha:+.4f} mHa   "
              f"dets = {n_det}   ({dt:.0f}s)", flush=True)
    print("CROSSCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
