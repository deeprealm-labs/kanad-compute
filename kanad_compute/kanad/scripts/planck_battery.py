"""Deeper planck-backend battery: different basis sets, PhysicsVQE, polyatomic
correlation. Runs planck vs statevector for parity + timing. Self-contained
(kanad + planck). Run with PYTHONPATH=<dir containing kanad> in the kanad venv.
"""
import argparse
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from kanad.core.bonds.bond_factory import BondFactory
from kanad.solvers.vqe_solver import VQESolver
from kanad.solvers.physics_vqe import PhysicsVQE


def _vqe(bond, backend, maxit=120):
    t = time.perf_counter()
    e = VQESolver(bond=bond, ansatz_type="hardware_efficient", backend=backend,
                  optimizer="L-BFGS-B", max_iterations=maxit,
                  enable_analysis=False).solve()["energy"]
    return e, time.perf_counter() - t


def _physvqe(bond, backend, max_exc=5):
    t = time.perf_counter()
    r = PhysicsVQE(bond=bond, backend=backend, max_excitations=max_exc).solve()
    e = getattr(r, "energy", None)
    if e is None:
        e = r["energy"] if isinstance(r, dict) else r
    return float(e), time.perf_counter() - t


def section(title):
    print(f"\n### {title} ###", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="H2 PhysicsVQE only (local smoke)")
    ap.add_argument("--basis", action="store_true")
    ap.add_argument("--physvqe", action="store_true")
    ap.add_argument("--poly", action="store_true")
    args = ap.parse_args()
    allsec = not (args.quick or args.basis or args.physvqe or args.poly)

    if args.quick:
        for be in ("statevector", "planck"):
            e, t = _physvqe(BondFactory.create_bond("H", "H", distance=0.74, basis="sto-3g"), be)
            print(f"PhysicsVQE H2 {be:11s} E={e:.8f} Ha  t={t:.2f}s", flush=True)
        return

    if allsec or args.basis:
        section("A. Basis sets (H2): sto-3g vs 6-31g, VQE planck vs CPU")
        for basis in ("sto-3g", "6-31g"):
            row = {}
            for be in ("statevector", "planck"):
                try:
                    bond = BondFactory.create_bond("H", "H", distance=0.74, basis=basis)
                    e, t = _vqe(bond, be)
                    row[be] = e
                    print(f"H2/{basis:6s} {be:11s} E={e:.8f} Ha  t={t:.2f}s", flush=True)
                except Exception as ex:
                    print(f"H2/{basis} {be}: ERROR {ex}", flush=True)
            if len(row) == 2:
                print(f"   -> |dE|={abs(row['planck']-row['statevector']):.2e} Ha", flush=True)

    if allsec or args.physvqe:
        section("B. PhysicsVQE (UCC-style excitations) planck vs CPU")
        for atoms, d in [(("H", "H"), 0.74), (("Li", "H"), 1.60)]:
            row = {}
            for be in ("statevector", "planck"):
                try:
                    bond = BondFactory.create_bond(atoms[0], atoms[1], distance=d, basis="sto-3g")
                    e, t = _physvqe(bond, be)
                    row[be] = e
                    print(f"PhysicsVQE {atoms} {be:11s} E={e:.8f} Ha  t={t:.2f}s", flush=True)
                except Exception as ex:
                    print(f"PhysicsVQE {atoms} {be}: ERROR {ex}", flush=True)
            if len(row) == 2:
                print(f"   -> |dE|={abs(row['planck']-row['statevector']):.2e} Ha", flush=True)

    if allsec or args.poly:
        section("C. More correlation: H2O / sto-3g via MolecularBuilder (planck)")
        try:
            from kanad.builder import MolecularBuilder
            qs = (MolecularBuilder.from_smiles("O").basis("sto-3g")
                  .solver("vqe", backend="planck").build())
            t = time.perf_counter()
            res = qs.solve()
            e = res.get("energy") if isinstance(res, dict) else res
            print(f"H2O builder/planck  E={e}  t={time.perf_counter()-t:.2f}s "
                  f"(n_qubits={getattr(qs,'n_qubits','?')})", flush=True)
        except Exception as ex:
            print(f"H2O builder: ERROR {ex}", flush=True)


if __name__ == "__main__":
    main()
