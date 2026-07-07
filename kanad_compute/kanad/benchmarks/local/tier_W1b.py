"""Wave-1 core-fix regression for B1 (CovalentHamiltonian.to_matrix frozen-core) and
B7 (active-space electron-count guard).

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_W1b
"""
from __future__ import annotations
import numpy as np


def _ham(atom_str, frozen=None, active=None):
    from kanad.core.molecule import Molecule
    from kanad.core.atom import Atom
    from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
    from kanad.core.representations.lcao_representation import LCAORepresentation
    atoms = [Atom(p.split()[0], [float(x) for x in p.split()[1:]]) for p in atom_str.split(";")]
    mol = Molecule(atoms, charge=0, spin=0)
    return CovalentHamiltonian(mol, LCAORepresentation(mol, basis_name="sto-3g"),
                               basis_name="sto-3g", use_pyscf_integrals=True,
                               frozen_orbitals=frozen, active_orbitals=active,
                               use_governance=False)


def main():
    print("=" * 92, flush=True)
    print("WAVE-1 CORE FIX REGRESSION B1 (to_matrix frozen-core) + B7 (e-count guard)", flush=True)
    print("=" * 92, flush=True)
    from pyscf import gto, scf, fci, mcscf

    # ---------- regression: FULL-space to_matrix still == FCI (must not have broken it) ----------
    for label, atom_str in (("H2", "H 0 0 0; H 0 0 0.74"), ("LiH", "Li 0 0 0; H 0 0 1.6")):
        mol = gto.M(atom=atom_str, basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        e_fci = float(fci.FCI(mf).kernel()[0])
        e_tm = float(np.linalg.eigvalsh(np.asarray(_ham(atom_str).to_matrix()))[0].real)
        ok = abs(e_tm - e_fci) < 1e-6
        print(f"[B1-full] {label:4} to_matrix={e_tm:.8f}  FCI={e_fci:.8f}  "
              f"-> {'PASS' if ok else '*** FAIL'}", flush=True)

    # ---------- B1: ACTIVE-space to_matrix == correct sibling to_sparse == CASCI ----------
    for label, atom_str, frozen, active in (
            ("LiH", "Li 0 0 0; H 0 0 1.6", [0], [1, 2, 3, 4, 5]),
            ("H2O", "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", [0], [1, 2, 3, 4, 5, 6])):
        mol = gto.M(atom=atom_str, basis="sto-3g", verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        n_act_e = mol.nelectron - 2 * len(frozen)
        cas = mcscf.CASCI(mf, ncas=len(active), nelecas=n_act_e)
        cas.fcisolver.conv_tol = 1e-12
        cas.fix_spin_(ss=0)
        e_cas = float(cas.run(verbose=0).e_tot)
        ham = _ham(atom_str, frozen=frozen, active=active)
        e_dense = float(np.linalg.eigvalsh(np.asarray(ham.to_matrix()))[0].real)
        e_sparse = float(np.linalg.eigvalsh(np.asarray(ham.to_sparse_hamiltonian().to_matrix()))[0].real)
        ok = abs(e_dense - e_cas) < 1e-3 and abs(e_dense - e_sparse) < 1e-6 and e_dense >= e_cas - 1e-4
        print(f"[B1-active] {label:4} to_matrix(dense)={e_dense:.6f}  to_sparse={e_sparse:.6f}  "
              f"CASCI={e_cas:.6f}  -> {'PASS' if ok else '*** FAIL'}", flush=True)

    # ---------- B7: over-filled active space must raise (not silently mis-count) ----------
    try:
        _ham("O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", frozen=[], active=[0, 1])
        print("[B7] *** FAIL: over-filled active space did NOT raise", flush=True)
    except ValueError as e:
        good = "over-filled" in str(e)
        print(f"[B7] over-fill guard fires: {'PASS' if good else '*** FAIL'}  ({str(e)[:60]}...)", flush=True)

    print("\nW1B_DONE", flush=True)


if __name__ == "__main__":
    main()
