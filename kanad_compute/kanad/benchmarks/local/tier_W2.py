"""Wave-2 SQD-core fix regression. Exercises the previously-UNCOVERED custom CI backend
and the out-of-sector / excited-state RDM paths.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_W2
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from kanad import MolecularBuilder

AU2D = 2.541746230211

SYSTEMS = [
    ("LiH", [('Li', (0, 0, 0)), ('H', (0, 0, 1.595))], [0], [1, 2, 3, 4, 5]),
    ("H4",  [('H', (0, 0, 0)), ('H', (0, 0, 1.0)), ('H', (0, 0, 2.0)), ('H', (0, 0, 3.0))], [], [0, 1, 2, 3]),
]


def all_sz0_dets(n_orb, n_a, n_b):
    """Every (n_a, n_b) determinant in interleaved-JW encoding (α=even, β=odd qubit)."""
    dets = []
    for aocc in combinations(range(n_orb), n_a):
        ad = sum(1 << (2 * p) for p in aocc)
        for bocc in combinations(range(n_orb), n_b):
            dets.append(ad | sum(1 << (2 * p + 1) for p in bocc))
    return dets


def fci_in_cas(atoms, basis, frozen, active):
    from pyscf import gto, scf, mcscf, ao2mo, fci
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    ncas, ne = len(active), mol.nelectron - 2 * len(frozen)
    cas = mcscf.CASCI(mf, ncas, ne)
    h1, ec = cas.get_h1eff(); h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
    e, _ = fci.direct_spin0.kernel(h1, h2, ncas, ne, ecore=ec)
    return float(e)


def main():
    print("=" * 92, flush=True)
    print("WAVE-2 SQD-CORE FIX REGRESSION (B14/B15 custom backend, B16 sector, B17 excited RDM)", flush=True)
    print("=" * 92, flush=True)

    for name, atoms, frozen, active in SYSTEMS:
        e_fci = fci_in_cas(atoms, 'sto-3g', frozen, active)
        qs = (MolecularBuilder.from_atoms(atoms).basis('sto-3g')
              .active_space('manual', frozen=frozen, active=active)
              .solver('sqd', backend='statevector', ci_backend='custom',
                      n_samples=2000, random_seed=0).build())
        qs.solve()
        solver = qs._sqd_solver
        n_orb = solver.hamiltonian.n_orbitals
        n_e = solver.hamiltonian.n_electrons
        n_a = n_b = n_e // 2
        dets = all_sz0_dets(n_orb, n_a, n_b)

        # ---- B14/B15: custom backend on the FULL CAS must equal exact FCI ----
        e_custom = float(solver._diagonalize_in_subspace(dets)['energy'])
        e_pyscf = float(solver._diagonalize_in_subspace_pyscf(dets)['energy'])
        okc = abs(e_custom - e_fci) < 1e-6 and abs(e_pyscf - e_fci) < 1e-6
        print(f"[B14/B15] {name:3} full-CAS({len(dets)} dets): custom={e_custom:.6f}  "
              f"pyscf={e_pyscf:.6f}  FCI={e_fci:.6f}  -> {'PASS' if okc else '*** FAIL'}", flush=True)

        # ---- B16: inject an out-of-(n_a,n_b)-sector det; must be dropped, no misalign ----
        bad = sum(1 << (2 * p) for p in range(n_a + 1))   # n_a+1 α electrons (wrong Sz/N)
        res = solver._diagonalize_in_subspace_pyscf(dets + [bad])
        okb = (abs(float(res['energy']) - e_fci) < 1e-6
               and len(res['determinants']) == len(res['eigenvector']) == len(dets))
        print(f"[B16]     {name:3} +out-of-sector det: E={float(res['energy']):.6f}  "
              f"len(dets)={len(res['determinants'])} len(evec)={len(res['eigenvector'])}  "
              f"-> {'PASS' if okb else '*** FAIL'}", flush=True)

        # ---- B17: excited-state state-0 1-RDM must match the ground-state 1-RDM ----
        # Both are the ground state; with the missing interleave-sign correction the
        # excited path's eigenvector had wrong fermionic phases -> wrong RDM.
        solver.solve()
        rdm_gs = np.asarray(solver.get_1rdm_active_mo())
        solver.solve_excited_states_iterative(n_states=2, max_iterations=4)
        rdm_ex0 = np.asarray(solver.get_1rdm_active_mo())
        d = float(np.max(np.abs(rdm_gs - rdm_ex0))) if rdm_gs.shape == rdm_ex0.shape else float('inf')
        noons = np.linalg.eigvalsh(rdm_ex0)
        valid = bool(np.all(noons > -1e-6) and np.all(noons < 2 + 1e-6)
                     and abs(np.trace(rdm_ex0) - n_e) < 1e-6)
        print(f"[B17]     {name:3} excited state-0 vs ground 1-RDM: max|Δ|={d:.2e}  "
              f"trace={np.trace(rdm_ex0):.3f} NOON∈[{noons.min():.3f},{noons.max():.3f}]  "
              f"-> {'PASS' if (d < 1e-3 and valid) else '*** FAIL'}", flush=True)

    print("\nW2_DONE", flush=True)


if __name__ == "__main__":
    main()
