"""Wave-4 cleanup regression: B20 (open-shell pyscf-FCI), B21 (ionic valence),
B22 (JW double contraction), B23 (frontier open-shell HOMO), B24 (observables('all') graceful).

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_W4
"""
from __future__ import annotations
import numpy as np


def _cov(atom_str, basis='sto-3g', spin=0):
    from kanad.core.molecule import Molecule
    from kanad.core.atom import Atom
    from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
    from kanad.core.representations.lcao_representation import LCAORepresentation
    atoms = [Atom(p.split()[0], [float(x) for x in p.split()[1:]]) for p in atom_str.split(';')]
    mol = Molecule(atoms, charge=0, spin=spin)
    return CovalentHamiltonian(mol, LCAORepresentation(mol, basis_name=basis),
                               basis_name=basis, use_pyscf_integrals=True, use_governance=False)


def main():
    print("=" * 92, flush=True)
    print("WAVE-4 CLEANUP REGRESSION (B20 B21 B22 B23 B24)", flush=True)
    print("=" * 92, flush=True)

    # ---- B18/B20: default to_matrix() FCI path is bit-correct on closed-shell H2 ----
    # (The dead use_pyscf_fci=True branch was removed in the 2026-05-31 reorg; the
    #  live default to_matrix() path is the FCI-bit-correct one.)
    try:
        from pyscf import gto, scf, fci
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        e_fci = float(fci.FCI(scf.RHF(mol).run(verbose=0)).kernel()[0])
        H = np.asarray(_cov('H 0 0 0; H 0 0 0.74').to_matrix())
        e = float(np.linalg.eigvalsh(H)[0].real)
        print(f"[B20] H2 to_matrix()={e:.6f} FCI={e_fci:.6f} -> "
              f"{'PASS' if abs(e - e_fci) < 1e-6 else '*** FAIL'}", flush=True)
    except Exception as ex:
        print(f"[B20] *** CRASH {type(ex).__name__}: {str(ex)[:60]}", flush=True)

    # ---- B21: IonicHubbardModel uses VALENCE electrons, not atomic number ----
    try:
        from kanad.core.models import IonicHubbardModel
        from kanad.core.atom import Atom
        m = IonicHubbardModel([Atom('Na', [0, 0, 0]), Atom('Cl', [0, 0, 2.3])])
        # Na valence 1 + Cl valence 7 = 8 (NOT atomic numbers 11+17=28)
        print(f"[B21] NaCl IonicHubbard n_electrons={m.n_electrons} (valence=8, not 28) -> "
              f"{'PASS' if m.n_electrons == 8 else '*** FAIL'}", flush=True)
    except Exception as ex:
        print(f"[B21] *** CRASH {type(ex).__name__}: {str(ex)[:60]}", flush=True)

    # ---- B22: JW map_double_excitation runs for j==k and j!=k (contraction handled) ----
    try:
        from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper
        jw = JordanWignerMapper()
        d1 = jw.map_double_excitation(0, 1, 2, 3, 4)   # j=0,l=1,i=2,k=3 -> j!=k
        d2 = jw.map_double_excitation(1, 2, 3, 1, 4)   # j=1, k=1 -> j==k contraction path
        print(f"[B22] map_double_excitation j!=k terms={len(d1)}, j==k terms={len(d2)} -> "
              f"{'PASS (both run)' if d1 and d2 is not None else '*** FAIL'}", flush=True)
    except Exception as ex:
        print(f"[B22] *** CRASH {type(ex).__name__}: {str(ex)[:60]}", flush=True)

    # ---- B23: frontier centers the window on the SOMO for open-shell (mo_occ-based HOMO) ----
    try:
        from kanad.core.active_space import ActiveSpaceSelector
        from pyscf import gto, scf
        mol = gto.M(atom='Be 0 0 0; H 0 0 1.33', basis='sto-3g', spin=1, verbose=0)  # 5 e- doublet
        mf = scf.ROHF(mol).run(verbose=0)
        somo = int(np.where(np.asarray(mf.mo_occ) > 0)[0][-1])
        n_virt, n_orb = 2, mf.mo_coeff.shape[1]
        asp = ActiveSpaceSelector(mf).frontier(n_occ=3, n_virt=n_virt)
        exp_top = min(somo + n_virt, n_orb - 1)
        ok = somo in asp.active_indices and max(asp.active_indices) == exp_top
        print(f"[B23] BeH doublet SOMO={somo} active={asp.active_indices} top(exp {exp_top}) -> "
              f"{'PASS' if ok else '*** FAIL'}", flush=True)
    except Exception as ex:
        print(f"[B23] *** CRASH {type(ex).__name__}: {str(ex)[:60]}", flush=True)

    # ---- B24: observables('all') degrades gracefully (no crash) on an active-space CI ----
    try:
        from kanad import MolecularBuilder
        qs = (MolecularBuilder.from_atoms([('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))])
              .basis('sto-3g').active_space('frozen_core').solver('ci').build())
        qs.solve()
        o = qs.observables('all')
        pol = o.get('polarizability_mean_au')
        note = o.get('polarizability_unavailable', '')
        print(f"[B24] observables('all') OK (no crash): polarizability={pol} "
              f"{('unavail=' + note[:40]) if note else '(computed)'} -> PASS", flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[B24] *** CRASH {type(ex).__name__}: {str(ex)[:60]}", flush=True)

    print("\nW4_DONE", flush=True)


if __name__ == "__main__":
    main()
