"""Wave-3 fix regression: lattice (B2,B8,B9), mappers (B10,B11), tapering (B12), conditions (B13).

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_W3
"""
from __future__ import annotations
import numpy as np


def mk_metallic(symbol, n, dist, U=0.0):
    from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian
    from kanad.core.representations.base_representation import BondMolecule
    from kanad.core.atom import Atom
    atoms = [Atom(symbol, np.array([0.0, 0.0, i * dist])) for i in range(n)]
    mol = BondMolecule(atoms)
    return MetallicHamiltonian(mol, lattice_type='1d_chain', hopping_parameter=-1.0,
                               onsite_energy=0.0, hubbard_u=U, periodic=False)


def _cov_h2():
    from kanad.core.molecule import Molecule
    from kanad.core.atom import Atom
    from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
    from kanad.core.representations.lcao_representation import LCAORepresentation
    atoms = [Atom('H', [0.0, 0.0, 0.0]), Atom('H', [0.0, 0.0, 0.74])]
    mol = Molecule(atoms, charge=0, spin=0)
    return CovalentHamiltonian(mol, LCAORepresentation(mol, basis_name='sto-3g'),
                               basis_name='sto-3g', use_pyscf_integrals=True, use_governance=False)


def main():
    print("=" * 92, flush=True)
    print("WAVE-3 FIX REGRESSION (lattice / mappers / tapering / conditions)", flush=True)
    print("=" * 92, flush=True)

    # ---- B2: non-monovalent metal over-fill raises; monovalent OK ----
    try:
        mk_metallic('C', 2, 1.5)
        print("[B2] *** FAIL: multivalent C2 over-fill did NOT raise", flush=True)
    except ValueError as e:
        print(f"[B2] multivalent over-fill raises: PASS ({str(e)[:45]}...)", flush=True)
    try:
        h = mk_metallic('Li', 2, 3.0)
        print(f"[B2] monovalent Li2 builds OK: PASS (n_e={h.n_electrons} n_orb={h.n_orbitals})", flush=True)
    except Exception as e:
        print(f"[B2] *** monovalent FAIL: {type(e).__name__}: {str(e)[:50]}", flush=True)

    # ---- B8: attractive Hubbard U (U<0) enters compute_energy ----
    try:
        dm = np.eye(2)
        e0 = mk_metallic('Li', 2, 3.0, U=0.0).compute_energy(dm)
        em = mk_metallic('Li', 2, 3.0, U=-4.0).compute_energy(dm)
        ok = abs(e0 - em) > 1e-9
        print(f"[B8] compute_energy U=0:{e0:.4f} vs U=-4:{em:.4f}  -> "
              f"{'PASS (U<0 included)' if ok else '*** FAIL (U<0 dropped)'}", flush=True)
    except Exception as e:
        print(f"[B8] *** CRASH {type(e).__name__}: {str(e)[:60]}", flush=True)

    # ---- B9: band gap clamps to 0 + metallic flag when bands overlap ----
    try:
        from kanad.core.hamiltonians.periodic_hamiltonian import PeriodicHamiltonian
        ph = PeriodicHamiltonian.__new__(PeriodicHamiltonian)
        ph.n_electrons, ph.n_orbitals, ph.k_points = 2, 2, [0, 1]
        ph.band_energies = np.array([[0.0, -0.1], [0.2, 0.1]])   # cbm < vbm → metal
        m = ph.get_band_gap()
        ph.band_energies = np.array([[-1.0, 1.0], [-0.9, 1.1]])  # clear gap → insulator
        ins = ph.get_band_gap()
        ok = m['metallic'] and m['gap'] == 0.0 and not ins['metallic'] and ins['gap'] > 0
        print(f"[B9] metal gap={m['gap']} type={m['type']} | insulator gap={ins['gap']:.2f} "
              f"type={ins['type']}  -> {'PASS' if ok else '*** FAIL'}", flush=True)
    except Exception as e:
        print(f"[B9] *** CRASH {type(e).__name__}: {str(e)[:60]}", flush=True)

    # ---- B10: BK operator differs from JW (not vacuously equal) but isospectral ----
    try:
        from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper; from kanad.core.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
        from kanad.core.hamiltonians.pauli_converter import PauliConverter
        ham = _cov_h2()
        jw = PauliConverter.to_sparse_pauli_op(ham, JordanWignerMapper(), use_qiskit_nature=False)
        bk = PauliConverter.to_sparse_pauli_op(ham, BravyiKitaevMapper(), use_qiskit_nature=False)
        ev_jw = np.sort(np.linalg.eigvalsh(jw.to_matrix()).real)
        ev_bk = np.sort(np.linalg.eigvalsh(bk.to_matrix()).real)
        isospectral = np.allclose(ev_jw, ev_bk, atol=1e-8)
        differ = not bool(jw.equiv(bk))
        print(f"[B10] fallback JW≠BK operator={differ}, isospectral={isospectral}  -> "
              f"{'PASS' if (isospectral and differ) else '*** FAIL'}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[B10] *** CRASH {type(e).__name__}: {str(e)[:60]}", flush=True)

    # ---- B11: mapper_type='parity' rejected loudly ----
    try:
        from kanad.solvers.vqe_solver import VQESolver
        VQESolver(hamiltonian=_cov_h2(), mapper_type='parity')
        print("[B11] *** FAIL: parity mapper did NOT raise", flush=True)
    except NotImplementedError:
        print("[B11] parity mapper rejected (NotImplementedError): PASS", flush=True)
    except Exception as e:
        print(f"[B11] *** wrong exception {type(e).__name__}: {str(e)[:50]}", flush=True)

    # ---- B12: tapering metadata exposes hf_index inside the sector (general path,
    #          n_qubits != 4 — H2's 4q case uses the specialized taper_h2 path) ----
    try:
        from kanad.core.molecule import Molecule
        from kanad.core.atom import Atom
        from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
        from kanad.core.representations.lcao_representation import LCAORepresentation
        from kanad.core.mappers.tapering import QubitTapering
        a4 = [Atom('H', [0, 0, 0]), Atom('H', [0, 0, 1.0]), Atom('H', [0, 0, 2.0]), Atom('H', [0, 0, 3.0])]
        m4 = Molecule(a4, charge=0, spin=0)
        ham = CovalentHamiltonian(m4, LCAORepresentation(m4, basis_name='sto-3g'),
                                  basis_name='sto-3g', use_pyscf_integrals=True, use_governance=False)
        sp = ham.to_sparse_hamiltonian()
        tap, meta = QubitTapering().taper_hamiltonian(sp, ham.n_electrons, sp.num_qubits)
        hf_i = meta.get('hf_index'); sec = meta.get('sector_indices')
        ok = hf_i is not None and sec is not None and hf_i in sec
        print(f"[B12] H4 taper meta hf_index={hf_i} in sector({len(sec) if sec else 0}), "
              f"tapered_hf_pos={sec.index(hf_i) if ok else '?'}  -> {'PASS' if ok else '*** FAIL'}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[B12] *** CRASH {type(e).__name__}: {str(e)[:60]}", flush=True)

    # ---- B13: conditions gated; solve(apply_conditions=False) is the bare energy ----
    try:
        from kanad import MolecularBuilder
        qs = (MolecularBuilder.from_atoms([('H', (0, 0, 0)), ('H', (0, 0, 0.74))])
              .basis('sto-3g').solver('ci').conditions(thermal=True).build())
        e_cond = qs.solve()['energy']
        e_bare = qs.solve(apply_conditions=False)['energy']
        ok = abs(e_cond - e_bare) > 1e-6
        print(f"[B13] thermal cond E={e_cond:.6f} vs bare E={e_bare:.6f}  -> "
              f"{'PASS (gated)' if ok else '*** FAIL (not gated)'}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[B13] *** CRASH {type(e).__name__}: {str(e)[:60]}", flush=True)

    print("\nW3_DONE", flush=True)


if __name__ == "__main__":
    main()
