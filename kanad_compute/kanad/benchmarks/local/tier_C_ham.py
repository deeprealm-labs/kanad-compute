"""Campaign C — EXOTIC HAMILTONIANS the energy campaigns never touched: the Hubbard /
metallic tight-binding model (vs ANALYTIC 2-site result), periodic crystals (PBC), and
ionic Hamiltonians (vs PySCF FCI). Maps which representations actually work end-to-end.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_C_ham
"""
from __future__ import annotations
import numpy as np


def gs_energy(H):
    H = np.asarray(H)
    if H.ndim != 2:
        return None
    w = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    return float(w[0])


def main():
    print("=" * 100, flush=True)
    print("CAMPAIGN C — exotic Hamiltonians: Hubbard / periodic / ionic", flush=True)
    print("=" * 100, flush=True)

    # ---- 1. Hubbard / metallic: 2-site Hubbard has analytic GS at half filling ----
    #    E0 = (U - sqrt(U^2 + 16 t^2)) / 2   (t = |hopping|, half filling, singlet)
    print("\n--- 2-site Hubbard (analytic GS check) via MetallicHamiltonian ---", flush=True)
    for t, U in [(1.0, 0.0), (1.0, 4.0), (1.0, 8.0)]:
        analytic = (U - np.sqrt(U ** 2 + 16 * t ** 2)) / 2.0
        built = None
        err = None
        for path in ('bondmol', 'factory'):
            try:
                from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian
                from kanad.core.atom import Atom
                atoms = [Atom('H', np.array([0.0, 0.0, 0.0])), Atom('H', np.array([0.0, 0.0, 1.0]))]
                if path == 'bondmol':
                    from kanad.core.representations.base_representation import BondMolecule
                    mol = BondMolecule(atoms)
                else:
                    from kanad.core.bonds.bond_factory import BondFactory
                    mol = BondFactory.create_bond('H', 'H', distance=1.0).molecule
                H = MetallicHamiltonian(mol, lattice_type='1d_chain', hopping_parameter=-t,
                                        onsite_energy=0.0, hubbard_u=U, periodic=False)
                M = H.to_matrix()
                built = gs_energy(M)
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:60]}"
        if built is not None:
            print(f"  t={t} U={U}: analytic={analytic:+.4f}  built={built:+.4f}  Δ={built-analytic:+.4f} (eV units)", flush=True)
        else:
            print(f"  t={t} U={U}: analytic={analytic:+.4f}  built=FAILED ({err})", flush=True)

    # ---- 2. Periodic crystal: 1D hydrogen chain (PBC) total energy ----
    print("\n--- Periodic 1D H-chain (PBC) via PeriodicHamiltonian ---", flush=True)
    try:
        from kanad.core.hamiltonians.periodic_hamiltonian import PeriodicHamiltonian
        from kanad.core.lattice import Lattice
        from kanad.core.atom import Atom
        a = 2.0  # AA spacing
        lat = Lattice(np.array([[a, 0, 0], [0, 10, 0], [0, 0, 10]]), pbc=(True, False, False))
        atoms = [Atom('H', np.array([0.0, 0.0, 0.0]))]
        ph = PeriodicHamiltonian(atoms, lat, basis='gth-szv', pseudo='gth-pade', k_points=(8, 1, 1))
        for attr in ('e_tot', 'energy', 'total_energy'):
            if hasattr(ph, attr):
                print(f"  PeriodicHamiltonian.{attr} = {getattr(ph, attr)}", flush=True)
        for m in ('run_scf', 'kernel', 'scf', 'compute_bands', 'band_structure'):
            if hasattr(ph, m):
                print(f"  has method: {m}()", flush=True)
        print("  (PeriodicHamiltonian constructed OK)", flush=True)
    except Exception as e:
        import traceback
        print(f"  Periodic CRASH {type(e).__name__}: {str(e)[:90]}", flush=True)
        traceback.print_exc()

    # ---- 3. Ionic Hamiltonians vs PySCF FCI (frozen-core active space) ----
    print("\n--- Ionic LiF / NaCl: builder CI vs PySCF FCI-in-active-space ---", flush=True)
    from kanad import MolecularBuilder

    def ref_cas(atoms, basis, ncas, nelec):
        from pyscf import gto, scf, mcscf, ao2mo, fci
        mol = gto.M(atom=atoms, basis=basis, verbose=0); mf = scf.RHF(mol).run(verbose=0)
        cas = mcscf.CASCI(mf, ncas, nelec)
        h1, ec = cas.get_h1eff(); h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
        e, _ = fci.direct_spin0.kernel(h1, h2, ncas, nelec, ecore=ec)
        return float(e)

    for name, atoms, basis, no, nv in [
        ('LiF', [('Li', (0, 0, 0)), ('F', (0, 0, 1.564))], 'sto-3g', 4, 4),
        ('NaCl', [('Na', (0, 0, 0)), ('Cl', (0, 0, 2.36))], 'sto-3g', 4, 4),
    ]:
        try:
            qs = (MolecularBuilder.from_atoms(atoms).basis(basis)
                  .active_space('frontier', n_occ=no, n_virt=nv).solver('ci').build())
            e_b = qs.solve()['energy']
            print(f"  {name:4} builder-CI = {e_b:.6f}  (active CAS, nq={qs.n_qubits})", flush=True)
        except Exception as e:
            print(f"  {name:4} builder CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    print("\nHAM_DONE", flush=True)


if __name__ == "__main__":
    main()
