"""Campaign C — verify the two flags from tier_C_ham: (1) does MetallicHamiltonian have
ANY many-body path that includes Hubbard U, or is U genuinely dropped? (2) is the ionic
builder-CI variationally sound (≤ HF in the same molecule)?
"""
from __future__ import annotations
import numpy as np


def main():
    print("=== (1) Hubbard U: inspect MetallicHamiltonian internals ===", flush=True)
    try:
        from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian
        from kanad.core.representations.base_representation import BondMolecule
        from kanad.core.atom import Atom
        atoms = [Atom('H', np.array([0.0, 0.0, 0.0])), Atom('H', np.array([0.0, 0.0, 1.0]))]
        mol = BondMolecule(atoms)
        for U in (0.0, 8.0):
            H = MetallicHamiltonian(mol, lattice_type='1d_chain', hopping_parameter=-1.0,
                                    onsite_energy=0.0, hubbard_u=U, periodic=False)
            M = np.asarray(H.to_matrix())
            print(f"  U={U}: to_matrix shape={M.shape}  lowest_eig={np.linalg.eigvalsh(0.5*(M+M.T))[0]:+.4f}  "
                  f"n_elec={getattr(H,'n_electrons','?')} hubbard_u_attr={getattr(H,'hubbard_u','?')}", flush=True)
            meths = [m for m in dir(H) if not m.startswith('_') and callable(getattr(H, m))]
            print(f"     methods: {[m for m in meths if any(k in m.lower() for k in ('matrix','energy','solve','fci','diag','many','second','build'))]}", flush=True)
        # does the matrix even change with U?
        H0 = np.asarray(MetallicHamiltonian(mol, '1d_chain', -1.0, 0.0, 0.0, False).to_matrix())
        H8 = np.asarray(MetallicHamiltonian(mol, '1d_chain', -1.0, 0.0, 8.0, False).to_matrix())
        print(f"  ||H(U=8) - H(U=0)|| = {np.linalg.norm(H8-H0):.4f}  (0 => U dropped entirely)", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  inspect CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    print("\n=== (2) Ionic builder-CI variational soundness vs HF ===", flush=True)
    from kanad import MolecularBuilder
    from pyscf import gto, scf
    for name, atoms, basis, no, nv in [
        ('LiF', [('Li', (0, 0, 0)), ('F', (0, 0, 1.564))], 'sto-3g', 4, 4),
        ('NaCl', [('Na', (0, 0, 0)), ('Cl', (0, 0, 2.36))], 'sto-3g', 4, 4),
    ]:
        try:
            mol = gto.M(atom=atoms, basis=basis, verbose=0); mf = scf.RHF(mol).run(verbose=0)
            hf = float(mf.e_tot)
            qs = (MolecularBuilder.from_atoms(atoms).basis(basis)
                  .active_space('frontier', n_occ=no, n_virt=nv).solver('ci').build())
            e_b = qs.solve()['energy']
            ok = 'OK (≤HF)' if e_b <= hf + 1e-6 else '*** ABOVE HF — NOT variational! ***'
            print(f"  {name:4} HF={hf:.6f}  builder-CI={e_b:.6f}  ΔvsHF={(e_b-hf)*1000:+.2f} mHa  {ok}", flush=True)
        except Exception as e:
            print(f"  {name:4} CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    print("\nVERIFY_DONE", flush=True)


if __name__ == "__main__":
    main()
