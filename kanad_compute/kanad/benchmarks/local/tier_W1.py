"""Wave-1 core-fix regression — exercises the previously-UNCOVERED rotated-orbital
(mp2no / AVAS) active-space paths that masked B3/B4/B5/B6.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_W1
"""
from __future__ import annotations
import numpy as np
from kanad import MolecularBuilder

AU2D = 2.541746230211

LIH = [('Li', (0, 0, 0)), ('H', (0, 0, 1.595))]
N2 = [('N', (0, 0, 0)), ('N', (0, 0, 1.10))]
H2O = [('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))]


def dipole_from_ao(mol, dm_ao):
    with mol.with_common_orig((0, 0, 0)):
        r = mol.intor('int1e_r', comp=3)
    el = -np.einsum('xij,ji->x', r, np.asarray(dm_ao))
    nuc = np.einsum('a,ax->x', mol.atom_charges(), mol.atom_coords())
    return float(np.linalg.norm(el + nuc) * AU2D)


def fci_dipole(atoms, basis):
    from pyscf import gto, scf, fci
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    cis = fci.FCI(mf)
    _, ci = cis.kernel()
    nmo = mf.mo_coeff.shape[1]
    dm1_mo = cis.make_rdm1(ci, nmo, mol.nelectron)
    dm1_ao = mf.mo_coeff @ dm1_mo @ mf.mo_coeff.T
    return dipole_from_ao(mol, dm1_ao), float(mf.e_tot)


def main():
    print("=" * 96, flush=True)
    print("WAVE-1 CORE FIX REGRESSION (B3 AVAS, B4/B6 rotated-MO density, B5 mp2no e-count)", flush=True)
    print("=" * 96, flush=True)

    # ---------- B3: AVAS builds an active space (was: ValueError every call) ----------
    try:
        qs = (MolecularBuilder.from_atoms(N2).basis('sto-3g')
              .active_space('avas', ao_labels=['N 2p']).solver('ci').build())
        e = qs.solve()['energy']
        print(f"[B3] AVAS N2 builds + solves OK: nq={qs.n_qubits}  E={e:.6f}", flush=True)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"[B3] *** AVAS FAILED: {type(ex).__name__}: {str(ex)[:90]}", flush=True)

    # ---------- B4/B6: rotated-MO (mp2no) density embedding ----------
    # The active 1-RDM lives in the NATURAL-orbital basis; embedding it with canonical
    # mf.mo_coeff (the bug) gives a physically wrong AO density (silent — trace still N).
    # Decisive check: dipole from the FIXED embedding must (a) match the full-FCI dipole
    # and (b) DIFFER from the buggy mf.mo_coeff embedding.
    from kanad.core.density.density_storage import embed_active_to_full_mo, mo_to_ao_1rdm
    qs = (MolecularBuilder.from_atoms(LIH).basis('sto-3g')
          .active_space('mp2no', occ_threshold=0.01).solver('ci').build())
    qs.solve()
    ham = qs.hamiltonian
    mol = ham.mol
    acs = ham.active_space
    # Known-correct reference: observables() override rebuilds D_AO from acs.mo_coeff
    # (the natural orbitals) for the frozen-non-empty case — independent of the path we fixed.
    mu_ref = float(np.linalg.norm(qs.observables('core')['dipole_debye']))
    rdm_act = qs._active_rdm1()
    # FIXED path under test (set_quantum_density_matrix now uses acs.mo_coeff)
    ham._quantum_density_matrix_ao = None
    ham.set_quantum_density_matrix(rdm_act)
    mu_fixed = dipole_from_ao(mol, ham.get_density_matrix('ao'))
    # BUGGY path reproduced inline (canonical mf.mo_coeff)
    Cmf = np.asarray(ham.mf.mo_coeff)
    full_buggy = embed_active_to_full_mo(rdm_act, list(acs.frozen_indices),
                                         list(acs.active_indices), Cmf.shape[1])
    mu_buggy = dipole_from_ao(mol, mo_to_ao_1rdm(full_buggy, Cmf))
    rot = float(np.linalg.norm(np.asarray(acs.mo_coeff) - Cmf))
    ok = abs(mu_fixed - mu_ref) < 1e-4 and abs(mu_buggy - mu_ref) > 0.05
    print(f"[B4] LiH mp2no dipole: FIXED={mu_fixed:.4f}  buggy(mf.C)={mu_buggy:.4f}  "
          f"correct_ref(acs.C)={mu_ref:.4f} D  |C_no-C_mf|={rot:.3f}  -> "
          f"{'PASS (fixed==ref, buggy!=ref)' if ok else '*** FAIL'}", flush=True)

    # ---------- B5: mp2no max_orbitals must conserve electrons (no over-fill / non-variational) ----------
    for name, atoms, mo in (('N2', N2, 3), ('N2', N2, 4)):
        try:
            mu_ref, hf = fci_dipole(atoms, 'sto-3g')
            qs = (MolecularBuilder.from_atoms(atoms).basis('sto-3g')
                  .active_space('mp2no', occ_threshold=0.02, max_orbitals=mo).solver('ci').build())
            r = qs.solve()
            e = r['energy']
            na = qs.n_electrons
            no = qs.n_orbitals
            okc = (na % 2 == 0) and (na <= 2 * no) and (e <= hf + 1e-6)
            print(f"[B5] {name} mp2no(max_orb={mo}): n_active_e={na} n_orb={no} "
                  f"E={e:.6f} (HF={hf:.4f}) variational&conserved={'PASS' if okc else '*** FAIL'}", flush=True)
        except Exception as ex:
            print(f"[B5] {name} *** FAILED: {type(ex).__name__}: {str(ex)[:80]}", flush=True)

    # ---------- canonical path unchanged (frozen_core dipole still sane) ----------
    qs = (MolecularBuilder.from_atoms(H2O).basis('sto-3g')
          .active_space('frozen_core').solver('ci').build())
    qs.solve()
    mu = float(np.linalg.norm(qs.observables('core')['dipole_debye']))
    print(f"[reg] H2O frozen_core dipole = {mu:.3f} D (canonical path, expect ~1.5-2.0)", flush=True)

    print("\nW1_DONE", flush=True)


if __name__ == "__main__":
    main()
