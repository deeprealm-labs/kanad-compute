"""Tier-5 frontier / domain / DFT-fails batch (cluster-feasible, statevector SQD).

Covers: Cr2 (sextuple-bond DFT disaster), Mott H10 ring (NOON collapse, exact 20q anchor),
trimethylene diradical, glycine (amino-acid bio fragment), pyridine (FBDD reactivity descriptors).
Each: SQD vs CASCI and vs DFT where meaningful; prints TIER5 lines + TIER5_DONE.

    PYTHONPATH=/tmp/kanad-pkg /root/miniconda3/bin/python -m benchmarks.tier5_frontier
"""
from __future__ import annotations
import time
import numpy as np
from kanad import MolecularBuilder


def _t(fn, *a, **k):
    t0 = time.time(); r = fn(*a, **k); return r, time.time() - t0


def cr2():
    """Cr2 — the multireference DFT disaster (formal sextuple bond)."""
    from pyscf import gto, scf, mcscf, dft
    try:
        mol = gto.M(atom='Cr 0 0 0; Cr 0 0 1.68', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        # AVAS on Cr 3d+4s
        from pyscf.mcscf import avas
        ncas, nelecas, mo, *_ = avas.avas(mf, ['Cr 3d', 'Cr 4s'], canonicalize=True)
        cas = mcscf.CASCI(mf, ncas, nelecas); cas.fcisolver.max_cycle = 300
        e_cas = float(cas.run(mo, verbose=0).e_tot)
        e_dft = float(dft.RKS(mol).set(xc='b3lyp').run(verbose=0).e_tot)
        qs = (MolecularBuilder.from_atoms([('Cr', (0, 0, 0)), ('Cr', (0, 0, 1.68))]).basis('cc-pvdz')
              .active_space('avas', ao_labels=['Cr 3d', 'Cr 4s'])
              .solver('sqd', n_samples=120000, max_iterations=5, energy_tol=1e-6, random_seed=0).build())
        res, dt = _t(qs.solve)
        nq = 2 * ncas
        print(f"TIER5 | Cr2 AVAS({nelecas},{ncas}) | {nq}q | CASCI={e_cas:.6f} | SQD={res['energy']:.6f} | "
              f"gap={((res['energy']-e_cas)*1000):+.3f}mHa | DFT(b3lyp)={e_dft:.6f} | dets={res.get('n_determinants')} | t={dt:.0f}s", flush=True)
    except Exception as e:
        print(f"TIER5 | Cr2 | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)


def mott_h10():
    """Half-filled H10 ring — Mott transition, NOON-collapse; 20q routes to exact CASCI anchor."""
    try:
        import numpy as np
        for R in (1.0, 1.8, 3.0):
            theta = 2 * np.pi / 10
            atoms = [('H', (1.0 / (2 * np.sin(theta / 2)) * R * np.cos(i * theta),
                            1.0 / (2 * np.sin(theta / 2)) * R * np.sin(i * theta), 0.0)) for i in range(10)]
            qs = (MolecularBuilder.from_atoms(atoms).basis('sto-3g')
                  .active_space('manual', frozen=[], active=list(range(10)))
                  .solver('sqd', n_samples=100000, max_iterations=5, energy_tol=1e-6, random_seed=0).build())
            res, dt = _t(qs.solve)
            obs = qs.observables('core') if hasattr(qs, 'observables') else {}
            noon = obs.get('natural_orbital_occupations') or obs.get('noon')
            noon_s = (np.round(np.asarray(noon), 3).tolist() if noon is not None else 'n/a')
            print(f"TIER5 | Mott H10 R={R}A CAS(10,10) | 20q | SQD={res['energy']:.6f} | dets={res.get('n_determinants')} | "
                  f"NOON={noon_s} | t={dt:.0f}s", flush=True)
    except Exception as e:
        print(f"TIER5 | Mott H10 | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)


def trimethylene():
    """Trimethylene 1,3-diradical (open ring) — CAS(6,6)/12q vs CASCI."""
    from pyscf import gto, scf, mcscf
    try:
        atom = 'C 0 0 0; C 1.5 0 0; C 3.0 0 0; H -0.6 0.9 0; H -0.6 -0.9 0; H 1.5 0.9 0.6; H 1.5 -0.9 0.6; H 3.6 0.9 0; H 3.6 -0.9 0'
        mol = gto.M(atom=atom, basis='sto-3g', spin=0, verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        ncas, ne = 6, 6
        cas = mcscf.CASCI(mf, ncas, ne); cas.fix_spin_(ss=0); e_cas = float(cas.run(verbose=0).e_tot)
        n_occ = mol.nelectron // 2
        qs = (MolecularBuilder.from_atoms([(p.split()[0], tuple(map(float, p.split()[1:]))) for p in atom.split(';')]).basis('sto-3g')
              .active_space('frontier', n_occ=3, n_virt=3).solver('sqd', n_samples=80000, max_iterations=5, random_seed=0).build())
        res, dt = _t(qs.solve)
        print(f"TIER5 | trimethylene CAS(6,6) | 12q | CASCI={e_cas:.6f} | SQD={res['energy']:.6f} | "
              f"gap={((res['energy']-e_cas)*1000):+.3f}mHa | t={dt:.0f}s", flush=True)
    except Exception as e:
        print(f"TIER5 | trimethylene | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)


def glycine():
    """Glycine (amino acid) — active-space SQD + properties (bio fragment)."""
    try:
        atom = [('N', (-1.5, 0.1, 0)), ('C', (0.0, 0.5, 0)), ('C', (1.1, -0.5, 0)),
                ('O', (2.3, -0.1, 0)), ('O', (0.9, -1.7, 0)),
                ('H', (-1.6, -0.9, 0)), ('H', (0.1, 1.2, 0.8)), ('H', (1.5, -2.2, 0))]
        qs = (MolecularBuilder.from_atoms(atom).basis('sto-3g')
              .active_space('frontier', n_occ=3, n_virt=3).solver('sqd', n_samples=80000, max_iterations=4, random_seed=0).build())
        res, dt = _t(qs.solve)
        obs = qs.observables('core') if hasattr(qs, 'observables') else {}
        dip = obs.get('dipole_magnitude_debye', obs.get('dipole_magnitude'))
        print(f"TIER5 | glycine CAS(6,6) frontier | 12q | SQD={res['energy']:.6f} | dipole={dip} | "
              f"dets={res.get('n_determinants')} | t={dt:.0f}s", flush=True)
    except Exception as e:
        print(f"TIER5 | glycine | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)


def pyridine_fbdd():
    """Pyridine — FBDD reactivity descriptors (Fukui / omega / eta / HOMO-LUMO)."""
    try:
        atom = [('N', (0, 1.40, 0)), ('C', (1.20, 0.70, 0)), ('C', (1.20, -0.70, 0)),
                ('C', (0, -1.40, 0)), ('C', (-1.20, -0.70, 0)), ('C', (-1.20, 0.70, 0)),
                ('H', (2.13, 1.25, 0)), ('H', (2.13, -1.25, 0)), ('H', (0, -2.49, 0)),
                ('H', (-2.13, -1.25, 0)), ('H', (-2.13, 1.25, 0))]
        qs = MolecularBuilder.from_atoms(atom).basis('sto-3g').solver('vqe').build()
        desc = qs.reactivity_descriptors() if hasattr(qs, 'reactivity_descriptors') else {}
        keys = {k: desc.get(k) for k in ('chi', 'eta', 'S', 'omega', 'homo', 'lumo', 'electronegativity', 'hardness', 'electrophilicity')}
        print(f"TIER5 | pyridine FBDD descriptors | {keys}", flush=True)
    except Exception as e:
        print(f"TIER5 | pyridine FBDD | FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)


def main():
    print("=" * 90, flush=True)
    print("TIER 5 — frontier / domain / DFT-fails (cluster-feasible)", flush=True)
    print("=" * 90, flush=True)
    mott_h10()
    trimethylene()
    glycine()
    pyridine_fbdd()
    cr2()  # heaviest last
    print("TIER5_DONE", flush=True)


if __name__ == "__main__":
    main()
