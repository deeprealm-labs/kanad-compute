"""Round 3 — harder correlations AT SCALE. Per-experiment isolated + failure-tolerant.
Energies vs CASCI/direct_spin0; NOON ladders; spin_s S-T gaps; open-shell full; 34q BlueQubit.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier3_scale [index]
"""
from __future__ import annotations
import sys, time, traceback
import numpy as np


def _atoms(geom):
    out = []
    for p in geom.strip().strip(';').split(';'):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def hring(n, R):
    r = R / (2 * np.sin(np.pi / n))
    return [('H', (r * np.cos(2 * np.pi * k / n), r * np.sin(2 * np.pi * k / n), 0.0)) for k in range(n)]


def casci_full(atoms, basis, spin=0):
    """direct_spin0 (singlet) energy + NOON for a full-active-space molecule."""
    from pyscf import gto, scf, ao2mo, fci, mcscf
    mol = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis=basis, spin=spin, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    ncas = mf.mo_coeff.shape[1]; ne = mol.nelectron
    cas = mcscf.CASCI(mf, ncas, ne); h1, ec = cas.get_h1eff(); h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
    e, v = fci.direct_spin0.kernel(h1, h2, ncas, ne, ecore=ec)
    dm = fci.direct_spin0.make_rdm1(v, ncas, ne)
    noon = np.sort(np.linalg.eigvalsh(dm))[::-1]
    return float(e), noon


def run(i):
    t0 = time.time()
    try:
        # ---------- 0: N2 dissociation curve (16q, manual CAS vs CASCI) ----------
        if i == 0:
            from kanad import MolecularBuilder
            from pyscf import gto, scf, mcscf
            out = []
            for R in (1.0, 1.10, 1.30, 1.60, 2.00, 2.50, 3.00):
                mol = gto.M(atom=f'N 0 0 0; N 0 0 {R}', basis='cc-pvdz', verbose=0); mf = scf.RHF(mol).run(verbose=0)
                cas = mcscf.CASCI(mf, 8, 6); cas.fix_spin_(ss=0); e_cas = float(cas.run(verbose=0).e_tot)
                qs = (MolecularBuilder.from_atoms([('N', (0, 0, 0)), ('N', (0, 0, R))]).basis('cc-pvdz')
                      .active_space('manual', frozen=[0, 1, 2, 3, 4], active=[5, 6, 7, 8, 9, 10, 11, 12])
                      .solver('sqd', n_samples=60000, max_iterations=5, random_seed=0, spin_s=0.0).build())
                e = qs.solve()['energy']; out.append(f"R={R}:{(e-e_cas)*1000:+.2f}mHa")
            return f"SCALE| N2_dissociation_curve_16q | gaps_vs_CASCI(8,6): {' '.join(out)} | t={time.time()-t0:.0f}s"

        # ---------- 1: H12/H14 Mott rings (full, vs direct_spin0) ----------
        if i == 1:
            from kanad import MolecularBuilder
            out = []
            for n in (12, 14):
                for R in (0.9, 2.5):
                    atoms = hring(n, R)
                    e_ref, noon_ref = casci_full(atoms, 'sto-3g')
                    qs = (MolecularBuilder.from_atoms(atoms).basis('sto-3g').active_space('full')
                          .solver('sqd', n_samples=40000, max_iterations=6, recovery_rounds=2, random_seed=0, spin_s=0.0).build())
                    r = qs.solve(); o = qs.observables('core')
                    noon = np.round(sorted(o['natural_orbital_occupations'], reverse=True)[:3], 3).tolist()
                    out.append(f"H{n}@{R}:gap={(r['energy']-e_ref)*1000:+.1f}mHa NOONtop={noon} M={o['m_diagnostic']:.2f}")
            return f"SCALE| H12_H14_Mott_rings | {' || '.join(out)} | t={time.time()-t0:.0f}s"

        # ---------- 2: naphthalene->anthracene AVAS-pi NOON ladder ----------
        if i == 2:
            from kanad import MolecularBuilder
            out = []
            for nm, smi in [('naphthalene', 'c1ccc2ccccc2c1'), ('anthracene', 'c1ccc2cc3ccccc3cc2c1')]:
                qs = (MolecularBuilder.from_smiles(smi, 'sto-3g').active_space('avas', ao_labels=['C 2pz'])
                      .solver('sqd', n_samples=80000, max_iterations=6, recovery_rounds=2, random_seed=0, spin_s=0.0).build())
                r = qs.solve(); o = qs.observables('core')
                noon = np.array(sorted(o['natural_orbital_occupations'], reverse=True))
                hono, luno = float(noon[len(noon)//2 - 1]), float(noon[len(noon)//2])
                out.append(f"{nm}({qs.n_qubits}q):HONO={hono:.3f} LUNO={luno:.3f} M={o['m_diagnostic']:.2f}")
            return f"SCALE| acene_NOON_ladder | {' | '.join(out)} | t={time.time()-t0:.0f}s"

        # ---------- 3: anthracene S-T gap via spin_s (28q AVAS-pi) ----------
        if i == 3:
            from kanad import MolecularBuilder
            def e_spin(s):
                qs = (MolecularBuilder.from_smiles('c1ccc2cc3ccccc3cc2c1', 'sto-3g').active_space('avas', ao_labels=['C 2pz'])
                      .solver('sqd', n_samples=80000, max_iterations=6, recovery_rounds=2, random_seed=0, spin_s=s).build())
                r = qs.solve(); o = qs.observables('core'); return r['energy'], o.get('s_squared'), qs.n_qubits
            es, sss, nq = e_spin(0.0); et, sst, _ = e_spin(1.0)
            return f"SCALE| anthracene_ST_gap_{nq}q | S0={es:.6f}(S2={sss}) T1={et:.6f}(S2={sst}) gap={(et-es)*627.509:+.2f}kcal | t={time.time()-t0:.0f}s"

        # ---------- 4: TiH open-shell 'full' (32q) + confirm AVAS wall ----------
        if i == 4:
            from kanad import MolecularBuilder
            walls = []
            for strat, kw in [('avas', dict(ao_labels=['Ti 3d'])), ('frozen_core', {})]:
                try:
                    MolecularBuilder.from_atoms([('Ti', (0, 0, 0)), ('H', (0, 0, 1.78))]).basis('sto-3g').spin(3).active_space(strat, **kw).solver('sqd', n_samples=100).build()
                    walls.append(f"{strat}:NO-WALL(built)")
                except Exception as e:
                    walls.append(f"{strat}:walled({type(e).__name__})")
            try:
                qs = (MolecularBuilder.from_atoms([('Ti', (0, 0, 0)), ('H', (0, 0, 1.78))]).basis('sto-3g').spin(3)
                      .active_space('full').solver('sqd', n_samples=40000, max_iterations=6, recovery_rounds=2, random_seed=0, spin_s=1.5).build())
                r = qs.solve(); o = qs.observables('core')
                full = f"E={r['energy']:.6f} S2={o.get('s_squared')} n_unp={o['n_unpaired_electrons']:.2f} dip={o['dipole_magnitude_debye']:.3f}D nq={qs.n_qubits}"
            except Exception as e:
                full = f"full-CRASH {type(e).__name__}: {str(e)[:80]}"
            return f"SCALE| TiH_openshell_full_32q | walls=[{','.join(walls)}] | quartet: {full} | t={time.time()-t0:.0f}s"

        # ---------- 5: pentacene mp2no CAS(<=17) 34q on BlueQubit ----------
        if i == 5:
            import os
            from kanad import MolecularBuilder
            be = 'bluequbit' if os.environ.get('BLUEQUBIT_API_KEY') else 'statevector'
            qs = (MolecularBuilder.from_smiles('c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1', 'sto-3g')
                  .active_space('mp2no', max_orbitals=17, occ_threshold=0.02).backend(be)
                  .solver('sqd', n_samples=50000, max_iterations=6, recovery_rounds=3, random_seed=0, spin_s=0.0).build())
            r = qs.solve(); o = qs.observables('core')
            noon = np.array(sorted(o['natural_orbital_occupations'], reverse=True))
            hono, luno = float(noon[len(noon)//2 - 1]), float(noon[len(noon)//2])
            return f"SCALE| pentacene_mp2no_{qs.n_qubits}q_{be} | E={r['energy']:.6f} dets={r.get('n_determinants')} HONO={hono:.3f} LUNO={luno:.3f} M={o['m_diagnostic']:.2f} | t={time.time()-t0:.0f}s"

    except Exception as e:
        return f"SCALE| EXP_{i} | CRASH {type(e).__name__}: {str(e)[:120]} | {traceback.format_exc().splitlines()[-2:]}"


N = 6
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(run(int(sys.argv[1])), flush=True)
    else:
        for i in range(N):
            print(run(i), flush=True)
        print("SCALE_DONE", flush=True)
