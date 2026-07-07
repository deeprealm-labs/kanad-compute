"""Campaign B — frontier feasible batch through the INTEGRATED builder path:
CASCI -> +CASSCF (orbital opt) -> +NEVPT2 (dynamical correlation), on multireference
systems where classical single-reference (DFT/CCSD(T)) fails, vs FCI/MRCI/experiment.
Demonstrates the new pt2_correction integration on real frontier chemistry.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_B_frontier
"""
from __future__ import annotations
import traceback
import numpy as np
KCAL = 627.509


def _atoms(geom):
    return [(p.split()[0], tuple(map(float, p.split()[1:4]))) for p in geom.strip().strip(';').split(';') if len(p.split()) >= 4]


def ci_ladder(geom, basis, asp, charge=0, spin=0):
    """builder CI energies: bare CASCI, +CASSCF, +CASSCF+NEVPT2 on the same active space."""
    from kanad import MolecularBuilder
    out = {}
    for lab, kw in (('casci', {}), ('casscf', {'orbital_optimization': True}),
                    ('nevpt2', {'pt2_correction': 'nevpt2'})):
        b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
        if charge: b = b.charge(charge)
        if spin: b = b.spin(spin)
        m = asp['m']
        if m == 'frontier': b = b.active_space('frontier', n_occ=asp['n_occ'], n_virt=asp['n_virt'])
        elif m == 'manual': b = b.active_space('manual', frozen=asp['frozen'], active=asp['active'])
        elif m == 'full': b = b.active_space('full')
        try:
            out[lab] = b.solver('ci', **kw).build().solve()['energy']
        except Exception as e:
            out[lab] = float('nan'); out[lab + '_err'] = f"{type(e).__name__}:{str(e)[:40]}"
    return out


def ccsdt(geom, basis, charge=0, spin=0):
    from pyscf import gto, scf, cc
    mol = gto.M(atom=_atoms(geom), basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    try:
        m = cc.CCSD(mf).run(verbose=0); return float(m.e_tot + m.ccsd_t())
    except Exception:
        return float('nan')


def main():
    print("=" * 110, flush=True)
    print("CAMPAIGN B — frontier multireference via integrated CASCI->CASSCF->NEVPT2 (vs FCI/MRCI/exp)", flush=True)
    print("=" * 110, flush=True)

    # ---- Ozone O3: notorious multireference, MRCI/exp anchor ----
    print("\n--- Ozone O3 vs O2+O atomization-ish / correlation recovery (multireference) ---", flush=True)
    o3 = 'O 0 0 0; O 1.089 0.681 0; O -1.089 0.681 0'
    try:
        l = ci_ladder(o3, 'cc-pvdz', {'m': 'frontier', 'n_occ': 6, 'n_virt': 6})
        ct = ccsdt(o3, 'cc-pvdz')
        print(f"  CCSD(T)={ct:.6f}", flush=True)
        for k in ('casci', 'casscf', 'nevpt2'):
            print(f"  CI[{k:7}] = {l.get(k, float('nan')):.6f}  (Δ vs CASCI = {(l.get(k,float('nan'))-l['casci'])*1000:+.1f} mHa) {l.get(k+'_err','')}", flush=True)
    except Exception as e:
        print(f"  O3 CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- C2: strongest small-molecule correlation ----
    print("\n--- C2 equilibrium (strong correlation) ---", flush=True)
    try:
        l = ci_ladder('C 0 0 0; C 0 0 1.2425', 'cc-pvdz', {'m': 'frontier', 'n_occ': 4, 'n_virt': 4})
        ct = ccsdt('C 0 0 0; C 0 0 1.2425', 'cc-pvdz')
        print(f"  CCSD(T)={ct:.6f}", flush=True)
        for k in ('casci', 'casscf', 'nevpt2'):
            print(f"  CI[{k:7}] = {l.get(k, float('nan')):.6f}  (Δ vs CASCI = {(l.get(k,float('nan'))-l['casci'])*1000:+.1f} mHa)", flush=True)
    except Exception as e:
        print(f"  C2 CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- Cyclobutadiene automerization barrier with the integrated ladder (TRUTH 8.9 kcal/mol) ----
    print("\n--- Cyclobutadiene barrier via integrated ladder (TRUTH FCI/CIPSI 8.9 kcal/mol) ---", flush=True)
    rect = 'C 0.673 0.783 0; C -0.673 0.783 0; C -0.673 -0.783 0; C 0.673 -0.783 0; H 1.41 1.43 0; H -1.41 1.43 0; H -1.41 -1.43 0; H 1.41 -1.43 0'
    sq = 'C 0.723 0.723 0; C -0.723 0.723 0; C -0.723 -0.723 0; C 0.723 -0.723 0; H 1.43 1.43 0; H -1.43 1.43 0; H -1.43 -1.43 0; H 1.43 -1.43 0'
    try:
        # bigger pi+sigma active space CAS(4,4)->try frontier(4,4)
        lr = ci_ladder(rect, 'cc-pvdz', {'m': 'frontier', 'n_occ': 2, 'n_virt': 2})
        ls = ci_ladder(sq, 'cc-pvdz', {'m': 'frontier', 'n_occ': 2, 'n_virt': 2})
        for k in ('casci', 'casscf', 'nevpt2'):
            b = (ls.get(k, float('nan')) - lr.get(k, float('nan'))) * KCAL
            print(f"  CI[{k:7}] barrier = {b:+6.2f} kcal/mol  (err vs 8.9 = {b-8.9:+.2f})", flush=True)
    except Exception as e:
        print(f"  cyclobutadiene CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- CrMo heterobimetallic multiple bond (TM multireference) ----
    print("\n--- CrMo quintuple-bond region (TM multireference) ---", flush=True)
    try:
        l = ci_ladder('Cr 0 0 0; Mo 0 0 1.81', 'cc-pvdz', {'m': 'frontier', 'n_occ': 6, 'n_virt': 6})
        for k in ('casci', 'casscf', 'nevpt2'):
            print(f"  CI[{k:7}] = {l.get(k, float('nan')):.6f}  (Δ vs CASCI = {(l.get(k,float('nan'))-l['casci'])*1000:+.1f} mHa) {l.get(k+'_err','')}", flush=True)
    except Exception as e:
        print(f"  CrMo CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    print("\nFRONTIER_B_DONE", flush=True)


if __name__ == "__main__":
    main()
