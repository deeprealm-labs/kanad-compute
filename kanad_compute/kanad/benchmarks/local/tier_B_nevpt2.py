"""Campaign B — quantify the no-PT2 vulnerability + prototype the NEVPT2/CASSCF fix.

The whole frontier matrix points to ONE root limit: Kanad's CASCI has no dynamical-
correlation layer (no CASPT2/NEVPT2) and no orbital optimization (canonical HF orbitals).
This decomposes the error vs TRUTH for each system into:
  CASCI(canonical)  ->  +CASSCF (orbital opt)  ->  +NEVPT2 (dynamical correlation)
and compares to FCI / CASPT2 / experiment. Also shows where CCSD(T) (classical gold
standard) FAILS so we see what the PT2-corrected multireference path buys.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_B_nevpt2
"""
from __future__ import annotations
import traceback
import numpy as np
EV = 27.211386245988
KCAL = 627.509


def cas_chain(atom, basis, ncas, nelecas, spin=0, charge=0, nroots=1):
    """Return HF, CCSD(T), CASCI, CASSCF, NEVPT2(CASCI), NEVPT2(CASSCF), FCI(if feasible)."""
    from pyscf import gto, scf, mcscf, cc, mrpt, fci, ao2mo
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    out = {'hf': float(mf.e_tot)}
    try:
        m = cc.CCSD(mf).run(verbose=0); out['ccsdt'] = float(m.e_tot + m.ccsd_t())
    except Exception:
        out['ccsdt'] = float('nan')
    # CASCI on canonical HF orbitals (what Kanad does)
    mc = mcscf.CASCI(mf, ncas, nelecas)
    if spin == 0:
        try: mc.fix_spin_(ss=0)
        except Exception: pass
    mc.run(verbose=0); out['casci'] = float(mc.e_tot)
    try: out['nevpt2_casci'] = float(mc.e_tot + mrpt.NEVPT(mc).kernel())
    except Exception as e: out['nevpt2_casci'] = float('nan'); out['nev_ci_err'] = str(e)[:40]
    # CASSCF (orbital optimization) + NEVPT2 (the full multireference fix)
    try:
        ms = mcscf.CASSCF(mf, ncas, nelecas)
        if spin == 0:
            try: ms.fix_spin_(ss=0)
            except Exception: pass
        ms.run(verbose=0); out['casscf'] = float(ms.e_tot)
        out['nevpt2_casscf'] = float(ms.e_tot + mrpt.NEVPT(ms).kernel())
    except Exception as e:
        out['casscf'] = float('nan'); out['nevpt2_casscf'] = float('nan'); out['cs_err'] = str(e)[:40]
    return out


def main():
    print("=" * 110, flush=True)
    print("CAMPAIGN B — no-PT2 vulnerability + NEVPT2/CASSCF prototype (vs FCI/CASPT2/experiment)", flush=True)
    print("=" * 110, flush=True)

    # ---- 1. Cyclobutadiene automerization barrier: CCSD(T) FAILS; FCI/CIPSI TBE = 8.9 kcal/mol ----
    print("\n--- Cyclobutadiene D2h->D4h automerization barrier (TRUTH: FCI/CIPSI ~8.9 kcal/mol) ---", flush=True)
    rect = 'C 0.673 0.783 0; C -0.673 0.783 0; C -0.673 -0.783 0; C 0.673 -0.783 0; H 1.41 1.43 0; H -1.41 1.43 0; H -1.41 -1.43 0; H 1.41 -1.43 0'
    sq = 'C 0.723 0.723 0; C -0.723 0.723 0; C -0.723 -0.723 0; C 0.723 -0.723 0; H 1.43 1.43 0; H -1.43 1.43 0; H -1.43 -1.43 0; H 1.43 -1.43 0'
    try:
        r = cas_chain(rect, 'cc-pvdz', 4, 4); s = cas_chain(sq, 'cc-pvdz', 4, 4)
        for lab in ('ccsdt', 'casci', 'nevpt2_casci', 'casscf', 'nevpt2_casscf'):
            b = (s[lab] - r[lab]) * KCAL
            print(f"  {lab:14} barrier = {b:+6.2f} kcal/mol  (err vs 8.9 = {b-8.9:+.2f})", flush=True)
    except Exception as e:
        print(f"  cyclobutadiene CRASH {type(e).__name__}: {str(e)[:90]}", flush=True)
        traceback.print_exc()

    # ---- 2. N2 strong-correlation point R=2.0 (TRUTH: FCI in CAS) ----
    print("\n--- N2 @ R=2.0 A stretched, CAS(10,8) (TRUTH: FCI-in-CAS) ---", flush=True)
    try:
        from pyscf import gto, scf, mcscf, ao2mo, fci
        mol = gto.M(atom='N 0 0 0; N 0 0 2.0', basis='cc-pvdz', verbose=0); mf = scf.RHF(mol).run(verbose=0)
        c = cas_chain('N 0 0 0; N 0 0 2.0', 'cc-pvdz', 8, 10)
        mc = mcscf.CASCI(mf, 8, 10); h1, ec = mc.get_h1eff(); h2 = ao2mo.restore(1, mc.get_h2eff(), 8)
        efci, _ = fci.direct_spin0.kernel(h1, h2, 8, 10, ecore=ec)
        for lab in ('ccsdt', 'casci', 'nevpt2_casci', 'casscf', 'nevpt2_casscf'):
            print(f"  {lab:14} = {c[lab]:.6f}  gap_vs_FCI(CAS) = {(c[lab]-efci)*1000:+.2f} mHa", flush=True)
        print(f"  FCI-in-CAS(10,8) = {efci:.6f}", flush=True)
    except Exception as e:
        print(f"  N2 CRASH {type(e).__name__}: {str(e)[:90]}", flush=True)

    # ---- 3. Formaldehyde n->pi* vertical excitation (TRUTH: exp ~3.94 eV) — state-specific via nroots ----
    print("\n--- H2CO n->pi* vertical (TRUTH: exp 3.94 eV, CASPT2 ~3.9-4.0) ---", flush=True)
    try:
        from pyscf import gto, scf, mcscf, mrpt
        mol = gto.M(atom='C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588', basis='cc-pvdz', verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        # CASCI nroots=2 (S0, S1) on canonical orbitals
        mc = mcscf.CASCI(mf, 6, 6); mc.fcisolver.nroots = 2; mc.fix_spin_(ss=0); mc.run(verbose=0)
        dE_casci = (mc.e_tot[1] - mc.e_tot[0]) * EV
        # SA-CASSCF(2 states) + NEVPT2 per state
        ms = mcscf.CASSCF(mf, 6, 6); ms.fcisolver.nroots = 2
        ms = mcscf.addons.state_average_(ms, [0.5, 0.5]); ms.run(verbose=0)
        e_states = ms.e_states if hasattr(ms, 'e_states') else ms.e_tot
        dE_sacas = (e_states[1] - e_states[0]) * EV
        print(f"  CASCI(6,6) n->pi*       = {dE_casci:.3f} eV  (err vs 3.94 = {dE_casci-3.94:+.3f})", flush=True)
        print(f"  SA-CASSCF(6,6) n->pi*   = {dE_sacas:.3f} eV  (err vs 3.94 = {dE_sacas-3.94:+.3f})", flush=True)
    except Exception as e:
        print(f"  H2CO CRASH {type(e).__name__}: {str(e)[:90]}", flush=True)
        traceback.print_exc()

    print("\nNEVPT2_DONE", flush=True)


if __name__ == "__main__":
    main()
