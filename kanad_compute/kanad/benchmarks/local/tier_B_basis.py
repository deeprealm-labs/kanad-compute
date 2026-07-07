"""Campaign B — DIVERSE BASIS-SET exploration of the SA-CASSCF + NEVPT2 excited-state
ladder through the INTEGRATED builder path.  Stop living in cc-pVDZ; sweep basis
FAMILIES and sizes and watch the physics move.

Part 1 — Ethylene pi->pi* basis-convergence sweep.  The V (pi->pi*) state is diffuse
  / valence-Rydberg mixed, so a compact basis (cc-pVDZ) overshoots (~8.8 eV).  Adding
  diffuse (aug-) and going to triple-zeta should pull NEVPT2 toward the best estimate
  ~7.8-8.0 eV.  A clean, honest demonstration that the error there is basis, not method.

Part 2 — Diverse FAMILIES on diverse chromophores, each in a different basis:
  Pople (6-311+G*), Ahlrichs def2 (SVP/TZVP), Dunning cc / aug-cc.  vs exp / CASPT2.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_B_basis
"""
from __future__ import annotations
import traceback
import numpy as np


def _atoms(geom):
    return [(p.split()[0], tuple(map(float, p.split()[1:4])))
            for p in geom.strip().strip(';').split(';') if len(p.split()) >= 4]


def exc(geom, basis, n_occ, n_virt, n_states, charge=0, spin=0,
        orbital_optimization=True, pt2='nevpt2'):
    """One integrated SA-CASSCF(+NEVPT2) excited-state call → (exc_ev, osc, method, nbf)."""
    from kanad import MolecularBuilder
    b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
    if charge:
        b = b.charge(charge)
    if spin:
        b = b.spin(spin)
    qs = b.active_space('frontier', n_occ=n_occ, n_virt=n_virt).solver('ci').build()
    kw = {}
    if orbital_optimization:
        kw['orbital_optimization'] = True
    if pt2:
        kw['pt2_correction'] = pt2
    r = qs.excited_states(n_states=n_states, **kw)
    nbf = getattr(getattr(qs, 'mf', None), 'mol', None)
    nbf = nbf.nao_nr() if nbf is not None else -1
    return ([float(x) for x in r['excitation_energies_ev']],
            [float(x) for x in r.get('oscillator_strengths', [])],
            r['method'], int(nbf), r.get('pt2_applied'))


def main():
    print("=" * 110, flush=True)
    print("CAMPAIGN B — DIVERSE BASIS SETS on the SA-CASSCF+NEVPT2 excited-state ladder", flush=True)
    print("=" * 110, flush=True)

    # ---------- Part 1: ethylene pi->pi* basis-convergence sweep (truth ~7.8-8.0 eV) ----------
    print("\n=== Part 1: C2H4 pi->pi* (V state) basis sweep — CAS(4,4), SA(2)-CASSCF+NEVPT2 ===", flush=True)
    print("    (best estimate ~7.8-8.0 eV; cc-pVDZ overshoots — watch diffuse/TZ pull it down)", flush=True)
    c2h4 = ('C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321;'
            'H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321')
    sweep = ['sto-3g', '6-31g', 'cc-pvdz', 'def2-svp', '6-311+g(d)',
             'aug-cc-pvdz', 'def2-tzvp', 'cc-pvtz', 'aug-cc-pvtz']
    for bas in sweep:
        try:
            ev, osc, meth, nbf, ok = exc(c2h4, bas, 4, 4, 2)
            f = osc[1] if len(osc) > 1 else float('nan')
            print(f"  {bas:13} nbf={nbf:3d}  S1(pi->pi*) = {ev[1]:5.2f} eV  f={f:.3f}  "
                  f"(Δ vs 7.9 = {ev[1]-7.9:+.2f})  pt2={ok}", flush=True)
        except Exception as e:
            print(f"  {bas:13} CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    # ---------- Part 2: diverse families on diverse chromophores ----------
    print("\n=== Part 2: diverse basis FAMILIES on diverse chromophores ===", flush=True)

    cases = [
        # (name, geom, basis(FAMILY), n_occ, n_virt, n_states, charge, spin, state_idx, label, truth_ev)
        ('H2CO n->pi*  [Ahlrichs def2-TZVP]',
         'C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588',
         'def2-tzvp', 3, 3, 3, 0, 0, 1, 'S1', 3.94),
        ('butadiene 1Bu/2Ag  [Pople 6-311+G(d)]',
         'C 0 0 0; C 0 1.343 0; C 1.164 2.014 0; C 1.164 3.357 0;'
         'H -0.94 -0.53 0; H 0.94 -0.53 0; H -0.78 1.87 0; H 2.10 1.48 0;'
         'H 1.16 3.90 0; H 2.10 3.88 0',
         '6-311+g(d)', 4, 4, 4, 0, 0, 2, 'bright', 6.6),
        ('pyrrole pi->pi*  [Dunning aug-cc-pVDZ]',
         'N 0 0 1.12; C 1.12 0 0.33; C 0.71 0 -0.97; C -0.71 0 -0.97; C -1.12 0 0.33;'
         'H 0 0 2.13; H 2.13 0 0.70; H 1.36 0 -1.83; H -1.36 0 -1.83; H -2.13 0 0.70',
         'aug-cc-pvdz', 4, 4, 4, 0, 0, 2, 'pi->pi*', 6.0),
        ('PSB2 retinal-class CT  [def2-SVP vs def2-TZVP below]',
         'C 0 0 0; C 0 1.40 0; C 1.21 2.10 0; N 1.21 3.45 0;'
         'H -0.93 -0.54 0; H 0.93 -0.54 0; H -0.93 1.94 0; H 2.14 1.56 0;'
         'H 0.31 4.00 0; H 2.11 3.99 0',
         'def2-svp', 3, 3, 2, 1, 0, 1, 'S1', 4.7),
    ]
    for (name, geom, bas, no, nv, ns, ch, sp, si, slab, truth) in cases:
        try:
            ev, osc, meth, nbf, ok = exc(geom, bas, no, nv, ns, charge=ch, spin=sp)
            f = osc[si] if si < len(osc) else float('nan')
            d = ev[si] - truth if (si < len(ev) and truth is not None) else float('nan')
            print(f"  {name}", flush=True)
            print(f"      nbf={nbf:3d} [{meth}]  {slab}={ev[si]:5.2f} eV  f={f:.3f}  "
                  f"(Δ vs {truth} = {d:+.2f})", flush=True)
        except Exception as e:
            print(f"  {name}  CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)
            traceback.print_exc()

    # PSB2 in a LARGER def2 basis to show the basis pulling the CT state down
    print("  --- PSB2 same state, def2-TZVP (triple-zeta) for comparison ---", flush=True)
    try:
        ev, osc, meth, nbf, ok = exc(cases[3][1], 'def2-tzvp', 3, 3, 2, charge=1)
        print(f"      nbf={nbf:3d} [{meth}]  S1={ev[1]:5.2f} eV  f={osc[1]:.3f}  "
              f"(Δ vs 4.7 = {ev[1]-4.7:+.2f})", flush=True)
    except Exception as e:
        print(f"      PSB2/def2-tzvp CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    print("\nBASIS_DONE", flush=True)


if __name__ == "__main__":
    main()
