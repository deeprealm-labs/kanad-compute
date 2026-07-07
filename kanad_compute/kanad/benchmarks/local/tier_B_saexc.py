"""Campaign B — SA-CASSCF + per-root NEVPT2 EXCITED STATES through the INTEGRATED
builder path (`excited_states(..., orbital_optimization=True, pt2_correction='nevpt2')`).

Walks the ladder  CASCI(canonical) -> SA-CASSCF -> SA-CASSCF+NEVPT2  on the SAME
active space and compares vertical excitation energies (and oscillator strengths)
to experiment / CASPT2 / MRCI. Includes the classic single-reference FAILURE cases:
  * butadiene 2¹Ag (doubly-excited DARK state) vs 1¹Bu (bright) — TD-DFT gets the
    ordering/character wrong; multireference is required.
  * PSB3 (minimal retinal protonated Schiff-base) S0->S1 — the chromophore class
    where the spectra workflow is meant to win.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_B_saexc
"""
from __future__ import annotations
import traceback
import numpy as np
EV = 27.211386245988


def _atoms(geom):
    return [(p.split()[0], tuple(map(float, p.split()[1:4])))
            for p in geom.strip().strip(';').split(';') if len(p.split()) >= 4]


def ladder(geom, basis, n_occ, n_virt, n_states, charge=0, spin=0):
    """Return per-rung dicts of {energies, exc_ev, osc} for
    CASCI(canonical) / SA-CASSCF / SA-CASSCF+NEVPT2 via the builder."""
    from kanad import MolecularBuilder
    rungs = (('casci',  {}),
             ('sacasscf', {'orbital_optimization': True}),
             ('nevpt2',  {'orbital_optimization': True, 'pt2_correction': 'nevpt2'}))
    out = {}
    for lab, kw in rungs:
        try:
            b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
            if charge:
                b = b.charge(charge)
            if spin:
                b = b.spin(spin)
            qs = b.active_space('frontier', n_occ=n_occ, n_virt=n_virt).solver('ci').build()
            r = qs.excited_states(n_states=n_states, **kw)
            out[lab] = {
                'exc_ev': [float(x) for x in r['excitation_energies_ev']],
                'osc': [float(x) for x in r.get('oscillator_strengths', [])],
                'method': r['method'],
            }
        except Exception as e:
            out[lab] = {'err': f"{type(e).__name__}: {str(e)[:70]}"}
            out[lab + '_tb'] = traceback.format_exc()
    return out


def show(name, out, anchors):
    """anchors: list of (state_index, label, truth_ev)."""
    print(f"\n--- {name} ---", flush=True)
    for lab in ('casci', 'sacasscf', 'nevpt2'):
        d = out.get(lab, {})
        if 'err' in d:
            print(f"  {lab:9} CRASH {d['err']}", flush=True)
            continue
        ev = d['exc_ev']
        osc = d['osc']
        cells = []
        for si, slabel, truth in anchors:
            if si < len(ev):
                f = osc[si] if si < len(osc) else float('nan')
                err = ev[si] - truth if truth is not None else float('nan')
                cells.append(f"{slabel}={ev[si]:5.2f}eV(f={f:.3f}|Δ{err:+.2f})")
        print(f"  {lab:9} [{d.get('method','')}]  " + "   ".join(cells), flush=True)


def main():
    print("=" * 110, flush=True)
    print("CAMPAIGN B — SA-CASSCF + NEVPT2 EXCITED STATES (integrated builder path) vs exp/CASPT2", flush=True)
    print("=" * 110, flush=True)

    # 1. Formaldehyde n->pi* — exp 3.94 eV (dark, symmetry-forbidden f~0). Plumbing + anchor.
    h2co = 'C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
    show('H2CO  n->pi* (exp 3.94 eV; dark)', ladder(h2co, 'cc-pvdz', 3, 3, 3),
         [(1, 'S1', 3.94)])

    # 2. Ethylene pi->pi* — vertical ~7.8 eV (bright, f large). Oscillator-strength sanity.
    c2h4 = 'C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321; H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321'
    show('C2H4  pi->pi* (vert ~7.8 eV; BRIGHT)', ladder(c2h4, 'cc-pvdz', 4, 4, 2),
         [(1, 'S1', 7.8)])

    # 3. Butadiene — the SINGLE-REFERENCE FAILURE: 1Bu bright (~6.2 eV) and 2Ag DARK
    #    doubly-excited (~6.6 eV exp/MRCI). TD-DFT misses 2Ag entirely (no double exc).
    #    SA-CASSCF+NEVPT2 should recover BOTH and put 2Ag in the right place.
    bd = ('C 0 0 0; C 0 1.343 0; C 1.164 2.014 0; C 1.164 3.357 0;'
          'H -0.94 -0.53 0; H 0.94 -0.53 0; H -0.78 1.87 0; H 2.10 1.48 0;'
          'H 1.16 3.90 0; H 2.10 3.88 0')
    show('butadiene 1Bu/2Ag (exp ~6.2/6.6 eV; 2Ag DOUBLY-EXC, TD-DFT FAILS)',
         ladder(bd, 'cc-pvdz', 4, 4, 4),
         [(1, 'St1', 6.2), (2, 'St2', 6.6), (3, 'St3', None)])

    # 4. PSB2 minimal protonated Schiff base CH2=CH-CH=NH2+ (retinal chromophore
    #    class), S0->S1 bright charge-transfer pi->pi*. C3H6N+ = 30 e- (even, S=0).
    #    Flagship for the spectra workflow; CASPT2(2-double-bond model) ~4.7 eV.
    psb2 = ('C 0 0 0; C 0 1.40 0; C 1.21 2.10 0; N 1.21 3.45 0;'
            'H -0.93 -0.54 0; H 0.93 -0.54 0; H -0.93 1.94 0; H 2.14 1.56 0;'
            'H 0.31 4.00 0; H 2.11 3.99 0')
    show('PSB2 retinal-class S0->S1 (CASPT2 ~4.7 eV; bright CT; flagship)',
         ladder(psb2, '6-31g', 3, 3, 2, charge=1),
         [(1, 'S1', 4.7)])

    print("\nSAEXC_DONE", flush=True)


if __name__ == "__main__":
    main()
