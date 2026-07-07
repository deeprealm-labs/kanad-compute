"""Campaign B — validate CHARACTER-BASED state selection (resolves the diffuse-basis
Rydberg-intrusion problem the basis sweep exposed). Confirms:
  * each state is tagged bright/dark + valence/rydberg via f and Δ⟨r²⟩,
  * select='brightest'/'valence' returns the intended valence band even when a dark
    Rydberg state is the lowest root (ethylene/aug-cc-pVDZ),
  * butadiene dark-2Ag vs bright-1Bu tags come out right.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_B_select
"""
from __future__ import annotations
import numpy as np


def _atoms(geom):
    return [(p.split()[0], tuple(map(float, p.split()[1:4])))
            for p in geom.strip().strip(';').split(';') if len(p.split()) >= 4]


def run(name, geom, basis, n_occ, n_virt, n_states, charge=0, **sel):
    from kanad import MolecularBuilder
    b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
    if charge:
        b = b.charge(charge)
    qs = b.active_space('frontier', n_occ=n_occ, n_virt=n_virt).solver('ci').build()
    full = qs.excited_states(n_states=n_states, orbital_optimization=True, pt2_correction='nevpt2')
    print(f"\n--- {name} [{basis}] ---", flush=True)
    ev = full['excitation_energies_ev']
    osc = full['oscillator_strengths']
    ext = full.get('state_extent_r2') or [None] * len(ev)
    ch = full.get('character') or []
    for i in range(len(ev)):
        r2 = f"{ext[i]:+6.1f}" if ext[i] is not None else "  n/a"
        print(f"   state {i}: {ev[i]:6.2f} eV  f={osc[i]:.3f}  Δ⟨r²⟩={r2}  -> {ch[i] if i < len(ch) else '?'}", flush=True)
    # selection variants
    for mode in ('brightest', 'valence', 'bright'):
        r = qs.excited_states(n_states=n_states, orbital_optimization=True,
                              pt2_correction='nevpt2', select=mode)
        idx = r.get('selected_indices')
        evs = [round(x, 2) for x in r['excitation_energies_ev']]
        fs = [round(x, 3) for x in r['oscillator_strengths']]
        print(f"   select={mode:9} -> kept idx {idx}  exc_eV={evs}  f={fs}", flush=True)


def main():
    print("=" * 100, flush=True)
    print("CAMPAIGN B — character-based state selection (bright/dark, valence/rydberg)", flush=True)
    print("=" * 100, flush=True)

    # Ethylene in aug-cc-pVDZ: the problem case — dark Rydberg pi->3s is the low root.
    # select='valence'/'brightest' must skip it and return the bright valence V state.
    run('C2H4 pi->pi* (aug → Rydberg intrusion)',
        'C 0 0 0.6695; C 0 0 -0.6695; H 0 0.9289 1.2321; H 0 -0.9289 1.2321;'
        'H 0 0.9289 -1.2321; H 0 -0.9289 -1.2321',
        'aug-cc-pvdz', 4, 4, 4)

    # Butadiene: dark doubly-excited 2Ag should tag 'dark/valence', bright 1Bu 'bright/valence'.
    run('butadiene 2Ag(dark)/1Bu(bright)',
        'C 0 0 0; C 0 1.343 0; C 1.164 2.014 0; C 1.164 3.357 0;'
        'H -0.94 -0.53 0; H 0.94 -0.53 0; H -0.78 1.87 0; H 2.10 1.48 0;'
        'H 1.16 3.90 0; H 2.10 3.88 0',
        'cc-pvdz', 4, 4, 4)

    print("\nSELECT_DONE", flush=True)


if __name__ == "__main__":
    main()
