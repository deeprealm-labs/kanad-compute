"""Campaign A — CBS (basis→reality) + spectroscopy vs EXPERIMENT.
Shows the gap to experiment is BASIS + active-space dynamic-correlation, and the
honest trade-off: Kanad's CAS captures static correlation (the H6/H8 win) but
misses out-of-CAS dynamic correlation that CCSD(T)/CBS gets for equilibrium De.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_A_cbs_spec
"""
from __future__ import annotations
import time, traceback
import numpy as np

EV = 27.211386245988
KCAL = 627.509


def ccsdt_cbs_atomization(mol_atom, atoms_list, charge=0):
    """De via CCSD(T)/cc-pV{D,T,Q}Z + Helgaker 2-pt CBS (TZ/QZ corr) — the dynamic-corr gold standard."""
    from pyscf import gto, scf, cc
    def e_ccsdt(atom, basis, spin=0, charge=0):
        mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
        mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
        m = cc.CCSD(mf).run(verbose=0); return float(m.e_tot + m.ccsd_t()), float(mf.e_tot)
    out = {}
    for b in ('cc-pvdz', 'cc-pvtz', 'cc-pvqz'):
        em, _ = e_ccsdt(mol_atom, b)
        ea = sum(e_ccsdt(a, b, spin=s, charge=0)[0] for a, s in atoms_list)
        out[b] = (ea - em) * EV  # De in eV
    # Helgaker X^-3 CBS extrapolation on TZ(3)/QZ(4) total De
    d3, d4 = out['cc-pvtz'], out['cc-pvqz']
    out['cbs'] = (4**3 * d4 - 3**3 * d3) / (4**3 - 3**3)
    return out


def run():
    from kanad import MolecularBuilder
    print("=" * 100, flush=True)
    print("CAMPAIGN A — CBS + SPECTROSCOPY vs EXPERIMENT", flush=True)
    print("=" * 100, flush=True)

    # ---- CBS dissociation energies vs experiment (CCSD(T)/CBS gold standard) ----
    # (name, molecule atom string, [(atom, spin)], exp De eV)
    CBS = [
        ('H2', 'H 0 0 0; H 0 0 0.7414', [('H', 1), ('H', 1)], 4.7466),
        ('HF', 'H 0 0 0; F 0 0 0.9168', [('H', 1), ('F', 1)], 6.121),
        ('N2', 'N 0 0 0; N 0 0 1.0977', [('N', 3), ('N', 3)], 9.902),
    ]
    for name, matom, atoms, exp in CBS:
        t0 = time.time()
        try:
            c = ccsdt_cbs_atomization(matom, atoms)
            print(f"CBS| {name:4} | exp De={exp:.3f}eV | CCSD(T): DZ={c['cc-pvdz']:.3f} TZ={c['cc-pvtz']:.3f} "
                  f"QZ={c['cc-pvqz']:.3f} CBS={c['cbs']:.3f}eV | CBS-exp={c['cbs']-exp:+.3f}eV | t={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"CBS| {name} | CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)

    # ---- Spectroscopy vs experiment ----
    # H2O dipole (exp 1.8546 D)
    try:
        qs = (MolecularBuilder.from_atoms([('O', (0, 0, 0)), ('H', (0, 0.7572, 0.5865)), ('H', (0, -0.7572, 0.5865))])
              .basis('cc-pvdz').active_space('frontier', n_occ=4, n_virt=4).solver('ci').build())
        qs.solve(); o = qs.observables('core')
        print(f"SPEC| H2O dipole | kanad={o['dipole_magnitude_debye']:.4f} D | exp=1.8546 D | dev={o['dipole_magnitude_debye']-1.8546:+.4f} D", flush=True)
    except Exception as e:
        print(f"SPEC| H2O dipole CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)

    # Formaldehyde n→π* (exp 3.94 eV) + ethylene π→π* (exp 7.93 eV) — singlet-pure excited states
    SPEC = [
        ('H2CO_npistar', [('C', (0, 0, 0)), ('O', (0, 0, 1.208)), ('H', (0, 0.943, -0.588)), ('H', (0, -0.943, -0.588))], 'cc-pvdz', 3, 3, 3.94),
        ('C2H4_pipistar', [('C', (0, 0, 0.668)), ('C', (0, 0, -0.668)), ('H', (0, 0.923, 1.231)), ('H', (0, -0.923, 1.231)), ('H', (0, 0.923, -1.231)), ('H', (0, -0.923, -1.231))], 'cc-pvdz', 3, 3, 7.93),
    ]
    for name, atoms, basis, no, nv, exp in SPEC:
        t0 = time.time()
        try:
            qs = MolecularBuilder.from_atoms(atoms).basis(basis).active_space('frontier', n_occ=no, n_virt=nv).solver('ci').build()
            ex = qs.excited_states(n_states=6)
            evs = [float(x) for x in ex['excitation_energies_ev']]
            osc = [float(x) for x in ex['oscillator_strengths']]
            # bright state = max oscillator strength
            bi = int(np.argmax(osc))
            print(f"SPEC| {name:14} | bright E={evs[bi]:.3f}eV f={osc[bi]:.3f} | exp={exp}eV | dev={evs[bi]-exp:+.3f}eV | all_ev={[round(e,2) for e in evs]} | t={time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"SPEC| {name} CRASH {type(e).__name__}: {str(e)[:70]}", flush=True)
    print("CBSSPEC_DONE", flush=True)


if __name__ == "__main__":
    run()
