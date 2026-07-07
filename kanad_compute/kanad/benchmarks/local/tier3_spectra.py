"""Round 3 — spectra vs REAL experimental measurements.
UV-Vis: pyscf TDA (CIS baseline) + framework CASCI excited_states, bright-band λmax vs experiment.
IR: framework FrequencyCalculator (scaled ~0.89 for sto-3g) vs experimental fundamentals.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier3_spectra
"""
from __future__ import annotations
import traceback
import numpy as np

EV2NM = 1239.84149


def _atoms(geom):
    out = []
    for p in geom.strip().strip(';').split(';'):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def _qs(smiles=None, geom=None, basis='sto-3g', n_occ=None, n_virt=None, charge=0, spin=0):
    from kanad import MolecularBuilder
    if smiles:
        b = MolecularBuilder.from_smiles(smiles, basis)
    else:
        b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
        if charge: b = b.charge(charge)
        if spin: b = b.spin(spin)
    if n_occ is not None:
        b = b.active_space('frontier', n_occ=n_occ, n_virt=n_virt)
    return b.solver('ci').build()


def pyscf_tda(mf, n_states=6):
    """CIS/TDA bright-state λmax (nm) from a RHF mean-field."""
    from pyscf import tdscf
    td = tdscf.TDA(mf); td.nstates = n_states; td.kernel()
    e_ev = np.asarray(td.e) * 27.211386
    try:
        f = np.asarray(td.oscillator_strength())
    except Exception:
        f = np.ones_like(e_ev)
    bright = int(np.argmax(f))
    return float(EV2NM / e_ev[bright]), [round(float(x), 2) for x in e_ev[:n_states]], [round(float(x), 3) for x in f[:n_states]]


def uvvis(name, exp_nm, exp_note, **kw):
    n_states = kw.pop('n_states', 6)
    r = {'name': name, 'exp_nm': exp_nm, 'note': exp_note}
    try:
        qs = _qs(**kw)
        # pyscf TDA (CIS baseline) on the mean field
        try:
            tda_nm, tda_ev, tda_f = pyscf_tda(qs.mf, n_states)
            r['tda_bright_nm'] = round(tda_nm, 1)
        except Exception as e:
            r['tda_err'] = f"{type(e).__name__}:{str(e)[:50]}"
        # framework CASCI excited states (active space)
        try:
            ex = qs.excited_states(n_states=n_states)
            ev = [float(x) for x in (ex.get('excitation_energies_ev') or ex.get('energies') or [])]
            ev = [e for e in ev if e > 0.05]
            osc = ex.get('oscillator_strengths')
            if osc and len(osc) == len(ev) + 1:
                osc = osc[1:]
            if ev:
                bright = int(np.argmax(osc)) if osc else 0
                r['casci_bright_nm'] = round(EV2NM / ev[bright], 1)
                r['casci_ev'] = [round(e, 2) for e in ev[:n_states]]
        except Exception as e:
            r['casci_err'] = f"{type(e).__name__}:{str(e)[:50]}"
        r['status'] = 'ok'
    except Exception as e:
        r['status'] = 'crash'; r['error'] = f"{type(e).__name__}: {str(e)[:90]}"
    return r


def ir(name, geom, exp_bands, basis='sto-3g', scale=0.89):
    r = {'name': name, 'exp_cm': exp_bands}
    try:
        from kanad.analysis.vibrational_analysis import FrequencyCalculator
        from kanad.core.molecule import Molecule
        from kanad.core.atom import Atom
        atoms = [Atom(e, np.array(x)) for e, x in _atoms(geom)]
        mol = Molecule(atoms, basis=basis)
        fc = FrequencyCalculator(mol)
        res = fc.compute_frequencies()
        freqs = res.get('frequencies_cm') or res.get('frequencies') or res
        freqs = np.asarray([f for f in np.atleast_1d(freqs) if np.isreal(f) and float(np.real(f)) > 200])
        top = sorted([round(scale * float(np.real(f)), 0) for f in freqs], reverse=True)[:6]
        r.update(status='ok', scaled_cm=top)
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:90]}")
    return r


def fmt(r):
    s = f"SPEC| {r['name']:30} [{r.get('status','?'):6}]"
    for k in ('exp_nm', 'tda_bright_nm', 'casci_bright_nm', 'casci_ev', 'exp_cm', 'scaled_cm', 'note', 'tda_err', 'casci_err', 'error'):
        if k in r: s += f" | {k}={r[k]}"
    return s


def main():
    print("=" * 110, flush=True)
    print("ROUND 3 — spectra vs experiment (UV-Vis: pyscf TDA + framework CASCI; IR: scaled harmonic)", flush=True)
    print("=" * 110, flush=True)
    UV = [
        dict(name='benzene', exp_nm='255(forbidden)/204/178(bright)', exp_note='vapor', smiles='c1ccccc1', n_occ=3, n_virt=3),
        dict(name='naphthalene', exp_nm=275, exp_note='1La cyclohexane', smiles='c1ccc2ccccc2c1', n_occ=4, n_virt=4),
        dict(name='anthracene', exp_nm=356, exp_note='1La/S1', smiles='c1ccc2cc3ccccc3cc2c1', n_occ=3, n_virt=3),
        dict(name='tetracene', exp_nm=474, exp_note='S1 visible', smiles='c1ccc2cc3cc4ccccc4cc3cc2c1', n_occ=4, n_virt=4),
        dict(name='pyrene', exp_nm='372(S1weak)/334(S2strong)', exp_note='cyclohexane', smiles='c1cc2ccc3cccc4ccc(c1)c2c34', n_occ=4, n_virt=4),
        dict(name='azulene_ANOMALY', exp_nm='580(S1)/340(S2)', exp_note='CIS-fails-doubles', smiles='c1ccc2cccc2cc1', n_occ=4, n_virt=4),
        dict(name='trans_azobenzene', exp_nm=320, exp_note='pi-pi*', smiles='c1ccc(cc1)/N=N/c1ccccc1', n_occ=4, n_virt=4),
        dict(name='pNitroaniline_CT', exp_nm=380, exp_note='charge-transfer', smiles='Nc1ccc(cc1)[N+](=O)[O-]', n_occ=4, n_virt=4),
    ]
    IRs = [
        dict(name='water', geom='O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469', exp_bands='1595(bend),3657,3756(O-H)'),
        dict(name='formaldehyde', geom='C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588', exp_bands='1746(C=O),2782,2843(C-H)'),
        dict(name='ammonia', geom='N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; H -0.81 -0.47 -0.27', exp_bands='950(umbrella),3337,3444(N-H)'),
        dict(name='methanol', geom='C 0 0 0; O 1.43 0 0; H -0.36 1.03 0; H -0.36 -0.51 0.89; H -0.36 -0.51 -0.89; H 1.74 0.89 0', exp_bands='1030(C-O),3681(O-H)'),
        dict(name='CO2', geom='C 0 0 0; O 0 0 1.16; O 0 0 -1.16', exp_bands='667(bend),2349(asym)'),
    ]
    res = []
    for u in UV:
        r = uvvis(**u); res.append(r); print(fmt(r), flush=True)
    for d in IRs:
        r = ir(**d); res.append(r); print(fmt(r), flush=True)
    from collections import Counter
    print("-" * 110, flush=True)
    print(f"SPEC_SUMMARY | {dict(Counter(r.get('status') for r in res))}", flush=True)
    print("SPEC_DONE", flush=True)


if __name__ == "__main__":
    main()
