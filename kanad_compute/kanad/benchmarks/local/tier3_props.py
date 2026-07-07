"""Round 3 — property / analysis calculator audit on diverse molecules.
Exercises observables('core'/'all'), reactivity descriptors, thermochemistry,
UV-Vis, Mayer/charges; compares dipole to pyscf RHF. Failure-tolerant.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier3_props
"""
from __future__ import annotations
import traceback
import numpy as np


def _atoms(geom):
    out = []
    for p in geom.strip().strip(';').split(';'):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def pyscf_rhf_dipole(geom, basis, charge=0, spin=0):
    from pyscf import gto, scf
    mol = gto.M(atom=geom, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    return float(np.linalg.norm(mf.dip_moment(unit='Debye', verbose=0)))


def _build(geom, basis, asp, solver='ci', charge=0, spin=0, **sk):
    from kanad import MolecularBuilder
    b = MolecularBuilder.from_atoms(_atoms(geom)).basis(basis)
    if charge: b = b.charge(charge)
    if spin: b = b.spin(spin)
    m = asp['m']
    if m == 'frontier': b = b.active_space('frontier', n_occ=asp['n_occ'], n_virt=asp['n_virt'])
    elif m == 'manual': b = b.active_space('manual', frozen=asp['frozen'], active=asp['active'])
    elif m == 'full': b = b.active_space('full')
    return b.solver(solver, **sk).build()


def case_dipole_charges(name, geom, basis, asp):
    """observables dipole vs pyscf RHF; report charges/NOON/M."""
    r = {'name': name, 'probe': 'dipole+charges+NOON'}
    try:
        ref = pyscf_rhf_dipole(geom, basis)
        qs = _build(geom, basis, asp, solver='ci'); qs.solve()
        o = qs.observables('core')
        dip = o['dipole_magnitude_debye']
        r.update(status='ok', dipole=round(dip, 4), dipole_pyscf_rhf=round(ref, 4),
                 dipole_dev=round(dip - ref, 4), density_source=o.get('density_source'),
                 m_diag=round(float(o['m_diagnostic']), 3),
                 noon_sum=round(float(sum(o['natural_orbital_occupations'])), 3),
                 homo_lumo_ev=round(float(o['homo_lumo_gap_ev']), 3))
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def case_observables_all(name, geom, basis, asp):
    """observables('all') polarizability + NMR (provisional)."""
    r = {'name': name, 'probe': "observables('all')"}
    try:
        qs = _build(geom, basis, asp, solver='ci'); qs.solve()
        o = qs.observables('all')
        r.update(status='ok',
                 polarizability_au=o.get('polarizability_mean_au'),
                 nmr_keys=[k for k in o if 'nmr' in k.lower() or 'shield' in k.lower()],
                 dipole=round(o['dipole_magnitude_debye'], 4))
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def case_thermo(name, geom, basis):
    r = {'name': name, 'probe': 'thermochemistry'}
    try:
        from kanad.analysis.thermochemistry import ThermochemistryCalculator
        from pyscf import gto, scf
        mol = gto.M(atom=geom, basis=basis, verbose=0); mf = scf.RHF(mol).run(verbose=0)
        tc = ThermochemistryCalculator(mf) if _accepts(ThermochemistryCalculator, mf) else ThermochemistryCalculator()
        # best-effort: many APIs — just probe what's callable
        meths = [m for m in dir(tc) if not m.startswith('_') and callable(getattr(tc, m))]
        r.update(status='ok', methods=meths[:8])
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def _accepts(cls, *a):
    try:
        cls(*a); return True
    except Exception:
        return False


def case_uvvis(name, geom, basis, asp, n_states=4):
    r = {'name': name, 'probe': 'UV-Vis CIS excitations'}
    try:
        qs = _build(geom, basis, asp, solver='ci')
        ex = qs.excited_states(n_states=n_states)
        r.update(status='ok', exc_ev=[round(float(x), 3) for x in
                 (ex.get('excitation_energies_ev') or ex.get('energies') or [])][:n_states],
                 osc=ex.get('oscillator_strengths'))
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def case_reactivity(name, smiles):
    r = {'name': name, 'probe': 'reactivity_descriptors'}
    try:
        from kanad import MolecularBuilder
        import dataclasses as dc
        qs = (MolecularBuilder.from_smiles(smiles, 'sto-3g')
              .active_space('frontier', n_occ=3, n_virt=3).solver('ci').build())
        qr = qs.reactivity_descriptors()['quantum_reactivity']
        d = dc.asdict(qr) if dc.is_dataclass(qr) else vars(qr)
        r.update(status='ok', chi=round(d['electronegativity_ev'], 3),
                 eta=round(d['hardness_ev'], 3), omega=round(d['electrophilicity_ev'], 3),
                 gap=round(d['gap_ev'], 3), source=d.get('source'))
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def case_cross_route(name, geom, basis, asp_ci, asp_sqd):
    """CI vs SQD dipole/NOON consistency + density_source labels."""
    r = {'name': name, 'probe': 'CI-vs-SQD consistency'}
    try:
        qci = _build(geom, basis, asp_ci, solver='ci'); qci.solve(); oci = qci.observables('core')
        qsd = _build(geom, basis, asp_sqd, solver='sqd', n_samples=60000, max_iterations=5,
                     random_seed=0); qsd.solve(); osd = qsd.observables('core')
        r.update(status='ok', ci_src=oci.get('density_source'), sqd_src=osd.get('density_source'),
                 dip_ci=round(oci['dipole_magnitude_debye'], 4), dip_sqd=round(osd['dipole_magnitude_debye'], 4),
                 dip_dev=round(oci['dipole_magnitude_debye'] - osd['dipole_magnitude_debye'], 4))
    except Exception as e:
        r.update(status='crash', error=f"{type(e).__name__}: {str(e)[:110]}")
    return r


def fmt(r):
    base = f"PROP| {r['name']:30} [{r.get('status','?'):6}] {r['probe']:24}"
    for k, v in r.items():
        if k in ('name', 'status', 'probe', 'error'): continue
        base += f" | {k}={v}"
    if r.get('error'): base += f" | {r['error']}"
    return base


def main():
    print("=" * 110, flush=True)
    print("ROUND 3 — property / analysis calculator audit", flush=True)
    print("=" * 110, flush=True)
    results = []
    # geometries
    H2O = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'
    NH3 = 'N 0 0 0.12; H 0 0.94 -0.27; H 0.81 -0.47 -0.27; H -0.81 -0.47 -0.27'
    urea = 'C 0 0 0.13; O 0 0 1.34; N 1.16 0 -0.60; N -1.16 0 -0.60; H 1.20 0 -1.61; H 2.00 0 -0.10; H -1.20 0 -1.61; H -2.00 0 -0.10'
    formamide = 'C 0 0 0; O 1.21 0 0; N -0.72 1.11 0; H -0.30 -0.94 0; H -0.30 1.97 0; H -1.73 1.06 0'
    CH4 = 'C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63'
    H2CO = 'C 0 0 0; O 0 0 1.208; H 0 0.943 -0.588; H 0 -0.943 -0.588'
    pyridine = 'N 0 1.40 0; C 1.20 0.70 0; C 1.20 -0.70 0; C 0 -1.40 0; C -1.20 -0.70 0; C -1.20 0.70 0; H 2.13 1.25 0; H 2.13 -1.25 0; H 0 -2.49 0; H -2.13 -1.25 0; H -2.13 1.25 0'
    O3 = 'O 0 0 0; O 1.143 0.571 0; O -1.143 0.571 0'

    # Lean, fast cases (small active spaces; CI optimization is the cost driver, so keep <=12q).
    # observables('all') finite-field polarizability dropped (re-solves x6 fields -> too slow at this scale).
    CASES = [
        lambda: case_dipole_charges('urea_dipole_Mayer', urea, 'sto-3g', dict(m='frontier', n_occ=2, n_virt=2)),
        lambda: case_dipole_charges('formamide_dipole', formamide, 'sto-3g', dict(m='frontier', n_occ=2, n_virt=2)),
        lambda: case_dipole_charges('H2O_dipole_vs_pyscf', H2O, 'sto-3g', dict(m='frontier', n_occ=2, n_virt=3)),
        lambda: case_uvvis('H2CO_uvvis', H2CO, 'sto-3g', dict(m='frontier', n_occ=3, n_virt=3)),
        lambda: case_dipole_charges('O3_multiref_indicators', O3, 'sto-3g', dict(m='frontier', n_occ=3, n_virt=3)),
        lambda: case_cross_route('H2O_CI_vs_SQD', H2O, 'sto-3g', dict(m='frontier', n_occ=2, n_virt=3), dict(m='frontier', n_occ=4, n_virt=4)),
        lambda: case_reactivity('aspirin_reactivity', 'CC(=O)Oc1ccccc1C(=O)O'),
    ]
    import os as _os
    only = _os.environ.get('PROP_ONLY')
    for j, fn in enumerate(CASES):
        if only is not None and str(j) != only:
            continue
        r = fn(); results.append(r); print(fmt(r), flush=True)
    from collections import Counter
    print("-" * 110, flush=True)
    print(f"PROP_SUMMARY | {dict(Counter(r.get('status') for r in results))}", flush=True)
    print("PROP_DONE", flush=True)


if __name__ == "__main__":
    main()
