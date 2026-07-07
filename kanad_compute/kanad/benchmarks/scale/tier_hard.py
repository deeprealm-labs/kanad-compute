"""Tier-HARD capability audit — diverse, challenging, failure-PROBING experiments.

Replaces toy H2 / repeated-N2. Each experiment targets a SPECIFIC predicted failure
(open-shell active space, SQD sample starvation on high configurational entropy,
frontier mis-selection, dipole embedding at large frozen-core, spin-state ordering,
diradical NOON, delocalization-error hole localization). FAILURE-TOLERANT: a crash /
large gap / variational violation is RECORDED, never aborts the batch.

Cluster only (235 GB, statevector):
    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg /root/miniconda3/bin/python -m benchmarks.tier_hard

One HARD| line per run + HARD_SUMMARY + HARD_DONE.
"""
from __future__ import annotations
import time, traceback
import numpy as np


# ---------- geometry helpers ----------
def _atoms(geom):
    out = []
    for p in geom.strip().strip(';').split(';'):
        t = p.split()
        if len(t) >= 4:
            out.append((t[0], (float(t[1]), float(t[2]), float(t[3]))))
    return out


def anthracene():
    """3 linearly-fused hexagons (C-C=1.40), 14 C + 10 H, planar z=0."""
    R = 1.40
    dx = 2 * R * np.cos(np.pi / 6)  # 2.424 horizontal period
    y = R * np.sin(np.pi / 6)       # 0.700
    centers = [0.0, dx, 2 * dx]
    C = []
    for cx in centers:                       # tops + bottoms
        C += [(cx, R), (cx, -R)]
    xs = sorted({round(cx + s * R * np.cos(np.pi / 6), 4) for cx in centers for s in (-1, 1)})
    for x in xs:                              # edge carbons at y=+/-0.7
        C += [(x, y), (x, -y)]
    C = sorted(set((round(a, 4), round(b, 4)) for a, b in C))
    cen = (dx, 0.0)
    atoms = [('C', (a, b, 0.0)) for a, b in C]
    # H on carbons with <3 C-neighbours (peripheral); point outward from centroid
    for (a, b) in C:
        nbr = sum(1 for (c, d) in C if 1e-3 < ((a - c) ** 2 + (b - d) ** 2) ** .5 < 1.55)
        if nbr < 3:
            v = np.array([a - cen[0], b - cen[1]]); v = v / (np.linalg.norm(v) + 1e-9)
            atoms.append(('H', (a + 1.09 * v[0], b + 1.09 * v[1], 0.0)))
    return atoms


# ---------- reference ----------
def casci_ref(geom_or_atoms, basis, n_orb, n_e, spin=0, charge=0, nroots=1):
    from pyscf import gto, scf, mcscf
    atom = geom_or_atoms if isinstance(geom_or_atoms, str) else \
        [(e, tuple(xyz)) for e, xyz in geom_or_atoms]
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    cas = mcscf.CASCI(mf, n_orb, n_e)
    cas.fcisolver.conv_tol = 1e-9; cas.fcisolver.max_cycle = 300
    if nroots > 1:
        cas.fcisolver.nroots = nroots
    if spin == 0:
        try: cas.fix_spin_(ss=0)
        except Exception: pass
    cas.run(verbose=0)
    if nroots > 1:
        return [float(x) for x in cas.e_tot]
    return float(cas.e_tot)


def _classify(gap_mha):
    a = abs(gap_mha)
    return 'pass' if a <= 1.6 else ('approx' if a <= 50 else 'large_gap')


# ---------- runner ----------
def run(exp):
    from kanad import MolecularBuilder
    r = {'name': exp['name'], 'nq': exp['n_qubits'], 'task': exp.get('task', 'energy'),
         'difficulty': exp.get('difficulty', '?')}
    t0 = time.time()
    try:
        atoms = anthracene() if exp.get('gen') == 'anthracene' else _atoms(exp['geometry'])
        b = MolecularBuilder.from_atoms(atoms).basis(exp['basis'])
        if exp.get('charge'): b = b.charge(exp['charge'])
        if exp.get('spin'):   b = b.spin(exp['spin'])
        asp = exp['as']
        if asp['m'] == 'manual':   b = b.active_space('manual', frozen=asp['frozen'], active=asp['active'])
        elif asp['m'] == 'frontier': b = b.active_space('frontier', n_occ=asp['n_occ'], n_virt=asp['n_virt'])
        elif asp['m'] == 'avas':   b = b.active_space('avas', ao_labels=asp['ao'])
        elif asp['m'] == 'mp2no':  b = b.active_space('mp2no', max_orbitals=asp.get('max_orbitals', 10), occ_threshold=asp.get('occ_threshold', 0.02))
        elif asp['m'] == 'full':   b = b.active_space('full')

        task = exp.get('task', 'energy')
        if task == 'excited':
            qs = b.solver('ci').build()
            ex = qs.excited_states(n_states=exp.get('n_states', 5), spin=exp.get('ex_spin'))
            r['excited'] = ex.get('excitation_energies_ev', ex.get('energies'))
            r['status'] = 'ran'
        else:
            sk = dict(n_samples=exp.get('n_samples', 80000), max_iterations=exp.get('max_iterations', 5),
                      energy_tol=1e-6, random_seed=0)
            qs = b.solver(exp.get('solver', 'sqd'), **sk).build()
            out = qs.solve()
            e = float(out['energy']); r['E'] = e; r['dets'] = out.get('n_determinants')
            ref = exp.get('_ref')
            if ref is None and exp.get('ref_calc'):
                ref = casci_ref(atoms, exp['basis'], exp['ref_calc'][0], exp['ref_calc'][1],
                                spin=exp.get('spin', 0), charge=exp.get('charge', 0))
            if ref is not None:
                r['ref'] = ref; gap = (e - ref) * 1000; r['gap'] = gap
                r['status'] = 'VARIATIONAL_VIOLATION' if e < ref - 1e-4 else _classify(gap)
            else:
                r['status'] = 'no_ref'
            if task == 'obs':
                try:
                    o = qs.observables('core')
                    r['dipole'] = round(float(o['dipole_magnitude_debye']), 3)
                    noon = o['natural_orbital_occupations']
                    r['noon_frontier'] = [round(float(x), 3) for x in sorted(noon)[len(noon)//2-2:len(noon)//2+2]]
                    r['m_diag'] = round(float(o['m_diagnostic']), 3)
                except Exception as e2:
                    r['obs_error'] = f"{type(e2).__name__}: {str(e2)[:70]}"
    except NotImplementedError as e:
        r['status'] = 'fenced'; r['error'] = f"NotImplementedError: {str(e)[:130]}"
    except Exception as e:
        r['status'] = 'crash'; r['error'] = f"{type(e).__name__}: {str(e)[:150]}"
        r['trace'] = traceback.format_exc().splitlines()[-2:]
    r['t'] = round(time.time() - t0, 1)
    return r


def fmt(r):
    s = f"HARD| {r['name']:44} | {r['nq']:2}q | {r['status']:20}"
    if 'E' in r: s += f" | E={r['E']:.6f}"
    if 'ref' in r: s += f" | ref={r['ref']:.6f} | gap={r.get('gap', float('nan')):+.3f}mHa"
    if 'excited' in r: s += f" | exc={r['excited']}"
    if 'dipole' in r: s += f" | dip={r['dipole']}D noon={r.get('noon_frontier')} M={r.get('m_diag')}"
    if r.get('obs_error'): s += f" | OBS_ERR={r['obs_error']}"
    if r.get('error'): s += f" | {r['error']}"
    s += f" | dets={r.get('dets')} t={r.get('t')}s"
    return s


def main():
    print("=" * 110, flush=True)
    print(f"TIER-HARD audit — {len(EXPERIMENTS)} failure-probing runs", flush=True)
    print("=" * 110, flush=True)
    results = []
    for exp in EXPERIMENTS:
        r = run(exp); results.append(r); print(fmt(r), flush=True)
    from collections import Counter
    print("-" * 110, flush=True)
    print(f"HARD_SUMMARY | {dict(Counter(x['status'] for x in results))}", flush=True)
    print("HARD_DONE", flush=True)
    return results


EXPERIMENTS = [
    # --- main-group multireference / dissociation ---
    {'name': 'C2_quadruple_bond_16q', 'geometry': 'C 0 0 0; C 0 0 1.2425', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 4}, 'n_qubits': 16, 'ref_calc': (8, 8), 'difficulty': 'hard', 'n_samples': 60000},
    {'name': 'O3_ozone_diradical_NOON_24q', 'geometry': 'O 0 0 0; O 1.1430 0.5713 0; O -1.1430 0.5713 0', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'ref_calc': (12, 12), 'task': 'obs', 'difficulty': 'hard', 'n_samples': 100000},
    {'name': 'F2_eq_28q', 'geometry': 'F 0 0 0; F 0 0 1.4119', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 7, 'n_virt': 7}, 'n_qubits': 28, 'ref_calc': (14, 14), 'difficulty': 'extreme', 'n_samples': 120000},
    {'name': 'F2_stretched_2re_28q', 'geometry': 'F 0 0 0; F 0 0 2.8238', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 7, 'n_virt': 7}, 'n_qubits': 28, 'ref_calc': (14, 14), 'difficulty': 'extreme', 'n_samples': 120000},
    {'name': 'P2_eq_24q', 'geometry': 'P 0 0 0; P 0 0 1.8934', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'ref_calc': (12, 12), 'difficulty': 'extreme', 'n_samples': 100000},
    {'name': 'P2_stretched_2re_24q', 'geometry': 'P 0 0 0; P 0 0 3.7868', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'ref_calc': (12, 12), 'difficulty': 'extreme', 'n_samples': 100000},
    {'name': 'Be2_vdw_min_24q', 'geometry': 'Be 0 0 0; Be 0 0 2.45', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 8}, 'n_qubits': 24, 'ref_calc': (8, 12), 'difficulty': 'extreme', 'n_samples': 100000},
    {'name': 'Be2_longrange_24q', 'geometry': 'Be 0 0 0; Be 0 0 4.00', 'basis': 'cc-pvdz',
     'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 8}, 'n_qubits': 24, 'ref_calc': (8, 12), 'difficulty': 'extreme', 'n_samples': 100000},
    # --- diradical / antiaromatic ---
    {'name': 'twisted_ethylene_90_ST_16q', 'geometry': 'C 0 0 0.667; C 0 0 -0.667; H 0.920 0 1.230; H -0.920 0 1.230; H 0 0.920 -1.230; H 0 -0.920 -1.230',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 4}, 'n_qubits': 16, 'ref_calc': (8, 8), 'task': 'obs', 'difficulty': 'hard', 'n_samples': 60000},
    {'name': 'cyclobutadiene_square_TS_16q', 'geometry': 'C 0.725 0.725 0; C -0.725 0.725 0; C -0.725 -0.725 0; C 0.725 -0.725 0; H 1.430 1.430 0; H -1.430 1.430 0; H -1.430 -1.430 0; H 1.430 -1.430 0',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 4}, 'n_qubits': 16, 'ref_calc': (8, 8), 'task': 'obs', 'difficulty': 'hard', 'n_samples': 60000},
    {'name': 'cyclobutadiene_rect_GS_16q', 'geometry': 'C 0.670 0.780 0; C -0.670 0.780 0; C -0.670 -0.780 0; C 0.670 -0.780 0; H 1.400 1.450 0; H -1.400 1.450 0; H -1.400 -1.450 0; H 1.400 -1.450 0',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 4, 'n_virt': 4}, 'n_qubits': 16, 'ref_calc': (8, 8), 'difficulty': 'hard', 'n_samples': 60000},
    {'name': 'para_benzyne_S0_24q', 'geometry': 'C 1.3970 0 0; C 0.6985 1.2098 0; C -0.6985 1.2098 0; C -1.3970 0 0; C -0.6985 -1.2098 0; C 0.6985 -1.2098 0; H 1.2410 2.1494 0; H -1.2410 2.1494 0; H -1.2410 -2.1494 0; H 1.2410 -2.1494 0',
     'basis': 'cc-pvdz', 'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'ref_calc': (12, 12), 'difficulty': 'hard', 'n_samples': 100000},
    # --- transition metal / open shell (predicted open-shell active-space failures) ---
    {'name': 'FeO_quintet_26q', 'geometry': 'Fe 0 0 0; O 0 0 1.616', 'basis': 'def2-svp', 'spin': 4,
     'as': {'m': 'manual', 'frozen': [], 'active': list(range(8, 21))}, 'n_qubits': 26, 'difficulty': 'hard', 'n_samples': 120000},
    {'name': 'FeO_triplet_26q', 'geometry': 'Fe 0 0 0; O 0 0 1.616', 'basis': 'def2-svp', 'spin': 2,
     'as': {'m': 'manual', 'frozen': [], 'active': list(range(8, 21))}, 'n_qubits': 26, 'difficulty': 'hard', 'n_samples': 120000},
    # --- delocalization error (open-shell cation, full space) ---
    {'name': 'H2O_dimer_cation_28q', 'geometry': 'O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0; O 0 0 2.800; H 0 -0.500 3.300; H 0.700 0.400 3.300',
     'basis': 'sto-3g', 'charge': 1, 'spin': 1, 'as': {'m': 'full'}, 'n_qubits': 28, 'task': 'obs', 'difficulty': 'extreme', 'n_samples': 120000},
    # --- polyradical scaling (AVAS pi) ---
    {'name': 'anthracene_pi_AVAS_28q', 'gen': 'anthracene', 'basis': 'sto-3g',
     'as': {'m': 'avas', 'ao': ['C 2pz']}, 'n_qubits': 28, 'task': 'obs', 'difficulty': 'extreme', 'n_samples': 120000},
    # --- bio / drug ---
    {'name': 'WC_GC_basepair_CT_24q', 'geometry': 'N -1.870 1.090 0; C -1.690 2.420 0; N -0.520 3.030 0; C 0.580 2.290 0; C 0.460 0.890 0; C -0.780 0.310 0; O -1.000 -0.910 0; N 1.790 2.840 0; C 2.870 2.010 0; N 2.760 0.680 0; N -2.790 3.200 0; H -2.690 0.490 0; H 3.840 2.430 0; H -3.690 2.760 0; H -2.690 4.210 0; H 3.560 0.060 0; N 1.700 -1.730 0; C 2.770 -0.890 0; O 3.940 -1.300 0; N 2.560 -3.080 0; C 1.370 -3.700 0; C 0.260 -2.930 0; C 0.470 -1.560 0; N 1.250 -5.040 0; H 3.430 -3.610 0; H -0.700 -3.390 0; H -0.390 -0.970 0; H 2.080 -5.600 0; H 0.350 -5.480 0',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'task': 'obs', 'ref_calc': (12, 12), 'difficulty': 'hard', 'n_samples': 100000},
    {'name': 'caffeine_dipole_24q', 'geometry': 'O 2.804 -1.453 0; C 1.838 -0.701 0; N 0.510 -1.155 0; C -0.589 -0.281 0; C -0.300 1.072 0; C 1.058 1.523 0; O 1.398 2.700 0; N 2.038 0.621 0; N -1.913 -0.524 0; C -2.479 0.681 0; N -1.581 1.667 0; C 0.279 -2.594 0; C 3.401 1.038 0; C -2.668 -1.760 0; H -3.553 0.812 0; H 0.793 -3.103 0.806; H 0.793 -3.103 -0.806; H -0.778 -2.836 0; H 3.974 0.745 0.879; H 3.974 0.745 -0.879; H 3.488 2.124 0; H -2.000 -2.620 0; H -3.310 -1.781 0.879; H -3.310 -1.781 -0.806',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 6, 'n_virt': 6}, 'n_qubits': 24, 'task': 'obs', 'ref_calc': (12, 12), 'difficulty': 'hard', 'n_samples': 100000},
    {'name': 'adenine_excited_npistar_20q', 'geometry': 'N 2.166 0.502 0; C 1.082 1.298 0; N -0.165 0.890 0; C -0.296 -0.460 0; C 0.847 -1.305 0; C 2.092 -0.852 0; N -1.604 -0.682 0; C -2.260 0.483 0; N -1.467 1.530 0; N 3.193 -1.640 0; H 1.231 2.371 0; H -3.337 0.560 0; H -1.794 2.487 0; H 4.085 -1.190 0; H 3.119 -2.647 0',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 5, 'n_virt': 5}, 'n_qubits': 20, 'task': 'excited', 'n_states': 5, 'difficulty': 'hard'},
    {'name': 'purine_mp2no_20q', 'geometry': 'N 1.336 1.117 0; C 0.219 1.853 0; N -0.997 1.391 0; C -1.080 0.044 0; C 0.057 -0.781 0; C 1.297 -0.243 0; N -2.150 -0.685 0; C -1.703 -1.927 0; N -0.383 -2.030 0; H 0.310 2.934 0; H 2.193 -0.857 0; H -2.357 -2.785 0; H 2.235 1.490 0',
     'basis': 'sto-3g', 'as': {'m': 'mp2no', 'max_orbitals': 10, 'occ_threshold': 0.02}, 'n_qubits': 20, 'difficulty': 'moderate', 'n_samples': 80000},
    {'name': 'purine_frontier_20q', 'geometry': 'N 1.336 1.117 0; C 0.219 1.853 0; N -0.997 1.391 0; C -1.080 0.044 0; C 0.057 -0.781 0; C 1.297 -0.243 0; N -2.150 -0.685 0; C -1.703 -1.927 0; N -0.383 -2.030 0; H 0.310 2.934 0; H 2.193 -0.857 0; H -2.357 -2.785 0; H 2.235 1.490 0',
     'basis': 'sto-3g', 'as': {'m': 'frontier', 'n_occ': 5, 'n_virt': 5}, 'n_qubits': 20, 'ref_calc': (10, 10), 'difficulty': 'moderate', 'n_samples': 80000},
]

if __name__ == "__main__":
    main()
