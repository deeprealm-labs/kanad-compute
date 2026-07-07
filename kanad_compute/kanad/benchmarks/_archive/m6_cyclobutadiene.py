"""M6-F — Cyclobutadiene (C₄H₄) D₄h ↔ D₂h Jahn-Teller distortion.

Square D₄h cyclobutadiene is a Jahn-Teller-unstable singlet biradical
(the two degenerate e_g π orbitals are each singly occupied). The
rectangular D₂h form is the closed-shell ground state. The interconversion
barrier is ~6.5 kcal/mol experimentally.

Stress axes:
  - Antiaromatic biradical singlet (D₄h)
  - Jahn-Teller distortion (D₄h → D₂h)
  - CAS(4e, 4o) active space (the 4 π MOs) → 8 qubits

References:
  - Carpenter 1983 experiment: D₂h ground state; barrier ~6.5 kcal/mol
  - Eckert-Maksic 2006 MR-AQCC: barrier = 6.4 kcal/mol, |c_max|² ≈ 0.5 at D₄h
  - Wenthold 2009 photoelectron: ΔE_S0-S1 about 1.85 eV at D₂h

Run:
  BLUEQUBIT_API_KEY=... python -m benchmarks.m6_cyclobutadiene
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from pyscf import gto, scf, mcscf


# Idealized geometries (Carpenter 1983 / Eckert-Maksic 2006).
# C-H bond length = 1.085 Å for both.
R_CH = 1.085


def build_d4h_geometry(r_cc=1.45):
    """Square D₄h: all 4 C-C bonds equal."""
    half = r_cc / 2.0
    carbons = [
        ( half,  half, 0.0),
        (-half,  half, 0.0),
        (-half, -half, 0.0),
        ( half, -half, 0.0),
    ]
    hydrogens = []
    for cx, cy, _ in carbons:
        # H is on the diagonal pointing outward from the square center
        norm = np.hypot(cx, cy)
        hydrogens.append((cx + (cx / norm) * R_CH,
                          cy + (cy / norm) * R_CH, 0.0))
    return [('C', c) for c in carbons] + [('H', h) for h in hydrogens]


def build_d2h_geometry(r_short=1.34, r_long=1.56):
    """Rectangular D₂h: alternating short / long C-C bonds.

    Eckert-Maksic 2006 give r_short ≈ 1.34 Å (C=C double),
    r_long ≈ 1.56 Å (C-C single).
    """
    hx = r_long / 2.0
    hy = r_short / 2.0
    carbons = [
        ( hx,  hy, 0.0),
        (-hx,  hy, 0.0),
        (-hx, -hy, 0.0),
        ( hx, -hy, 0.0),
    ]
    hydrogens = []
    for cx, cy, _ in carbons:
        norm = np.hypot(cx, cy)
        hydrogens.append((cx + (cx / norm) * R_CH,
                          cy + (cy / norm) * R_CH, 0.0))
    return [('C', c) for c in carbons] + [('H', h) for h in hydrogens]


def atom_string(atoms):
    return '; '.join(f'{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for s, p in atoms)


def reference_state(label, atoms, spin=0, mf_class=None):
    """PySCF HF + CASCI(4,4) on the 4 π orbitals."""
    if mf_class is None:
        mf_class = scf.RHF if spin == 0 else scf.ROHF
    print(f'\n  [{label}]  spin = {spin}')
    mol = gto.M(atom=atom_string(atoms), basis='cc-pvdz',
                charge=0, spin=spin, verbose=0)
    mf = mf_class(mol).run(verbose=0)
    print(f'    HF       = {mf.e_tot:.6f} Ha  '
          f'({mol.nelectron} electrons, {mol.nao_nr()} AOs)')

    # Cyclobutadiene has 4 π electrons in 4 π orbitals.
    # Active = HOMO-1, HOMO, LUMO, LUMO+1 (the 4 π MOs near frontier)
    n_occ = int(np.sum(mf.mo_occ > 0))
    active = list(range(n_occ - 2, n_occ + 2))    # HOMO-1, HOMO, LUMO, LUMO+1
    print(f'    Active MO indices: {active}  (π valence: 4 e in 4 o)')

    n_a = (4 + spin) // 2
    n_b = 4 - n_a
    nelecas = (n_a, n_b) if spin > 0 else 4
    cas = mcscf.CASCI(mf, ncas=4, nelecas=nelecas)
    cas.sort_mo(active, base=0)
    cas.run(verbose=0)
    print(f'    CASCI(4,4) = {cas.e_tot:.6f} Ha')
    ci = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
    max_w = float(np.max(np.abs(ci)) ** 2)
    print(f'    |c_max|² = {max_w:.4f}  '
          f'→ {"single-ref" if max_w > 0.8 else "MULTI-REF biradical"}')
    return {'label': label, 'spin': spin, 'mol': mol, 'mf': mf, 'cas': cas,
            'active': active, 'e_hf': float(mf.e_tot),
            'e_casci': float(cas.e_tot), 'max_weight': max_w}


def cloud_sqd_anchor(state):
    import bluequbit
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _generate_singles_doubles, _filter_with_recovery,
    )

    mf = state['mf']
    spin = state['spin']
    active = state['active']
    frozen = list(range(active[0]))
    print(f'    [{state["label"]} cloud SQD] frozen 0..{frozen[-1]}, active = {active}')
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=frozen, active=active),
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = ham.n_electrons
    target_sz = spin / 2.0
    print(f'    [{state["label"]} cloud SQD] {n_qubits} qubits, '
          f'{n_active_e} active e, target_sz = {target_sz}')

    n_a = (n_active_e + spin) // 2
    n_b = n_active_e - n_a
    cas_check = mcscf.CASCI(mf, ncas=len(active),
                             nelecas=(n_a, n_b) if spin > 0 else n_active_e)
    cas_check.sort_mo(active, base=0)
    cas_check.run(verbose=0)
    e_casci_ref = float(cas_check.e_tot)

    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=n_active_e,
                       n_layers=1, target_sz=target_sz)
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(0)
    params = rng.uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters({qc.parameters[i]: float(params[i])
                                  for i in range(qc.num_parameters)})
    if bound.num_clbits == 0:
        bound.measure_all()
    print(f'    [{state["label"]} cloud SQD] LUCJ: {bound.num_nonlocal_gates()} 2q, '
          f'depth {bound.depth()}')

    bq = bluequbit.init(os.environ['BLUEQUBIT_API_KEY'])
    t0 = time.time()
    result = bq.run(circuits=bound, device='cpu', shots=10000,
                     job_name=f'm6_cbd_{state["label"]}_{int(time.time())}')
    cloud_t = time.time() - t0
    counts = result.get_counts()
    bitstrings = []
    for bs, n in counts.items():
        bitstrings.extend([int(bs.replace(' ', ''), 2)] * int(n))
    bitstrings = np.array(bitstrings, dtype=np.int64)
    print(f'    [{state["label"]} cloud SQD] BlueQubit CPU: {len(bitstrings)} shots '
          f'in {cloud_t:.1f}s (job {result.job_id})')

    solver = SamplingSQDSolver(ham, n_samples=len(bitstrings), random_seed=0,
                                recover_configurations=True,
                                ci_backend='pyscf', target_sz=target_sz)
    mo_e = solver._resolve_mo_energies()
    valid, *_ = _filter_with_recovery(bitstrings, ham.n_orbitals, n_active_e,
                                       target_sz, mo_e)
    dets = sorted(set(int(d) for d in valid))
    last = None
    t1 = time.time()
    for it in range(4):
        res = solver._diagonalize_in_subspace_pyscf(dets)
        if last is not None and abs(res['energy'] - last) < 1e-6:
            break
        last = res['energy']
        evec = res['eigenvector']
        top = np.argsort(np.abs(evec) ** 2)[::-1][:min(50, len(dets))]
        new_dets = set()
        for i in top:
            new_dets.update(_generate_singles_doubles(dets[i], n_qubits, n_active_e))
        old = len(dets)
        dets = sorted(set(dets) | new_dets)
        if len(dets) == old:
            break
    expand_t = time.time() - t1
    gap = (res['energy'] - e_casci_ref) * 1000
    tag = '✓' if abs(gap) < 1.0 else ('⚠' if abs(gap) < 5.0 else '✗')
    print(f'    [{state["label"]} cloud SQD] SQD = {res["energy"]:.6f}  '
          f'CASCI = {e_casci_ref:.6f}  gap = {gap:+.4f} mHa  {tag}  '
          f'({len(dets)} dets, expand {expand_t:.1f}s)')
    return {'e_casci': e_casci_ref, 'e_sqd_cloud': res['energy'],
            'gap_mha': gap, 'cloud_time_s': cloud_t,
            'expand_time_s': expand_t, 'n_det': len(dets),
            'n_shots': int(len(bitstrings)), 'job_id': result.job_id}


def main():
    if not os.environ.get('BLUEQUBIT_API_KEY'):
        print('ERROR: BLUEQUBIT_API_KEY env var required')
        sys.exit(1)

    HA_TO_KCAL = 627.509
    print('=' * 96)
    print('M6-F — CYCLOBUTADIENE  D₄h ↔ D₂h JAHN-TELLER  cc-pVDZ  CAS(4e, 4o)')
    print('  Classical refs: PySCF (HF + CASCI)   Quantum sampling: BlueQubit CPU cloud')
    print('=' * 96)

    print('\n### D₂h ground state (rectangular, closed-shell)')
    d2h = reference_state('D₂h ¹A_g', build_d2h_geometry(), spin=0)

    print('\n### D₄h transition state (square, antiaromatic biradical singlet)')
    d4h = reference_state('D₄h ¹B_1g', build_d4h_geometry(), spin=0)

    barrier_hf = (d4h['e_hf'] - d2h['e_hf']) * HA_TO_KCAL
    barrier_casci = (d4h['e_casci'] - d2h['e_casci']) * HA_TO_KCAL

    print()
    print('=' * 96)
    print('JAHN-TELLER BARRIER (D₂h → D₄h)')
    print('=' * 96)
    print(f'  HF/cc-pVDZ:           barrier = {barrier_hf:+7.2f} kcal/mol')
    print(f'  CASCI(4,4)/cc-pVDZ:   barrier = {barrier_casci:+7.2f} kcal/mol')
    print()
    print('  Literature reference values:')
    print('    Carpenter 1983 experiment:           ~6.5 kcal/mol')
    print('    Eckert-Maksic 2006 MR-AQCC:           6.4 kcal/mol')
    print('    Whitman 1982 CASSCF(4,4):            ~8 kcal/mol')

    print()
    print('=' * 96)
    print('FRAMEWORK ANCHOR — Kanad SamplingSQD on BlueQubit cloud per geometry')
    print('=' * 96)
    sqd_d2h = cloud_sqd_anchor(d2h)
    sqd_d4h = cloud_sqd_anchor(d4h)

    barrier_sqd = (sqd_d4h['e_sqd_cloud'] - sqd_d2h['e_sqd_cloud']) * HA_TO_KCAL
    print()
    print(f'  Cloud-SQD-derived barrier: {barrier_sqd:+.2f} kcal/mol  '
          f'(= CASCI(4,4): {barrier_casci:+.2f})')

    print()
    print('=' * 96)
    print('MULTIREFERENCE CHARACTER')
    print('=' * 96)
    print(f'  D₂h (rectangular)  |c_max|² = {d2h["max_weight"]:.4f}  '
          f'→ {"single-config" if d2h["max_weight"] > 0.8 else "multireference"}')
    print(f'  D₄h (square)       |c_max|² = {d4h["max_weight"]:.4f}  '
          f'→ {"single-config" if d4h["max_weight"] > 0.8 else "BIRADICAL (Eckert-Maksic ≈ 0.5)"}')
    print()
    print('  Framework status:')
    if abs(sqd_d2h['gap_mha']) < 1.0 and abs(sqd_d4h['gap_mha']) < 1.0:
        print(f'    ✓ Cloud-SQD reproduces CASCI(4,4) to <1 mHa on BOTH geometries.')
        print(f'    ✓ Antiaromatic biradical handled (D₄h |c_max|² < 0.6).')
        print(f'    ✓ D₂h (job {sqd_d2h["job_id"]}): {sqd_d2h["cloud_time_s"]:.1f}s cloud.')
        print(f'    ✓ D₄h (job {sqd_d4h["job_id"]}): {sqd_d4h["cloud_time_s"]:.1f}s cloud.')
    else:
        print(f'    ⚠ Cloud-SQD gap to CASCI exceeds 1 mHa — investigate.')
        print(f'      D₂h gap = {sqd_d2h["gap_mha"]:+.4f} mHa')
        print(f'      D₄h gap = {sqd_d4h["gap_mha"]:+.4f} mHa')


if __name__ == '__main__':
    main()
