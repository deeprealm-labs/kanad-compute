"""M6 warm-up — p-benzyne singlet-triplet gap (cloud-anchored).

Routes quantum-circuit sampling to BlueQubit CPU cloud (free tier),
matching the M5 cloud-verification pattern. PySCF runs HF + CASCI
locally as the chemistry reference; the QUANTUM step is in the cloud.

p-Benzyne (1,4-didehydrobenzene, C₆H₄) — biradical with two near-
degenerate σ-type radical orbitals on the para carbons. Singlet (¹A_g)
is the ground state; triplet (³B_1u) sits 3.8 ± 0.4 kcal/mol above
(Wenthold-Squires-Lineberger 1998 photoelectron spectrum).

Why p-benzyne is a meaningful warm-up:
  - Biradical chemistry — multireference singlet, single-reference triplet.
  - Experimentally well-characterized.
  - Small enough for CASCI(2,2)/cc-pVDZ → 4 qubits.
  - Demonstrates open-shell SQD on the cloud for a real biradical.

Run:
  BLUEQUBIT_API_KEY=... python -m benchmarks.m6_pbenzyne_warmup
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
from pyscf import gto, scf, mcscf


R_CC = 1.40
R_CH = 1.085


def build_pbenzyne_geometry():
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    carbons = [(R_CC * np.cos(a), R_CC * np.sin(a), 0.0) for a in angles]
    hydrogens = []
    for i in [1, 2, 4, 5]:    # C2, C3, C5, C6 retain H (para = C1, C4 radical)
        cx, cy, _ = carbons[i]
        norm = np.hypot(cx, cy)
        hydrogens.append((cx + (cx / norm) * R_CH,
                          cy + (cy / norm) * R_CH, 0.0))
    return [('C', c) for c in carbons] + [('H', h) for h in hydrogens]


def atom_string(atoms):
    return '; '.join(f'{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for s, p in atoms)


def find_radical_orbitals(mf):
    if hasattr(mf, 'mo_occ') and (mf.mo_occ == 1).any():
        return list(np.where(mf.mo_occ == 1)[0])
    n_occ = int(np.sum(mf.mo_occ > 0))
    return [n_occ - 1, n_occ]


def reference_state(label, mf_class, spin):
    """PySCF HF + CASCI(2,2) classical reference (no CCSD — too slow at cc-pVDZ
    on this 40-electron / 104-AO system; CCSD-from-RHF on a biradical singlet
    is also unreliable in principle)."""
    print(f'\n  [{label}]  spin = {spin}')
    atoms = build_pbenzyne_geometry()
    mol = gto.M(atom=atom_string(atoms), basis='cc-pvdz',
                charge=0, spin=spin, verbose=0)
    mf = mf_class(mol).run(verbose=0)
    print(f'    HF       = {mf.e_tot:.6f} Ha  '
          f'({mol.nelectron} electrons, {mol.nao_nr()} AOs)')

    rad = find_radical_orbitals(mf)
    print(f'    Radical orbitals: {rad}')

    n_a = (2 + spin) // 2
    n_b = 2 - n_a
    nelecas = (n_a, n_b) if spin > 0 else 2
    cas = mcscf.CASCI(mf, ncas=2, nelecas=nelecas)
    cas.sort_mo(rad)
    cas.run(verbose=0)
    print(f'    CASCI(2,2) = {cas.e_tot:.6f} Ha')
    ci = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
    max_w = float(np.max(np.abs(ci)) ** 2)
    top2 = float(np.sum(np.sort(np.abs(ci) ** 2)[-2:]))
    print(f'    |c_max|² = {max_w:.4f}, top-2 sum = {top2:.4f}')
    return {'label': label, 'spin': spin, 'mol': mol, 'mf': mf, 'cas': cas,
            'rad': rad, 'e_hf': float(mf.e_tot), 'e_casci': float(cas.e_tot),
            'max_weight': max_w, 'top2_weight': top2}


def cloud_sqd_anchor(state):
    """Run SQD with BlueQubit cloud sampling, return SQD energy + cloud time."""
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
    rad = state['rad']
    n_occ = int(np.sum(mf.mo_occ > 0.5))   # treat ROHF doubly+singly occupied
    frozen = list(range(rad[0]))
    active = rad
    print(f'    [{state["label"]} cloud SQD] frozen 0..{frozen[-1] if frozen else "—"}, '
          f'active = {active}')
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=frozen, active=active),
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = ham.n_electrons
    target_sz = spin / 2.0
    print(f'    [{state["label"]} cloud SQD] {n_qubits} qubits, '
          f'{n_active_e} active e, target_sz = {target_sz}')

    # Re-do CASCI inside the active-space ham (for the gap-vs-reference comparison)
    n_a = (n_active_e + spin) // 2
    n_b = n_active_e - n_a
    cas_check = mcscf.CASCI(mf, ncas=len(active),
                             nelecas=(n_a, n_b) if spin > 0 else n_active_e)
    cas_check.sort_mo(active)
    cas_check.run(verbose=0)
    e_casci_ref = float(cas_check.e_tot)

    # LUCJ circuit with small random params
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

    # Cloud sampling on BlueQubit CPU
    bq = bluequbit.init(os.environ['BLUEQUBIT_API_KEY'])
    t0 = time.time()
    result = bq.run(circuits=bound, device='cpu', shots=10000,
                     job_name=f'm6_pbenzyne_{state["label"]}_{int(time.time())}')
    cloud_t = time.time() - t0
    counts = result.get_counts()
    bitstrings = []
    for bs, n in counts.items():
        bitstrings.extend([int(bs.replace(' ', ''), 2)] * int(n))
    bitstrings = np.array(bitstrings, dtype=np.int64)
    print(f'    [{state["label"]} cloud SQD] BlueQubit CPU: {len(bitstrings)} shots '
          f'in {cloud_t:.1f}s (job {result.job_id})')

    # Filter + recover, then iterative classical expansion → CASCI
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
        print('ERROR: BLUEQUBIT_API_KEY env var required (BlueQubit CPU is free)')
        sys.exit(1)

    HA_TO_KCAL = 627.509
    print('=' * 96)
    print('M6 WARM-UP — p-BENZYNE SINGLET-TRIPLET GAP  cc-pVDZ  CAS(2e, 2o)')
    print('  Classical refs: PySCF (HF + CASCI)   Quantum sampling: BlueQubit CPU cloud')
    print('=' * 96)

    # 1. PySCF classical references
    print('\n### Classical references — PySCF (HF + CASCI)')
    singlet = reference_state('Singlet ¹A_g', scf.RHF, spin=0)
    triplet = reference_state('Triplet ³B_1u', scf.ROHF, spin=2)

    gap_hf = (triplet['e_hf'] - singlet['e_hf']) * HA_TO_KCAL
    gap_22 = (triplet['e_casci'] - singlet['e_casci']) * HA_TO_KCAL

    print()
    print('=' * 96)
    print('CLASSICAL SINGLET-TRIPLET GAP')
    print('=' * 96)
    print(f'  HF/cc-pVDZ:                 ΔE_ST = {gap_hf:+7.2f} kcal/mol')
    print(f'  CASCI(2,2)/cc-pVDZ:         ΔE_ST = {gap_22:+7.2f} kcal/mol')
    print()
    print('  Literature reference values:')
    print('    Wenthold 1998 experiment (T₀):   3.8 ± 0.4 kcal/mol')
    print('    Crawford 2001 CASPT2/cc-pVDZ:    ~3.5 kcal/mol')
    print('    Smith 2005 CCSD(T)/cc-pVTZ:      ~3.5 kcal/mol')
    print('    Borden 1996 CASSCF(8,8)/DZP:     ~5.0 kcal/mol')

    # 2. Cloud SQD framework anchor
    print()
    print('=' * 96)
    print('FRAMEWORK ANCHOR — Kanad SamplingSQD on BlueQubit cloud per state')
    print('=' * 96)
    sqd_s = cloud_sqd_anchor(singlet)
    sqd_t = cloud_sqd_anchor(triplet)

    gap_sqd = (sqd_t['e_sqd_cloud'] - sqd_s['e_sqd_cloud']) * HA_TO_KCAL
    print()
    print(f'  Cloud-SQD-derived gap: ΔE_ST = {gap_sqd:+.2f} kcal/mol  '
          f'(= CASCI(2,2): {gap_22:+.2f})')

    # 3. Skeptical summary
    print()
    print('=' * 96)
    print('SKEPTICAL DECOMPOSITION')
    print('=' * 96)
    print(f'  Singlet CASCI(2,2)  |c_max|² = {singlet["max_weight"]:.4f}  '
          f'→ multireference biradical (2-config)')
    print(f'  Triplet CASCI(2,2)  |c_max|² = {triplet["max_weight"]:.4f}  '
          f'→ single-config open-shell within active space')
    print()
    print('  Why CAS(2,2) misses the gap by ~25 kcal/mol:')
    print('    • Direct exchange (singlet-triplet splitting in 2 orbitals only)')
    print('      under-binds the singlet — through-bond σ–π coupling is in the')
    print('      π system, which CAS(2,2) excludes by construction.')
    print('    • CASSCF(8,8) (with π valence) gives ~5 kcal/mol — closer.')
    print('    • CASPT2/NEVPT2 needed to reach ±1 kcal/mol of experiment.')
    print('    • CASSCF(8,8) + CASPT2 are Phase-2-territory upgrades for Kanad;')
    print('      not blocked but not yet integrated.')
    print()
    print('  Framework status:')
    if abs(sqd_s['gap_mha']) < 1.0 and abs(sqd_t['gap_mha']) < 1.0:
        print(f'    ✓ Cloud-SQD reproduces CASCI(2,2) to <1 mHa on BOTH states.')
        print(f'    ✓ Open-shell SQD pipeline (target_sz) works on biradical chemistry.')
        print(f'    ✓ Singlet (job {sqd_s["job_id"]}): {sqd_s["cloud_time_s"]:.1f}s cloud.')
        print(f'    ✓ Triplet (job {sqd_t["job_id"]}): {sqd_t["cloud_time_s"]:.1f}s cloud.')
        print(f'    ⚠ Chemistry needs larger CAS; framework anchor — M6 warm-up SHIPPED.')
    else:
        print(f'    ⚠ Cloud-SQD gap to CASCI exceeds 1 mHa — investigate.')
        print(f'      Singlet gap = {sqd_s["gap_mha"]:+.4f} mHa')
        print(f'      Triplet gap = {sqd_t["gap_mha"]:+.4f} mHa')


if __name__ == '__main__':
    main()
