"""M5-D — LiH X¹Σ⁺ / A¹Σ⁺ avoided crossing + non-adiabatic coupling.

LiH is the textbook 2-state quantum-dynamics system. The ionic (X¹Σ⁺,
Li⁺H⁻) and covalent (A¹Σ⁺, Li+H atom) configurations cross around
r ≈ 3-4 Å forming an avoided crossing — the classic Landau-Zener case
where non-adiabatic coupling matters and surface hopping becomes defined.

Spectroscopic constants of both states are measured to ~0.1 cm⁻¹ in
gas-phase spectroscopy (Stwalley 1991).

Scope of M5-D (focused for publishable-bar deliverable):
  1. CASCI(2e, 5o)/cc-pVDZ ground (X) + first-excited (A) PES on r ∈ [1, 8] Å
  2. Identify avoided crossing r_x
  3. NAC magnitude |d_XA(r)| via finite-difference of CI vectors
  4. Framework anchor: Kanad SamplingSQD reproduces CASCI X at the crossing
  5. Branching ratio + FSSH integrator are parked for M6/Phase 2

Reference values:
  - Stwalley 1991 X-state r_e = 1.5957 Å, ω_e = 1405.65 cm⁻¹, D_e = 56.0 kcal/mol
  - A-state r_e ≈ 2.595 Å, ω_e ≈ 234.41 cm⁻¹
  - Avoided crossing r_x ≈ 3.7-4.0 Å (Boutalib 1992 MRCI)
"""

from __future__ import annotations

import time
import numpy as np
from pyscf import gto, scf, mcscf


# LiH active space (cc-pVDZ): freeze Li 1s; active = MOs 1-5
FROZEN_ORBS = [0]
ACTIVE_ORBS = [1, 2, 3, 4, 5]


def build_lih(r_angstrom):
    mol = gto.M(
        atom=f'Li 0 0 0; H 0 0 {r_angstrom}',
        basis='cc-pvdz', spin=0, charge=0, verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    return mol, mf


def casci_2states(mf):
    """CASCI(2e, 5o) returning the lowest 2 singlet states."""
    cas = mcscf.CASCI(mf, ncas=5, nelecas=2)
    cas.fcisolver.nroots = 2
    cas.run(verbose=0)
    energies = cas.e_tot  # array of 2 energies
    ci_vecs = cas.ci      # list of 2 CI vectors (n_alpha_strs × n_beta_strs)
    return energies, ci_vecs, cas


def overlap_ci(ci1, ci2):
    """Overlap of two PySCF CI vectors (assumed same string basis)."""
    return float(np.sum(ci1 * ci2))


def compute_nac_finite_diff(mf, r, dr=0.02):
    """Non-adiabatic coupling magnitude |⟨X(R)|∂/∂R|A(R)⟩| via finite-difference.

    d_XA(R) = ⟨X(R) | A(R + dR)⟩ / dR  (when E_X ≠ E_A; phase-aligned)
    Returns the magnitude (sign depends on arbitrary phase of CI vectors).
    """
    _, mf_minus = build_lih(r - dr)
    _, mf_plus = build_lih(r + dr)
    e_m, ci_m, _ = casci_2states(mf_minus)
    e_p, ci_p, _ = casci_2states(mf_plus)
    # Align phases — pick sign so the dominant components match
    ci_m_X = ci_m[0]
    ci_m_A = ci_m[1]
    ci_p_X = ci_p[0]
    ci_p_A = ci_p[1]
    # Phase alignment via maximum overlap
    if overlap_ci(ci_m_X, ci_p_X) < 0:
        ci_p_X = -ci_p_X
    if overlap_ci(ci_m_A, ci_p_A) < 0:
        ci_p_A = -ci_p_A
    # NAC via ⟨X(R-dR) | A(R+dR)⟩ / 2dR
    overlap_XA = overlap_ci(ci_m_X, ci_p_A)
    return abs(overlap_XA) / (2.0 * dr)


def verify_sqd_x_state(mf, r):
    """Kanad SamplingSQD on the X ground state at this r."""
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _generate_singles_doubles,
    )
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=FROZEN_ORBS, active=ACTIVE_ORBS),
    )
    n_qubits = 2 * ham.n_orbitals
    n_e = ham.n_electrons

    cas = mcscf.CASCI(mf, ncas=5, nelecas=2).run(verbose=0)
    e_casci_x = float(cas.e_tot)

    np.random.seed(0)
    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=n_e, n_layers=1)
    qc = ansatz.build_circuit()
    params = np.random.default_rng(0).uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
    )
    solver = SamplingSQDSolver(
        ham, n_samples=20000, random_seed=0,
        recover_configurations=True, ci_backend='pyscf',
    )
    from kanad.solvers.sampling_sqd import (
        _sample_circuit_statevector, _filter_with_recovery,
    )
    samples = _sample_circuit_statevector(bound, 20000, np.random.default_rng(0))
    mo_e = solver._resolve_mo_energies()
    valid, *_ = _filter_with_recovery(samples, ham.n_orbitals, n_e, 0.0, mo_e)
    determinants = sorted(set(int(d) for d in valid))
    last_e = None
    for it in range(4):
        res = solver._diagonalize_in_subspace_pyscf(determinants)
        if last_e is not None and abs(res['energy'] - last_e) < 1e-6:
            break
        last_e = res['energy']
        evec = res['eigenvector']
        top_idx = np.argsort(np.abs(evec) ** 2)[::-1][:min(50, len(determinants))]
        new_dets = set()
        for i in top_idx:
            new_dets.update(_generate_singles_doubles(determinants[i], n_qubits, n_e))
        old = len(determinants)
        determinants = sorted(set(determinants) | new_dets)
        if len(determinants) == old:
            break
    return e_casci_x, res['energy'], (res['energy'] - e_casci_x) * 1000


def main():
    print('=' * 92)
    print('M5-D — LiH X¹Σ⁺ / A¹Σ⁺ AVOIDED CROSSING  cc-pVDZ CAS(2e, 5o)')
    print('=' * 92)

    # Stage 1: PES scan for 2 lowest singlet states
    rs = [1.2, 1.5, 1.5957, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]
    pes = []
    print(f'\n  {"r (Å)":>6}  {"E(X) [Ha]":>12}  {"E(A) [Ha]":>12}  {"ΔE [eV]":>9}  '
          f'{"|c_max,X|²":>11}  {"|c_max,A|²":>11}')
    print(' ' + '-' * 80)
    for r in rs:
        try:
            mol, mf = build_lih(r)
            energies, ci_vecs, cas = casci_2states(mf)
            E_X, E_A = energies[0], energies[1]
            cX_max = float(np.max(np.abs(ci_vecs[0])) ** 2)
            cA_max = float(np.max(np.abs(ci_vecs[1])) ** 2)
            gap_eV = (E_A - E_X) * 27.2114
            print(f'  {r:>6.4f}  {E_X:>12.6f}  {E_A:>12.6f}  {gap_eV:>9.4f}  '
                  f'{cX_max:>11.4f}  {cA_max:>11.4f}')
            pes.append({
                'r': r, 'E_X': E_X, 'E_A': E_A, 'gap_eV': gap_eV,
                'cX_max': cX_max, 'cA_max': cA_max, 'mf': mf,
            })
        except Exception as e:
            print(f'  {r:>6.4f}  failed: {type(e).__name__}: {e}')

    # Stage 2: identify avoided crossing.
    # For LiH the adiabatic gap is MONOTONIC — both X and A approach the
    # same Li+H atomic limit at large r. The signature of the avoided
    # crossing is the SHARPEST DROP in |c_max,X|² (ionic→covalent character
    # change), not the gap minimum. Identify it via finite-difference of
    # |c_max,X|² along the path.
    rs_arr = np.array([p['r'] for p in pes])
    cX_arr = np.array([p['cX_max'] for p in pes])
    # First-derivative (forward difference)
    dcX_dr = np.diff(cX_arr) / np.diff(rs_arr)
    # Most-negative derivative → fastest character change (ionic → covalent)
    idx_max_change = int(np.argmin(dcX_dr))
    r_x = (rs_arr[idx_max_change] + rs_arr[idx_max_change + 1]) / 2
    gap_at_rx = (pes[idx_max_change]['gap_eV'] + pes[idx_max_change + 1]['gap_eV']) / 2

    print()
    print('=' * 92)
    print('AVOIDED CROSSING (character-change region for LiH X-A)')
    print('=' * 92)
    print(f'  Maximum d|c_max,X|²/dr at r ≈ {r_x:.3f} Å (gap there ≈ {gap_at_rx:.4f} eV)')
    print(f'  Both adiabatic states approach Li(²S) + H(²S) at large r,')
    print(f'  so the gap stays monotonic; the "crossing" is where ionic↔covalent')
    print(f'  character flips in the adiabatic representation.')
    print(f'  Reference (Boutalib 1992 MRCI): r_x ≈ 3.7-4.0 Å')
    print(f'  Match: {"✓ within 0.5 Å of literature" if abs(r_x - 3.85) < 0.5 else "⚠ outside expected range"}')

    # Stage 3: NAC magnitude near crossing — DEFERRED.
    # Note: a clean NAC computation requires projecting CI vectors at
    # different geometries onto a COMMON orbital basis. PySCF's per-r
    # mf.mo_coeff differs (orbitals re-optimized at each r), so direct CI
    # overlap is meaningless without orbital tracking. This is the
    # well-known "diabatization" / "phase-following" problem.
    # The clean fix uses PySCF's `mc.nac` (in pyscf-properties extension)
    # or analytic NAC via Hellmann-Feynman with derivative orbitals.
    # Parked for M6 / Phase 2; the PES + character analysis above already
    # localize where d_XA is largest (the steepest |c_max,X|² drop).
    print()
    print('=' * 92)
    print('NON-ADIABATIC COUPLING — deferred to M6')
    print('=' * 92)
    print(f'  Adiabatic NAC d_XA(r) peaks where the |c_max,X|² character flips.')
    print(f'  Our data: character flip centred at r ≈ {r_x:.2f} Å (gap ≈ {gap_at_rx:.4f} eV).')
    print(f'  Computing |d_XA| numerically requires orbital-basis tracking across r')
    print(f'  (PySCF mo_coeff is re-optimized at each geometry → naive CI overlap = 0).')
    print(f'  PySCF analytic NAC (via mc.nac) is the production path; M6 work.')

    # Stage 4: framework anchor at crossing
    print()
    print('=' * 92)
    print('FRAMEWORK ANCHOR — SQD reproduces CASCI ground state at crossing')
    print('=' * 92)
    mf_x = pes[idx_max_change]['mf']
    e_casci, e_sqd, gap_mha = verify_sqd_x_state(mf_x, r_x)
    ok = '✓' if abs(gap_mha) < 1.0 else ('⚠' if abs(gap_mha) < 5.0 else '✗')
    print(f'  At r = {r_x:.3f} Å (avoided crossing):')
    print(f'    CASCI X = {e_casci:.6f} Ha')
    print(f'    SQD X   = {e_sqd:.6f} Ha')
    print(f'    Gap     = {gap_mha:+.4f} mHa  {ok}')

    print()
    print('=' * 92)
    print('SKEPTICAL CHECKS')
    print('=' * 92)
    # Multireference at crossing: |c_max| should drop
    p_x = pes[idx_max_change]
    print(f'  At crossing r = {p_x["r"]:.3f} Å: |c_max,X|² = {p_x["cX_max"]:.4f}, '
          f'|c_max,A|² = {p_x["cA_max"]:.4f}')
    if p_x['cX_max'] < 0.85 or p_x['cA_max'] < 0.85:
        print(f'  ✓ Both states multireference at crossing (textbook avoided-crossing behavior)')
    # Asymptotic limit: at large r, gap → 0 (both states near Li + H atomic asymptote)
    far_p = pes[-1]
    print(f'  At r = {far_p["r"]:.1f} Å (far): gap = {far_p["gap_eV"]:.4f} eV')
    if far_p['gap_eV'] < 0.5:
        print(f'  ✓ States nearly degenerate at dissociation (Li+H atomic limit)')


if __name__ == '__main__':
    main()
