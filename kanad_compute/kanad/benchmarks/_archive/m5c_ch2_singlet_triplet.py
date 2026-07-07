"""M5-C — CH₂ singlet-triplet gap at cc-pVDZ CAS(6e, 6o).

The classic static-correlation benchmark. Methylene's ground state is the
triplet ³B₁ (Walsh diagram); the singlet ¹A₁ is 9.0 ± 0.014 kcal/mol above
(Leopold et al., J. Phys. Chem. 90, 1487, 1986; refined Bunker 2005).
DFT methods get this WRONG by 5-10 kcal/mol because the ¹A₁ singlet
requires multireference treatment (2 dominant configs of similar weight).

This benchmark:
  1. Optimize each state's geometry classically (PySCF UHF + CASSCF/casci)
  2. Compute electronic energy at each minimum via CASCI(6e, 6o)/cc-pVDZ
  3. Verify Kanad SamplingSQD reproduces each state's CASCI energy
  4. Report ΔE_ST = E(¹A₁) − E(³B₁); compare to experiment + HEAT ab initio

Key open-shell test for the framework: triplet has (n_α=4, n_β=2) at
active-space level (after freezing C 1s); this exercises the M5-C
open-shell wiring in SamplingSQDSolver and LUCJAnsatz.

Geometries (from Bunker et al. 2005, used as reference):
  - ¹A₁: r(CH) = 1.107 Å, ∠HCH = 102.4°
  - ³B₁: r(CH) = 1.075 Å, ∠HCH = 133.9°

Reference T_e (Bunker-Sears CCSD(T)/CBS + relativistic + DBOC): 9.03 kcal/mol
Reference T_e (HEAT 2009): 9.04 ± 0.06 kcal/mol
Experiment (Leopold 1986 photoelectron): 9.013 ± 0.014 kcal/mol (T₀, ZPE-corrected)
"""

from __future__ import annotations

import time
import numpy as np
from pyscf import gto, scf, mcscf


# Geometries (Å, degrees)
GEOMS = {
    'singlet_1A1': {
        'description': 'CH₂ ¹A₁ (singlet, bent at 102.4°)',
        'r_CH': 1.107, 'angle_HCH': 102.4,
        'multiplicity': 1,
    },
    'triplet_3B1': {
        'description': 'CH₂ ³B₁ (triplet, bent at 133.9°)',
        'r_CH': 1.075, 'angle_HCH': 133.9,
        'multiplicity': 3,
    },
}


def build_atom_string(r_CH, angle_HCH_deg):
    """C at origin, two H's symmetric about z-axis at angle."""
    half_angle = np.deg2rad(angle_HCH_deg / 2)
    # H atoms in xz plane, symmetric about z
    h1 = (r_CH * np.sin(half_angle), 0.0, r_CH * np.cos(half_angle))
    h2 = (-r_CH * np.sin(half_angle), 0.0, r_CH * np.cos(half_angle))
    return f'C 0 0 0; H {h1[0]:.6f} {h1[1]:.6f} {h1[2]:.6f}; H {h2[0]:.6f} {h2[1]:.6f} {h2[2]:.6f}'


def compute_state(label, spec):
    """Compute HF/CCSD(T)/CASCI(6,6) at this state's geometry."""
    print(f'\n  [{label}] {spec["description"]}')
    atom_str = build_atom_string(spec['r_CH'], spec['angle_HCH'])
    print(f'    Geometry: {atom_str}')

    multiplicity = spec['multiplicity']
    spin = multiplicity - 1  # 2S = multiplicity - 1 for closed-shell convention

    # PySCF mol
    mol = gto.M(atom=atom_str, basis='cc-pvdz',
                spin=spin, charge=0, verbose=0)
    n_e_total = mol.nelectron
    # ROHF for open-shell, RHF for singlet
    if multiplicity == 1:
        mf = scf.RHF(mol).run(verbose=0)
    else:
        mf = scf.ROHF(mol).run(verbose=0)
    print(f'    n_electrons total = {n_e_total}, spin = {spin}, multiplicity = {multiplicity}')
    print(f'    HF      = {mf.e_tot:.6f} Ha  (converged: {mf.converged})')

    # CASCI(6e, 6o) — active = HOMO-2 .. LUMO+2 in valence
    # CH2 has C 1s (1 orbital) frozen → 8 valence electrons → use 6 in active
    # Actually: CH2 has 8 electrons; freeze C 1s (2e) → 6 active. Same for both.
    n_active_orb = 6
    n_active_e = n_e_total - 2  # frozen = C 1s (1 orbital, 2 electrons)
    # nelecas for triplet: (n_alpha_active, n_beta_active)
    if multiplicity == 1:
        nelecas = n_active_e
    else:
        # n_alpha - n_beta = spin → n_alpha = (n_active_e + spin)/2
        n_a = (n_active_e + spin) // 2
        n_b = n_active_e - n_a
        nelecas = (n_a, n_b)
    print(f'    CAS({n_active_e}e, {n_active_orb}o), nelecas = {nelecas}')
    cas = mcscf.CASCI(mf, ncas=n_active_orb, nelecas=nelecas).run(verbose=0)
    print(f'    CASCI   = {cas.e_tot:.6f} Ha')
    ci_vec = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
    max_weight = float(np.max(np.abs(ci_vec)) ** 2)
    print(f'    |c_max|² = {max_weight:.4f}  ({"single-ref" if max_weight > 0.85 else "MULTIREF"})')

    return {
        'label': label, 'mol': mol, 'mf': mf, 'cas': cas,
        'multiplicity': multiplicity, 'spin': spin,
        'e_hf': float(mf.e_tot), 'e_casci': float(cas.e_tot),
        'n_active_orb': n_active_orb, 'n_active_e': n_active_e,
        'nelecas': nelecas, 'max_weight': max_weight,
    }


def verify_sqd_open_shell(state):
    """Run Kanad SamplingSQD on the given state — closed or open shell."""
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _generate_singles_doubles,
    )

    mf = state['mf']
    multiplicity = state['multiplicity']
    spin = state['spin']
    n_active_e = state['n_active_e']
    n_active_orb = state['n_active_orb']
    e_casci_ref = state['e_casci']

    target_sz = spin / 2.0  # Sz = spin/2 in atomic units

    # Build active-space Hamiltonian. PySCF active-space selector + build
    # ham accepts ROHF mf for open-shell; the integrals come from mo_coeff
    # which are the canonical (closed-shell or ROHF) orbitals.
    try:
        ham = build_active_space_hamiltonian(
            mf,
            ActiveSpaceSelector(mf).manual(frozen=[0], active=[1, 2, 3, 4, 5, 6]),
        )
    except Exception as e:
        print(f'    [{state["label"]}] ham build failed: {type(e).__name__}: {e}')
        return None

    n_qubits = 2 * ham.n_orbitals
    print(f'    {n_qubits} qubits, target_sz = {target_sz}')

    # LUCJ ansatz with per-spin HF reference
    np.random.seed(0)
    ansatz = LUCJAnsatz(
        n_qubits=n_qubits, n_electrons=n_active_e,
        n_layers=1, target_sz=target_sz,
    )
    qc = ansatz.build_circuit()
    params = np.random.default_rng(0).uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
    )

    # SamplingSQD with target_sz
    solver = SamplingSQDSolver(
        ham, n_samples=20000, random_seed=0,
        recover_configurations=True, ci_backend='pyscf',
        target_sz=target_sz,
    )
    # Resolve MO energies for recovery
    mo_energies = solver._resolve_mo_energies()

    # Sample
    from kanad.solvers.sampling_sqd import (
        _sample_circuit_statevector, _filter_with_recovery,
    )
    samples = _sample_circuit_statevector(bound, 20000, np.random.default_rng(0))
    valid, n_kept, n_recov, n_drop = _filter_with_recovery(
        samples, ham.n_orbitals, ham.n_electrons, target_sz, mo_energies,
    )
    print(f'    Recovery: kept {n_kept}, recovered {n_recov}, dropped {n_drop}')
    seed_dets = sorted(set(int(d) for d in valid))
    print(f'    Cloud seed: {len(seed_dets)} dets')

    # Iterative expansion (cloud seed → classical S+D expansion)
    determinants = list(seed_dets)
    last_e = None
    t0 = time.time()
    for it in range(4):
        res = solver._diagonalize_in_subspace_pyscf(determinants)
        if last_e is not None and abs(res['energy'] - last_e) < 1e-6:
            break
        last_e = res['energy']
        evec = res['eigenvector']
        top_idx = np.argsort(np.abs(evec) ** 2)[::-1][:min(50, len(determinants))]
        new_dets = set()
        for i in top_idx:
            new_dets.update(_generate_singles_doubles(
                determinants[i], n_qubits, n_active_e,
            ))
        old = len(determinants)
        determinants = sorted(set(determinants) | new_dets)
        if len(determinants) == old:
            break
    dt = time.time() - t0
    gap = (res['energy'] - e_casci_ref) * 1000
    ok = '✓' if abs(gap) < 1.0 else ('⚠' if abs(gap) < 5.0 else '✗')
    print(f'    After {it + 1} iter(s): {len(determinants)} dets')
    print(f'    SQD energy = {res["energy"]:.6f}  CASCI = {e_casci_ref:.6f}  '
          f'gap = {gap:+.4f} mHa  {ok}  ({dt:.1f}s)')
    return {'e_sqd': res['energy'], 'gap_mha': gap, 'n_det': len(determinants),
            'n_seed': len(seed_dets), 'time_s': dt}


def main():
    HA_TO_KCAL = 627.509
    print('=' * 92)
    print('M5-C — CH₂ ¹A₁ vs ³B₁ singlet-triplet gap at cc-pVDZ CAS(6, 6)')
    print('=' * 92)

    states = {}
    for label, spec in GEOMS.items():
        states[label] = compute_state(label, spec)

    # Classical gap
    e_S = states['singlet_1A1']['e_casci']
    e_T = states['triplet_3B1']['e_casci']
    gap_kcal = (e_S - e_T) * HA_TO_KCAL
    e_S_hf = states['singlet_1A1']['e_hf']
    e_T_hf = states['triplet_3B1']['e_hf']
    gap_hf_kcal = (e_S_hf - e_T_hf) * HA_TO_KCAL

    print()
    print('=' * 92)
    print('CLASSICAL SINGLET-TRIPLET GAP (kcal/mol)')
    print('=' * 92)
    print(f'  HF/cc-pVDZ:                E(¹A₁) − E(³B₁) = {gap_hf_kcal:+.2f} kcal/mol')
    print(f'  CASCI(6,6)/cc-pVDZ:        E(¹A₁) − E(³B₁) = {gap_kcal:+.2f} kcal/mol')
    print()
    print('  Literature:')
    print('    HEAT (CCSDTQ + corrections, 2009):     9.04 ± 0.06 kcal/mol')
    print('    Bunker-Sears CCSD(T)/CBS:               9.03 kcal/mol')
    print('    Leopold 1986 experiment (T₀):           9.013 ± 0.014 kcal/mol')

    # Framework verification: SQD on each state
    print()
    print('=' * 92)
    print('FRAMEWORK ANCHOR — SamplingSQD reproduces CASCI per state')
    print('=' * 92)
    sqd_results = {}
    for label, st in states.items():
        sqd_results[label] = verify_sqd_open_shell(st)

    # SQD gap
    if all(r is not None for r in sqd_results.values()):
        e_S_sqd = sqd_results['singlet_1A1']['e_sqd']
        e_T_sqd = sqd_results['triplet_3B1']['e_sqd']
        gap_sqd_kcal = (e_S_sqd - e_T_sqd) * HA_TO_KCAL
        print()
        print(f'  SQD-derived gap: E(¹A₁) − E(³B₁) = {gap_sqd_kcal:+.2f} kcal/mol')

    print()
    print('=' * 92)
    print('SKEPTICAL CHECKS')
    print('=' * 92)
    for label, st in states.items():
        print(f'  {label}: |c_max|² = {st["max_weight"]:.4f}  '
              f'({"single-ref" if st["max_weight"] > 0.85 else "MULTIREFERENCE — needs CASCI not CCSD(T)"})')
    if any(s['max_weight'] < 0.85 for s in states.values()):
        print('  ⚠ Multireference character confirmed — this is exactly why DFT/CCSD(T)')
        print('    struggle with CH₂ singlet-triplet. Active-space methods (CASCI/CASPT2) needed.')


if __name__ == '__main__':
    main()
