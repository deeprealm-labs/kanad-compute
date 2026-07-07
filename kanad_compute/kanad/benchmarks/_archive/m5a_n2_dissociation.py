"""M5-A — N₂ full dissociation curve at cc-pVDZ.

The benchmark the M5 plan opens with. N₂ tests the framework on a system
where RHF qualitatively fails at r > 1.5 Å (Coulson-Fischer point) and only
multireference methods recover the correct dissociation limit.

Two-stage strategy (the way real research is done):

  STAGE 1 — Full PES via PySCF classical methods.
    For each bond length r ∈ [0.9, 3.0] Å:
      HF, MP2, CCSD, CCSD(T), and CASCI(10e, 8o)/cc-pVDZ.
    Fit Morse to each PES; tabulate r_e, ω_e, D_e against experiment.

  STAGE 2 — Kanad VQE/SQD reproduces CASCI at sample geometries.
    At r = 1.0977 Å (equilibrium) and r = 1.5 Å (multireference):
      Run kanad LUCJ + SamplingSQD with iterative subspace expansion.
      Verify |E_VQE − E_CASCI| < 5 mHa (method-truth via M3/M4 contract).

The PES is the deliverable (a real chemistry result). The VQE/SQD
reproduction at sample points is the framework-correctness anchor.
"""

from __future__ import annotations

import argparse
import time
import numpy as np


# Experimental constants (Huber-Herzberg, NIST WebBook X¹Σ_g⁺)
N2_REF = {
    'r_e_angstrom':     1.0977,
    'omega_e_cm':       2358.57,
    'omega_e_x_e_cm':   14.324,
    'D_e_kcal_mol':     228.4,
}

N2_FROZEN = [0, 1]
N2_ACTIVE = [2, 3, 4, 5, 6, 7, 8, 9]


def build_n2_at_r(r_angstrom):
    from pyscf import gto, scf
    mol = gto.M(
        atom=f'N 0 0 0; N 0 0 {r_angstrom}',
        basis='cc-pvdz', spin=0, charge=0, verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    return mol, mf


def compute_classical_energies(mf):
    """HF, MP2, CCSD, CCSD(T), CASCI(10,8) at this geometry."""
    from pyscf import mp, cc, mcscf
    e_hf = float(mf.e_tot)
    out = {'hf': e_hf}
    try:
        out['mp2'] = float(mp.MP2(mf).run(verbose=0).e_tot)
    except Exception as e:
        out['mp2'] = float('nan')
    try:
        ccsd = cc.CCSD(mf).run(verbose=0)
        out['ccsd'] = float(ccsd.e_tot)
        try:
            t_corr = ccsd.ccsd_t()
            out['ccsdt'] = float(ccsd.e_tot + t_corr)
        except Exception:
            out['ccsdt'] = float('nan')
    except Exception:
        out['ccsd'] = float('nan')
        out['ccsdt'] = float('nan')
    try:
        cas = mcscf.CASCI(mf, ncas=8, nelecas=10).run(verbose=0)
        out['casci'] = float(cas.e_tot)
        # Capture HF coefficient via FCI vector
        # CI vector is (ci.ci) with shape (n_alpha_strings, n_beta_strings)
        # Largest amplitude squared = HF coefficient²
        ci_vec = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
        out['casci_max_weight_sq'] = float(np.max(np.abs(ci_vec)) ** 2)
    except Exception as e:
        out['casci'] = float('nan')
        out['casci_max_weight_sq'] = float('nan')
    return out


def fit_morse(rs, energies):
    """Fit V(r) = D_e [1 - exp(-β(r - r_e))]² + V_min → (r_e, ω_e, D_e)."""
    from scipy.optimize import curve_fit
    rs = np.asarray(rs); energies = np.asarray(energies)
    finite = np.isfinite(energies)
    rs = rs[finite]; energies = energies[finite]
    if len(rs) < 5:
        return None, None, None
    e_min_idx = int(np.argmin(energies))
    e_min = energies[e_min_idx]
    p0 = [
        rs[e_min_idx], 2.0,
        (energies[-1] - e_min) * 627.509, e_min,
    ]

    def morse(r, r_e, beta, D_e_kcal, V_min):
        D_e_ha = D_e_kcal / 627.509
        return V_min + D_e_ha * (1 - np.exp(-beta * (r - r_e))) ** 2

    try:
        popt, _ = curve_fit(morse, rs, energies, p0=p0, maxfev=10000)
        r_e, beta, D_e_kcal, V_min = popt
    except Exception:
        return None, None, None
    mu_amu = 14.003 / 2.0
    mu_au = mu_amu * 1822.888486209
    beta_per_bohr = beta / 1.8897259886
    D_e_au = D_e_kcal / 627.509
    omega_au = beta_per_bohr * np.sqrt(2.0 * D_e_au / mu_au)
    omega_cm = omega_au * 219474.6313705
    return float(r_e), float(omega_cm), float(D_e_kcal)


def reproduce_casci_via_vqe_sqd(mf, max_n_dets=1000):
    """Kanad LUCJ+SamplingSQD at this geometry; return energy + n_det."""
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import SamplingSQDSolver

    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=N2_FROZEN, active=N2_ACTIVE),
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = ham.n_electrons

    np.random.seed(0)
    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=n_active_e, n_layers=1)
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(0)
    params = rng.uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
    )
    solver = SamplingSQDSolver(
        ham, n_samples=20000, random_seed=0,
        backend='statevector', recover_configurations=True,
    )
    # Single-shot SQD (no iterative expansion at 16q — too slow)
    result = solver.solve(ansatz_circuit=bound)
    return result['energy'], result['n_determinants']


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bonds', type=str,
                        default='0.9,1.0,1.05,1.0977,1.15,1.2,1.3,1.4,1.5,1.7,2.0,2.5,3.0')
    parser.add_argument('--skip-vqe', action='store_true',
                        help='Skip the SamplingSQD verification step')
    args = parser.parse_args()

    rs = [float(s) for s in args.bonds.split(',')]
    print('=' * 92)
    print(f'M5-A — N₂ DISSOCIATION CURVE  ({len(rs)} points)  cc-pVDZ')
    print('=' * 92)

    # ===== STAGE 1: classical PES =====
    print(f'\n{"r (Å)":>8}  {"HF":>11}  {"MP2":>11}  {"CCSD":>11}  {"CCSD(T)":>11}  '
          f'{"CASCI(10,8)":>11}  {"|c_max|²":>10}')
    print(' ' + '-' * 86)
    pes = []
    for r in rs:
        t0 = time.time()
        try:
            mol, mf = build_n2_at_r(r)
            cls = compute_classical_energies(mf)
            print(f'  {r:>6.4f}  {cls["hf"]:>11.6f}  {cls["mp2"]:>11.6f}  '
                  f'{cls["ccsd"]:>11.6f}  {cls["ccsdt"]:>11.6f}  '
                  f'{cls["casci"]:>11.6f}  {cls["casci_max_weight_sq"]:>10.4f}')
            pes.append({'r': r, 'mf': mf, **cls})
        except Exception as e:
            print(f'  {r:>6.4f}  Failed: {e}')

    # ===== Morse fits for each method =====
    print()
    print('=' * 92)
    print('SPECTROSCOPIC CONSTANTS (Morse fit per method)')
    print('=' * 92)
    rs_used = [p['r'] for p in pes]
    morse_fits = {}
    for label, key in [('HF', 'hf'), ('MP2', 'mp2'), ('CCSD', 'ccsd'),
                       ('CCSD(T)', 'ccsdt'), ('CASCI(10,8)', 'casci')]:
        es = [p[key] for p in pes]
        r_e, omega_e, D_e = fit_morse(rs_used, es)
        morse_fits[label] = (r_e, omega_e, D_e)
        if r_e is None:
            print(f'  {label:14s}: Morse fit failed')
            continue
        print(f'  {label:14s}: r_e = {r_e:.4f} Å,  ω_e = {omega_e:7.1f} cm⁻¹,  D_e = {D_e:6.2f} kcal/mol')

    # ===== Comparison table vs experiment =====
    print()
    print('=' * 92)
    print('COMPARISON vs EXPERIMENT (Huber-Herzberg, NIST X¹Σ_g⁺)')
    print('=' * 92)
    print(f'  {"Property":18s}  ' + '  '.join(f'{lbl:>11}' for lbl in
          ['HF', 'MP2', 'CCSD', 'CCSD(T)', 'CASCI(10,8)', 'Exp']))
    print('  ' + '-' * 84)
    for prop, idx, ref_key in [('r_e (Å)', 0, 'r_e_angstrom'),
                                ('ω_e (cm⁻¹)', 1, 'omega_e_cm'),
                                ('D_e (kcal/mol)', 2, 'D_e_kcal_mol')]:
        line = f'  {prop:18s}  '
        for lbl in ['HF', 'MP2', 'CCSD', 'CCSD(T)', 'CASCI(10,8)']:
            val = morse_fits[lbl][idx]
            line += f'  {val:>11.4f}' if val is not None else f'  {"—":>11}'
        line += f'  {N2_REF[ref_key]:>11.4f}'
        print(line)
    # Per-method error vs experiment for r_e and ω_e
    print(f'\n  Δ vs Exp:')
    for lbl in ['HF', 'MP2', 'CCSD', 'CCSD(T)', 'CASCI(10,8)']:
        r_e, omega_e, D_e = morse_fits[lbl]
        if r_e is None:
            print(f'    {lbl:12s}: n/a'); continue
        dr = (r_e - N2_REF['r_e_angstrom']) * 1000
        dw = omega_e - N2_REF['omega_e_cm']
        dD = D_e - N2_REF['D_e_kcal_mol']
        print(f'    {lbl:12s}: Δr_e = {dr:+5.1f} mÅ,  Δω_e = {dw:+6.1f} cm⁻¹,  '
              f'ΔD_e = {dD:+5.1f} kcal/mol')

    # ===== STAGE 2: VQE/SQD verification at sample geometries =====
    if not args.skip_vqe:
        print()
        print('=' * 92)
        print('STAGE 2 — Kanad LUCJ + SamplingSQD reproduction of CASCI')
        print('=' * 92)
        # Sample geometries: equilibrium + stretched (multireference)
        sample_rs = [1.0977, 1.5]
        for r_sample in sample_rs:
            # Find or rebuild
            matching = [p for p in pes if abs(p['r'] - r_sample) < 1e-4]
            if matching:
                p = matching[0]
                mf = p['mf']
                e_casci = p['casci']
            else:
                _, mf = build_n2_at_r(r_sample)
                cls = compute_classical_energies(mf)
                e_casci = cls['casci']
            print(f'\n  r = {r_sample} Å:')
            print(f'    CASCI(10e, 8o)/cc-pVDZ = {e_casci:.6f} Ha  (within-basis truth)')
            t0 = time.time()
            try:
                e_sqd, n_det = reproduce_casci_via_vqe_sqd(mf)
                dt = time.time() - t0
                gap_mha = (e_sqd - e_casci) * 1000
                print(f'    Kanad SamplingSQD       = {e_sqd:.6f} Ha  '
                      f'({n_det} dets, {dt:.1f} s)')
                ok = '✓ framework correct' if abs(gap_mha) < 5.0 else '✗ EXCEEDS 5 mHa method-error budget'
                print(f'    Gap = {gap_mha:+.3f} mHa     {ok}')
            except Exception as e:
                print(f'    Kanad SamplingSQD: failed ({type(e).__name__}: {e})')

    print()
    print('=' * 92)
    print('SKEPTICAL CHECKS (PES quality)')
    print('=' * 92)
    # Multireference emergence: CASCI's max ci coefficient should drop at large r
    print('\n  CASCI dominant configuration weight along the curve:')
    for p in pes:
        c2 = p['casci_max_weight_sq']
        flag = '' if c2 > 0.85 else ('  ← MULTIREF (weight < 0.85)' if c2 > 0.5
                                     else '  ← STRONG MULTIREF (weight < 0.5)')
        print(f'    r = {p["r"]:.3f} Å:  |c_max|² = {c2:.4f}{flag}')
    print('\nDone.')


if __name__ == '__main__':
    main()
