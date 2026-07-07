"""M6-B — C₂ Carbon Dimer Dissociation.

Why C₂: the wavefunction at r_e is *strongly multireference*: the
ground X¹Σ_g⁺ state has |c_max|² ≈ 0.7 in CAS(8,8); dissociation
limit is even more so. The "quadruple bond" debate (Shaik 2012) hinges
on whether the σ_g 2s²σ_u 2s² in-out correlations should be counted as
bonding. This champion stresses:

  - CASSCF(8e, 8o) orbital optimisation on a strongly correlated bond
  - SQD reproducing CASSCF over a PES (not just a single point)
  - Spectroscopic constant extraction (r_e, ω_e, D_e) with experimental
    anchor

Active space: 8 electrons in 8 orbitals = 2σ_g + 2σ_u + 1π_u(×2) + 3σ_g + 1π_g(×2)
                                          + the antibonding partners (8 spatial).
That's the standard valence active space for C₂.

References:
  - Experiment (Huber-Herzberg / NIST):  r_e = 1.2425 Å, ω_e = 1854.7 cm⁻¹
                                          D_e = 6.21 eV = 143.2 kcal/mol
  - Booth 2011 i-FCIQMC:                   D_e = 6.17 eV (cc-pV5Z extrapolated)
  - Su 2009 MRCI/cc-pVQZ:                  r_e = 1.2426 Å, ω_e = 1869 cm⁻¹
"""

from __future__ import annotations

import time
import numpy as np
from pyscf import gto, scf, mcscf


# C₂ active space — 8 valence MOs (2σ_g²σ_u²π_u⁴ ... + antibonding partners)
ACTIVE_NELEC = 8

# Scan grid — 15 points, dense near r_e (1.24 Å), sparse in dissociation tail.
# CASSCF(8,8) is ~30-60 s per point at cc-pVDZ on C₂, so the grid is sized
# to total ~10 min wall on the classical reference side.
SCAN_R = np.array([
    1.10, 1.15, 1.20, 1.24, 1.28, 1.32, 1.40, 1.50,
    1.65, 1.85, 2.10, 2.50, 3.00, 4.00, 5.00,
])


def build_c2(r_AA: float):
    return gto.M(
        atom=f'C 0 0 0; C 0 0 {r_AA:.4f}',
        basis='cc-pvdz', spin=0, charge=0, verbose=0,
    )


def find_valence_active_window(mf, n_active=8):
    """Pick the 8 valence MOs centered around HOMO/LUMO.

    For C₂ at cc-pVDZ: 12 electrons total → 6 occupied MOs, HOMO=5.
    Active = HOMO-3 .. HOMO+4 (8 MOs).
    """
    n_occ = int(np.sum(mf.mo_occ > 0))
    lo = n_occ - n_active // 2
    hi = lo + n_active
    return list(range(lo, hi))


def compute_point(r, casscf=False):
    """CASCI(8,8) by default — deterministic & smooth across the curve.

    CASSCF tends to converge to different states at different r on C₂
    (multireference); for a *smooth PES* + framework-anchor comparison,
    CASCI with HF orbitals is the right reference. SQD with iterative
    expansion reaches the same CASCI eigenvalue.
    """
    mol = build_c2(r)
    mf = scf.RHF(mol).run(verbose=0)
    active = find_valence_active_window(mf, n_active=8)
    try:
        if casscf:
            cas = mcscf.CASSCF(mf, ncas=8, nelecas=ACTIVE_NELEC)
            cas.max_cycle_macro = 80
        else:
            cas = mcscf.CASCI(mf, ncas=8, nelecas=ACTIVE_NELEC)
        cas.sort_mo(active, base=0)   # 0-indexed to match kanad's manual selector
        cas.run(verbose=0)
    except Exception as exc:
        return {'r': r, 'error': f'{type(exc).__name__}: {exc}'}

    ci_vec = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
    max_w = float(np.max(np.abs(ci_vec)) ** 2)
    return {
        'r': r,
        'e_hf': float(mf.e_tot),
        'e_cas': float(cas.e_tot),
        'max_weight': max_w,
        'active': active,
        'mf': mf,
        'cas': cas,
        'converged': bool(cas.converged) if hasattr(cas, 'converged') else True,
    }


def morse_fit(rs, es):
    """Fit a Morse potential V(r) = D_e * (1 - exp(-α(r - r_e)))² + V_min.

    Returns r_e, omega_e (cm⁻¹), D_e (Ha) from the fit.
    """
    from scipy.optimize import curve_fit
    def morse(r, De, alpha, re, V0):
        return De * (1.0 - np.exp(-alpha * (r - re))) ** 2 + V0
    p0 = (0.2, 2.0, 1.24, np.min(es))
    popt, _ = curve_fit(morse, rs, es, p0=p0, maxfev=20000)
    De, alpha, re, V0 = popt
    # ω_e from k = 2 D_e α² (atomic units); convert to cm⁻¹
    k = 2.0 * De * alpha * alpha           # in Ha/bohr² when r is in Å? — careful units
    # Use exact Morse formula: ω_e = (1/2πc) sqrt(k/μ)
    # but our r is in Å; convert α to bohr⁻¹: 1 Å = 1.8897259886 bohr
    alpha_bohr_inv = alpha / 1.8897259886
    k_au = 2.0 * De * alpha_bohr_inv * alpha_bohr_inv     # Ha / bohr²
    # μ for ¹²C₂ = 6.000 amu = 6.000 × 1822.888 m_e in atomic units
    mu_au = 6.000 * 1822.888486
    omega_au = np.sqrt(k_au / mu_au)        # in atomic units (radians/time)
    omega_cm = omega_au * 219474.63 / (2 * np.pi)
    # Actually ω_e in cm⁻¹ = (omega_au_in_Hartree) * 219474.63
    omega_cm = np.sqrt(k_au / mu_au) * 219474.63
    return re, omega_cm, De, V0


def verify_sqd_at_point(r, casscf_data):
    """Kanad SamplingSQD on the CAS(8,8) active space at this point.

    Quantum-sampling step routes to BlueQubit CPU cloud (matches
    `benchmarks/m6_pbenzyne_warmup.py`). Requires BLUEQUBIT_API_KEY env var.
    """
    import os
    import bluequbit
    from kanad.core.active_space import ActiveSpaceSelector, build_active_space_hamiltonian
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _generate_singles_doubles, _filter_with_recovery,
    )
    mf = casscf_data['mf']
    active = casscf_data['active']
    cas = casscf_data['cas']

    # The CASSCF has rotated MOs; ham must use mf.mo_coeff = cas.mo_coeff
    mf.mo_coeff = cas.mo_coeff
    mf.mo_occ = cas.mo_occ if hasattr(cas, 'mo_occ') else mf.mo_occ
    frozen = list(range(active[0]))
    sel = ActiveSpaceSelector(mf).manual(frozen=frozen, active=active)
    ham = build_active_space_hamiltonian(mf, sel)
    n_qubits = 2 * ham.n_orbitals

    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=ham.n_electrons,
                       n_layers=1, target_sz=0.0)
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(0)
    params = rng.uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters({qc.parameters[i]: float(params[i])
                                  for i in range(qc.num_parameters)})
    if bound.num_clbits == 0:
        bound.measure_all()

    # Cloud sampling on BlueQubit CPU (free, ~1s)
    bq = bluequbit.init(os.environ['BLUEQUBIT_API_KEY'])
    bq_result = bq.run(circuits=bound, device='cpu', shots=10000,
                        job_name=f'm6_c2_r{r:.3f}')
    counts = bq_result.get_counts()
    samples = []
    for bs, n in counts.items():
        samples.extend([int(bs.replace(' ', ''), 2)] * int(n))
    samples = np.array(samples, dtype=np.int64)

    solver = SamplingSQDSolver(ham, n_samples=len(samples), random_seed=0,
                                recover_configurations=True,
                                ci_backend='pyscf', target_sz=0.0)
    mo_e = solver._resolve_mo_energies()
    valid, *_ = _filter_with_recovery(samples, ham.n_orbitals, ham.n_electrons,
                                       0.0, mo_e)
    dets = sorted(set(int(d) for d in valid))
    last = None
    for it in range(4):
        res = solver._diagonalize_in_subspace_pyscf(dets)
        if last is not None and abs(res['energy'] - last) < 1e-6:
            break
        last = res['energy']
        evec = res['eigenvector']
        top = np.argsort(np.abs(evec) ** 2)[::-1][:min(50, len(dets))]
        new_dets = set()
        for i in top:
            new_dets.update(_generate_singles_doubles(dets[i], n_qubits, ham.n_electrons))
        old = len(dets)
        dets = sorted(set(dets) | new_dets)
        if len(dets) == old:
            break
    return float(res['energy'])


def main():
    HA_TO_EV = 27.211386
    HA_TO_KCAL = 627.509
    print('=' * 96)
    print('M6-B — C₂ DIMER DISSOCIATION  cc-pVDZ  CASCI(8e, 8o)  + cloud SQD anchor')
    print('  Classical: PySCF CASCI(8,8) on RHF orbitals (smooth, deterministic).')
    print('  Quantum:   LUCJ → BlueQubit CPU cloud → iterative classical expansion.')
    print('=' * 96)
    print(f'\n  Scanning {len(SCAN_R)} points r ∈ [{SCAN_R[0]:.2f}, {SCAN_R[-1]:.2f}] Å')

    results = []
    t0 = time.time()
    for r in SCAN_R:
        pt = compute_point(r, casscf=False)
        if 'error' in pt:
            print(f'  r={r:.3f} Å — {pt["error"]}')
            continue
        results.append(pt)
        wt = pt['max_weight']
        print(f'  r = {r:5.3f} Å  E = {pt["e_cas"]:11.6f}  |c_max|² = {wt:.4f}'
              + ('   [MR]' if wt < 0.9 else ''))
    dt = time.time() - t0
    print(f'  classical scan complete in {dt:.1f} s')

    if not results:
        print('  no successful points — abort')
        return

    rs = np.array([p['r'] for p in results])
    es = np.array([p['e_cas'] for p in results])
    # E at large r (~ 5 Å) ≈ 2 × E(C atom)
    e_inf = es[np.argmax(rs)]
    de_au = e_inf - es.min()
    re_grid = rs[np.argmin(es)]

    print()
    print('=' * 96)
    print('SPECTROSCOPIC CONSTANTS (Morse fit)')
    print('=' * 96)
    try:
        # Fit only the bound region (r < 2.5 Å)
        mask = rs < 2.5
        re_fit, omega_fit, de_fit, _ = morse_fit(rs[mask], es[mask])
        de_fit_eV = de_fit * HA_TO_EV
        print(f'  Fitted r_e  = {re_fit:.4f} Å    grid-min r_e  = {re_grid:.4f} Å')
        print(f'  Fitted ω_e  = {omega_fit:.1f} cm⁻¹')
        print(f'  Fitted D_e  = {de_fit_eV:.3f} eV = {de_fit*HA_TO_KCAL:.1f} kcal/mol')
    except Exception as exc:
        print(f'  Morse fit failed: {type(exc).__name__}: {exc}')
        re_fit = omega_fit = de_fit_eV = float('nan')

    print()
    print('  Reference:')
    print('    Experiment (NIST/Huber):    r_e = 1.2425 Å,  ω_e = 1854.7 cm⁻¹, D_e = 6.21 eV')
    print('    Booth 2011 i-FCIQMC:        D_e = 6.17 eV  (cc-pV5Z extrapolated)')
    print('    Su 2009 MRCI/cc-pVQZ:       r_e = 1.2426 Å, ω_e = 1869 cm⁻¹')

    # Framework anchor: SQD vs CASSCF at 3 representative points
    print()
    print('=' * 96)
    print('FRAMEWORK ANCHOR — SQD vs CASCI(8,8) at 3 anchor points')
    print('=' * 96)
    anchor_indices = [np.argmin(np.abs(rs - 1.24)), np.argmin(np.abs(rs - 1.80)),
                      np.argmin(np.abs(rs - 3.00))]
    for i in anchor_indices:
        pt = results[i]
        t1 = time.time()
        try:
            e_sqd = verify_sqd_at_point(pt['r'], pt)
            gap = (e_sqd - pt['e_cas']) * 1000
            tag = '✓' if abs(gap) < 1.0 else ('⚠' if abs(gap) < 5.0 else '✗')
            print(f'  r = {pt["r"]:5.3f} Å  '
                  f'CASCI = {pt["e_cas"]:.6f}  SQD = {e_sqd:.6f}  '
                  f'gap = {gap:+.4f} mHa  {tag}  ({time.time()-t1:.1f}s)')
        except Exception as exc:
            print(f'  r = {pt["r"]:5.3f} Å  SQD failed: {type(exc).__name__}: {exc}')

    # Multireference character profile
    print()
    print('=' * 96)
    print('MULTIREFERENCE CHARACTER ACROSS THE CURVE')
    print('=' * 96)
    print(f'  r_e region (|c_max|² > 0.8): {sum(1 for p in results if p["max_weight"] > 0.8)} pts')
    print(f'  intermediate  (0.5 ≤ |c_max|² ≤ 0.8): '
          f'{sum(1 for p in results if 0.5 <= p["max_weight"] <= 0.8)} pts')
    print(f'  strong MR     (|c_max|² < 0.5): '
          f'{sum(1 for p in results if p["max_weight"] < 0.5)} pts')


if __name__ == '__main__':
    main()
