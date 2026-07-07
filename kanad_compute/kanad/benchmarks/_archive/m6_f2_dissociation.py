"""M6-C — F₂ Bond Dissociation.

F₂ is the textbook case where Hartree-Fock is *qualitatively wrong*:
RHF predicts F₂ is unbound (D_e < 0). The bond is entirely a correlation
effect — the σ → σ* doubles in the σ-electron-pair description.

This champion demonstrates the HF → correlated transition end-to-end:
  - HF: D_e should be negative (textbook pathology)
  - CCSD: D_e ≈ 1.5 eV
  - CCSD(T): D_e ≈ 1.65 eV (matches experiment)
  - CASSCF(2,2): captures the σ/σ* static correlation alone
  - SQD on CAS(2,2) reproduces CASSCF

References:
  - Experiment (Huber-Herzberg / NIST):  r_e = 1.4119 Å, ω_e = 916.6 cm⁻¹,
                                          D_e = 1.66 eV
  - Peterson 1998 CCSD(T)/aug-cc-pV5Z:    D_e = 1.65 eV
  - Bytautas 2007 CASPT2/aug-cc-pVQZ:     D_e = 1.64 eV
"""

from __future__ import annotations

import time
import numpy as np
from pyscf import gto, scf, mcscf, cc


SCAN_R = np.concatenate([
    np.arange(1.20, 1.50, 0.02),
    np.arange(1.50, 2.50, 0.05),
    np.arange(2.50, 5.01, 0.20),
])


def build_f2(r):
    return gto.M(atom=f'F 0 0 0; F 0 0 {r:.4f}',
                 basis='cc-pvdz', spin=0, verbose=0)


def compute_point(r, do_ccsdt=True):
    mol = build_f2(r)
    mf = scf.RHF(mol).run(verbose=0)
    out = {'r': r, 'e_hf': float(mf.e_tot)}
    if do_ccsdt:
        try:
            ccobj = cc.CCSD(mf).run(verbose=0)
            out['e_ccsd'] = float(ccobj.e_tot)
            try:
                out['e_ccsdt'] = float(ccobj.e_tot + ccobj.ccsd_t())
            except Exception:
                out['e_ccsdt'] = float('nan')
        except Exception:
            out['e_ccsd'] = out['e_ccsdt'] = float('nan')
    # CAS(2,2) — σ / σ* of the F-F bond
    n_occ = int(np.sum(mf.mo_occ > 0))
    active = [n_occ - 1, n_occ]
    try:
        cas = mcscf.CASSCF(mf, ncas=2, nelecas=2)
        cas.sort_mo(active)
        cas.run(verbose=0)
        out['e_casscf'] = float(cas.e_tot)
        ci = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
        out['max_weight'] = float(np.max(np.abs(ci)) ** 2)
        out['mf'] = mf; out['cas'] = cas; out['active'] = active
    except Exception as exc:
        out['e_casscf'] = float('nan')
        out['error_casscf'] = str(exc)
    return out


def morse_fit(rs, es):
    from scipy.optimize import curve_fit
    def morse(r, De, alpha, re, V0):
        return De * (1.0 - np.exp(-alpha * (r - re))) ** 2 + V0
    p0 = (0.06, 2.0, 1.41, np.min(es))
    popt, _ = curve_fit(morse, rs, es, p0=p0, maxfev=20000)
    De, alpha, re, V0 = popt
    alpha_bohr_inv = alpha / 1.8897259886
    k_au = 2.0 * De * alpha_bohr_inv * alpha_bohr_inv
    # F atom: 18.998 amu × 1822.888 = m_e units
    mu_au = (18.998 * 18.998 / (2 * 18.998)) * 1822.888486
    omega_cm = np.sqrt(k_au / mu_au) * 219474.63
    return re, omega_cm, De


def verify_sqd_at_point(pt):
    """Quantum sampling routes to BlueQubit CPU cloud. Requires BLUEQUBIT_API_KEY."""
    import os
    import bluequbit
    from kanad.core.active_space import ActiveSpaceSelector, build_active_space_hamiltonian
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _generate_singles_doubles, _filter_with_recovery,
    )
    mf = pt['mf']
    cas = pt['cas']
    active = pt['active']
    mf.mo_coeff = cas.mo_coeff
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
    bq = bluequbit.init(os.environ['BLUEQUBIT_API_KEY'])
    bq_result = bq.run(circuits=bound, device='cpu', shots=10000,
                        job_name=f'm6_f2_r{pt["r"]:.3f}')
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
    res = solver._diagonalize_in_subspace_pyscf(dets)
    return float(res['energy'])


def main():
    HA_TO_EV = 27.211386
    HA_TO_KCAL = 627.509
    print('=' * 96)
    print('M6-C — F₂ DISSOCIATION  cc-pVDZ  HF+CCSD+CCSD(T)+CASSCF(2,2)')
    print('=' * 96)
    print(f'\n  Scanning {len(SCAN_R)} points r ∈ [{SCAN_R[0]:.2f}, {SCAN_R[-1]:.2f}] Å')
    t0 = time.time()
    pts = []
    for r in SCAN_R:
        pt = compute_point(r, do_ccsdt=True)
        pts.append(pt)
        wt = pt.get('max_weight', float('nan'))
        print(f'  r = {r:5.3f}  HF = {pt["e_hf"]:.6f}  CCSD = {pt.get("e_ccsd", "nan"):>11}'
              f'  CCSD(T) = {pt.get("e_ccsdt", "nan"):>11}  '
              f'CASSCF(2,2) = {pt.get("e_casscf", "nan"):>11}  |c_max|² = {wt:.3f}')
    print(f'  scan complete in {time.time()-t0:.1f} s')

    rs = np.array([p['r'] for p in pts])
    print()
    print('=' * 96)
    print('SPECTROSCOPIC CONSTANTS (Morse fit)')
    print('=' * 96)
    for method in ['e_hf', 'e_ccsd', 'e_ccsdt', 'e_casscf']:
        es = np.array([p.get(method, float('nan')) for p in pts])
        mask = ~np.isnan(es) & (rs < 2.5)
        if mask.sum() < 5:
            continue
        try:
            re, ome, de = morse_fit(rs[mask], es[mask])
            de_eV = de * HA_TO_EV
            print(f'  {method:<10} r_e = {re:.4f} Å  ω_e = {ome:.1f} cm⁻¹  D_e = {de_eV:+.3f} eV')
        except Exception as exc:
            print(f'  {method:<10} fit failed ({type(exc).__name__})')

    print()
    print('  Reference:  r_e = 1.4119 Å,  ω_e = 916.6 cm⁻¹,  D_e = 1.66 eV (Huber-Herzberg)')
    print('              Peterson 1998 CCSD(T)/aug-cc-pV5Z: D_e = 1.65 eV')

    # SQD framework anchor at 3 points
    print()
    print('=' * 96)
    print('FRAMEWORK ANCHOR — SQD vs CASSCF(2,2) at 3 anchor points')
    print('=' * 96)
    for r_target in [1.40, 2.00, 3.00]:
        i = int(np.argmin(np.abs(rs - r_target)))
        pt = pts[i]
        if 'cas' not in pt:
            continue
        try:
            e_sqd = verify_sqd_at_point(pt)
            gap = (e_sqd - pt['e_casscf']) * 1000
            tag = '✓' if abs(gap) < 1.0 else '✗'
            print(f'  r = {pt["r"]:.3f} Å  CASSCF = {pt["e_casscf"]:.6f}'
                  f'  SQD = {e_sqd:.6f}  gap = {gap:+.4f} mHa  {tag}')
        except Exception as exc:
            print(f'  r = {pt["r"]:.3f} Å  SQD failed: {exc}')

    # The HF pathology check
    print()
    print('=' * 96)
    print('HF PATHOLOGY CHECK — does HF predict F₂ unbound?')
    print('=' * 96)
    hf_es = np.array([p['e_hf'] for p in pts])
    hf_min_idx = int(np.argmin(hf_es))
    hf_inf_idx = int(np.argmax(rs))
    hf_de = (hf_es[hf_inf_idx] - hf_es[hf_min_idx]) * HA_TO_EV
    sign = 'BOUND' if hf_de > 0 else 'UNBOUND'
    print(f'  HF D_e (E(r_∞) − E(r_min)) = {hf_de:+.3f} eV  →  {sign}')
    if hf_de < 0:
        print('  ✓ Reproduces the known HF pathology (F₂ unbound at the HF level).')
    else:
        print('  ⚠ HF reports F₂ bound — pathology not reproduced; investigate.')


if __name__ == '__main__':
    main()
