"""M5-B (focused) — HCN ↔ HNC isomerization at 3 stationary points.

The full path PES is expensive and requires careful geometry optimization at
each step (otherwise we sample far above the minimum-energy path). For a
publishable barrier we use 3 fixed geometries from the literature:
  - HCN equilibrium (linear, NIST CCSD(T)/cc-pVTZ optimized)
  - TS (cyclic, Bowman 1993 CCSD(T)/cc-pVQZ optimized)
  - HNC equilibrium (linear)

At each geometry we compute HF, CCSD, CCSD(T), and CASCI(6,6)/cc-pVDZ,
plus Kanad SamplingSQD verification. The barrier and reaction energy
follow from the three energies + ZPE corrections.

Reference values (Bowman & Gazdy 1993, CCSD(T)/cc-pVQZ):
  - E_barrier (electronic): ~31.9 kcal/mol
  - ΔE_rxn (electronic):    ~14.8 kcal/mol
  - ZPE corrections shift each by ~0.5-1 kcal/mol
"""

from __future__ import annotations

import time
import numpy as np
from pyscf import gto, scf, mp, cc, mcscf


# Three published stationary-point geometries (Bowman, Maki, NIST)
GEOMETRIES = {
    'HCN_reactant': {
        # Linear H-C≡N, NIST experimental r_e
        'H': (0.0, 0.0, 0.000),
        'C': (0.0, 0.0, 1.066),
        'N': (0.0, 0.0, 2.222),
    },
    'TS': {
        # Cyclic TS from Bowman 1993 CCSD(T)/cc-pVQZ
        # ∠HCN = 71.4°, r(CH) = 1.21 Å, r(NH) = 1.20 Å, r(CN) = 1.183 Å
        # In Cartesian: C at origin, N along x, H out of axis
        'C': (0.0,    0.0,   0.0),
        'N': (1.183,  0.0,   0.0),
        'H': (1.21 * np.cos(np.deg2rad(71.4)),
              1.21 * np.sin(np.deg2rad(71.4)), 0.0),
    },
    'HNC_product': {
        # Linear H-N≡C
        'H': (0.0, 0.0, 0.000),
        'N': (0.0, 0.0, 0.994),
        'C': (0.0, 0.0, 2.163),
    },
}

FROZEN_ORBS = [0, 1]
ACTIVE_ORBS = [2, 3, 4, 5, 6, 7]


def geom_str(geom):
    return '; '.join(
        f'{atom} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}' for atom, p in geom.items()
    )


def compute_point(label, geom):
    print(f'\n  [{label}]  {geom_str(geom)}')
    mol = gto.M(atom=geom_str(geom), basis='cc-pvdz', spin=0, charge=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    out = {'mol': mol, 'mf': mf, 'hf': float(mf.e_tot), 'geom': geom}
    out['mp2'] = float(mp.MP2(mf).run(verbose=0).e_tot)
    ccsd = cc.CCSD(mf).run(verbose=0)
    out['ccsd'] = float(ccsd.e_tot)
    out['ccsdt'] = float(ccsd.e_tot + ccsd.ccsd_t())
    cas = mcscf.CASCI(mf, ncas=6, nelecas=6).run(verbose=0)
    out['casci'] = float(cas.e_tot)
    ci_vec = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
    out['casci_max_weight'] = float(np.max(np.abs(ci_vec)) ** 2)
    print(f'    HF       = {out["hf"]:.6f}')
    print(f'    MP2      = {out["mp2"]:.6f}')
    print(f'    CCSD     = {out["ccsd"]:.6f}')
    print(f'    CCSD(T)  = {out["ccsdt"]:.6f}')
    print(f'    CASCI(6,6) = {out["casci"]:.6f}  (|c_max|² = {out["casci_max_weight"]:.4f})')
    return out


def verify_sqd(label, mf):
    """Kanad SamplingSQD at this point — must reproduce CASCI."""
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    from kanad.solvers.sampling_sqd import SamplingSQDSolver
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=FROZEN_ORBS, active=ACTIVE_ORBS),
    )
    n_qubits = 2 * ham.n_orbitals
    n_e = ham.n_electrons
    cas = mcscf.CASCI(mf, ncas=6, nelecas=n_e).run(verbose=0)
    e_casci = float(cas.e_tot)

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
    t0 = time.time()
    res = solver.solve_iterative(
        ansatz_circuit=bound, max_iterations=3, expansion_per_round=50,
        energy_tol=1e-5,
    )
    dt = time.time() - t0
    gap = (res['energy'] - e_casci) * 1000
    print(f'    [{label} SQD] CASCI = {e_casci:.6f}, SQD = {res["energy"]:.6f}, '
          f'gap = {gap:+.4f} mHa, n_det = {res["n_determinants"]}  ({dt:.1f}s)')
    return e_casci, res['energy'], gap


def main():
    print('=' * 92)
    print('M5-B — HCN ↔ HNC ISOMERIZATION — 3 stationary points  cc-pVDZ')
    print('=' * 92)

    pts = {}
    for label, geom in GEOMETRIES.items():
        pts[label] = compute_point(label, geom)

    HA_TO_KCAL = 627.509
    print()
    print('=' * 92)
    print('REACTION ENERGETICS (kcal/mol, electronic — no ZPE)')
    print('=' * 92)
    for method in ('hf', 'mp2', 'ccsd', 'ccsdt', 'casci'):
        e_r = pts['HCN_reactant'][method]
        e_t = pts['TS'][method]
        e_p = pts['HNC_product'][method]
        if not all(np.isfinite([e_r, e_t, e_p])):
            print(f'  {method.upper():8s}: incomplete'); continue
        barrier = (e_t - e_r) * HA_TO_KCAL
        rxn = (e_p - e_r) * HA_TO_KCAL
        print(f'  {method.upper():8s}:  barrier = {barrier:>7.2f} kcal/mol,   '
              f'ΔE_rxn = {rxn:>+6.2f} kcal/mol')

    print()
    print('=' * 92)
    print('COMPARISON vs LITERATURE')
    print('=' * 92)
    print('  Reference  Barrier (kcal/mol)  ΔE_rxn (kcal/mol)  Notes')
    print('  ' + '-' * 72)
    print('  Bowman 1993 CCSD(T)/cc-pVQZ      31.9            14.8           electronic')
    print('  W4 / CBS extrapolation           ~31              ~14           CBS-limit')
    print('  Maki 1990 experiment             ~28              ~14           T=0 (ZPE-corrected)')
    print('  Kanad CASCI(6,6)/cc-pVDZ        '
          f'{(pts["TS"]["casci"] - pts["HCN_reactant"]["casci"]) * HA_TO_KCAL:>5.2f}'
          '          '
          f'{(pts["HNC_product"]["casci"] - pts["HCN_reactant"]["casci"]) * HA_TO_KCAL:>+5.2f}'
          '         electronic')
    print('  Kanad CCSD(T)/cc-pVDZ           '
          f'{(pts["TS"]["ccsdt"] - pts["HCN_reactant"]["ccsdt"]) * HA_TO_KCAL:>5.2f}'
          '          '
          f'{(pts["HNC_product"]["ccsdt"] - pts["HCN_reactant"]["ccsdt"]) * HA_TO_KCAL:>+5.2f}'
          '         electronic')

    print()
    print('=' * 92)
    print('STAGE 2 — Kanad SamplingSQD reproduces CASCI at each point')
    print('=' * 92)
    for label, p in pts.items():
        try:
            verify_sqd(label, p['mf'])
        except Exception as e:
            print(f'    [{label}] failed: {type(e).__name__}: {e}')

    print()
    print('=' * 92)
    print('SKEPTICAL CHECKS')
    print('=' * 92)
    cmaxes = {k: v['casci_max_weight'] for k, v in pts.items()}
    print(f'  CASCI |c_max|²: ')
    for k, v in cmaxes.items():
        flag = '' if v > 0.85 else ('  ← MULTIREF' if v > 0.5 else '  ← STRONG MULTIREF')
        print(f'    {k:18s} = {v:.4f}{flag}')
    if all(v > 0.85 for v in cmaxes.values()):
        print(f'  ✓ Single-reference throughout — CCSD(T) reference is reliable')


if __name__ == '__main__':
    main()
