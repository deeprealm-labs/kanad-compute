"""Campaign A — STRONG-CORRELATION truth test (the Campaign-B bridge).
On H-chains (full space → FCI and CCSD(T) BOTH computable on the SAME space),
scan from compressed to dissociated and show: CCSD(T) (classical gold standard)
DIVERGES from exact FCI as static correlation grows, while Kanad SQD/CI tracks
truth. This is where 'gold-standard classical' fails and a quantum/CI method wins.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_A_strong
"""
from __future__ import annotations
import time, traceback
import numpy as np


def hchain(n, R):
    return [('H', (0.0, 0.0, i * R)) for i in range(n)]


def refs(atoms, basis):
    """HF, CCSD(T), and exact FCI on the SAME full space."""
    from pyscf import gto, scf, cc, fci, ao2mo
    mol = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    e_hf = float(mf.e_tot)
    # CCSD(T)
    try:
        mcc = cc.CCSD(mf).run(verbose=0); et = mcc.ccsd_t()
        e_ccsdt = float(mcc.e_tot + et); e_ccsd = float(mcc.e_tot)
    except Exception:
        e_ccsdt = float('nan'); e_ccsd = float('nan')
    # exact FCI (full space)
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    ne = mol.nelectron; na = ne // 2
    e_fci, _ = fci.direct_spin0.kernel(h1, eri, norb, (na, ne - na), ecore=float(mol.energy_nuc()))
    return e_hf, e_ccsd, e_ccsdt, float(e_fci)


def run():
    from kanad import MolecularBuilder
    print("=" * 110, flush=True)
    print("CAMPAIGN A — STRONG CORRELATION: CCSD(T) vs exact FCI vs Kanad SQD on H-chains", flush=True)
    print("=" * 110, flush=True)
    CASES = [(6, 'sto-6g'), (8, 'sto-6g'), (10, 'sto-6g')]
    RSCAN = [1.0, 1.5, 2.0, 2.6, 3.2]
    for n, basis in CASES:
        for R in RSCAN:
            t0 = time.time(); atoms = hchain(n, R)
            try:
                e_hf, e_ccsd, e_ccsdt, e_fci = refs(atoms, basis)
                qs = (MolecularBuilder.from_atoms(atoms).basis(basis).active_space('full')
                      .solver('sqd', n_samples=40000, max_iterations=4, recovery_rounds=2,
                              random_seed=0, spin_s=0.0).build())
                e_sqd = qs.solve()['energy']
                ccsdt_err = (e_ccsdt - e_fci) * 1000   # CCSD(T) error vs truth
                sqd_err = (e_sqd - e_fci) * 1000        # Kanad error vs truth
                # multireference strength: how much correlation HF misses
                corr = (e_fci - e_hf) * 1000
                print(f"STRONG| H{n}@R={R:.1f}Å {2*n}q | FCI={e_fci:.5f} corr={corr:.1f}mHa | "
                      f"CCSD(T)_err={ccsdt_err:+.2f}mHa | SQD_err={sqd_err:+.3f}mHa | "
                      f"{'CCSD(T) FAILS' if abs(ccsdt_err) > abs(sqd_err) + 5 else ''} | t={time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                print(f"STRONG| H{n}@R={R:.1f} | CRASH {type(e).__name__}: {str(e)[:90]}", flush=True)
    print("STRONG_DONE", flush=True)


if __name__ == "__main__":
    run()
