"""RE-TEST R5 cloud frontier — SQD on BlueQubit cloud statevector at 20→28 qubits vs EXACT FCI-in-CAS.
Pushes the subspace SCALE beyond the QPU 24q run (noise-free cloud), and tests strong correlation at
scale (stretched N2). Honest decomposition: pure-sample vs +recovery vs +cisd-seed.

BLUEQUBIT_API_KEY must be in the environment (passed at launch, NOT written to this file).

    cd /root/kanad-framework && BLUEQUBIT_API_KEY=... PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.cloud_frontier
"""
from __future__ import annotations
import os
import time
import numpy as np

BASIS = 'cc-pvdz'
N_TOTAL_E = 14  # N2


def geom(r):
    return [('N', (0, 0, 0)), ('N', (0, 0, r))]


def fci_in_cas(ncas, ne, r):
    from pyscf import gto, scf, mcscf, ao2mo, fci
    mol = gto.M(atom=geom(r), basis=BASIS, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    cas = mcscf.CASCI(mf, ncas, ne)              # auto ncore = (14-ne)//2 lowest orbitals frozen
    h1, ec = cas.get_h1eff()
    h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
    e, _ = fci.direct_spin0.kernel(h1, h2, ncas, ne, ecore=ec)
    return float(e)


def sqd_cloud(ncas, ne, r, recover=True, cisd=False):
    from kanad import MolecularBuilder
    ncore = (N_TOTAL_E - ne) // 2
    frozen = list(range(ncore))
    active = list(range(ncore, ncore + ncas))
    qs = (MolecularBuilder.from_atoms(geom(r)).basis(BASIS)
          .active_space('manual', frozen=frozen, active=active)
          .solver('sqd', backend='bluequbit', bq_device='cpu', n_samples=12000,
                  recovery_rounds=(1 if recover else 0), recover_configurations=recover,
                  cisd_seed=cisd, sampling_init='physical', random_seed=0,
                  spin_s=0.0).build())
    assert qs.spec.backend == 'bluequbit', "NOT routed to BlueQubit cloud!"
    res = qs.solve()
    return float(res['energy']), res.get('n_determinants')


def run(label, ncas, ne, r):
    print(f"\n--- {label}: N2 CAS({ne},{ncas})={2*ncas}q  r={r} ---", flush=True)
    try:
        e_fci = fci_in_cas(ncas, ne, r)
        print(f"  EXACT FCI-in-CAS = {e_fci:.6f} Ha", flush=True)
        for tag, rec, cis in (('pure-sample', False, False), ('+recovery', True, False), ('+recovery+cisd', True, True)):
            t0 = time.time()
            try:
                e, d = sqd_cloud(ncas, ne, r, recover=rec, cisd=cis)
                print(f"  {tag:16} = {e:.6f} | gap_vs_FCI={(e-e_fci)*1000:+.2f} mHa | dets={d} | t={time.time()-t0:.0f}s", flush=True)
            except Exception as ex:
                print(f"  {tag:16} FAILED: {type(ex).__name__}: {str(ex)[:90]}", flush=True)
    except Exception as e:
        print(f"  reference CRASH {type(e).__name__}: {str(e)[:80]}", flush=True)


def main():
    print("=" * 100, flush=True)
    print("RE-TEST R5 CLOUD FRONTIER — SQD on BlueQubit 20-28q vs EXACT FCI-in-CAS", flush=True)
    print("=" * 100, flush=True)
    if not os.environ.get('BLUEQUBIT_API_KEY'):
        print("*** BLUEQUBIT_API_KEY not set — aborting (pass it in the launch env).", flush=True)
        return
    # SQD subspace-scale ladder at equilibrium
    run('20q ladder', ncas=10, ne=10, r=1.10)
    run('24q ladder', ncas=12, ne=10, r=1.10)
    run('28q frontier', ncas=14, ne=10, r=1.10)
    # strong correlation at scale (stretched bond)
    run('20q stretched', ncas=10, ne=10, r=2.10)
    print("\nCLOUD_FRONTIER_DONE", flush=True)


if __name__ == "__main__":
    main()
