"""Campaign A — REAL-QPU SQD vs EXACT FCI (honest benchmark).
Large-ish FCI-feasible active space on IBM Heron. Reports, vs the EXACT in-active-space
FCI: (a) the PURE quantum-sample energy (sampled dets only, no recovery, no seed),
(b) standard SQD (recovery), (c) CISD-seeded SQD. This disambiguates what the QPU
SAMPLE actually delivers from what classical post-processing adds — the honesty the
IP audit demanded. Confirms backend is real Heron (not the CI auto-route).

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_A_qpu_fci
"""
from __future__ import annotations
import os, time
import numpy as np

os.environ.setdefault('IBM_QUANTUM_TOKEN', 'kpAbwR0cleN0VLpj5fpbpEmsThZZtvS3T4L_rxrB7i2S')
os.environ.setdefault('IBM_QUANTUM_CRN', 'crn:v1:bluemix:public:quantum-computing:us-east:a/e8d9164b3ff748daa345dce1fcc0437e:fa184d28-2d7a-471b-8cc4-8986436c7439::')

# N2 CAS(12,12) = 24 qubits. FCI dim C(12,6)^2 = 853,776 — exact FCI feasible on the cluster.
ATOMS = [('N', (0, 0, 0)), ('N', (0, 0, 1.10))]
BASIS = 'cc-pvdz'
FROZEN = [0, 1]
ACTIVE = list(range(2, 14))   # 12 active orbitals -> 24 qubits
NE_ACTIVE = 10                # 14 - 2*2 frozen


def exact_fci_in_cas():
    """Exact FCI in the SAME CAS(12,12) the SQD uses (the true reference)."""
    from pyscf import gto, scf, mcscf, ao2mo, fci
    mol = gto.M(atom=ATOMS, basis=BASIS, verbose=0); mf = scf.RHF(mol).run(verbose=0)
    ncas = len(ACTIVE); nelecas = NE_ACTIVE
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1, ecore = cas.get_h1eff(); h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
    e, _ = fci.direct_spin0.kernel(h1, h2, ncas, nelecas, ecore=ecore)
    return float(e)


def heron_sqd(recover, cisd):
    from kanad import MolecularBuilder
    qs = (MolecularBuilder.from_atoms(ATOMS).basis(BASIS)
          .active_space('manual', frozen=FROZEN, active=ACTIVE).backend('ibm')
          .solver('sqd', n_samples=12000, max_iterations=1,
                  recovery_rounds=(1 if recover else 0),
                  recover_configurations=recover,
                  random_seed=0, spin_s=0.0, cisd_seed=cisd,
                  ibm_backend_name='ibm_marrakesh').build())
    # confirm it's actually the SQD/ibm route, not CI auto
    assert qs.spec.solver == 'sqd' and qs.spec.backend == 'ibm', "NOT on the IBM SQD route!"
    r = qs.solve()
    return float(r['energy']), r.get('n_determinants')


def main():
    print("=" * 100, flush=True)
    print("CAMPAIGN A — REAL-QPU SQD vs EXACT FCI (N2 CAS(12,12) 24q on IBM Heron)", flush=True)
    print("=" * 100, flush=True)
    e_fci = exact_fci_in_cas()
    print(f"EXACT FCI(12,12)/cc-pVDZ = {e_fci:.6f} Ha (the true reference)", flush=True)
    # (a) PURE quantum sample (no recovery, no seed) — what the QPU sample alone gives
    t0 = time.time()
    try:
        e_q, d_q = heron_sqd(recover=False, cisd=False)
        print(f"QPU-PURE   (sample only)      = {e_q:.6f} | gap_vs_FCI={(e_q-e_fci)*1000:+.2f} mHa | dets={d_q} | t={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"QPU-PURE FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    # (b) standard SQD (recovery) — what config recovery adds
    t0 = time.time()
    try:
        e_r, d_r = heron_sqd(recover=True, cisd=False)
        print(f"QPU+RECOVERY                  = {e_r:.6f} | gap_vs_FCI={(e_r-e_fci)*1000:+.2f} mHa | dets={d_r} | t={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"QPU+RECOVERY FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    # (c) CISD-seeded SQD — the robustness fix
    t0 = time.time()
    try:
        e_c, d_c = heron_sqd(recover=True, cisd=True)
        print(f"QPU+RECOVERY+CISD-SEED        = {e_c:.6f} | gap_vs_FCI={(e_c-e_fci)*1000:+.2f} mHa | dets={d_c} | t={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"QPU+CISD FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)
    print("QPUFCI_DONE", flush=True)


if __name__ == "__main__":
    main()
