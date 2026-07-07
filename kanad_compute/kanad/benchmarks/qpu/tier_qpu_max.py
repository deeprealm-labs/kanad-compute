"""Round 3 MAX-SCALE — push real-QPU SQD past the 77-qubit mark on IBM Heron (156q).

Large dilute active spaces (few electrons, many orbitals) in cc-pVTZ keep the sparse
Slater-Condon ENERGY path tractable (no full-FCI tensor) while the qubit count goes
60 -> 76 -> 100+. spin_s=None (sparse path), observables skipped (1-RDM would need the
full tensor). Reference: CCSD(T)/cc-pVTZ (dynamic-correlation gold standard) + HF bound.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_qpu_max <index> <mode>
  mode = heron | preflight | refs
"""
from __future__ import annotations
import os, sys, time, traceback
import numpy as np

os.environ.setdefault('IBM_QUANTUM_TOKEN', 'kpAbwR0cleN0VLpj5fpbpEmsThZZtvS3T4L_rxrB7i2S')
os.environ.setdefault('IBM_QUANTUM_CRN', 'crn:v1:bluemix:public:quantum-computing:us-east:a/e8d9164b3ff748daa345dce1fcc0437e:fa184d28-2d7a-471b-8cc4-8986436c7439::')

# (name, atom, basis, frozen, n_active_orb, n_active_e)  -> qubits = 2*n_active_orb
EXPERIMENTS = [
    ('N2_ccpvtz_CAS10-30_60q',  'N 0 0 0; N 0 0 1.10', 'cc-pvtz', [0, 1], 30, 10),
    ('N2_ccpvtz_CAS10-38_76q',  'N 0 0 0; N 0 0 1.10', 'cc-pvtz', [0, 1], 38, 10),  # beat IBM's 77q
    ('CO_ccpvtz_CAS10-38_76q',  'C 0 0 0; O 0 0 1.128', 'cc-pvtz', [0, 1], 38, 10),
    ('N2_ccpvtz_CAS10-45_90q',  'N 0 0 0; N 0 0 1.10', 'cc-pvtz', [0, 1], 45, 10),
    ('N2_ccpvtz_CAS10-50_100q', 'N 0 0 0; N 0 0 1.10', 'cc-pvtz', [0, 1], 50, 10),
]


def _active(frozen, n_orb):
    start = max(frozen) + 1 if frozen else 0
    return list(range(start, start + n_orb))


def refs(atom, basis, n_act_orb, n_act_e, frozen):
    """HF + CCSD(T) (gold standard) + small-CAS CASCI correlation anchor."""
    from pyscf import gto, scf, cc, mcscf
    mol = gto.M(atom=atom, basis=basis, verbose=0); mf = scf.RHF(mol).run(verbose=0)
    e_hf = float(mf.e_tot)
    out = {'hf': e_hf}
    try:
        mcc = cc.CCSD(mf, frozen=len(frozen)).run(verbose=0)
        et = mcc.ccsd_t()
        out['ccsd'] = float(mcc.e_tot); out['ccsd_t'] = float(mcc.e_tot + et)
    except Exception as e:
        out['ccsd_err'] = f"{type(e).__name__}:{str(e)[:50]}"
    return out


def build(exp, backend):
    from kanad import MolecularBuilder
    name, atom, basis, frozen, n_orb, ne = exp
    active = _active(frozen, n_orb)
    atoms = [(p.split()[0], tuple(map(float, p.split()[1:4]))) for p in atom.split(';')]
    b = (MolecularBuilder.from_atoms(atoms).basis(basis)
         .active_space('manual', frozen=frozen, active=active).backend(backend))
    sk = dict(n_samples=(10000 if backend == 'ibm' else 40000),
              max_iterations=1, recovery_rounds=1, random_seed=0)  # spin_s=None -> sparse path
    if backend == 'ibm':
        sk['ibm_backend_name'] = 'ibm_marrakesh'
    return b.solver('sqd', **sk).build()


def run(i, mode):
    exp = EXPERIMENTS[i]
    name, atom, basis, frozen, n_orb, ne = exp
    nq = 2 * n_orb
    if mode == 'preflight':
        # build + transpile-to-Heron depth check (no submission)
        from qiskit import transpile
        from qiskit_ibm_runtime import QiskitRuntimeService
        qs = build(exp, 'statevector')
        circ = qs._build_sampling_circuit(qs._resolve_ansatz_type('sqd'))
        svc = QiskitRuntimeService(channel='ibm_cloud', token=os.environ['IBM_QUANTUM_TOKEN'], instance=os.environ['IBM_QUANTUM_CRN'])
        bk = svc.backend('ibm_marrakesh')
        ct = transpile(circ, backend=bk, optimization_level=1)
        n2q = sum(1 for inst in ct.data if inst.operation.num_qubits == 2)
        return f"PREMAX| {name} | nq={qs.n_qubits}(target {nq}) | circ depth={circ.depth()} | Heron-transpiled depth={ct.depth()} 2q={n2q}"
    if mode == 'refs':
        r = refs(atom, basis, n_orb, ne, frozen)
        return f"REFMAX| {name} | {nq}q | HF={r['hf']:.6f} CCSD={r.get('ccsd')} CCSD(T)={r.get('ccsd_t')} {r.get('ccsd_err','')}"
    # heron real-QPU SQD (energy only; observables skipped — full-tensor 1-RDM intractable here)
    t0 = time.time()
    try:
        r = refs(atom, basis, n_orb, ne, frozen)
        qs = build(exp, 'ibm')
        res = qs.solve(); dt = time.time() - t0
        e = res['energy']; ccsdt = r.get('ccsd_t')
        gap = (e - ccsdt) * 1000 if ccsdt else None
        below_hf = e < r['hf']
        return (f"MAXQPU| {name} | {nq}q | HERON E={e:.6f} | HF={r['hf']:.6f} below_HF={below_hf} | "
                f"CCSD(T)={ccsdt} gap_vs_CCSDT={('%+.2f mHa' % gap) if gap is not None else 'n/a'} | "
                f"dets={res.get('n_determinants')} S2={res.get('s_squared')} t={dt:.0f}s")
    except Exception as e:
        return f"MAXQPU| {name} | {nq}q | CRASH {type(e).__name__}: {str(e)[:130]} | {traceback.format_exc().splitlines()[-2:]}"


N = len(EXPERIMENTS)
if __name__ == "__main__":
    i = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else 'heron'
    print(run(i, mode), flush=True)
