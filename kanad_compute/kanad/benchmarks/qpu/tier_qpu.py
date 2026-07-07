"""Round 3 QPU — large real-QPU SQD on IBM Heron (156q), 30-50 qubits, distributed:
each system runs on IBM Heron (real QPU) and, where <=34q, a BlueQubit exact-statevector
SQD reference; <=CAS(14,14) also gets a CASCI reference. Quantifies real-hardware SQD
accuracy at scale + the spin_s singlet enforcement on noisy hardware.

Per-experiment isolated (one Heron job batch per run). Cluster-driven (submits to cloud, polls, diagonalizes):
    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_qpu <index> <mode>
  mode = heron | bluequbit | preflight
"""
from __future__ import annotations
import os, sys, time, traceback
import numpy as np

os.environ.setdefault('IBM_QUANTUM_TOKEN', 'kpAbwR0cleN0VLpj5fpbpEmsThZZtvS3T4L_rxrB7i2S')
os.environ.setdefault('IBM_QUANTUM_CRN', 'crn:v1:bluemix:public:quantum-computing:us-east:a/e8d9164b3ff748daa345dce1fcc0437e:fa184d28-2d7a-471b-8cc4-8986436c7439::')
os.environ.setdefault('BLUEQUBIT_API_KEY', 'm2WUitTn3XRpjFEZnKvfk05E14iW9SKD')

# (name, atoms, basis, frozen, active_orbs, n_active_e, spin, has_casci_ref)
# qubit count = 2*len(active_orbs)
EXPERIMENTS = [
    ('N2_CAS10-15_30q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,17)), 10, 0, False),
    ('CO_CAS10-15_30q',  [('C',(0,0,0)),('O',(0,0,1.128))], 'cc-pvdz', [0,1], list(range(2,17)), 10, 0, False),
    ('C2_CAS8-15_30q',   [('C',(0,0,0)),('C',(0,0,1.2425))], 'cc-pvdz', [0,1], list(range(2,17)), 8, 0, False),
    ('BeH2_CAS6-15_30q', [('Be',(0,0,0)),('H',(0,0,1.334)),('H',(0,0,-1.334))], 'cc-pvdz', [0], list(range(1,16)), 4, 0, False),
    ('H2O_CAS8-15_30q',  [('O',(0,0,0.117)),('H',(0,0.757,-0.469)),('H',(0,-0.757,-0.469))], 'cc-pvdz', [0], list(range(1,16)), 8, 0, False),
    ('N2_CAS10-16_32q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,18)), 10, 0, False),
    ('C2H4_CAS12-16_32q',[('C',(0,0,0.667)),('C',(0,0,-0.667)),('H',(0.92,0,1.23)),('H',(-0.92,0,1.23)),('H',(0.92,0,-1.23)),('H',(-0.92,0,-1.23))], 'cc-pvdz', [0,1], list(range(2,18)), 12, 0, False),
    ('N2_CAS10-17_34q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,19)), 10, 0, False),
    # ---- beyond BlueQubit exact (>34q): Heron only, no exact ref ----
    ('N2_CAS10-20_40q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,22)), 10, 0, False),
    ('CO_CAS10-20_40q',  [('C',(0,0,0)),('O',(0,0,1.128))], 'cc-pvdz', [0,1], list(range(2,22)), 10, 0, False),
    ('C2_CAS8-20_40q',   [('C',(0,0,0)),('C',(0,0,1.2425))], 'cc-pvdz', [0,1], list(range(2,22)), 8, 0, False),
    ('H2O_CAS8-22_44q',  [('O',(0,0,0.117)),('H',(0,0.757,-0.469)),('H',(0,-0.757,-0.469))], 'cc-pvdz', [0], list(range(1,23)), 8, 0, False),
    ('N2_CAS10-23_46q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,25)), 10, 0, False),
    ('N2_CAS10-25_50q',  [('N',(0,0,0)),('N',(0,0,1.10))], 'cc-pvdz', [0,1], list(range(2,27)), 10, 0, False),
]


def casci_ref(atoms, basis, n_act_orb, n_act_e):
    from pyscf import gto, scf, mcscf
    mol = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis=basis, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    cas = mcscf.CASCI(mf, n_act_orb, n_act_e); cas.fix_spin_(ss=0); cas.fcisolver.max_cycle = 200
    return float(cas.run(verbose=0).e_tot)


def build(exp, backend):
    from kanad import MolecularBuilder
    name, atoms, basis, frozen, active, ne, spin, _ = exp
    b = MolecularBuilder.from_atoms(atoms).basis(basis)
    if spin: b = b.spin(spin)
    b = b.active_space('manual', frozen=frozen, active=active).backend(backend)
    sk = dict(n_samples=(8000 if backend == 'ibm' else 50000),
              max_iterations=1, recovery_rounds=(1 if backend == 'ibm' else 2),
              random_seed=0, spin_s=(spin / 2.0))
    if backend == 'ibm':
        sk['ibm_backend_name'] = 'ibm_marrakesh'
    return b.solver('sqd', **sk).build()


def run(i, mode):
    exp = EXPERIMENTS[i]
    name, atoms, basis, frozen, active, ne, spin, _ = exp
    nq = 2 * len(active)
    if mode == 'preflight':
        qs = build(exp, 'statevector')
        return f"PREFLIGHT| {name} | built nq={qs.n_qubits} (target {nq}) {'OK' if qs.n_qubits == nq else 'MISMATCH'}"
    t0 = time.time()
    try:
        qs = build(exp, 'ibm' if mode == 'heron' else 'bluequbit')
        res = qs.solve(); o = qs.observables('core'); dt = time.time() - t0
        line = (f"QPU| {name} | {nq}q | {mode} | E={res['energy']:.6f} | "
                f"S2={res.get('s_squared')} dets={res.get('n_determinants')} "
                f"M={o.get('m_diagnostic')} t={dt:.0f}s")
        return line
    except Exception as e:
        return f"QPU| {name} | {nq}q | {mode} | CRASH {type(e).__name__}: {str(e)[:120]} | {traceback.format_exc().splitlines()[-2:]}"


N = len(EXPERIMENTS)
if __name__ == "__main__":
    i = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else 'heron'
    print(run(i, mode), flush=True)
