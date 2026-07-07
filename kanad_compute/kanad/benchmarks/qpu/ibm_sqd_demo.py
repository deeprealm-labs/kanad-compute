"""IBM Quantum SamplingSQD demonstration — submits NH3 cc-pVDZ to a 156-qubit
Heron r3 processor (the same family Robledo-Moreno 2025 used) and persists
the job ID for later retrieval.

IBM hardware jobs queue from minutes to hours; this script submits async and
exits. Re-run with `--poll JOB_ID` to fetch results once the job completes.

Run:
  export IBM_QUANTUM_TOKEN=...
  export IBM_QUANTUM_CRN='crn:v1:bluemix:public:quantum-computing:...'
  python -m benchmarks.ibm_sqd_demo                       # submit
  python -m benchmarks.ibm_sqd_demo --poll <JOB_ID>        # fetch when ready
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np


def submit_ibm(circuit, shots: int = 4096, backend_name: str = None):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import transpile
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    if backend_name:
        backend = svc.backend(backend_name)
    else:
        backends = svc.backends(operational=True, simulator=False)
        backend = min(backends, key=lambda b: b.status().pending_jobs)

    circ = circuit.copy()
    if circ.num_clbits == 0:
        circ.measure_all()
    circ_native = transpile(circ, backend=backend, optimization_level=1)

    sampler = Sampler(mode=backend)
    job = sampler.run([circ_native], shots=shots)
    return svc, backend, job


def poll_ibm(job_id: str):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    job = svc.job(job_id)
    print(f'IBM job {job_id} status: {job.status()}')
    if str(job.status()) != 'DONE':
        return None
    res = job.result()
    # SamplerV2 result format: result[0].data.<clreg_name>
    pub_result = res[0]
    counts = pub_result.data.meas.get_counts() if hasattr(pub_result.data, 'meas') \
        else list(pub_result.data.values())[0].get_counts()
    return counts


def bitstrings_from_counts(counts: dict) -> np.ndarray:
    out = []
    for bstr, n in counts.items():
        clean = bstr.replace(' ', '').replace('0x', '')
        val = int(clean, 2)
        out.extend([val] * int(n))
    return np.array(out, dtype=np.int64)


def run_local_sqd(counts: dict, name: str = 'NH3'):
    """Run SamplingSQD locally on the IBM hardware samples."""
    from pyscf import gto, scf, mcscf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.solvers.sampling_sqd import SamplingSQDSolver

    if name == 'NH3':
        mol = gto.M(
            atom='N 0 0 0; H 0 0.9377 0.3816; H 0.8121 -0.4689 0.3816; H -0.8121 -0.4689 0.3816',
            basis='cc-pvdz', verbose=0,
        )
        frozen, active = [0], [1, 2, 3, 4, 5, 6, 7]
    elif name == 'LiH':
        mol = gto.M(atom='Li 0 0 0; H 0 0 1.5957', basis='cc-pvdz', verbose=0)
        frozen, active = [0], [1, 2, 3, 4, 5, 6]
    elif name == 'H2':
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz', verbose=0)
        frozen, active = [], [0, 1, 2, 3, 4, 5]
    else:
        raise ValueError(f'Unknown molecule {name}')

    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=frozen, active=active)
    )
    n_active_e = mol.nelectron - 2 * len(frozen)

    bitstrings = bitstrings_from_counts(counts)
    print(f'Total samples: {len(bitstrings)}')

    # SamplingSQD using the same code path as cloud_sqd_demo
    solver = SamplingSQDSolver(ham, n_samples=len(bitstrings), random_seed=0)
    # Override the sampling step by injecting bitstrings directly
    # via the internal pipeline (we don't actually re-sample)
    from kanad.solvers.sampling_sqd import (
        _filter_by_n_sz, _h_diag, _diff_spin_orbitals,
    )
    valid = _filter_by_n_sz(bitstrings, ham.n_orbitals, ham.n_electrons, 0.0)
    print(f'Valid (N, Sz) fraction: {len(valid)/len(bitstrings)*100:.1f}%')
    determinants = sorted(set(int(d) for d in valid))
    print(f'Unique valid determinants: {len(determinants)}')

    result = solver._diagonalize_in_subspace(determinants)
    e_sqd = result['energy']

    cas = mcscf.CASCI(mf, ncas=len(active), nelecas=n_active_e).run(verbose=0)
    gap_mha = (e_sqd - cas.e_tot) * 1000
    print(f'IBM SamplingSQD energy = {e_sqd:.8f} Ha')
    print(f'PySCF CASCI reference  = {cas.e_tot:.8f} Ha')
    print(f'Gap to CASCI           = {gap_mha:+.3f} mHa')
    return {'energy': e_sqd, 'casci_ref': cas.e_tot, 'gap_mha': gap_mha,
            'n_determinants': len(determinants), 'valid_fraction': len(valid)/len(bitstrings)}


def build_nh3_circuit():
    from pyscf import gto, scf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import GivensSDAnsatz
    mol = gto.M(
        atom='N 0 0 0; H 0 0.9377 0.3816; H 0.8121 -0.4689 0.3816; H -0.8121 -0.4689 0.3816',
        basis='cc-pvdz', verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=[0], active=[1,2,3,4,5,6,7])
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = mol.nelectron - 2
    ansatz = GivensSDAnsatz(n_qubits=n_qubits, n_electrons=n_active_e, n_layers=1)
    qc = ansatz.build_circuit()
    qiskit_circ = qc.to_qiskit() if hasattr(qc, 'to_qiskit') else qc
    rng = np.random.default_rng(42)
    n_params = qiskit_circ.num_parameters
    params = rng.uniform(-0.1, 0.1, size=n_params)
    bound = qiskit_circ.assign_parameters(
        {qiskit_circ.parameters[i]: float(params[i]) for i in range(n_params)}
    )
    return bound, n_qubits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll', help='Existing IBM job ID to fetch results from')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--molecule', default='NH3', choices=['NH3', 'LiH', 'H2'])
    parser.add_argument('--backend', default=None,
                        help='IBM backend name (default: least-busy 156q Heron)')
    args = parser.parse_args()

    if not os.environ.get('IBM_QUANTUM_TOKEN'):
        print('ERROR: IBM_QUANTUM_TOKEN env var required'); sys.exit(1)
    if not os.environ.get('IBM_QUANTUM_CRN'):
        print('ERROR: IBM_QUANTUM_CRN env var required'); sys.exit(1)

    print('=' * 80)
    print(f'IBM QUANTUM SAMPLING-SQD — {args.molecule} cc-pVDZ')
    print('=' * 80)

    if args.poll:
        print(f'\nPolling job {args.poll}...')
        counts = poll_ibm(args.poll)
        if counts is None:
            print('Job not yet complete; try again later.')
            sys.exit(0)
        print(f'Job complete — {len(counts)} unique bitstrings')
        result = run_local_sqd(counts, args.molecule)
        sys.exit(0)

    # Build + submit
    print(f'\nBuilding {args.molecule} cc-pVDZ ansatz circuit...')
    if args.molecule == 'NH3':
        circuit, n_qubits = build_nh3_circuit()
    else:
        raise NotImplementedError(f'Builder for {args.molecule} not yet wired in.')
    print(f'  {n_qubits}-qubit circuit ready.')

    print(f'\nSubmitting to IBM Quantum ({args.shots} shots)...')
    t0 = time.time()
    svc, backend, job = submit_ibm(circuit, shots=args.shots,
                                   backend_name=args.backend)
    submit_time = time.time() - t0
    print(f'  Job submitted to {backend.name} in {submit_time:.1f}s')
    print(f'  Job ID: {job.job_id()}')
    print(f'  Status: {job.status()}')

    # Persist
    jobfile = f'/tmp/ibm_sqd_{args.molecule}_{job.job_id()}.json'
    with open(jobfile, 'w') as f:
        json.dump({
            'molecule': args.molecule,
            'backend': backend.name,
            'job_id': job.job_id(),
            'shots': args.shots,
            'n_qubits': n_qubits,
            'submitted_at': time.time(),
        }, f, indent=2)
    print(f'  Persisted to {jobfile}')
    print(f'\nRe-run with `--poll {job.job_id()}` to fetch results when complete.')


if __name__ == '__main__':
    main()
