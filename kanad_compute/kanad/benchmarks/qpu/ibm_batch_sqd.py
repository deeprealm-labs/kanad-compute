"""IBM Quantum batch-mode SamplingSQD.

`Batch` mode submits multiple circuits as one reservation: the QPU runs them
back-to-back without re-queueing. Much faster end-to-end than separate
single-circuit jobs when you have multiple molecules / multiple parameter
sweeps to characterise.

This script batches H2 + LiH + NH3 cc-pVDZ sampling circuits (4-, 12-, and
14-qubit) into one `Batch` on a 156-qubit Heron r3 — total 30 qubits used,
roughly 7-15 minutes for the batch on a moderate-load day.

Run:
  export IBM_QUANTUM_TOKEN=...
  export IBM_QUANTUM_CRN=...
  python -m benchmarks.ibm_batch_sqd                          # submit
  python -m benchmarks.ibm_batch_sqd --poll <JOB_ID>          # fetch
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np


MOLECULES = {
    'H2': {
        'atom_str': 'H 0 0 0; H 0 0 0.74',
        'charge': 0,
        'frozen': [], 'active': [0, 1, 2, 3, 4, 5],   # CAS(2, 6) → 12 qubits
    },
    'LiH': {
        'atom_str': 'Li 0 0 0; H 0 0 1.5957',
        'charge': 0,
        'frozen': [0], 'active': [1, 2, 3, 4, 5, 6],   # CAS(2, 6) → 12 qubits
    },
    'NH3': {
        'atom_str': ('N 0 0 0; '
                     'H 0 0.9377 0.3816; '
                     'H 0.8121 -0.4689 0.3816; '
                     'H -0.8121 -0.4689 0.3816'),
        'charge': 0,
        'frozen': [0], 'active': [1, 2, 3, 4, 5, 6, 7],   # CAS(10, 7) → 14 qubits
    },
}


def build_ansatz_circuit(spec, seed=42, n_layers=1):
    from pyscf import gto, scf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import GivensSDAnsatz
    mol = gto.M(atom=spec['atom_str'], basis='cc-pvdz',
                charge=spec['charge'], spin=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=spec['frozen'], active=spec['active'])
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = mol.nelectron - 2 * len(spec['frozen'])
    ansatz = GivensSDAnsatz(n_qubits=n_qubits, n_electrons=n_active_e, n_layers=n_layers)
    qc = ansatz.build_circuit()
    qiskit_circ = qc.to_qiskit() if hasattr(qc, 'to_qiskit') else qc
    rng = np.random.default_rng(seed)
    n_params = qiskit_circ.num_parameters
    params = rng.uniform(-0.1, 0.1, size=n_params)
    bound = qiskit_circ.assign_parameters(
        {qiskit_circ.parameters[i]: float(params[i]) for i in range(n_params)}
    )
    return bound, ham, mol, mf, n_active_e


def submit_batch(circuits_dict, shots=4096, backend_name=None):
    """Submit all circuits in one IBM Batch — back-to-back execution.

    Returns (svc, backend, job, ordering) where ordering is the list of
    circuit-names matching the per-pub result order.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
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
    print(f'IBM backend: {backend.name} ({backend.num_qubits}q, '
          f'pending jobs: {backend.status().pending_jobs})')

    ordering = list(circuits_dict.keys())
    transpiled = []
    for name in ordering:
        c = circuits_dict[name].copy()
        if c.num_clbits == 0:
            c.measure_all()
        ct = transpile(c, backend=backend, optimization_level=1)
        transpiled.append(ct)
        print(f'  {name}: transpiled depth = {ct.depth()}, '
              f'2q ops = {sum(1 for inst in ct.data if inst.operation.num_qubits == 2)}')

    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        job = sampler.run(transpiled, shots=shots)
    return svc, backend, job, ordering


def poll_batch(job_id, ordering):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    job = svc.job(job_id)
    print(f'IBM batch {job_id} status: {job.status()}')
    if str(job.status()) != 'DONE':
        return None
    result = job.result()
    counts_per_circuit = {}
    for idx, name in enumerate(ordering):
        pub_result = result[idx]
        if hasattr(pub_result.data, 'meas'):
            counts_per_circuit[name] = pub_result.data.meas.get_counts()
        else:
            # First data field
            field_name = list(pub_result.data.keys())[0]
            counts_per_circuit[name] = getattr(pub_result.data, field_name).get_counts()
    return counts_per_circuit


def bitstrings_from_counts(counts: dict) -> np.ndarray:
    out = []
    for bstr, n in counts.items():
        clean = bstr.replace(' ', '').replace('0x', '')
        val = int(clean, 2)
        out.extend([val] * int(n))
    return np.array(out, dtype=np.int64)


def run_sqd_on_samples(name, ham, mol, mf, n_active_e, counts):
    """Run SamplingSQD locally on hardware counts."""
    from pyscf import mcscf
    from kanad.solvers.sampling_sqd import SamplingSQDSolver, _filter_by_n_sz

    bitstrings = bitstrings_from_counts(counts)
    valid = _filter_by_n_sz(bitstrings, ham.n_orbitals, ham.n_electrons, 0.0)
    valid_frac = len(valid) / max(len(bitstrings), 1)
    determinants = sorted(set(int(d) for d in valid))

    solver = SamplingSQDSolver(ham, n_samples=len(bitstrings), random_seed=0)
    res = solver._diagonalize_in_subspace(determinants)

    spec = MOLECULES[name]
    cas = mcscf.CASCI(mf, ncas=len(spec['active']), nelecas=n_active_e).run(verbose=0)
    gap_mha = (res['energy'] - cas.e_tot) * 1000

    print(f'  Total samples: {len(bitstrings)}')
    print(f'  Valid (N, Sz): {len(valid)} ({100*valid_frac:.1f}%)')
    print(f'  Unique dets:   {len(determinants)}')
    print(f'  Energy (IBM):  {res["energy"]:.8f} Ha')
    print(f'  CASCI ref:     {cas.e_tot:.8f} Ha')
    print(f'  Gap:           {gap_mha:+.3f} mHa')
    return {
        'name': name, 'energy': res['energy'], 'casci': cas.e_tot,
        'gap_mha': gap_mha, 'n_dets': len(determinants),
        'valid_fraction': valid_frac,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll', help='Existing IBM batch job ID')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--molecules', default='H2,LiH,NH3',
                        help='Comma-separated subset')
    parser.add_argument('--backend', default=None)
    args = parser.parse_args()

    if not os.environ.get('IBM_QUANTUM_TOKEN'):
        print('ERROR: IBM_QUANTUM_TOKEN required'); sys.exit(1)
    if not os.environ.get('IBM_QUANTUM_CRN'):
        print('ERROR: IBM_QUANTUM_CRN required'); sys.exit(1)

    chosen = [m.strip() for m in args.molecules.split(',')]

    print('=' * 80)
    print(f'IBM QUANTUM BATCH SAMPLING-SQD — {chosen}')
    print('=' * 80)

    # Build all spec dicts (always — for poll-mode we still need ham, mol, mf)
    print()
    print('Building cc-pVDZ ansatz circuits...')
    built = {}
    for name in chosen:
        if name not in MOLECULES:
            print(f'  Unknown molecule {name}, skipping')
            continue
        circ, ham, mol, mf, n_active_e = build_ansatz_circuit(MOLECULES[name])
        built[name] = {
            'circuit': circ, 'ham': ham, 'mol': mol, 'mf': mf,
            'n_active_e': n_active_e, 'n_qubits': 2 * ham.n_orbitals,
        }
        print(f'  {name}: {built[name]["n_qubits"]} qubits, {n_active_e} active e')

    if args.poll:
        print(f'\nPolling batch {args.poll}...')
        counts_dict = poll_batch(args.poll, list(built.keys()))
        if counts_dict is None:
            print('Batch not yet complete; try later.')
            sys.exit(0)
        print()
        results = []
        for name, counts in counts_dict.items():
            print(f'\n[{name}]')
            r = run_sqd_on_samples(
                name, built[name]['ham'], built[name]['mol'],
                built[name]['mf'], built[name]['n_active_e'], counts,
            )
            results.append(r)
        print()
        print('=' * 80)
        print('IBM Batch SamplingSQD summary')
        print('=' * 80)
        print(f"  {'Molecule':10s}  {'Energy (Ha)':>14}  {'CASCI ref':>14}  {'Gap (mHa)':>12}  {'N_det':>6}")
        for r in results:
            print(f"  {r['name']:10s}  {r['energy']:>14.8f}  {r['casci']:>14.8f}  {r['gap_mha']:>+11.3f}  {r['n_dets']:>6d}")
        sys.exit(0)

    # Submit batch
    print(f'\nSubmitting batch ({args.shots} shots per circuit) to IBM...')
    t0 = time.time()
    circuits_dict = {name: data['circuit'] for name, data in built.items()}
    svc, backend, job, ordering = submit_batch(
        circuits_dict, shots=args.shots, backend_name=args.backend
    )
    submit_time = time.time() - t0
    print(f'\nBatch submitted in {submit_time:.1f}s')
    print(f'  Backend:  {backend.name}')
    print(f'  Job ID:   {job.job_id()}')
    print(f'  Status:   {job.status()}')
    print(f'  Ordering: {ordering}')

    jobfile = f'/tmp/ibm_batch_{job.job_id()}.json'
    with open(jobfile, 'w') as f:
        json.dump({
            'molecules': ordering,
            'backend': backend.name,
            'job_id': job.job_id(),
            'shots': args.shots,
            'submitted_at': time.time(),
        }, f, indent=2)
    print(f'  Persisted to {jobfile}')
    print(f'\nRe-run with `--poll {job.job_id()}` to fetch when complete.')


if __name__ == '__main__':
    main()
