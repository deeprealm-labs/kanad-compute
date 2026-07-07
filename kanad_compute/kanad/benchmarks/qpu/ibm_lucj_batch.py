"""IBM batch with LUCJ ansatz + configuration recovery (M4-A + M4-D).

Submits H2 + LiH + NH3 cc-pVDZ as before, but with:
- **LUCJ ansatz**: depth O(n_orb), ~500× fewer 2q gates than Givens-SD
- **Configuration recovery**: bit-flip recovery for samples that fall out of
  (N, S_z) due to hardware noise

Expected vs the Givens-SD batch:
- Valid-sample fraction: 5% → 50%+
- Energy gap to CASCI: tighter (more determinants per shot)

Run: same env-var pattern as ibm_batch_sqd.py.
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
        'frozen': [], 'active': [0, 1, 2, 3, 4, 5],
    },
    'LiH': {
        'atom_str': 'Li 0 0 0; H 0 0 1.5957',
        'charge': 0,
        'frozen': [0], 'active': [1, 2, 3, 4, 5, 6],
    },
    'NH3': {
        'atom_str': ('N 0 0 0; '
                     'H 0 0.9377 0.3816; '
                     'H 0.8121 -0.4689 0.3816; '
                     'H -0.8121 -0.4689 0.3816'),
        'charge': 0,
        'frozen': [0], 'active': [1, 2, 3, 4, 5, 6, 7],
    },
}


def build_lucj_circuit(spec, seed=42, n_layers=1):
    from pyscf import gto, scf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz

    mol = gto.M(atom=spec['atom_str'], basis='cc-pvdz',
                charge=spec['charge'], spin=0, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=spec['frozen'], active=spec['active'])
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = mol.nelectron - 2 * len(spec['frozen'])

    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=n_active_e, n_layers=n_layers)
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(seed)
    n_params = qc.num_parameters
    params = rng.uniform(-0.3, 0.3, size=n_params)
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(params[i]) for i in range(n_params)}
    )
    return bound, ham, mol, mf, n_active_e


def submit_batch(circuits_dict, shots=4096, backend_name=None):
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
            field_name = list(pub_result.data.keys())[0]
            counts_per_circuit[name] = getattr(pub_result.data, field_name).get_counts()
    return counts_per_circuit


def bitstrings_from_counts(counts):
    out = []
    for bstr, n in counts.items():
        clean = bstr.replace(' ', '').replace('0x', '')
        val = int(clean, 2)
        out.extend([val] * int(n))
    return np.array(out, dtype=np.int64)


def run_sqd_on_samples(name, ham, mf, n_active_e, counts):
    """Two passes: (a) drop-filter (baseline), (b) drop+recovery (M4-D)."""
    from pyscf import mcscf
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _filter_by_n_sz, _filter_with_recovery,
    )

    bitstrings = bitstrings_from_counts(counts)
    n_orb = ham.n_orbitals
    n_e = ham.n_electrons

    # Resolve MO energies for the active subset
    mo_e = None
    try:
        active = list(ham.active_space.active_indices)
        mo_e = np.asarray(mf.mo_energy)[active]
    except Exception:
        pass

    # Pass A: drop-filter
    valid_drop = _filter_by_n_sz(bitstrings, n_orb, n_e, 0.0)
    dets_drop = sorted(set(int(d) for d in valid_drop))

    # Pass B: drop + recovery
    valid_rec, n_kept, n_recov, n_dropped = _filter_with_recovery(
        bitstrings, n_orb, n_e, 0.0, mo_e,
    )
    dets_rec = sorted(set(int(d) for d in valid_rec))

    spec = MOLECULES[name]
    cas = mcscf.CASCI(mf, ncas=len(spec['active']), nelecas=n_active_e).run(verbose=0)
    print(f'\n[{name}]  CASCI ref = {cas.e_tot:.8f} Ha')
    print(f'  Total shots: {len(bitstrings)}')

    # Diagonalize each subspace
    for label, dets, frac_str in [
        ('drop-only ', dets_drop, f'{len(valid_drop)/len(bitstrings)*100:.1f}%'),
        ('recovery  ', dets_rec, f'{len(valid_rec)/len(bitstrings)*100:.1f}%'),
    ]:
        solver = SamplingSQDSolver(ham, n_samples=len(bitstrings), random_seed=0)
        res = solver._diagonalize_in_subspace(dets) if dets else {'energy': float('nan')}
        gap_mha = (res['energy'] - cas.e_tot) * 1000
        print(f'  {label} valid={frac_str:>6}  N_det={len(dets):>5}  '
              f'E={res["energy"]:>14.8f}  gap={gap_mha:+.3f} mHa')
    print(f'  Recovery breakdown: kept={n_kept}, recovered={n_recov}, dropped={n_dropped}')

    return {'name': name, 'casci': cas.e_tot}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll', help='Existing IBM batch job ID')
    parser.add_argument('--shots', type=int, default=4096)
    parser.add_argument('--molecules', default='H2,LiH,NH3')
    parser.add_argument('--backend', default=None)
    parser.add_argument('--layers', type=int, default=1, help='LUCJ layers')
    args = parser.parse_args()

    if not os.environ.get('IBM_QUANTUM_TOKEN') or not os.environ.get('IBM_QUANTUM_CRN'):
        print('ERROR: IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN required'); sys.exit(1)

    chosen = [m.strip() for m in args.molecules.split(',')]
    print('=' * 80)
    print(f'IBM BATCH SamplingSQD with LUCJ + Configuration Recovery — {chosen}')
    print(f'LUCJ layers: {args.layers}')
    print('=' * 80)

    print('\nBuilding LUCJ cc-pVDZ ansatz circuits...')
    built = {}
    for name in chosen:
        circ, ham, mol, mf, n_active_e = build_lucj_circuit(
            MOLECULES[name], n_layers=args.layers,
        )
        built[name] = {
            'circuit': circ, 'ham': ham, 'mol': mol, 'mf': mf,
            'n_active_e': n_active_e, 'n_qubits': 2 * ham.n_orbitals,
        }
        nlocal = circ.num_nonlocal_gates()
        print(f'  {name}: {built[name]["n_qubits"]}q, {n_active_e} active e, '
              f'circuit 2q gates = {nlocal}')

    if args.poll:
        print(f'\nPolling batch {args.poll}...')
        counts_dict = poll_batch(args.poll, list(built.keys()))
        if counts_dict is None:
            print('Batch not yet complete; try later.'); sys.exit(0)
        for name, counts in counts_dict.items():
            run_sqd_on_samples(
                name, built[name]['ham'], built[name]['mf'],
                built[name]['n_active_e'], counts,
            )
        sys.exit(0)

    print(f'\nSubmitting batch ({args.shots} shots) to IBM...')
    t0 = time.time()
    circuits_dict = {name: data['circuit'] for name, data in built.items()}
    svc, backend, job, ordering = submit_batch(
        circuits_dict, shots=args.shots, backend_name=args.backend,
    )
    print(f'\nBatch submitted in {time.time()-t0:.1f}s')
    print(f'  Backend: {backend.name}   Job ID: {job.job_id()}   Status: {job.status()}')

    jobfile = f'/tmp/ibm_lucj_batch_{job.job_id()}.json'
    with open(jobfile, 'w') as f:
        json.dump({
            'molecules': ordering,
            'backend': backend.name,
            'job_id': job.job_id(),
            'shots': args.shots,
            'lucj_layers': args.layers,
            'submitted_at': time.time(),
        }, f, indent=2)
    print(f'  Persisted to {jobfile}')
    print(f'\nRe-run with `--poll {job.job_id()}` when complete.')


if __name__ == '__main__':
    main()
