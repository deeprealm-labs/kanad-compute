"""Cloud SamplingSQD demonstration: NH3 cc-pVDZ on BlueQubit GPU + CH4 on IonQ simulator.

Submits each ansatz circuit to its respective cloud backend, collects samples,
then runs the local SamplingSQDSolver (Slater-Condon + sparse diagonalization)
to recover the ground-state energy. This is the IBM-style workflow: the
quantum hardware contributes ONLY the sampling distribution; everything
else is classical.

Backends used:
- BlueQubit ``device='gpu'``: NVIDIA-backed statevector simulator (~36 qubits).
- IonQ ``ionq_simulator``: noise-free quantum simulator (counts back, no real
  hardware billing). Real QPU usage would require ``ionq_qpu`` and significant
  shot budget — not appropriate for a demo.

Both submissions are ASYNC; job IDs are saved so the same script can be
re-run to poll for results.

Run:  PYTHONUNBUFFERED=1 python -m benchmarks.cloud_sqd_demo [--poll JOB_ID_FILE]
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

# These keys are passed via env only — not committed.
BLUEQUBIT_API_KEY = os.environ.get('BLUEQUBIT_API_KEY')
IONQ_API_KEY = os.environ.get('IONQ_API_KEY')


def build_polyatomic_ansatz(name: str, atom_str: str, charge: int,
                            frozen: list, active: list, n_layers: int = 1):
    """Build a kanad VQE Givens-SD ansatz circuit at cc-pVDZ active space.

    Uses small random parameters (no VQE optimization) so the circuit
    produces a HF-like state with weight on the dominant single/double
    excitations. That's enough for a SamplingSQD demo — the noisy hardware
    will spread weight further via decoherence, and SamplingSQD recovers
    the best determinants via N/Sz filtering + diagonalization.
    """
    from pyscf import gto, scf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import GivensSDAnsatz

    mol = gto.M(atom=atom_str, basis='cc-pvdz', charge=charge, spin=0,
                unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(frozen=frozen, active=active),
    )
    n_active_e = mol.nelectron - 2 * len(frozen)
    n_qubits = 2 * ham.n_orbitals

    ansatz = GivensSDAnsatz(
        n_qubits=n_qubits, n_electrons=n_active_e, n_layers=n_layers,
    )
    qc = ansatz.build_circuit()
    qiskit_circ = qc.to_qiskit() if hasattr(qc, 'to_qiskit') else qc

    # Small random parameters to spread weight off HF
    rng = np.random.default_rng(42)
    n_params = qiskit_circ.num_parameters
    params = rng.uniform(-0.1, 0.1, size=n_params)
    bound = qiskit_circ.assign_parameters(
        {qiskit_circ.parameters[i]: float(params[i]) for i in range(n_params)}
    )
    return {
        'name': name,
        'circuit': bound,
        'n_qubits': n_qubits,
        'n_active_electrons': n_active_e,
        'ham': ham,
        'mol': mol,
        'mf': mf,
    }


def submit_bluequbit(circuit, name: str, shots: int = 10000,
                     device: str = 'cpu'):
    """Submit a circuit to BlueQubit and return the job handle.

    Device options: 'cpu' (free), 'mps.cpu' (free), 'gpu' (paid),
    'mps.gpu' (paid), 'quantum' (paid, real hardware).
    """
    import bluequbit
    bq = bluequbit.init(BLUEQUBIT_API_KEY)
    job = bq.run(
        circuits=circuit,
        device=device,
        shots=shots,
        asynchronous=True,
        job_name=f'sqd_{name}',
    )
    return bq, job


def submit_ibm(circuit, name: str, shots: int = 4096,
               backend_name: Optional[str] = None):
    """Submit a circuit to IBM Quantum Cloud via Sampler primitive.

    IBM authentication uses the `IBM_QUANTUM_TOKEN` + `IBM_QUANTUM_CRN`
    environment variables. Default backend is the least-busy 156-qubit
    Heron r3 processor (same family as Robledo-Moreno 2025).
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit import transpile
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    crn = os.environ.get('IBM_QUANTUM_CRN')
    if not token or not crn:
        raise RuntimeError('IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN env vars required')
    service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=crn)
    if backend_name:
        backend = service.backend(backend_name)
    else:
        # Pick the least-busy Heron r3 (156-qubit)
        backends = service.backends(operational=True, simulator=False)
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    print(f'  Selected IBM backend: {backend.name} ({backend.num_qubits}q, '
          f'pending jobs: {backend.status().pending_jobs})')
    circ = circuit.copy()
    if circ.num_clbits == 0:
        circ.measure_all()
    # IBM requires transpilation to its native gateset + heavy-hex connectivity
    circ_native = transpile(circ, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run([circ_native], shots=shots)
    return backend, job


def submit_ionq(circuit, name: str, shots: int = 10000,
                use_qpu: bool = False):
    """Submit a circuit to IonQ.

    By default uses the free `ionq_simulator`. Real `ionq_qpu` requires
    `use_qpu=True` and costs real money — only flip for funded experiments.

    IonQ QIS backends accept only their native gateset (gpi/gpi2/ms/cx/rxx
    + universal subset). We transpile via Qiskit at optimization_level=1
    (the level IonQ recommends — higher levels over-resynthesize).
    """
    from qiskit_ionq import IonQProvider
    from qiskit import transpile
    provider = IonQProvider(IONQ_API_KEY)
    backend_name = 'ionq_qpu' if use_qpu else 'ionq_simulator'
    backend = provider.get_backend(backend_name)
    circ = circuit.copy()
    if circ.num_clbits == 0:
        circ.measure_all()
    # Transpile to IonQ native gateset
    circ_native = transpile(circ, backend=backend, optimization_level=1)
    job = backend.run(circ_native, shots=shots)
    return backend, job


def poll_bluequbit(bq, job, max_wait_s: int = 1800):
    """Poll BlueQubit until job completes."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            updated = bq.get(job.job_id)
            if updated.run_status in ('COMPLETED', 'FAILED', 'CANCELED'):
                return updated
        except Exception as e:
            print(f'BlueQubit poll error: {e}')
        time.sleep(5)
    raise TimeoutError(f'BlueQubit job {job.job_id} did not finish in {max_wait_s}s')


def counts_from_bluequbit_result(result):
    """Extract a dict[bitstring] -> count from a BlueQubit result."""
    # BlueQubit returns a dict {bitstring: count} or {bitstring: prob}
    counts = result.get_counts()
    if isinstance(counts, dict):
        # Check if these are probabilities or counts
        total = sum(counts.values())
        if abs(total - 1.0) < 1e-3:
            # Probabilities — convert to integer counts
            n_shots = 10000  # arbitrary; only ratios matter
            return {k: int(round(v * n_shots)) for k, v in counts.items()}
        return counts
    return counts


def counts_from_ionq_result(job):
    """Block on IonQ job and return measurement counts."""
    res = job.result()
    return res.get_counts()


def bitstrings_from_counts(counts: dict, n_qubits: int) -> np.ndarray:
    """Expand counts dict to a flat array of integers (one per shot)."""
    out = []
    for bstr, n in counts.items():
        # IonQ/BlueQubit may include spaces or leading '0x'
        clean = bstr.replace(' ', '').replace('0x', '')
        # Parse as binary; little-endian (Qiskit convention)
        val = int(clean, 2)
        out.extend([val] * int(n))
    return np.array(out, dtype=np.int64)


def run_sqd_on_samples(spec: dict, bitstrings: np.ndarray,
                       backend_label: str) -> dict:
    """Run local SamplingSQD on hardware/cloud samples."""
    from kanad.solvers.sampling_sqd import (
        _filter_by_n_sz, _h_diag, _diff_spin_orbitals,
    )
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import eigsh

    ham = spec['ham']
    n_orb = ham.n_orbitals
    n_e = ham.n_electrons
    n_qubits = spec['n_qubits']

    # Filter
    valid = _filter_by_n_sz(bitstrings, n_orb, n_e, target_sz=0.0)
    valid_fraction = len(valid) / max(len(bitstrings), 1)
    if len(valid) == 0:
        return {
            'energy': float('nan'),
            'n_determinants': 0,
            'valid_fraction': 0.0,
            'backend': backend_label,
        }

    determinants = sorted(set(int(d) for d in valid))
    n_det = len(determinants)

    # Build CI matrix
    h1, h2 = ham.h_core, ham.eri
    nuc = float(ham.nuclear_repulsion)
    H = lil_matrix((n_det, n_det), dtype=float)
    for i, di in enumerate(determinants):
        H[i, i] = _h_diag(di, h1, h2, n_qubits) + nuc
        for j in range(i + 1, n_det):
            dj = determinants[j]
            diff = _diff_spin_orbitals(di, dj)
            if diff is None or diff[0] == 0:
                continue
            # Reuse the same logic as in SamplingSQDSolver.solve
            val = 0.0
            sign = diff[-1]
            if diff[0] == 1:
                p, q = diff[1]
                if (p % 2) != (q % 2):
                    continue
                p_sp, q_sp = p // 2, q // 2
                val = h1[p_sp, q_sp] * sign
                common = di & dj
                for r_q in range(n_qubits):
                    if (common >> r_q) & 1:
                        r_sp = r_q // 2
                        val += h2[p_sp, q_sp, r_sp, r_sp] * sign
                        if (r_q % 2) == (p % 2):
                            val -= h2[p_sp, r_sp, r_sp, q_sp] * sign
            elif diff[0] == 2:
                p, q, r, s = diff[1]
                if (p % 2) != (q % 2) or (r % 2) != (s % 2):
                    continue
                p_sp, q_sp = p // 2, q // 2
                r_sp, s_sp = r // 2, s // 2
                val = h2[p_sp, q_sp, r_sp, s_sp]
                if (p % 2) == (r % 2):
                    val -= h2[p_sp, s_sp, r_sp, q_sp]
                val *= sign
            H[i, j] = val
            H[j, i] = val
    H = H.tocsr()

    if n_det <= 200:
        evals = np.linalg.eigvalsh(H.toarray())
    else:
        evals = eigsh(H, k=1, which='SA', return_eigenvectors=False)
    e_ground = float(min(evals))

    return {
        'energy': e_ground,
        'n_determinants': n_det,
        'n_samples_total': int(len(bitstrings)),
        'n_valid_samples': int(len(valid)),
        'valid_fraction': float(valid_fraction),
        'backend': backend_label,
    }


# ---------------------------------------------------------------------------
# Molecule specs at cc-pVDZ active space
# ---------------------------------------------------------------------------

NH3_SPEC = {
    'name': 'NH3',
    'atom_str': ('N 0 0 0; '
                 'H 0 0.9377 0.3816; '
                 'H 0.8121 -0.4689 0.3816; '
                 'H -0.8121 -0.4689 0.3816'),
    'charge': 0,
    'frozen': [0], 'active': [1, 2, 3, 4, 5, 6, 7],  # (10e, 7o) → 14 qubits
}
CH4_SPEC = {
    'name': 'CH4',
    'atom_str': ('C 0 0 0; '
                 'H 0.6276 0.6276 0.6276; '
                 'H 0.6276 -0.6276 -0.6276; '
                 'H -0.6276 0.6276 -0.6276; '
                 'H -0.6276 -0.6276 0.6276'),
    'charge': 0,
    'frozen': [0], 'active': [1, 2, 3, 4, 5, 6, 7, 8],  # (8e, 8o) → 16 qubits
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll', help='JSON file with previously submitted job IDs')
    parser.add_argument('--shots', type=int, default=10000)
    args = parser.parse_args()

    print('=' * 80)
    print('CLOUD SAMPLING-SQD DEMO — NH3 (BlueQubit GPU) + CH4 (IonQ sim)')
    print('=' * 80)

    if not BLUEQUBIT_API_KEY:
        print('ERROR: BLUEQUBIT_API_KEY environment variable not set.')
        sys.exit(1)
    if not IONQ_API_KEY:
        print('ERROR: IONQ_API_KEY environment variable not set.')
        sys.exit(1)

    if args.poll:
        with open(args.poll) as f:
            jobs = json.load(f)
        print(f'Polling jobs from {args.poll}')
    else:
        jobs = {}

    # Build circuits + submit
    for spec_dict in (NH3_SPEC, CH4_SPEC):
        name = spec_dict['name']
        if name in jobs:
            print(f'\n{name}: previously submitted — skipping resubmit.')
            continue
        print(f'\n[{name}] Building cc-pVDZ ansatz circuit...')
        t0 = time.time()
        spec = build_polyatomic_ansatz(**spec_dict)
        print(f'  Built {spec["n_qubits"]}-qubit circuit, {spec["n_active_electrons"]} active e ({time.time()-t0:.1f}s)')

        if name == 'NH3':
            print(f'  Submitting to BlueQubit CPU async ({args.shots} shots)...')
            bq, job = submit_bluequbit(spec['circuit'], name,
                                       shots=args.shots, device='cpu')
            print(f'  → BlueQubit job ID: {job.job_id}')
            jobs[name] = {
                'backend': 'bluequbit_cpu',
                'job_id': job.job_id,
                'n_qubits': spec['n_qubits'],
            }
        else:  # CH4 → IonQ simulator
            print(f'  Submitting to IonQ simulator ({args.shots} shots)...')
            backend, job = submit_ionq(spec['circuit'], name, shots=args.shots)
            print(f'  → IonQ job ID: {job.job_id()}')
            jobs[name] = {
                'backend': 'ionq_simulator',
                'job_id': job.job_id(),
                'n_qubits': spec['n_qubits'],
            }

    # Persist job IDs
    job_file = '/tmp/cloud_sqd_jobs.json'
    with open(job_file, 'w') as f:
        json.dump(jobs, f, indent=2)
    print(f'\nJob IDs persisted to {job_file}')
    print('Re-run with `--poll /tmp/cloud_sqd_jobs.json` to fetch results when ready.')

    # Try to poll immediately for results if non-async
    print()
    print('Polling for completion (BlueQubit usually < 60 s, IonQ sim usually < 5 min)...')
    for name, info in jobs.items():
        print(f'\n[{name}] Fetching {info["backend"]} job {info["job_id"]}...')
        t0 = time.time()
        try:
            if info['backend'].startswith('bluequbit'):
                import bluequbit
                bq = bluequbit.init(BLUEQUBIT_API_KEY)
                job_obj = bq.get(info['job_id'])
                # Wait for completion
                while job_obj.run_status not in ('COMPLETED', 'FAILED', 'CANCELED'):
                    time.sleep(3)
                    job_obj = bq.get(info['job_id'])
                if job_obj.run_status != 'COMPLETED':
                    print(f'  Status: {job_obj.run_status}; skipping.')
                    continue
                counts = job_obj.get_counts()
                print(f'  BlueQubit result: {len(counts)} unique bitstrings, status={job_obj.run_status}')
            else:  # ionq
                from qiskit_ionq import IonQProvider
                p = IonQProvider(IONQ_API_KEY)
                backend = p.get_backend('ionq_simulator')
                job_obj = backend.retrieve_job(info['job_id'])
                # Block on result
                res = job_obj.result()
                counts = res.get_counts()
                print(f'  IonQ result: {len(counts)} unique bitstrings')
        except Exception as e:
            print(f'  Fetch failed: {type(e).__name__}: {e}')
            continue
        elapsed = time.time() - t0
        print(f'  ({elapsed:.1f} s end-to-end)')

        # Run SamplingSQD locally on the cloud samples
        spec_dict = NH3_SPEC if name == 'NH3' else CH4_SPEC
        spec = build_polyatomic_ansatz(**spec_dict)
        bitstrings = bitstrings_from_counts(counts, spec['n_qubits'])
        sqd_result = run_sqd_on_samples(spec, bitstrings, info['backend'])
        print(f'  Local SamplingSQD on {info["backend"]} counts:')
        print(f'    energy            = {sqd_result["energy"]:.8f} Ha')
        print(f'    determinants      = {sqd_result["n_determinants"]}')
        print(f'    valid sample fr.  = {100*sqd_result["valid_fraction"]:.1f}%')

        # Reference: classical CASCI in same active space
        try:
            from pyscf import mcscf
            cas = mcscf.CASCI(spec['mf'], ncas=len(spec_dict['active']),
                              nelecas=spec['n_active_electrons']).run(verbose=0)
            print(f'    PySCF CASCI ref   = {cas.e_tot:.8f} Ha  (gap = {(sqd_result["energy"]-cas.e_tot)*1000:+.3f} mHa)')
        except Exception:
            pass


if __name__ == '__main__':
    main()
