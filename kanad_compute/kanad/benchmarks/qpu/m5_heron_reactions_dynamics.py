"""M5 on real IBM Heron — reactions + dynamics at SQD scale.

Extends the M5 local-statevector verification to REAL hardware on larger
molecules. Key efficiency: the geometries within one task are independent,
so they batch into a SINGLE Heron allocation:

  - **Reaction profile**: all scan points (reactant → TS → product, or a
    dissociation curve) are independent → one Heron batch of N circuits.
  - **Force at one geometry**: the 6·N_atoms+1 displaced geometries are
    independent → one Heron batch.

MD is inherently sequential (each step's force depends on the previous
position), so a K-step trajectory is K sequential Heron batches.

Default system: N₂(10,10)/cc-pVDZ → 20 qubits (the M11a setup) — a real
bond-breaking "reaction" at SQD scale, far larger than the toy H₃/STO-3G.

Run:
  # Reaction: N2 dissociation profile on Heron (one batch)
  python -m benchmarks.m5_heron_reactions_dynamics --reaction --submit
  python -m benchmarks.m5_heron_reactions_dynamics --reaction --poll <jobid>

  # Force at one geometry on Heron (one batch)
  python -m benchmarks.m5_heron_reactions_dynamics --force --submit
  python -m benchmarks.m5_heron_reactions_dynamics --force --poll <jobid>

Env: IBM_QUANTUM_TOKEN, IBM_QUANTUM_CRN
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np

ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
HA_TO_KCAL = 627.509


def build_n2(r_ang):
    from pyscf import gto
    return gto.M(atom=f'N 0 0 0; N 0 0 {r_ang:.6f}', basis='cc-pvdz',
                 spin=0, charge=0, verbose=0)


def n2_ham_and_circuit(r_ang, seed=42):
    """Build N2(10,10) active-space ham + LUCJ circuit at bond length r."""
    from pyscf import scf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )
    from kanad.core.ansatze import LUCJAnsatz
    mol = build_n2(r_ang)
    mf = scf.RHF(mol).run(verbose=0)
    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(
            frozen=[0, 1], active=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    )
    n_qubits = 2 * ham.n_orbitals
    ansatz = LUCJAnsatz(n_qubits=n_qubits, n_electrons=ham.n_electrons,
                        n_layers=1, target_sz=0.0)
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.3, 0.3, size=qc.num_parameters)
    bound = qc.assign_parameters({qc.parameters[i]: float(params[i])
                                  for i in range(qc.num_parameters)})
    if bound.num_clbits == 0:
        bound.measure_all()
    return ham, mf, bound


def submit_batch(circuits, shots, backend_name=None, anchor_job=None):
    """Submit a list of circuits as ONE Heron batch allocation.

    IBM Cloud's backend-discovery endpoint (``svc.backend(name)`` /
    ``svc.backends()`` / ``least_busy``) intermittently hangs for minutes
    on this instance, while job operations stay responsive. When
    ``anchor_job`` (a known prior job id) is given we derive the backend
    object from ``svc.job(anchor).backend()``, which resolves instantly and
    bypasses discovery entirely.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    if anchor_job:
        backend = svc.job(anchor_job).backend()
    elif backend_name:
        backend = svc.backend(backend_name)
    else:
        backends = svc.backends(operational=True, simulator=False,
                                 min_num_qubits=circuits[0].num_qubits)
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    print(f'  Backend: {backend.name} ({backend.num_qubits}q); '
          f'{len(circuits)} circuits', flush=True)
    transpiled = [transpile(c, backend=backend, optimization_level=1)
                  for c in circuits]
    max_2q = max(sum(1 for inst in ct.data if inst.operation.num_qubits == 2)
                 for ct in transpiled)
    print(f'  Max 2q gates across circuits: {max_2q}', flush=True)
    # All circuits go in ONE multi-PUB job (job mode). This is already a
    # single Heron allocation and posts to /api/v1/jobs — avoiding the
    # /api/v1/sessions endpoint that a Batch() context requires, which has
    # been returning HTTP 520s during IBM Cloud degradations.
    sampler = SamplerV2(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    return svc, backend, job


def poll_counts(job_id, n_circuits):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    job = svc.job(job_id)
    if str(job.status()) != 'DONE':
        return None
    result = job.result()
    counts_list = []
    for idx in range(n_circuits):
        pub = result[idx]
        if hasattr(pub.data, 'meas'):
            counts_list.append(pub.data.meas.get_counts())
        else:
            field = list(pub.data.keys())[0]
            counts_list.append(getattr(pub.data, field).get_counts())
    return counts_list


def bits_from_counts(counts):
    out = []
    for bs, n in counts.items():
        out.extend([int(str(bs).replace(' ', '').replace('0x', ''), 2)] * int(n))
    return np.array(out, dtype=np.int64)


def sqd_energy_from_counts(ham, mf, counts, max_iter=4, expansion=50):
    """Post-process one Heron count set → SQD energy (recovery + expansion)."""
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver, _filter_with_recovery, _generate_singles_doubles,
    )
    bits = bits_from_counts(counts)
    solver = SamplingSQDSolver(ham, n_samples=len(bits), backend='statevector',
                                recover_configurations=True, ci_backend='pyscf',
                                target_sz=0.0, random_seed=0)
    mo_e = solver._resolve_mo_energies()
    valid, *_ = _filter_with_recovery(bits, ham.n_orbitals, ham.n_electrons, 0.0, mo_e)
    dets = sorted(set(int(d) for d in valid))
    n_qubits = 2 * ham.n_orbitals
    last = None
    for it in range(max_iter):
        res = solver._diagonalize_in_subspace_pyscf(dets)
        if last is not None and abs(res['energy'] - last) < 5e-6:
            break
        last = res['energy']
        evec = res['eigenvector']
        top = np.argsort(np.abs(evec) ** 2)[::-1][:min(expansion, len(dets))]
        new = set()
        for i in top:
            new.update(_generate_singles_doubles(dets[i], n_qubits, ham.n_electrons))
        old = len(dets)
        dets = sorted(set(dets) | new)
        if len(dets) == old:
            break
    return float(res['energy'])


# ---- Reaction: N2 dissociation profile ----
REACTION_SCAN = [1.00, 1.10, 1.25, 1.50, 2.00]   # Å — bond-breaking coordinate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reaction', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--poll')
    ap.add_argument('--shots', type=int, default=20000)
    ap.add_argument('--backend', default=None)
    ap.add_argument('--anchor-job', default=None,
                    help='Derive backend from this prior job id (bypasses '
                         'IBM discovery endpoint when it hangs)')
    ap.add_argument('--r0', type=float, default=1.15, help='force ref bond length (Å)')
    args = ap.parse_args()

    if not (os.environ.get('IBM_QUANTUM_TOKEN') and os.environ.get('IBM_QUANTUM_CRN')):
        print('ERROR: IBM_QUANTUM_TOKEN + IBM_QUANTUM_CRN required'); sys.exit(1)

    print('=' * 90)
    print('M5 ON REAL HERON — N2(10,10)/cc-pVDZ @ 20 qubits')
    print('=' * 90)

    if args.reaction:
        geometries = REACTION_SCAN
        label = 'reaction'
    elif args.force:
        # Diatomic force: ±δ on the bond (z of atom 1) + reference
        delta_ang = 0.01 * BOHR_TO_ANGSTROM
        geometries = [args.r0, args.r0 + delta_ang, args.r0 - delta_ang]
        label = 'force'
    else:
        print('Pass --reaction or --force'); sys.exit(1)

    if args.submit:
        print(f'\n### Building {len(geometries)} circuits for {label}')
        circuits = []
        meta = []
        for r in geometries:
            ham, mf, qc = n2_ham_and_circuit(r)
            circuits.append(qc)
            meta.append(r)
        print(f'\n### Submitting batch of {len(circuits)} circuits to Heron')
        svc, backend, job = submit_batch(circuits, args.shots, args.backend,
                                         anchor_job=args.anchor_job)
        jid = job.job_id()
        print(f'  Job ID: {jid}  status {job.status()}')
        manifest = {'milestone': 'M5-Heron', 'task': label,
                    'geometries_ang': geometries, 'shots': args.shots,
                    'backend': backend.name, 'job_id': jid, 'r0': args.r0}
        with open(f'/tmp/m5heron_{label}_{jid}.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'  Manifest /tmp/m5heron_{label}_{jid}.json')
        print(f'\n→ Poll: python -m benchmarks.m5_heron_reactions_dynamics '
              f'--{label} --poll {jid}')
        return

    if args.poll:
        # Reconstruct geometries from manifest
        man_path = None
        for f in os.listdir('/tmp'):
            if f.startswith('m5heron_') and args.poll in f:
                man_path = f'/tmp/{f}'; break
        if man_path:
            with open(man_path) as f:
                man = json.load(f)
            geometries = man['geometries_ang']
            label = man['task']
            args.r0 = man.get('r0', args.r0)

        print(f'\n### Polling {args.poll} ({len(geometries)} circuits)')
        counts_list = poll_counts(args.poll, len(geometries))
        if counts_list is None:
            print('  Not DONE yet.'); return

        print('\n### Post-processing each geometry (SQD on cluster)')
        energies = []
        for r, counts in zip(geometries, counts_list):
            ham, mf, _ = n2_ham_and_circuit(r)
            e = sqd_energy_from_counts(ham, mf, counts)
            energies.append(e)
            print(f'  r = {r:.4f} Å  E_SQD = {e:.6f} Ha')

        if label == 'reaction':
            print('\n=== N2 DISSOCIATION PROFILE (Heron 20q) ===')
            e0 = energies[0]
            for r, e in zip(geometries, energies):
                print(f'  r = {r:.3f} Å   E = {e:.6f}   ΔE = {(e-e0)*HA_TO_KCAL:+8.2f} kcal/mol')
            # Compare to classical CASCI at each point
            print('\n  Classical CASCI(10,10) reference:')
            from pyscf import mcscf, scf
            for r in geometries:
                mol = build_n2(r); mf = scf.RHF(mol).run(verbose=0)
                cas = mcscf.CASCI(mf, ncas=10, nelecas=10)
                cas.sort_mo([2,3,4,5,6,7,8,9,10,11], base=0)
                cas.run(verbose=0)
                print(f'    r = {r:.3f} Å   CASCI = {cas.e_tot:.6f}')
        elif label == 'force':
            e_ref, e_plus, e_minus = energies
            delta_bohr = 0.01
            force_z = -(e_plus - e_minus) / (2.0 * delta_bohr)
            print('\n=== N2 FORCE FROM HERON (20q) ===')
            print(f'  Bond length r0 = {args.r0} Å')
            print(f'  F_z (on atom 1) = {force_z:+.6f} Ha/Bohr (SQD/Heron)')
            # CASCI analytic reference
            from pyscf import mcscf, scf
            mol = build_n2(args.r0); mf = scf.RHF(mol).run(verbose=0)
            cas = mcscf.CASCI(mf, ncas=10, nelecas=10)
            cas.sort_mo([2,3,4,5,6,7,8,9,10,11], base=0); cas.run(verbose=0)
            try:
                g = cas.nuc_grad_method().kernel()
                print(f'  F_z CASCI analytic = {-g[1,2]:+.6f} Ha/Bohr')
                print(f'  |Δ| = {abs(force_z - (-g[1,2])):.6f} Ha/Bohr')
            except Exception as e:
                print(f'  (CASCI analytic gradient unavailable: {e})')
        return


if __name__ == '__main__':
    main()
