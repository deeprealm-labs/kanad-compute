"""M11c — Naphthalene(16,16) at 32 qubits on real IBM Heron.

Third step of PLAN.md M11 (Tier H2 qubit-count push). Builds on M11a (N₂/20q)
and M11b (C₂/24q):
  - 24q → 32q (4× FCI scaling)
  - N_FCI: 854,000 → 165,636,900 (≈195× larger CI space)
  - Aromatic π system; less multireference than C₂ but bigger active space.

CAS(16e, 16o) at cc-pVDZ:
  - 16 active electrons in 16 active orbitals.
  - Active window centered on HOMO/LUMO captures the π valence (10 π e/o)
    plus 3 occ + 3 virt σ-frontier orbitals.
  - 16 spatial orbs × 2 spin = 32 qubits.

Why this is the right next step:
  - 32q is the BlueQubit-CPU ceiling — first M11 run where local sim is
    impossible. Hardware-required from here on.
  - Naphthalene is well-characterized; classical CASCI(16,16) is the
    in-active-space truth.
  - Tests whether the M3 sparse-SC + matrix-free CI scales gracefully
    above the 24q threshold.

Designed to RUN ON THE CLUSTER (235 GB RAM, 20 CPUs) — the iterative
expansion at 150k+ dets requires the RAM headroom.

References:
  - Hashimoto 1998 MR-CI/cc-pVTZ: E_total ≈ -383.5 Ha
  - Experiment: planar D_2h; π-system aromatic

Submission/poll workflow:
  $ python -m benchmarks.m11c_naphthalene_heron --submit
  $ python -m benchmarks.m11c_naphthalene_heron --poll <jobid>

Env vars: IBM_QUANTUM_TOKEN, IBM_QUANTUM_CRN
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse
import numpy as np


# Naphthalene C10H8 (D_2h), experimental geometry from NIST CCCBDB.
# 10 C atoms (8 peripheral with H, 2 bridge without) + 8 H atoms.
NAPHTHALENE_XYZ = """
C   1.2415   0.7104   0.0000
C   2.4282   1.4020   0.0000
C   2.4282   2.7960   0.0000
C   1.2415   3.4876   0.0000
C   0.0000   2.7960   0.0000
C   0.0000   1.4020   0.0000
C  -1.2415   0.7104   0.0000
C  -2.4282   1.4020   0.0000
C  -2.4282   2.7960   0.0000
C  -1.2415   3.4876   0.0000
H   2.1561   0.1893   0.0000
H   3.3754   0.8809   0.0000
H   3.3754   3.3171   0.0000
H   2.1561   4.0087   0.0000
H  -2.1561   0.1893   0.0000
H  -3.3754   0.8809   0.0000
H  -3.3754   3.3171   0.0000
H  -2.1561   4.0087   0.0000
"""


NAPH_SPEC = {
    'atom_str': NAPHTHALENE_XYZ.strip(),
    'basis': 'cc-pvdz',
    'charge': 0,
    'spin': 0,
    # 68 total electrons. CAS(16e, 16o) freezes the lowest 26 doubly-occ
    # MOs (= 52 electrons), leaving 16 electrons in 16 active MOs centered
    # around the HOMO/LUMO.
    'frozen': list(range(26)),
    'active': list(range(26, 42)),
    'description': 'Naphthalene C₁₀H₈ (CAS(16e,16o)/cc-pVDZ → 32 qubits)',
}


CACHE_PATH = '/tmp/m11c_naph_classical_cache.json'


def build_naph_setup():
    """Build PySCF + kanad ham + CASCI(16,16) classical reference.

    Caches HF + CASCI energies to ``CACHE_PATH`` so re-polls don't rerun
    the 5-min Davidson. mol/mf/ham still get rebuilt (cheap) since the
    SQD post-processing needs them.
    """
    from pyscf import gto, scf, mcscf
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )

    spec = NAPH_SPEC
    mol = gto.M(atom=spec['atom_str'], basis=spec['basis'],
                charge=spec['charge'], spin=spec['spin'], verbose=0)
    print(f'  Molecule: {spec["description"]}')
    print(f'  Electrons: {mol.nelectron}, AOs: {mol.nao_nr()}')
    mf = scf.RHF(mol).run(verbose=0)
    print(f'  HF/cc-pVDZ:       E = {mf.e_tot:.6f} Ha  (converged: {mf.converged})')

    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(
            frozen=spec['frozen'], active=spec['active'],
        ),
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = ham.n_electrons
    print(f'  Active space:     {len(spec["active"])} orbs, {n_active_e} electrons')
    print(f'  Qubit count:      {n_qubits}')
    print(f'  N_FCI in active:  C({len(spec["active"])},{n_active_e//2})^2 = '
          f'{math.comb(len(spec["active"]), n_active_e//2)**2}')

    # CASCI cache check
    cache = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if cache.get('e_hf') and abs(cache['e_hf'] - float(mf.e_tot)) < 1e-6:
        e_casci = float(cache['e_casci'])
        max_w = float(cache['max_weight'])
        print(f'  CASCI(16,16):     E = {e_casci:.6f} Ha  |c_max|² = {max_w:.4f}  '
              f'(cached from {CACHE_PATH})')
    else:
        print(f'  Running PySCF CASCI(16,16) (~5 min, big Davidson)...')
        t0 = time.time()
        cas = mcscf.CASCI(mf, ncas=len(spec['active']), nelecas=n_active_e)
        cas.sort_mo(spec['active'], base=0)
        cas.run(verbose=0)
        e_casci = float(cas.e_tot)
        ci = cas.ci.flatten() if cas.ci.ndim == 2 else cas.ci
        max_w = float(np.max(np.abs(ci)) ** 2)
        print(f'  CASCI(16,16):     E = {e_casci:.6f} Ha  |c_max|² = {max_w:.4f}  '
              f'(took {time.time()-t0:.1f}s)')
        try:
            with open(CACHE_PATH, 'w') as f:
                json.dump({'e_hf': float(mf.e_tot), 'e_casci': e_casci,
                            'max_weight': max_w}, f)
        except Exception:
            pass

    return {
        'spec': spec, 'mol': mol, 'mf': mf, 'ham': ham,
        'n_qubits': n_qubits, 'n_active_e': n_active_e,
        'e_hf': float(mf.e_tot), 'e_casci': e_casci, 'max_weight': max_w,
    }


def build_lucj_circuit(setup, seed=42, n_layers=1):
    from kanad.core.ansatze import LUCJAnsatz
    ansatz = LUCJAnsatz(
        n_qubits=setup['n_qubits'], n_electrons=setup['n_active_e'],
        n_layers=n_layers, target_sz=0.0,
    )
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(seed)
    # Modest param range — large active space, want to spread weight gently
    params = rng.uniform(-0.4, 0.4, size=qc.num_parameters)
    bound = qc.assign_parameters(
        {qc.parameters[i]: float(params[i]) for i in range(qc.num_parameters)}
    )
    return bound, ansatz


def submit_to_heron(circuit, shots, backend_name=None):
    from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2
    from qiskit import transpile
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    if backend_name:
        backend = svc.backend(backend_name)
    else:
        backends = svc.backends(operational=True, simulator=False,
                                 min_num_qubits=32)
        backend = min(backends, key=lambda b: b.status().pending_jobs)
    print(f'\n  IBM backend: {backend.name} ({backend.num_qubits}q, '
          f'pending: {backend.status().pending_jobs})')

    c = circuit.copy()
    if c.num_clbits == 0:
        c.measure_all()
    ct = transpile(c, backend=backend, optimization_level=1)
    n_2q = sum(1 for inst in ct.data if inst.operation.num_qubits == 2)
    print(f'  Transpiled depth:    {ct.depth()}')
    print(f'  Transpiled 2q gates: {n_2q}')

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        job = sampler.run([ct], shots=shots)
    return svc, backend, job, ct, n_2q


def poll_and_get_counts(job_id):
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(
        channel='ibm_cloud',
        token=os.environ['IBM_QUANTUM_TOKEN'],
        instance=os.environ['IBM_QUANTUM_CRN'],
    )
    job = svc.job(job_id)
    status = str(job.status())
    print(f'  Job {job_id} status: {status}')
    if status != 'DONE':
        return None, status
    result = job.result()
    pub = result[0]
    if hasattr(pub.data, 'meas'):
        counts = pub.data.meas.get_counts()
    else:
        field = list(pub.data.keys())[0]
        counts = getattr(pub.data, field).get_counts()
    return counts, status


def bitstrings_from_counts(counts):
    out = []
    for bs, n in counts.items():
        clean = str(bs).replace(' ', '').replace('0x', '')
        out.extend([int(clean, 2)] * int(n))
    return np.array(out, dtype=np.int64)


def run_sqd_post(setup, counts):
    """Single-shot recovery + iterative classical SD expansion.
    M11c skips multi-round at 32q (RAM headroom for iterative expansion is
    the real win on the cluster). Multi-round D2 would be a future
    optimization if iterative expansion plateaus above 1 mHa."""
    from kanad.solvers.sampling_sqd import (
        SamplingSQDSolver,
        _filter_by_n_sz, _filter_with_recovery, _generate_singles_doubles,
    )

    bitstrings = bitstrings_from_counts(counts)
    ham = setup['ham']
    n_orb = ham.n_orbitals
    n_e = ham.n_electrons
    n_qubits = setup['n_qubits']
    total = len(bitstrings)
    print(f'\n  Heron samples:   {total} shots')

    solver = SamplingSQDSolver(
        ham, n_samples=total, backend='statevector',
        recover_configurations=True, ci_backend='pyscf',
        target_sz=0.0, random_seed=0,
    )
    mo_e = solver._resolve_mo_energies()

    valid_drop = _filter_by_n_sz(bitstrings, n_orb, n_e, 0.0)
    dets_drop = sorted(set(int(d) for d in valid_drop))
    print(f'  Drop-only:       {len(valid_drop)} valid '
          f'({100*len(valid_drop)/total:.1f}%), {len(dets_drop)} unique dets')

    valid_ss, n_kept, n_rec, n_drop = _filter_with_recovery(
        bitstrings, n_orb, n_e, 0.0, mo_e,
    )
    dets_ss = sorted(set(int(d) for d in valid_ss))
    print(f'  Single-shot rec: {len(valid_ss)} valid '
          f'({100*len(valid_ss)/total:.1f}%), {len(dets_ss)} unique dets')

    results = {}
    for label, dets in [('drop-only', dets_drop), ('single-shot', dets_ss)]:
        if not dets:
            results[label] = {'energy': float('nan'), 'gap_mha': float('nan'),
                              'n_det': 0}
            continue
        t0 = time.time()
        res = solver._diagonalize_in_subspace_pyscf(dets)
        gap = (res['energy'] - setup['e_casci']) * 1000
        tag = '✓' if abs(gap) < 1.0 else ('⚠' if abs(gap) < 10.0 else '✗')
        print(f'\n  [{label}]  E_SQD = {res["energy"]:.6f}  '
              f'gap vs CASCI = {gap:+.3f} mHa  {tag}  '
              f'(N_det = {len(dets)}, {time.time()-t0:.1f}s)')
        results[label] = {
            'energy': float(res['energy']), 'gap_mha': float(gap),
            'n_det': len(dets),
            'eigenvector': res.get('eigenvector'),
        }

    # Iterative classical expansion with **adaptive top-K** (M11c fix
    # 2026-05-28). Choose K each iteration so the resulting subspace
    # stays in the sparse-SC regime (≤ N_target dets) — avoids the
    # matrix-free Davidson cliff at very large subspaces.
    N_TARGET_MAX = 150_000   # sparse-SC ceiling; matrix-free above
    print(f'\n  Iterative classical expansion (adaptive top-K, target≤{N_TARGET_MAX} dets):')
    dets = list(dets_ss)
    last = None
    t0 = time.time()
    # Estimate connections per det: CAS(16,16) ≈ 3500 SD partners
    PER_DET_CONNS = 3500
    for it in range(6):
        res = solver._diagonalize_in_subspace_pyscf(dets)
        if last is not None and abs(res['energy'] - last) < 5e-6:
            print(f'    iter {it+1}: converged (Δ < 5 µHa)')
            break
        last = res['energy']
        evec = res['eigenvector']
        # Adaptive top-K: target ≤1.5× current OR ≤N_TARGET_MAX
        room = max(0, N_TARGET_MAX - len(dets))
        target_growth = min(room, int(len(dets) * 0.8))
        # Each top-K det adds ~PER_DET_CONNS connections (most novel ones)
        K = max(3, min(100, target_growth // PER_DET_CONNS))
        top = np.argsort(np.abs(evec) ** 2)[::-1][:min(K, len(dets))]
        new_dets = set()
        for i in top:
            new_dets.update(_generate_singles_doubles(dets[i], n_qubits, n_e))
        old = len(dets)
        dets = sorted(set(dets) | new_dets)
        print(f'    iter {it+1}: K={K}, N_det {old} → {len(dets)}, '
              f'E = {res["energy"]:.6f}')
        if len(dets) == old:
            print(f'    iter {it+1}: no new dets, done.')
            break
        if len(dets) >= N_TARGET_MAX:
            print(f'    iter {it+1}: hit target cap {N_TARGET_MAX}, final diag next')
    expand_t = time.time() - t0
    final_e = float(res['energy'])
    final_gap = (final_e - setup['e_casci']) * 1000
    tag = '✓' if abs(final_gap) < 1.0 else ('⚠' if abs(final_gap) < 5.0 else '✗')
    print(f'\n  Final E_SQD (after expansion) = {final_e:.6f}')
    print(f'  CASCI(16,16) ref               = {setup["e_casci"]:.6f}')
    print(f'  Gap = {final_gap:+.3f} mHa  {tag}  ({len(dets)} dets, {expand_t:.1f}s)')

    results['iterative'] = {
        'energy': final_e, 'gap_mha': float(final_gap),
        'n_det': len(dets), 'expand_time_s': expand_t,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--poll', help='Existing IBM job ID')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--shots', type=int, default=30000)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--layers', type=int, default=1)
    args = parser.parse_args()

    if not (args.submit or args.poll or args.dry_run):
        print('Pass --submit, --poll <jobid>, or --dry-run'); sys.exit(1)
    if (args.submit or args.poll) and not (os.environ.get('IBM_QUANTUM_TOKEN')
                                            and os.environ.get('IBM_QUANTUM_CRN')):
        print('ERROR: IBM_QUANTUM_TOKEN + IBM_QUANTUM_CRN env vars required')
        sys.exit(1)

    print('=' * 96)
    print('M11c — NAPHTHALENE(16,16)/cc-pVDZ @ 32 qubits on REAL IBM HERON')
    print('  Sparse SC + matrix-free CI + iterative expansion (cluster-class)')
    print('=' * 96)

    print('\n### Classical references (PySCF)')
    setup = build_naph_setup()

    if args.dry_run:
        print('\n### Building LUCJ circuit (dry-run; NO submission)')
        circuit, ansatz = build_lucj_circuit(setup, n_layers=args.layers)
        n_2q_pre = circuit.num_nonlocal_gates()
        print(f'  LUCJ {args.layers}-layer:')
        print(f'    n_qubits     = {setup["n_qubits"]}')
        print(f'    n_active_e   = {setup["n_active_e"]}')
        print(f'    parameters   = {ansatz.num_parameters}')
        print(f'    depth (raw)  = {circuit.depth()}')
        print(f'    2q (raw)     = {n_2q_pre}')
        print(f'    Heron est. 2q post-transpile: ~{n_2q_pre*3}-{n_2q_pre*5}')
        print(f'    Per-shot fid. est @ 10⁻³ 2q err: '
              f'~{100*(1-1e-3)**(n_2q_pre*3):.1f}% best, '
              f'~{100*(1-1e-3)**(n_2q_pre*5):.1f}% worst')
        return

    if args.submit:
        print('\n### Building LUCJ circuit')
        circuit, ansatz = build_lucj_circuit(setup, n_layers=args.layers)
        n_2q_pre = circuit.num_nonlocal_gates()
        print(f'  LUCJ {args.layers}-layer: depth {circuit.depth()}, '
              f'2q gates (pre-transpile) = {n_2q_pre}')
        print(f'  Parameters: {ansatz.num_parameters} (bound to random U(-0.4, 0.4))')

        print('\n### Submitting to Heron')
        t0 = time.time()
        svc, backend, job, transpiled, n_2q_post = submit_to_heron(
            circuit, shots=args.shots, backend_name=args.backend,
        )
        job_id = job.job_id()
        print(f'  Submitted in {time.time()-t0:.1f}s')
        print(f'  Job ID: {job_id}')
        print(f'  Initial status: {job.status()}')

        manifest = {
            'milestone': 'M11c',
            'molecule': 'naphthalene', 'cas': '(16,16)', 'basis': 'cc-pvdz',
            'n_qubits': setup['n_qubits'], 'n_active_e': setup['n_active_e'],
            'lucj_layers': args.layers, 'shots': args.shots,
            'backend': backend.name, 'job_id': job_id,
            'submitted_at': time.time(),
            'n_2q_pre_transpile': int(n_2q_pre),
            'n_2q_post_transpile': int(n_2q_post),
            'transpiled_depth': int(transpiled.depth()),
            'classical_refs': {
                'e_hf': setup['e_hf'],
                'e_casci_16_16': setup['e_casci'],
                'casci_max_weight': setup['max_weight'],
            },
        }
        jobfile = f'/tmp/m11c_naph_{job_id}.json'
        with open(jobfile, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'  Manifest: {jobfile}')
        print(f'\n→ Poll with:  python -m benchmarks.m11c_naphthalene_heron --poll {job_id}')
        return

    if args.poll:
        print(f'\n### Polling job {args.poll}')
        counts, status = poll_and_get_counts(args.poll)
        if counts is None:
            print(f'\n  Job not complete (status: {status}). Try later.')
            return

        print(f'\n### SQD post-processing')
        results = run_sqd_post(setup, counts)

        print('\n' + '=' * 96)
        print('M11c RESULT SUMMARY')
        print('=' * 96)
        print(f'  Classical anchors:')
        print(f'    HF:           {setup["e_hf"]:.6f}')
        print(f'    CASCI(16,16): {setup["e_casci"]:.6f}  ← in-active-space truth')
        print(f'    |c_max|²:     {setup["max_weight"]:.4f}')
        print(f'  Heron SQD result (32q, free tier):')
        for k in ('drop-only', 'single-shot', 'iterative'):
            r = results.get(k, {})
            print(f'    {k:13}: E = {r.get("energy", float("nan")):.6f}  '
                  f'gap = {r.get("gap_mha", float("nan")):+.3f} mHa  '
                  f'(N_det = {r.get("n_det", 0)})')

        out = {
            'milestone': 'M11c', 'job_id': args.poll,
            'classical': {
                'e_hf': setup['e_hf'], 'e_casci_16_16': setup['e_casci'],
                'casci_max_weight': setup['max_weight'],
            },
            'heron_sqd': {
                k: {kk: vv for kk, vv in v.items() if kk != 'eigenvector'}
                for k, v in results.items()
            },
        }
        outfile = f'/tmp/m11c_naph_{args.poll}_result.json'
        with open(outfile, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n  Persisted to {outfile}')


if __name__ == '__main__':
    main()
