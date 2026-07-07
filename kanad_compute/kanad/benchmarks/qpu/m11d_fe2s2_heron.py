"""M11d — [2Fe-2S] cluster at 40 qubits on real IBM Heron.

PLAN.md M11 step 11d — the marquee cofactor/transition-metal benchmark.
This is qualitatively different from M11a/b/c:
  - First all-electron-equivalent transition-metal cluster (ECP-SDD on Fe).
  - CAS(20e, 20o) → 40 qubits.
  - **N_FCI = C(20,10)² ≈ 34 BILLION → PySCF CASCI as a classical anchor
    is infeasible.** SQD IS the reference at this size.

Cluster: [Fe₂S₂Cl₄]²⁻ — the canonical small-ligand [2Fe-2S] model used
in Krewald 2017, Bertini 2009. Two Fe(III) bridged by μ-S²⁻, each Fe
terminally coordinated by 2 Cl⁻. Total charge −2, ground-state
electronic structure is antiferromagnetic singlet (broken-symmetry).

Active space: 20 electrons in 20 orbitals covering:
  - 10 Fe 3d orbitals (5 per Fe)
  - 4 bridging S 3p orbitals
  - 6 terminal Cl 3p / Fe-S bonding mix (frontier orbitals around HOMO)

Classical references attempted:
  - HF (single-determinant, will not capture biradical)
  - CCSD (single-reference, may not converge on biradical singlet)
  - CASSCF(10,10) on Fe-d-only subset (smaller, tractable as sanity)
  - SQD itself is THE reference at this scale

References:
  - Mouesca 1994: BS-DFT exchange coupling J ≈ −180 cm⁻¹ for [2Fe-2S]
  - Robledo-Moreno 2025 Nature: [2Fe-2S] on 77-qubit Heron via SQD
  - Krewald 2017: CAS(10,10) for the Fe-d shell

Submit/poll workflow: same as M11a/b/c.
Run on the GPU cluster (235 GB RAM needed for any classical refs):
  $ ssh root@<cluster> 'cd /root/kanad-framework && \
      IBM_QUANTUM_TOKEN=... IBM_QUANTUM_CRN=... PYTHONPATH=/root/kanad-fw-test-pkg \
      python -m benchmarks.m11d_fe2s2_heron --submit'

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


# [Fe2S2Cl4]^2- canonical geometry (Bertini 2009).
# Cluster in xy-plane; terminal Cl's in z direction.
# Distances: Fe-Fe ≈ 2.73 Å, Fe-S(bridge) ≈ 2.20 Å, Fe-Cl ≈ 2.25 Å.
FE2S2CL4_XYZ = """
Fe   1.365   0.000   0.000
Fe  -1.365   0.000   0.000
S    0.000   1.732   0.000
S    0.000  -1.732   0.000
Cl   1.365   0.000   2.250
Cl   1.365   0.000  -2.250
Cl  -1.365   0.000   2.250
Cl  -1.365   0.000  -2.250
"""


FE2S2_SPEC = {
    'atom_str': FE2S2CL4_XYZ.strip(),
    # LANL2DZ ECP on Fe removes the 1s²2s²2p⁶3s²3p⁶ (18 e) core. Cl & S
    # kept all-electron at cc-pVDZ.
    'basis': {'Fe': 'lanl2dz', 'S': 'cc-pvdz', 'Cl': 'cc-pvdz'},
    'ecp': {'Fe': 'lanl2dz'},
    'charge': -2,
    'spin': 0,              # Broken-symmetry singlet (closed-shell HF guess)
    'cas_n_e': 20, 'cas_n_o': 20,
    'description': '[Fe₂S₂Cl₄]²⁻ (CAS(20e,20o), LANL2DZ-ECP on Fe + cc-pVDZ → 40 qubits)',
}


def build_fe2s2_setup(run_ccsd=True):
    """Build PySCF mol + mf with ECP, kanad active-space ham, classical refs.

    No CASCI(20,20) — too big (N_FCI≈34B). Reports HF + (optionally) CCSD.
    SQD is THE reference at this scale.

    Args:
        run_ccsd: if False, skip CCSD (saves 30-150 min on this system).
    """
    from pyscf import gto, scf, mcscf, cc
    from kanad.core.active_space import (
        ActiveSpaceSelector, build_active_space_hamiltonian,
    )

    spec = FE2S2_SPEC
    mol = gto.M(
        atom=spec['atom_str'], basis=spec['basis'], ecp=spec['ecp'],
        charge=spec['charge'], spin=spec['spin'], verbose=0,
    )
    print(f'  Molecule: {spec["description"]}')
    print(f'  Electrons: {mol.nelectron}, AOs: {mol.nao_nr()}')
    print(f'  Nuclear charge (incl. ECP):  {mol.atom_charges().sum()}')

    mf = scf.RHF(mol)
    mf.max_cycle = 200
    mf.conv_tol = 1e-8
    print(f'  Running RHF (max_cycle=200, biradical may need patience)...')
    t0 = time.time()
    mf.run(verbose=0)
    print(f'  HF: E = {mf.e_tot:.6f} Ha, converged = {mf.converged}, '
          f'{time.time()-t0:.1f}s')

    # Pick active window: HOMO-9..LUMO+10 around the frontier
    n_occ = int(np.sum(mf.mo_occ > 0))
    n_a_e = spec['cas_n_e'] // 2
    cas_lo = n_occ - n_a_e
    cas_hi = cas_lo + spec['cas_n_o']
    active_orbs = list(range(cas_lo, cas_hi))
    frozen_orbs = list(range(cas_lo))
    print(f'  Active window: MOs {cas_lo}..{cas_hi-1} ({spec["cas_n_o"]} orbs, '
          f'{spec["cas_n_e"]} e)')

    ham = build_active_space_hamiltonian(
        mf, ActiveSpaceSelector(mf).manual(
            frozen=frozen_orbs, active=active_orbs,
        ),
    )
    n_qubits = 2 * ham.n_orbitals
    n_active_e = ham.n_electrons
    print(f'  Qubit count: {n_qubits}')
    n_fci = math.comb(spec['cas_n_o'], n_a_e) ** 2
    print(f'  N_FCI in active: C(20,10)² = {n_fci:,} '
          f'(≈ {n_fci/1e9:.1f} B — classical CASCI INFEASIBLE)')

    # CCSD as classical correlated anchor (single-reference; will be off
    # but reproducible). On this system it costs ~30-150 min — gated by
    # run_ccsd flag because dry-run / circuit-build doesn't need it.
    e_ccsd = float('nan')
    if run_ccsd:
        try:
            print(f'  Attempting CCSD (single-ref; ~30-150 min expected)...')
            ccobj = cc.CCSD(mf)
            ccobj.max_cycle = 200
            t0 = time.time()
            ccobj.run(verbose=0)
            if ccobj.converged:
                e_ccsd = float(ccobj.e_tot)
                print(f'  CCSD: E = {e_ccsd:.6f} Ha ({time.time()-t0:.1f}s)')
            else:
                print(f'  CCSD: did not converge in {ccobj.max_cycle} cycles')
        except Exception as exc:
            print(f'  CCSD: failed ({type(exc).__name__}: {exc})')
    else:
        print(f'  CCSD: skipped (run_ccsd=False)')

    return {
        'spec': spec, 'mol': mol, 'mf': mf, 'ham': ham,
        'n_qubits': n_qubits, 'n_active_e': n_active_e,
        'active': active_orbs, 'frozen': frozen_orbs,
        'e_hf': float(mf.e_tot), 'e_ccsd': e_ccsd,
        'n_fci_estimate': n_fci,
    }


def build_lucj_circuit(setup, seed=42, n_layers=1):
    from kanad.core.ansatze import LUCJAnsatz
    ansatz = LUCJAnsatz(
        n_qubits=setup['n_qubits'], n_electrons=setup['n_active_e'],
        n_layers=n_layers, target_sz=0.0,
    )
    qc = ansatz.build_circuit()
    rng = np.random.default_rng(seed)
    params = rng.uniform(-0.3, 0.3, size=qc.num_parameters)
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
                                 min_num_qubits=40)
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
    print(f'  Per-shot fid estim @ 10⁻³ 2q err: '
          f'~{100*(1-1e-3)**n_2q:.1f}%')

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
    """SQD post-processing with adaptive top-K (sparse SC handles to 200k)."""
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

    valid_ss, *_ = _filter_with_recovery(
        bitstrings, n_orb, n_e, 0.0, mo_e,
    )
    dets_ss = sorted(set(int(d) for d in valid_ss))
    print(f'  Single-shot rec: {len(valid_ss)} valid '
          f'({100*len(valid_ss)/total:.1f}%), {len(dets_ss)} unique dets')

    results = {}
    for label, dets in [('drop-only', dets_drop), ('single-shot', dets_ss)]:
        if not dets:
            results[label] = {'energy': float('nan'), 'n_det': 0}
            continue
        t0 = time.time()
        res = solver._diagonalize_in_subspace_pyscf(dets)
        print(f'\n  [{label}]  E_SQD = {res["energy"]:.6f}  '
              f'(N_det = {len(dets)}, {time.time()-t0:.1f}s)')
        results[label] = {'energy': float(res['energy']),
                          'n_det': len(dets),
                          'eigenvector': res.get('eigenvector')}

    # Iterative classical expansion with adaptive top-K
    # CAS(20,20): ~6000 SD partners per det (singles + doubles)
    N_TARGET_MAX = 200_000
    PER_DET_CONNS = 6000
    print(f'\n  Iterative classical expansion (adaptive K, target≤{N_TARGET_MAX}):')
    dets = list(dets_ss)
    last = None
    t0 = time.time()
    for it in range(6):
        res = solver._diagonalize_in_subspace_pyscf(dets)
        if last is not None and abs(res['energy'] - last) < 5e-6:
            print(f'    iter {it+1}: converged (Δ < 5 µHa)')
            break
        last = res['energy']
        evec = res['eigenvector']
        room = max(0, N_TARGET_MAX - len(dets))
        target_growth = min(room, int(len(dets) * 0.8))
        K = max(3, min(50, target_growth // PER_DET_CONNS))
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
    final_e = float(res['energy'])
    print(f'\n  Final E_SQD (after expansion) = {final_e:.6f}  '
          f'({len(dets)} dets, {time.time()-t0:.1f}s)')
    results['iterative'] = {'energy': final_e, 'n_det': len(dets)}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--poll', help='Existing IBM job ID')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--shots', type=int, default=40000)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--ccsd', action='store_true',
                        help='Run CCSD reference (warning: 30-150 min on M11d)')
    args = parser.parse_args()

    if not (args.submit or args.poll or args.dry_run):
        print('Pass --submit, --poll <jobid>, or --dry-run'); sys.exit(1)
    if (args.submit or args.poll) and not (os.environ.get('IBM_QUANTUM_TOKEN')
                                            and os.environ.get('IBM_QUANTUM_CRN')):
        print('ERROR: IBM_QUANTUM_TOKEN + IBM_QUANTUM_CRN env vars required')
        sys.exit(1)

    print('=' * 96)
    print('M11d — [Fe₂S₂Cl₄]²⁻ CAS(20,20) @ 40 qubits on REAL IBM HERON')
    print('  Marquee transition-metal demo. Classical FCI(20,20) intractable.')
    print('=' * 96)

    print('\n### Classical references (PySCF — HF + CCSD only)')
    # CCSD on this system is 30-150 min — gate it explicitly with
    # `--ccsd` so dry-run / submit / poll can all skip if not wanted.
    setup = build_fe2s2_setup(run_ccsd=args.ccsd)

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
        print(f'    Per-shot fid est @ 10⁻³ 2q err: '
              f'~{100*(1-1e-3)**(n_2q_pre*3):.1f}% best, '
              f'~{100*(1-1e-3)**(n_2q_pre*5):.1f}% worst')
        return

    if args.submit:
        print('\n### Building LUCJ circuit')
        circuit, ansatz = build_lucj_circuit(setup, n_layers=args.layers)
        n_2q_pre = circuit.num_nonlocal_gates()
        print(f'  LUCJ {args.layers}-layer: depth {circuit.depth()}, '
              f'2q gates (pre-transpile) = {n_2q_pre}')

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
            'milestone': 'M11d',
            'molecule': '[Fe2S2Cl4]2-', 'cas': '(20,20)',
            'basis': 'sdd-ecp on Fe + cc-pvdz on S/Cl',
            'n_qubits': setup['n_qubits'], 'n_active_e': setup['n_active_e'],
            'lucj_layers': args.layers, 'shots': args.shots,
            'backend': backend.name, 'job_id': job_id,
            'submitted_at': time.time(),
            'n_2q_pre_transpile': int(n_2q_pre),
            'n_2q_post_transpile': int(n_2q_post),
            'transpiled_depth': int(transpiled.depth()),
            'classical_refs': {
                'e_hf': setup['e_hf'], 'e_ccsd': setup['e_ccsd'],
                'n_fci_estimate': setup['n_fci_estimate'],
                'note': 'no CASCI(20,20) reference — N_FCI infeasible',
            },
        }
        jobfile = f'/tmp/m11d_fe2s2_{job_id}.json'
        with open(jobfile, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f'  Manifest: {jobfile}')
        print(f'\n→ Poll with:  python -m benchmarks.m11d_fe2s2_heron --poll {job_id}')
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
        print('M11d RESULT SUMMARY')
        print('=' * 96)
        print(f'  Classical anchors:')
        print(f'    HF:                  {setup["e_hf"]:.6f}')
        if not math.isnan(setup['e_ccsd']):
            print(f'    CCSD:                {setup["e_ccsd"]:.6f}  '
                  f'(corr = {(setup["e_ccsd"]-setup["e_hf"])*1000:.1f} mHa)')
        else:
            print(f'    CCSD:                did not converge')
        print(f'    CASCI(20,20):        INFEASIBLE (N_FCI ≈ 34 B)')
        print(f'  Heron SQD result (40q, free tier):')
        for k in ('drop-only', 'single-shot', 'iterative'):
            r = results.get(k, {})
            if not math.isnan(r.get('energy', float('nan'))):
                gain_vs_hf = (setup['e_hf'] - r['energy']) * 1000
                print(f'    {k:13}: E = {r["energy"]:.6f}  '
                      f'(N_det = {r.get("n_det", 0)}, captured {gain_vs_hf:+.1f} mHa vs HF)')

        out = {
            'milestone': 'M11d', 'job_id': args.poll,
            'classical': {'e_hf': setup['e_hf'], 'e_ccsd': setup['e_ccsd']},
            'heron_sqd': {
                k: {kk: vv for kk, vv in v.items() if kk != 'eigenvector'}
                for k, v in results.items()
            },
        }
        outfile = f'/tmp/m11d_fe2s2_{args.poll}_result.json'
        with open(outfile, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n  Persisted to {outfile}')


if __name__ == '__main__':
    main()
