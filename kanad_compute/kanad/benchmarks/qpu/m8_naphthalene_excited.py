"""M8 scale test — naphthalene low-lying spectrum at 32 qubits from real Heron data.

Reuses the existing M11c Heron job (`d8bk0bb8amns73bia0dg`, 30k shots,
CAS(16,16)/cc-pVDZ): rebuild the converged selected-CI subspace from the
hardware bitstrings, then extract the lowest states via
`SamplingSQDSolver.solve_excited_states`. This is excited states at production
scale on real-QPU samples — no new queue wait.

Caveat under test: the subspace was expanded to converge the GROUND state; if it
is too ground-state-biased the excited roots will be high. A bad result here is
the signal to add excited-state-targeted subspace expansion (store as benchmark
if good, fix if bad).

Run:  python -m benchmarks.m8_naphthalene_excited
"""

import sys
import json

import numpy as np

from benchmarks.qpu.m11c_naphthalene_heron import (
    build_naph_setup, poll_and_get_counts, bitstrings_from_counts,
)
from kanad.solvers.sampling_sqd import (
    SamplingSQDSolver, _filter_with_recovery, _generate_singles_doubles,
    _build_sparse_h_subspace,
)

JOB = 'd8bk0bb8amns73bia0dg'
HA_TO_EV = 27.211386245988


def main():
    setup = build_naph_setup()
    ham = setup['ham']
    n_orb, n_e, n_qubits = ham.n_orbitals, ham.n_electrons, setup['n_qubits']

    counts, status = poll_and_get_counts(JOB)
    if counts is None:
        print(f'Job not DONE ({status}).', flush=True)
        sys.exit(1)
    bits = bitstrings_from_counts(counts)
    print(f'Naphthalene CAS(16,16) {n_qubits}q — {len(bits)} Heron shots '
          f'(job {JOB}); E_CASCI = {setup["e_casci"]:.6f}', flush=True)

    solver = SamplingSQDSolver(ham, n_samples=len(bits), backend='statevector',
                               recover_configurations=True, ci_backend='pyscf',
                               target_sz=0.0, random_seed=0)
    mo_e = solver._resolve_mo_energies()
    valid, *_ = _filter_with_recovery(bits, n_orb, n_e, 0.0, mo_e)
    dets = sorted(set(int(d) for d in valid))
    print(f'recovered {len(dets)} determinants; converging subspace...', flush=True)

    # STATE-AVERAGED expansion: diagonalize for N_STATES roots each round and
    # expand from the top determinants of ALL of them, so the subspace spans the
    # excited manifold (a ground-only expansion misses the low-lying triplet).
    # Small top-K + a hard cap keep the sparse-SC + k-root eigsh tractable.
    from scipy.sparse.linalg import eigsh
    N_TARGET_MAX = 120_000
    N_STATES = 6
    HA_TO_EV = 27.211386245988
    h1, h2 = solver._h1, solver._h2
    nuc = float(ham.nuclear_repulsion)
    last = None
    for it in range(8):
        nd = len(dets)
        k = min(N_STATES, nd - 1)
        H, _ = _build_sparse_h_subspace(dets, h1, h2, nuc, n_orb)
        ncv = max(2 * k + 1, min(nd - 1, 60))
        ev, evec = eigsh(H, k=k, which='SA', tol=1e-8, ncv=ncv)
        o = np.argsort(ev); ev = ev[o]; evec = evec[:, o]
        print(f'  iter {it}: N_det={nd}  E0={ev[0]:.6f}  dE1={(ev[1]-ev[0])*HA_TO_EV:.3f}eV',
              flush=True)
        if last is not None and len(last) == len(ev) \
                and max(abs(a - b) for a, b in zip(ev, last)) < 5e-6:
            break
        last = list(ev)
        if nd >= N_TARGET_MAX:
            break
        new = set()
        for col in range(evec.shape[1]):
            top = np.argsort(np.abs(evec[:, col]) ** 2)[::-1][:20]
            for i in top:
                new.update(_generate_singles_doubles(dets[i], n_qubits, n_e))
        old = len(dets)
        dets = sorted(set(dets) | new)
        if len(dets) == old:
            break

    solver.results = {'determinants': dets}
    ex = solver.solve_excited_states(n_states=N_STATES)

    print(f'\nNaphthalene 32q low-lying spectrum (N_det = {len(dets)}):', flush=True)
    for i, (e, ev) in enumerate(zip(ex['energies'], ex['excitation_energies_ev'])):
        print(f'  state {i}: E = {e:.6f} Ha   ΔE = {ev:7.3f} eV')
    print('\nLiterature naphthalene verticals: T1 ~2.6 eV, S1(1Lb) ~4.0 eV, 1La ~4.5 eV',
          flush=True)

    out = {
        'system': 'naphthalene', 'cas': '(16,16)', 'n_qubits': n_qubits,
        'job_id': JOB, 'n_det': len(dets), 'e_casci_ground': setup['e_casci'],
        'energies_ha': ex['energies'],
        'excitation_energies_ev': ex['excitation_energies_ev'],
        'lit_verticals_ev': {'T1': 2.6, 'S1_1Lb': 4.0, '1La': 4.5},
    }
    with open('/tmp/m8_naph_excited_result.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('M8_NAPH_EXCITED_DONE', flush=True)


if __name__ == '__main__':
    main()
