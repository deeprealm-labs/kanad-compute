"""Campaign C — SOLVERS × ANSÄTZE × MAPPERS cross-validation. Stop recycling one config:
run each molecule through CI/FCI, VQE(givens_sd / hardware_efficient), SQD, and both
fermion->qubit mappers (Jordan-Wigner vs Bravyi-Kitaev), all vs exact FCI. Designed to
EXPOSE where methods fail (HEA plateau / N-conservation, VQE convergence, mapper drift).

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework \
        /root/miniconda3/bin/python -m benchmarks.tier_C_solvers
"""
from __future__ import annotations
import time
import numpy as np
MHA = 1000.0


def fci_ref(atoms, basis, charge=0, spin=0):
    from pyscf import gto, scf, fci
    mol = gto.M(atom=atoms, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    e = fci.FCI(mf).kernel()[0]
    return float(mf.e_tot), float(e)


def via(atoms, basis, asp, solver, **kw):
    from kanad import MolecularBuilder
    b = MolecularBuilder.from_atoms(atoms).basis(basis)
    if asp == 'frozen_core':
        b = b.active_space('frozen_core')
    if 'mapper' in kw:
        b = b.mapper(kw.pop('mapper'))
    if 'ansatz' in kw:
        b = b.ansatz(kw.pop('ansatz'), **kw.pop('ansatz_kw', {}))
    qs = b.solver(solver, **kw).build()
    t0 = time.time()
    r = qs.solve()
    return float(r['energy']), qs.n_qubits, time.time() - t0


def main():
    print("=" * 110, flush=True)
    print("CAMPAIGN C — solvers × ansätze × mappers vs exact FCI", flush=True)
    print("=" * 110, flush=True)

    systems = [
        ('H2',   [('H', (0, 0, 0)), ('H', (0, 0, 0.74))], 'sto-3g', None),
        ('LiH',  [('Li', (0, 0, 0)), ('H', (0, 0, 1.595))], 'sto-3g', 'frozen_core'),
        ('H2O',  [('O', (0, 0, 0)), ('H', (0, 0.757, 0.587)), ('H', (0, -0.757, 0.587))], 'sto-3g', 'frozen_core'),
        ('BeH2', [('Be', (0, 0, 0)), ('H', (0, 0, 1.33)), ('H', (0, 0, -1.33))], 'sto-3g', 'frozen_core'),
    ]
    for name, atoms, basis, asp in systems:
        print(f"\n--- {name}/{basis}  active_space={asp} ---", flush=True)
        hf, fci = fci_ref(atoms, basis)
        print(f"  HF={hf:.6f}  FCI={fci:.6f}  (corr={(fci-hf)*MHA:.1f} mHa)", flush=True)
        trials = [
            ('CI (JW)',          'ci',  {}),
            ('CI (Bravyi-Kit.)', 'ci',  {'mapper': 'bravyi_kitaev'}),
            ('VQE givens_sd',    'vqe', {'ansatz': 'givens_sd', 'backend': 'statevector', 'max_iterations': 300}),
            ('VQE HEA(2L)',      'vqe', {'ansatz': 'hardware_efficient', 'ansatz_kw': {'n_layers': 2},
                                        'backend': 'statevector', 'max_iterations': 300}),
            ('SQD (statevec)',   'sqd', {'backend': 'statevector', 'n_samples': 8000, 'random_seed': 0}),
        ]
        for label, solver, kw in trials:
            try:
                e, nq, dt = via(atoms, basis, asp, solver, **kw)
                gap = (e - fci) * MHA
                flag = 'FCI' if abs(gap) < 1.6 else ('ok' if abs(gap) < 50 else 'OFF')
                print(f"  {label:18} E={e:.6f}  gap={gap:+8.2f} mHa  [{flag}] nq={nq} t={dt:.1f}s", flush=True)
            except Exception as ex:
                print(f"  {label:18} CRASH {type(ex).__name__}: {str(ex)[:70]}", flush=True)

    print("\nSOLVERS_DONE", flush=True)


if __name__ == "__main__":
    main()
