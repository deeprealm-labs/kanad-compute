"""Campaign A — FCI-exact validation. Full-space Kanad SQD (statevector) + CI vs
EXACT FCI (pyscf), in the SAME basis. No active-space truncation, so any gap is
TRUE sampling/method completeness — the real distance to exact-in-basis.

    cd /root/kanad-framework && PYTHONPATH=/tmp/kanad-pkg:/root/kanad-framework /root/miniconda3/bin/python -m benchmarks.tier_A_fci
"""
from __future__ import annotations
import time, traceback
import numpy as np

# (name, atoms, basis, charge, spin)  — full-space, sto-3g
SYS = [
    ('H2',    [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))], 'sto-3g', 0, 0),
    ('HeH+',  [('He', (0, 0, 0)), ('H', (0, 0, 0.7743))], 'sto-3g', 1, 0),
    ('LiH',   [('Li', (0, 0, 0)), ('H', (0, 0, 1.5949))], 'sto-3g', 0, 0),
    ('BH',    [('B', (0, 0, 0)), ('H', (0, 0, 1.2324))], 'sto-3g', 0, 0),
    ('NH_triplet', [('N', (0, 0, 0)), ('H', (0, 0, 1.036))], 'sto-3g', 0, 2),
    ('H2O',   [('O', (0, 0, 0)), ('H', (0, 0.7572, 0.5865)), ('H', (0, -0.7572, 0.5865))], 'sto-3g', 0, 0),
    ('N2',    [('N', (0, 0, 0)), ('N', (0, 0, 1.0977))], 'sto-3g', 0, 0),
    ('C2',    [('C', (0, 0, 0)), ('C', (0, 0, 1.2425))], 'sto-3g', 0, 0),
    ('Li2',   [('Li', (0, 0, 0)), ('Li', (0, 0, 2.673))], 'sto-3g', 0, 0),
    ('LiF',   [('Li', (0, 0, 0)), ('F', (0, 0, 1.5639))], 'sto-3g', 0, 0),
]


def fci_energy(atoms, basis, charge, spin):
    from pyscf import gto, scf, fci, ao2mo
    mol = gto.M(atom=[(e, tuple(x)) for e, x in atoms], basis=basis, charge=charge, spin=spin, verbose=0)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).run(verbose=0)
    norb = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norb)
    nelec = mol.nelectron
    na = (nelec + spin) // 2; nb = nelec - na
    solver = fci.direct_spin0 if spin == 0 else fci.direct_spin1
    e, _ = solver.kernel(h1, eri, norb, (na, nb), ecore=float(mol.energy_nuc()))
    return float(e), float(mf.e_tot)


def run():
    from kanad import MolecularBuilder
    print("=" * 110, flush=True)
    print("CAMPAIGN A — FCI-EXACT validation (full-space SQD/CI vs exact FCI, same basis)", flush=True)
    print("=" * 110, flush=True)
    results = []
    for name, atoms, basis, charge, spin in SYS:
        r = {'name': name, 'spin': spin}
        t0 = time.time()
        try:
            e_fci, e_hf = fci_energy(atoms, basis, charge, spin)
            r['fci'] = e_fci; r['hf'] = e_hf
            b = MolecularBuilder.from_atoms(atoms).basis(basis)
            if charge: b = b.charge(charge)
            if spin: b = b.spin(spin)
            b = b.active_space('full')
            # CI route (classical exact-in-subspace)
            try:
                eci = b.solver('ci').build().solve()['energy']
                r['ci'] = eci; r['ci_gap_mha'] = (eci - e_fci) * 1000
            except Exception as e:
                r['ci_err'] = f"{type(e).__name__}:{str(e)[:45]}"
            # SQD statevector (true sampling completeness)
            try:
                sk = dict(n_samples=50000, max_iterations=4, recovery_rounds=2, random_seed=0)
                if spin: sk['spin_s'] = spin / 2.0
                bb = MolecularBuilder.from_atoms(atoms).basis(basis)
                if charge: bb = bb.charge(charge)
                if spin: bb = bb.spin(spin)
                out = bb.active_space('full').solver('sqd', **sk).build().solve()
                r['sqd'] = out['energy']; r['sqd_gap_mha'] = (out['energy'] - e_fci) * 1000
                r['sqd_dets'] = out.get('n_determinants'); r['sqd_s2'] = out.get('s_squared')
            except Exception as e:
                r['sqd_err'] = f"{type(e).__name__}:{str(e)[:45]}"
            r['status'] = 'ok'
        except Exception as e:
            r['status'] = 'crash'; r['error'] = f"{type(e).__name__}: {str(e)[:90]}"
        r['t'] = round(time.time() - t0, 1)
        results.append(r)
        # one line
        s = f"AFCI| {name:12} (S={spin}) | FCI={r.get('fci', float('nan')):.6f}"
        if 'ci_gap_mha' in r: s += f" | CI_gap={r['ci_gap_mha']:+.4f}mHa"
        if 'sqd_gap_mha' in r: s += f" | SQD_gap={r['sqd_gap_mha']:+.4f}mHa (dets={r.get('sqd_dets')},S2={r.get('sqd_s2')})"
        for k in ('ci_err', 'sqd_err', 'error'):
            if k in r: s += f" | {k}={r[k]}"
        s += f" | t={r['t']}s"
        print(s, flush=True)
    print("AFCI_DONE", flush=True)
    return results


if __name__ == "__main__":
    run()
