"""M10 scale test — density / ESP .cube export on a large molecule (benzene).

Benzene/cc-pVDZ is 114 AOs (vs 24 for H2O): a real test that the wavefunction-
derived cube export scales. Solves a frontier active space through the builder,
writes density + ESP cubes, and checks the density integrates to the electron
count. Run on the cluster.

    python -m benchmarks.m10_cube_scale
"""

import sys
import numpy as np

from kanad import MolecularBuilder


def benzene_atoms():
    """D6h benzene: C-C 1.39 Å, C-H 1.09 Å, planar."""
    r_c, r_h = 1.39, 1.39 + 1.09
    atoms = []
    for k in range(6):
        th = np.deg2rad(60 * k)
        atoms.append(('C', (r_c * np.cos(th), r_c * np.sin(th), 0.0)))
    for k in range(6):
        th = np.deg2rad(60 * k)
        atoms.append(('H', (r_h * np.cos(th), r_h * np.sin(th), 0.0)))
    return atoms


def _integrate_cube(path):
    with open(path) as f:
        lines = f.readlines()
    natm = abs(int(lines[2].split()[0]))
    vox = np.array([[float(x) for x in lines[3 + i].split()[1:4]] for i in range(3)])
    dV = abs(np.linalg.det(vox))
    vals = []
    for ln in lines[6 + natm:]:
        vals += [float(x) for x in ln.split()]
    return sum(vals) * dV


def main():
    atoms = benzene_atoms()
    n_elec = 6 * 6 + 6 * 1   # 42 electrons
    print(f'Benzene/cc-pVDZ — {len(atoms)} atoms, {n_elec} electrons', flush=True)
    qs = (MolecularBuilder.from_atoms(atoms).basis('cc-pvdz')
          .active_space('frontier', n_occ=3, n_virt=3).solver('ci').build())
    n_ao = qs.mf.mo_coeff.shape[0]
    print(f'  AOs: {n_ao} | active qubits: {qs.n_qubits}', flush=True)
    res = qs.solve()
    print(f'  E = {res["energy"]:.6f} Ha (route {res["solver"]})', flush=True)

    df = qs.export_cube('/tmp/benzene_density.cube', kind='density', nx=80, ny=80, nz=80)
    integ = _integrate_cube(df)
    print(f'  density cube: ∫ρ = {integ:.3f} (expect ~{n_elec})', flush=True)
    ef = qs.export_cube('/tmp/benzene_esp.cube', kind='esp', nx=60, ny=60, nz=60)

    obs = qs.observables('core')
    print(f'  observables: |mu|={obs["dipole_magnitude_debye"]:.4f} D (D6h → ~0), '
          f'M-diag={obs["m_diagnostic"]:.4f}', flush=True)
    ok = abs(integ - n_elec) < 1.0 and obs['dipole_magnitude_debye'] < 0.05
    print('M10_CUBE_SCALE_OK' if ok else 'M10_CUBE_SCALE_CHECK', flush=True)


if __name__ == '__main__':
    main()
