"""Response spectroscopy from the correlated wavefunction: polarizability tensor and
Raman activities, both driven by the solver ``field_response`` capability (Phase 3) and
the ``hessian`` capability (Phase 1).

- ``polarizability_tensor(solver, atoms)``  — full symmetric α_ij = −∂²E/∂E_i∂E_j (a.u.)
  by finite field over ``solver.energy_under_field``.
- ``raman_spectrum(solver, atoms)``         — per-mode Raman activity S = 45ᾱ'² + 7γ'² and
  depolarization ratio, from ∂α/∂Q along the quantum-Hessian normal modes (a nested
  field × geometry finite difference).

The classic validation this unlocks: H₂ is IR-INACTIVE (no dipole derivative, ν≈0 km/mol)
but Raman-ACTIVE (its polarizability changes along the stretch) — the two spectroscopies are
complementary, and only a correlated α captures the right activity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def polarizability_tensor(solver, atoms_bohr: np.ndarray, *, field_au: float = 0.005,
                          warm_state=None) -> np.ndarray:
    """Full symmetric static dipole polarizability tensor (3×3, a.u.) at ``atoms_bohr``.

    α_ii = −(E(+F_i) + E(−F_i) − 2E₀)/F²   (diagonal, central 2nd derivative)
    α_ij = −(E(++) − E(+−) − E(−+) + E(−−))/(4F²)   (off-diagonal, mixed field)

    Requires ``solver`` to implement the ``field_response`` capability
    (``energy_under_field(atoms, e_field, b_field)``). 19 energy evaluations.
    """
    def ef(fx, fy, fz):
        return float(solver.energy_under_field(atoms_bohr, [fx, fy, fz], None,
                                               warm_state=warm_state)[0])
    F = float(field_au)
    e0 = ef(0, 0, 0)
    alpha = np.zeros((3, 3))
    axes = [(F, 0, 0), (0, F, 0), (0, 0, F)]
    for i, ax in enumerate(axes):
        ep = ef(*ax)
        em = ef(*(-v for v in ax))
        alpha[i, i] = -(ep + em - 2 * e0) / (F * F)
    for i in range(3):
        for j in range(i + 1, 3):
            def fld(si, sj):
                v = [0.0, 0.0, 0.0]; v[i] = si * F; v[j] = sj * F; return ef(*v)
            aij = -(fld(+1, +1) - fld(+1, -1) - fld(-1, +1) + fld(-1, -1)) / (4 * F * F)
            alpha[i, j] = alpha[j, i] = aij
    return alpha


def _placzek_invariants(dalpha: np.ndarray) -> tuple:
    """Placzek invariants of a polarizability derivative tensor ∂α/∂Q (3×3).

    Returns (mean ᾱ', anisotropy² γ'², Raman activity S = 45ᾱ'² + 7γ'²,
    depolarization ratio ρ = 3γ'²/(45ᾱ'² + 4γ'²))."""
    a_mean = np.trace(dalpha) / 3.0
    dxx, dyy, dzz = dalpha[0, 0], dalpha[1, 1], dalpha[2, 2]
    gamma2 = 0.5 * ((dxx - dyy) ** 2 + (dyy - dzz) ** 2 + (dzz - dxx) ** 2) \
        + 3.0 * (dalpha[0, 1] ** 2 + dalpha[1, 2] ** 2 + dalpha[0, 2] ** 2)
    activity = 45.0 * a_mean ** 2 + 7.0 * gamma2
    denom = 45.0 * a_mean ** 2 + 4.0 * gamma2
    rho = (3.0 * gamma2 / denom) if denom > 1e-30 else 0.0
    return float(a_mean), float(gamma2), float(activity), float(rho)


def raman_spectrum(solver, atoms_bohr: np.ndarray, *, field_au: float = 0.005,
                   q_step_bohr: float = 0.02, hessian_result=None) -> dict:
    """Per-mode Raman activities from ∂α/∂Q along the quantum-Hessian normal modes.

    For each real vibrational mode k with (mass-weighted, normalized) eigenvector L_k, the
    Cartesian displacement per unit normal coordinate is Δx = M^(−1/2) L_k. The polarizability
    tensor is finite-differenced along ±Δx, giving ∂α/∂Q_k, then reduced to the Placzek
    invariants. This is a nested finite difference (field × geometry) — expensive; gate on size.

    Args:
        solver: must implement ``field_response`` (energy_under_field) and ``hessian``.
        atoms_bohr: (N,3) geometry (Bohr); should be a stationary point for meaningful modes.
        field_au: electric-field amplitude for the polarizability finite difference.
        q_step_bohr: normal-coordinate displacement amplitude.
        hessian_result: optional precomputed HessianResult (skips re-running the Hessian).

    Returns dict with ``frequencies_cm``, ``raman_activities`` (45ᾱ'²+7γ'², a.u.),
    ``depolarization_ratios``, ``mean_derivatives``, ``anisotropy2`` and ``source='quantum'``.
    """
    hr = hessian_result if hessian_result is not None else solver.hessian(atoms_bohr)
    freqs = np.asarray(hr.frequencies_cm, dtype=float)
    modes = np.asarray(hr.normal_modes, dtype=float)   # (3N, n_modes), mass-weighted, normalized
    masses = np.asarray(solver._hessian_masses_amu(), dtype=float)
    sqrt_m_inv = 1.0 / np.sqrt(np.repeat(masses, 3))   # (3N,)

    atoms = np.asarray(atoms_bohr, dtype=float)
    n_atoms = atoms.shape[0]
    activities, rhos, means, aniso = [], [], [], []
    for k in range(modes.shape[1]):
        dx = (modes[:, k] * sqrt_m_inv).reshape(n_atoms, 3)   # Cartesian displacement / unit Q
        a_plus = polarizability_tensor(solver, atoms + q_step_bohr * dx, field_au=field_au)
        a_minus = polarizability_tensor(solver, atoms - q_step_bohr * dx, field_au=field_au)
        dalpha = (a_plus - a_minus) / (2.0 * q_step_bohr)
        a_mean, gamma2, activity, rho = _placzek_invariants(dalpha)
        activities.append(activity); rhos.append(rho); means.append(a_mean); aniso.append(gamma2)

    return {
        'frequencies_cm': freqs,
        'raman_activities': np.array(activities),
        'depolarization_ratios': np.array(rhos),
        'mean_derivatives': np.array(means),
        'anisotropy2': np.array(aniso),
        'source': 'quantum',
    }
