"""Pure harmonic vibrational analysis.

Turns a Cartesian Hessian into vibrational frequencies, normal modes, reduced masses,
force constants and the zero-point energy. This is the single source of truth for the
harmonic math, shared by:

- the solver ``hessian`` capability (``solvers.capabilities.FiniteDifferenceHessianMixin``),
  which finite-differences the *quantum* nuclear gradient into a Hessian, and
- ``analysis.vibrational_analysis.FrequencyCalculator``, the classical/analysis entry point.

It is deliberately free of any Molecule / solver / PySCF dependency — it takes plain arrays,
so both a wavefunction Hessian and an HF Hessian flow through identical, tested math.

Units contract (matches ``solvers/capabilities.py``):
    hessian     (3N, 3N)  Ha / Bohr²   (Cartesian second derivatives of the energy)
    atoms_bohr  (N, 3)     Bohr
    masses_amu  (N,)        amu
Returns frequencies in cm⁻¹ (negative = imaginary), normal_modes (3N, n_vib),
reduced_masses in amu, force_constants in mdyn/Å, zpe in Hartree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Physical constants (CODATA 2018) — identical to FrequencyCalculator so the two
# code paths agree to the last digit.
_Ha_to_J = 4.3597447222071e-18      # Hartree -> Joule
_Bohr_to_m = 5.29177210903e-11      # Bohr -> metre
_amu_to_kg = 1.66053906660e-27      # amu -> kg
_c_cm_s = 2.99792458e10             # speed of light, cm/s
_h = 6.62607015e-34                 # Planck constant, J·s


def is_linear(atoms_bohr: np.ndarray, masses_amu: np.ndarray, *, tol: float = 1e-3) -> bool:
    """True if the nuclei are collinear (smallest principal moment of inertia ≈ 0).

    Diatomics and single atoms are linear by definition.
    """
    atoms = np.asarray(atoms_bohr, dtype=float)
    n = atoms.shape[0]
    if n <= 2:
        return True
    masses = np.asarray(masses_amu, dtype=float)
    com = (masses[:, None] * atoms).sum(axis=0) / masses.sum()
    rel = atoms - com
    inertia = np.zeros((3, 3))
    for pos, m in zip(rel, masses):
        inertia[0, 0] += m * (pos[1] ** 2 + pos[2] ** 2)
        inertia[1, 1] += m * (pos[0] ** 2 + pos[2] ** 2)
        inertia[2, 2] += m * (pos[0] ** 2 + pos[1] ** 2)
        inertia[0, 1] -= m * pos[0] * pos[1]
        inertia[0, 2] -= m * pos[0] * pos[2]
        inertia[1, 2] -= m * pos[1] * pos[2]
    inertia[1, 0], inertia[2, 0], inertia[2, 1] = inertia[0, 1], inertia[0, 2], inertia[1, 2]
    moments = np.linalg.eigvalsh(inertia)
    return bool(moments[0] < tol * moments[2])


def harmonic_analysis(
    hessian: np.ndarray,
    atoms_bohr: np.ndarray,
    masses_amu: np.ndarray,
    *,
    linear: Optional[bool] = None,
) -> dict:
    """Diagonalize the mass-weighted Hessian and return the harmonic spectrum.

    Args:
        hessian: (3N, 3N) Cartesian Hessian in Ha/Bohr².
        atoms_bohr: (N, 3) nuclear positions in Bohr (for translation/rotation projection).
        masses_amu: (N,) atomic masses in amu, in the SAME atom order as ``atoms_bohr``.
        linear: override linearity detection; if None it is inferred from the geometry.

    Returns a dict with:
        frequencies_cm     (n_vib,)      cm⁻¹, ascending by |ν|, negative = imaginary
        normal_modes       (3N, n_vib)   mass-weighted eigenvectors
        reduced_masses_amu (n_vib,)      amu
        force_constants     (n_vib,)      mdyn/Å
        zpe_ha             float          Hartree (real modes only)
        n_imaginary        int
    """
    H = np.asarray(hessian, dtype=float)
    masses = np.asarray(masses_amu, dtype=float)
    n_atoms = masses.shape[0]
    dim = 3 * n_atoms
    if H.shape != (dim, dim):
        raise ValueError(f"hessian shape {H.shape} inconsistent with {n_atoms} atoms (expected {(dim, dim)})")

    masses_per_dof = np.repeat(masses, 3)  # (3N,) amu

    # Mass-weight: H̃_ij = H_ij / √(m_i m_j)
    H_mw = H / np.sqrt(np.outer(masses_per_dof, masses_per_dof))
    H_mw = 0.5 * (H_mw + H_mw.T)  # enforce symmetry before eigh

    eigenvalues, eigenvectors = np.linalg.eigh(H_mw)

    # Expected number of internal (vibrational) modes.
    if linear is None:
        linear = is_linear(atoms_bohr, masses)
    if n_atoms == 1:
        n_vib = 0
    elif linear:
        n_vib = 3 * n_atoms - 5
    else:
        n_vib = 3 * n_atoms - 6
    n_vib = max(n_vib, 0)

    empty = {
        'frequencies_cm': np.array([]),
        'normal_modes': np.zeros((dim, 0)),
        'reduced_masses_amu': np.array([]),
        'force_constants': np.array([]),
        'zpe_ha': 0.0,
        'n_imaginary': 0,
    }
    if n_vib == 0:
        return empty

    # Vibrational modes carry the largest-|eigenvalue| — sorting by magnitude keeps a
    # genuine imaginary (most-negative) transition-state mode instead of discarding it
    # in favour of a near-zero translation/rotation.
    order = np.argsort(np.abs(eigenvalues))[::-1][:n_vib]
    vib_vals = eigenvalues[order]
    vib_vecs = eigenvectors[:, order]

    # Eigenvalue (Ha/Bohr²·amu⁻¹) -> frequency (cm⁻¹), sign preserved for imaginaries.
    vals_SI = vib_vals * (_Ha_to_J / _Bohr_to_m ** 2) / _amu_to_kg  # s⁻²
    freqs = np.where(
        vals_SI >= 0.0,
        np.sqrt(np.abs(vals_SI)) / (2 * np.pi * _c_cm_s),
        -np.sqrt(np.abs(vals_SI)) / (2 * np.pi * _c_cm_s),
    )

    # Present ascending by magnitude.
    srt = np.argsort(np.abs(freqs))
    freqs = freqs[srt]
    vib_vecs = vib_vecs[:, srt]
    vib_vals = vib_vals[srt]

    # Physical Cartesian reduced mass: μ_k = 1 / Σ_i (L_ik/√m_i)².
    cartesian = vib_vecs / np.sqrt(masses_per_dof)[:, None]
    reduced_masses = 1.0 / np.sum(cartesian ** 2, axis=0)  # amu

    # Force constant k = λ (N/m) -> mdyn/Å (1 mdyn/Å = 100 N/m).
    force_constants = vib_vals * (_Ha_to_J / _Bohr_to_m ** 2) * 1e-2

    # ZPE = Σ ½hν over real modes.
    real = freqs[freqs > 0]
    zpe_J = np.sum(0.5 * _h * (real * _c_cm_s))
    zpe_ha = float(zpe_J / _Ha_to_J)

    return {
        'frequencies_cm': freqs,
        'normal_modes': vib_vecs,
        'reduced_masses_amu': reduced_masses,
        'force_constants': force_constants,
        'zpe_ha': zpe_ha,
        'n_imaginary': int(np.sum(freqs < 0)),
    }
