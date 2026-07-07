"""Density-matrix storage helpers used by every Hamiltonian class.

Two transforms recur in M3:

1. **Active-MO → full-MO embedding** — for active-space VQE (the only path
   that reaches Tier-1 chemical accuracy on systems past H₂). The active
   1-RDM is the VQE wavefunction's contribution; frozen orbitals are
   doubly occupied (closed-shell convention), virtuals are empty. Embedding
   reconstructs the full-MO 1-RDM whose trace equals the full electron count.

2. **MO → AO transform** — `D_AO = C · D_MO · Cᵀ` where `C = mf.mo_coeff`
   (AO × MO). PySCF property routines (`mol.intor('int1e_r')`, GIAO, etc.)
   consume AO-basis density; VQE produces MO-basis density. Failing to
   convert is the pre-M3 silent bug at `solvers/vqe_solver.py:2451` where
   an MO-basis 1-RDM was passed straight to `scf.hf.dip_moment(mol, dm)`.

Every Hamiltonian's `set_quantum_density_matrix()` calls these helpers so the
basis conversion lives in exactly one place and the stored density carries
explicit AO and full-MO copies for downstream property calculators.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def embed_active_to_full_mo(
    rdm_active: np.ndarray,
    frozen_indices: list,
    active_indices: list,
    n_mo_full: int,
) -> np.ndarray:
    """Embed an active-space 1-RDM into the full-MO 1-RDM.

    Closed-shell convention:
    - Frozen orbitals are doubly occupied: `D_full[i, i] = 2`.
    - Active block carries the VQE result.
    - Virtuals are empty: `D_full[v, v] = 0`.
    - Off-diagonal blocks are zero (canonical orbitals).

    The resulting trace equals `2·len(frozen) + tr(rdm_active)` which is
    `n_electrons_total` if `tr(rdm_active) = n_active_electrons`.

    Args:
        rdm_active: Active-space 1-RDM `(n_active, n_active)` in MO basis.
        frozen_indices: Indices of frozen-core orbitals (in the full MO list).
        active_indices: Indices of active orbitals (in the full MO list).
        n_mo_full: Total number of MOs (frozen + active + virtual).

    Returns:
        Full-MO 1-RDM `(n_mo_full, n_mo_full)`.
    """
    rdm_full = np.zeros((n_mo_full, n_mo_full), dtype=float)
    for i in frozen_indices:
        rdm_full[i, i] = 2.0
    act = np.asarray(active_indices, dtype=int)
    rdm_full[np.ix_(act, act)] = np.asarray(rdm_active, dtype=float)
    return rdm_full


def mo_to_ao_1rdm(rdm_mo: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """Transform a full-MO 1-RDM to AO basis: `D_AO = C · D_MO · Cᵀ`.

    PySCF's `mo_coeff` is shaped `(n_ao, n_mo)`; we use the same convention.
    The transform preserves trace ONLY in non-orthonormal AO bases when
    contracted with the overlap matrix `S`, but the AO 1-RDM in this form is
    what `mol.intor('int1e_r')` and `scf.hf.dip_moment` consume.

    Args:
        rdm_mo: Full-MO 1-RDM `(n_mo, n_mo)`.
        mo_coeff: AO×MO coefficient matrix `(n_ao, n_mo)`.

    Returns:
        AO-basis 1-RDM `(n_ao, n_ao)`.
    """
    rdm_mo = np.asarray(rdm_mo, dtype=float)
    C = np.asarray(mo_coeff, dtype=float)
    return C @ rdm_mo @ C.T


def validate_trace(
    rdm: np.ndarray,
    expected_trace: float,
    label: str,
    tol: float = 1e-4,
    overlap: Optional[np.ndarray] = None,
) -> None:
    """Raise `RuntimeError` if `|tr(rdm) − expected_trace| > tol`.

    For AO-basis 1-RDMs in a non-orthonormal AO basis, the meaningful
    invariant is `tr(D · S) = n_electrons`, so pass the overlap matrix.

    Args:
        rdm: 1-RDM to validate (any basis).
        expected_trace: Target (e.g. `n_electrons`).
        label: Human-readable name for the error message (e.g. ``"AO 1-RDM"``).
        tol: Absolute tolerance on `|tr − expected|`.
        overlap: If provided, validate `tr(D · S)` instead of `tr(D)`.
    """
    if overlap is not None:
        trace = float(np.trace(np.asarray(rdm) @ np.asarray(overlap)))
    else:
        trace = float(np.trace(np.asarray(rdm)))
    err = abs(trace - float(expected_trace))
    if err > tol:
        raise RuntimeError(
            f"{label} trace = {trace:.8f} (expected {expected_trace}); "
            f"deviation {err:.3e} exceeds tolerance {tol:.3e}. "
            "Likely causes: wrong basis (MO vs AO), wrong spin convention, "
            "or an ansatz that does not conserve particle number."
        )
