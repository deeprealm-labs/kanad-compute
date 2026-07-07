"""One-electron property integrals (core.integrals.property_integrals).

Indigenous home for the AO-basis one-electron *property* integrals — electric
dipole, 1/|r-R| (for diamagnetic NMR shielding), and r^2 — that were re-derived
from raw ``mol.intor`` calls in ~12 places across analysis/builder/excited-states
(reorg Phase B3.2, 2026-05-31).

Thin PySCF dispatchers, mirroring the existing inline calls bit-for-bit (the NMR
gate asserts <1e-6 vs ``mol.intor('int1e_rinv')``, so PySCF stays the backend).
A native fallback is intentionally NOT provided here — callers that have their own
native branch keep it; these helpers require a PySCF ``Mole``.
"""

from __future__ import annotations

import numpy as np


def compute_dipole(mol, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """AO electric-dipole integrals ``<p|(r - origin)|q>``, shape ``(3, nao, nao)``.

    Mirrors ``mol.intor('int1e_r')`` with the origin folded in via the overlap S
    (``dip[i] -= origin[i] * S``), matching the prior inline implementation.
    """
    dip = np.asarray(mol.intor('int1e_r'))  # (3, nao, nao)
    origin = np.asarray(origin, dtype=float)
    if not np.allclose(origin, 0.0):
        S = mol.intor('int1e_ovlp')
        for i in range(3):
            dip[i] = dip[i] - origin[i] * S
    return dip


def compute_rinv(mol, origin) -> np.ndarray:
    """AO ``<p| 1/|r - origin| |q>`` integrals, shape ``(nao, nao)``.

    Used by diamagnetic NMR shielding (Lamb term). Mirrors
    ``with mol.with_rinv_origin(origin): mol.intor('int1e_rinv')``.
    """
    with mol.with_rinv_origin(tuple(origin)):
        return np.asarray(mol.intor('int1e_rinv'))


def compute_r2(mol, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    """AO ``<p|(r - origin)^2|q>`` integrals, shape ``(nao, nao)``.

    Mirrors ``with mol.with_common_orig(origin): mol.intor('int1e_r2')``.
    """
    with mol.with_common_orig(tuple(origin)):
        return np.asarray(mol.intor('int1e_r2'))
