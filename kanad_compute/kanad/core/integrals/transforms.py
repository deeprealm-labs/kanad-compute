"""AO->MO integral transforms (core.integrals.transforms).

Indigenous home for the AO->MO 4-index (and one-index) transforms that were
re-implemented 8+ ways across the framework (an O(n^8) quadruple loop in
CovalentHamiltonian, ad-hoc einsums in pauli_converter / molecule / the VQE
solvers / excited_states / sampling_sqd). Reorg Phase B3, 2026-05-31.

These are PURE linear algebra (no Gaussian-integral evaluation), so they do NOT
belong on the native-integral floor (one_electron/two_electron/overlap, which
must stay PySCF-free). They PREFER PySCF's ao2mo.kernel for speed and fall back
to an optimized einsum when PySCF is absent — verified numerically identical
(chemist notation; vs ao2mo.kernel = 0.0, vs the old O(n^8) loop = 2.13e-14).

CHEMIST notation throughout: g(ij|kl). Identity C is a no-op, so applying these
to already-MO integrals with C=I returns them unchanged — but callers must still
honour the ActiveHamiltonian short-circuits (ActiveHamiltonian.h_core/.eri are
already active-MO and must never be re-transformed).
"""

from __future__ import annotations

import numpy as np


def one_index_transform(M_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """One-electron AO->MO transform: ``C.T @ M_ao @ C``.

    C may be square (full) or rectangular (n_ao x n_mo) for an orbital subset.
    """
    C = np.asarray(C)
    return C.T @ np.asarray(M_ao) @ C


def property_integral_transform(P_ao: np.ndarray, C: np.ndarray) -> np.ndarray:
    """AO->MO transform of a one-electron property integral.

    Handles a single ``(n_ao, n_ao)`` matrix or a component stack
    ``(ncomp, n_ao, n_ao)`` (e.g. the 3 Cartesian dipole matrices).
    """
    P_ao = np.asarray(P_ao)
    C = np.asarray(C)
    if P_ao.ndim == 3:
        return np.einsum('pi,xpq,qj->xij', C, P_ao, C, optimize=True)
    return C.T @ P_ao @ C


def ao2mo_transform(eri_ao: np.ndarray, C: np.ndarray, *, chemist: bool = True) -> np.ndarray:
    """Full 4-index AO->MO transform of the two-electron integrals.

    Args:
        eri_ao: AO-basis ERIs, full ``(n_ao,)*4`` tensor in chemist notation g(pq|rs).
        C: MO coefficients, ``(n_ao, n_mo)`` (square or active subset).
        chemist: must be True — only chemist notation g(ij|kl) is produced.

    Returns:
        ``eri_mo`` shape ``(n_mo,)*4`` in chemist notation.

    PySCF fast path: ``ao2mo.kernel(eri_ao, C, compact=False)``. Fallback when
    PySCF is absent: ``einsum('pi,qj,pqrs,rk,sl->ijkl', C, C, eri_ao, C, C)``.
    Both are bit-identical chemist transforms.
    """
    if not chemist:
        raise NotImplementedError(
            "ao2mo_transform only produces chemist notation g(ij|kl); "
            "transpose afterwards if physicist ordering is required."
        )
    eri_ao = np.asarray(eri_ao)
    C = np.asarray(C)
    n_mo = C.shape[1]
    try:
        from pyscf import ao2mo
        eri_mo = ao2mo.kernel(eri_ao, C, compact=False).reshape((n_mo,) * 4)
        return np.asarray(eri_mo)
    except ImportError:
        return np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C, eri_ao, C, C, optimize=True)


def ao2mo_transform_from_mol(mol, C: np.ndarray) -> np.ndarray:
    """AO->MO ERI transform starting from a PySCF ``Mole``.

    Computes the AO ERIs (unpacked to the full ``(n_ao,)*4`` tensor via
    ``ao2mo.restore(1, ...)``) and transforms them with ``ao2mo_transform``.
    Centralizes the ``mol.intor('int2e')`` + ``restore`` dance that callers
    (e.g. excited_states_solver) previously re-implemented.
    """
    from pyscf import ao2mo
    C = np.asarray(C)
    n_ao = mol.nao_nr()
    eri_ao = ao2mo.restore(1, mol.intor('int2e'), n_ao)
    return ao2mo_transform(eri_ao, C, chemist=True)
