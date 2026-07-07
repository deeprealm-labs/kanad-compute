"""Selected-CI / sampled RDMs in the active-space MO basis (core.density.sampled_rdm).

Indigenous home for building 1-/2-RDMs from a selected-CI eigenvector — a list of
Jordan-Wigner-interleaved determinant bitstrings plus their coefficients, as
produced by SamplingSQDSolver. Extracted from solvers/sampling_sqd.py
(reorg Phase B2, 2026-05-31) so the RDM-from-subspace logic is owned by core and
shared, not re-implemented in the solver.

Distinct from ``QuantumRDMExtractor`` (core.density.quantum_rdm), which reads a
full dense statevector and a fermion->qubit mapper. Here the coefficients live on
an explicit determinant subspace; we embed them into PySCF's full FCI layout
(zeros at non-sampled positions) and use ``direct_spin1.make_rdm1`` /
``make_rdm12`` for the canonical Slater-Condon sum.

JW spin convention: alpha at even qubit 2p, beta at odd 2p+1. Determinants outside
the requested (n_a, n_b) sector are dropped during the embed.
"""

from __future__ import annotations

import logging

import numpy as np

from kanad.core.ci.slater_condon import _split_alpha_beta

logger = logging.getLogger(__name__)


def embed_ci_vector(determinants, coeffs, n_orb: int, n_a: int, n_b: int) -> np.ndarray:
    """Embed (determinants, coeffs) into PySCF's (n_a, n_b) full FCI tensor, normalized.

    Determinants outside the (n_a, n_b) sector are dropped. Raises ``RuntimeError``
    if nothing lands in-sector (zero norm).
    """
    from pyscf.fci import cistring
    strs_a = cistring.make_strings(range(n_orb), n_a)
    strs_b = cistring.make_strings(range(n_orb), n_b)
    a_idx = {int(s): i for i, s in enumerate(strs_a)}
    b_idx = {int(s): i for i, s in enumerate(strs_b)}

    ci_full = np.zeros((len(strs_a), len(strs_b)))
    for d, c in zip(determinants, coeffs):
        a, b = _split_alpha_beta(int(d), n_orb)
        if a in a_idx and b in b_idx:
            ci_full[a_idx[a], b_idx[b]] = c

    nrm = float(np.linalg.norm(ci_full))
    if nrm < 1e-12:
        raise RuntimeError(
            "sampled_rdm: embedded CI vector has zero norm "
            "(no sampled determinant fell in the (n_a, n_b) sector)."
        )
    ci_full /= nrm
    return ci_full


def rdm1_from_ci_vector(determinants, coeffs, n_orb: int, n_a: int, n_b: int,
                        n_e_expected=None) -> np.ndarray:
    """Spin-summed 1-RDM (n_orb x n_orb) from a selected-CI eigenvector.

    Trace equals the number of (active) electrons. If ``n_e_expected`` is given,
    a trace mismatch beyond 1e-4 emits a warning (does not raise).
    """
    from pyscf.fci import direct_spin1
    ci_full = embed_ci_vector(determinants, coeffs, n_orb, n_a, n_b)
    rdm1 = direct_spin1.make_rdm1(ci_full, n_orb, (n_a, n_b))
    if n_e_expected is not None:
        tr = float(np.trace(rdm1))
        if abs(tr - n_e_expected) > 1e-4:
            logger.warning(
                "rdm1_from_ci_vector: trace %.6f differs from n_active_e %s by %.2e",
                tr, n_e_expected, tr - n_e_expected,
            )
    return rdm1


def rdm12_from_ci_vector(determinants, coeffs, n_orb: int, n_a: int, n_b: int):
    """Spin-summed (1-RDM, 2-RDM) in chemist notation from a selected-CI eigenvector.

    ``rdm2[p, q, r, s] = <psi| a_p^dag a_r^dag a_s a_q |psi>``.
    """
    from pyscf.fci import direct_spin1
    ci_full = embed_ci_vector(determinants, coeffs, n_orb, n_a, n_b)
    return direct_spin1.make_rdm12(ci_full, n_orb, (n_a, n_b))
