"""1-RDM extraction from a converged VQE statevector.

The 1-RDM element is `ρ_pq = ⟨ψ| a†_p a_q |ψ⟩`. For closed-shell systems we
sum over the two spin sectors: `ρ^{spatial}_pq = ρ^α_pq + ρ^β_pq`.

JW spin convention (matches `core.operators.jordan_wigner`):
- spin-α at qubit `2*p`
- spin-β at qubit `2*p + 1`

The implementation builds the Pauli image of each `a†_p a_q` once at
construction time, then computes the matrix element as a single
`statevector.expectation_value(op)` per (p, q) pair — `O(n_orbitals²)`
expectation values total.

Trace validation: `|trace(ρ) − n_electrons| < tol`. The pre-M3
implementation in `solvers/vqe_solver.py` had this off by ~35% on HeH⁺
(reported trace 2.71 vs target 2.0); this module catches that class of
bug at the source by raising `RuntimeError` rather than silently
proceeding to a wrong dipole/NMR/IR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector

from kanad.core.operators.fermion_operator import FermionOperator
from kanad.core.operators.jordan_wigner import jordan_wigner


class QuantumRDMExtractor:
    """Extract the spatial 1-RDM (and optionally 2-RDM) from a VQE statevector.

    Parameters
    ----------
    n_orbitals : int
        Number of spatial orbitals.
    n_electrons : int
        Total electron count (for trace validation).
    mapper : str
        Fermion→qubit mapper. Only ``'jordan_wigner'`` is supported in v1;
        the JW spin convention is α at even qubits, β at odd. Other mappers
        (e.g. ``bravyi_kitaev``) will be added when a use case demands it.
    trace_tol : float
        Maximum allowed deviation of ``|trace(ρ) − n_electrons|`` before the
        extractor raises. Default 1e-4 — matched to the 2-RDM trace check and
        chosen so optimizer/float noise from a well-converged statevector
        (~1e-6) passes, while genuine particle-non-conservation (O(0.01–1), e.g.
        a barren-plateaued hardware-efficient ansatz) is still rejected.
    """

    def __init__(
        self,
        n_orbitals: int,
        n_electrons: int,
        mapper: str = 'jordan_wigner',
        trace_tol: float = 1e-4,
    ):
        if mapper.lower() not in ('jordan_wigner', 'jw'):
            raise NotImplementedError(
                f"QuantumRDMExtractor v1 supports only Jordan-Wigner; got mapper={mapper!r}. "
                "Bravyi-Kitaev support is a follow-up if needed for hardware tapering."
            )
        self.n_orbitals = int(n_orbitals)
        self.n_electrons = int(n_electrons)
        self.n_qubits = 2 * self.n_orbitals
        self.trace_tol = float(trace_tol)

        # Pre-compute the JW image of every (p, q) excitation operator
        # summed over spin. This is `O(n²)` Pauli operators each with `O(n)`
        # Pauli terms; for small actives (n ≤ 10) this is microseconds.
        self._rdm_paulis: dict[tuple[int, int], SparsePauliOp] = {}
        for p in range(self.n_orbitals):
            for q in range(self.n_orbitals):
                self._rdm_paulis[(p, q)] = self._build_spatial_excitation_op(p, q)

    # ----- public API ---------------------------------------------------

    def extract_2rdm(self, statevector) -> np.ndarray:
        """Return the spatial 2-RDM `(n_orb, n_orb, n_orb, n_orb)` in chemist's
        notation: ``D₂[p, q, r, s] = ⟨ψ| a†_p a†_r a_s a_q |ψ⟩`` summed over
        the four spin combinations (αα, αβ, βα, ββ).

        This is the convention PySCF uses (`mcscf.CASCI.fcisolver.make_rdm12`).
        The energy reconstructed via

        ::

            E = E_nuc + Σ_pq h_pq D₁[p,q] + ½ Σ_pqrs g_pqrs D₂[p,q,r,s]

        equals the VQE energy to machine precision when the wavefunction and
        the 1/2-RDMs all come from the same statevector.

        Cost is `O(n_orb⁴)` SparsePauliOp expectation values — manageable for
        ``n_orb ≤ 10``. For larger active spaces, prefer cumulant-truncated
        representations (M4 work).

        Returns:
            ``np.ndarray`` of shape ``(n, n, n, n)`` (real-valued, symmetric
            under particle-exchange).
        """
        sv = statevector if isinstance(statevector, Statevector) else Statevector(statevector)
        if sv.num_qubits != self.n_qubits:
            raise ValueError(
                f"Statevector has {sv.num_qubits} qubits, but extractor expects "
                f"{self.n_qubits} (2 × {self.n_orbitals} spatial orbitals)."
            )

        n = self.n_orbitals
        if not hasattr(self, '_rdm2_paulis'):
            self._rdm2_paulis = {}
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            self._rdm2_paulis[(p, q, r, s)] = self._build_spatial_2rdm_op(p, q, r, s)

        d2 = np.zeros((n, n, n, n), dtype=float)
        for (p, q, r, s), op in self._rdm2_paulis.items():
            d2[p, q, r, s] = float(sv.expectation_value(op).real)
        self._validate_2rdm_trace(d2)
        return d2

    def extract_1rdm(self, statevector) -> np.ndarray:
        """Return the spatial 1-RDM `(n_orbitals, n_orbitals)` in MO basis.

        Args:
            statevector: Qiskit `Statevector` or a 1-D numpy array on
                `2**n_qubits` amplitudes (Qiskit little-endian).

        Returns:
            Real-valued symmetric matrix `ρ_pq`. Trace equals
            `n_electrons` to within `trace_tol`; else raises.
        """
        sv = statevector if isinstance(statevector, Statevector) else Statevector(statevector)
        if sv.num_qubits != self.n_qubits:
            raise ValueError(
                f"Statevector has {sv.num_qubits} qubits, but extractor expects "
                f"{self.n_qubits} (2 × {self.n_orbitals} spatial orbitals)."
            )

        rho = np.zeros((self.n_orbitals, self.n_orbitals), dtype=float)
        for (p, q), op in self._rdm_paulis.items():
            rho[p, q] = float(sv.expectation_value(op).real)

        self._validate_trace(rho)
        # Hermitize (the matrix is real-symmetric by construction; numerical
        # asymmetry comes from the order of statevector multiplications).
        return 0.5 * (rho + rho.T)

    # ----- internals ----------------------------------------------------

    def _build_spatial_excitation_op(self, p: int, q: int) -> SparsePauliOp:
        """Build the JW image of (a†_pα a_qα + a†_pβ a_qβ) as a SparsePauliOp."""
        ops = []
        for spin_offset in (0, 1):  # 0 = α at even qubit, 1 = β at odd
            p_spin = 2 * p + spin_offset
            q_spin = 2 * q + spin_offset
            ferm = FermionOperator(((p_spin, 1), (q_spin, 0)))
            d = jordan_wigner(ferm, n_qubits=self.n_qubits)
            if not d:
                continue
            ops.append(SparsePauliOp(list(d.keys()), list(d.values())))
        if not ops:
            return SparsePauliOp(['I' * self.n_qubits], [0.0])
        combined = ops[0]
        for o in ops[1:]:
            combined = combined + o
        return combined.simplify()

    def _build_spatial_2rdm_op(self, p: int, q: int, r: int, s: int) -> SparsePauliOp:
        """Build the JW image of `Σ_{στ} a†_pσ a†_rτ a_sτ a_qσ` as a SparsePauliOp.

        Chemist's notation: the operator's expectation value is
        ``D₂[p, q, r, s] = ⟨ψ| a†_p a†_r a_s a_q |ψ⟩`` (spin-summed).
        """
        ops = []
        for sigma in (0, 1):
            for tau in (0, 1):
                p_s = 2 * p + sigma
                r_s = 2 * r + tau
                s_s = 2 * s + tau
                q_s = 2 * q + sigma
                ferm = FermionOperator(((p_s, 1), (r_s, 1), (s_s, 0), (q_s, 0)))
                d = jordan_wigner(ferm, n_qubits=self.n_qubits)
                if not d:
                    continue
                ops.append(SparsePauliOp(list(d.keys()), list(d.values())))
        if not ops:
            return SparsePauliOp(['I' * self.n_qubits], [0.0])
        combined = ops[0]
        for o in ops[1:]:
            combined = combined + o
        return combined.simplify()

    def _validate_2rdm_trace(self, d2: np.ndarray) -> None:
        """Standard chemist's-notation trace identity:
        ``Σ_pq D₂[p, p, q, q] = N(N − 1)`` (= <N̂(N̂−1)>).

        Derivation: ``D₂[p, p, q, q] = ⟨a†_p a†_q a_q a_p⟩`` (since
        ``D₂[p,q,r,s] = ⟨a†_p a†_r a_s a_q⟩``); summing over p, q gives
        ``⟨Σ_pq a†_p a†_q a_q a_p⟩ = ⟨Σ_p a†_p N̂ a_p⟩ = ⟨N̂(N̂−1)⟩``.
        """
        n = d2.shape[0]
        trace = float(sum(d2[p, p, q, q] for p in range(n) for q in range(n)))
        expected = self.n_electrons * (self.n_electrons - 1)
        err = abs(trace - expected)
        if err > 1e-4:
            raise RuntimeError(
                f"2-RDM trace Σ_pq D₂[p,p,q,q] = {trace:.8f} "
                f"(expected N(N-1) = {expected}); deviation {err:.3e}."
            )

    def _validate_trace(self, rho: np.ndarray) -> None:
        trace = float(np.trace(rho))
        err = abs(trace - self.n_electrons)
        if err > self.trace_tol:
            raise RuntimeError(
                f"Quantum 1-RDM trace = {trace:.8f} (n_electrons = {self.n_electrons}); "
                f"deviation {err:.3e} exceeds tolerance {self.trace_tol:.3e}. "
                "Likely causes: (a) the wavefunction is not particle-conserving "
                "(check ansatz_type — use 'givens_sd' for chemistry), or (b) the "
                "JW spin convention assumed by the extractor (α at even qubit, β at "
                "odd) does not match the convention the ansatz was built against."
            )


# ---------------------------------------------------------------------------
# Energy from RDMs (independent verification path)
# ---------------------------------------------------------------------------

def energy_from_rdms(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    h_core: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion: float,
) -> float:
    """Total electronic energy from spatial 1-RDM and 2-RDM in chemist's notation.

    ::

        E = E_nuc + Σ_pq h_pq D₁[p,q] + ½ Σ_pqrs g_pqrs D₂[p,q,r,s]

    where ``g_pqrs = (pq|rs)`` (chemist's notation) and ``D₂[p,q,r,s] =
    ⟨a†_p a†_r a_s a_q⟩`` (PySCF spatial convention).

    For an active-space VQE solve, pass the active-space ``h_core`` (=h_eff),
    ``eri`` (=g_eff), and ``nuclear_repulsion`` (=E_inactive) — the result is
    the **total** energy including the frozen-core contribution baked into
    E_inactive.
    """
    e_one = float(np.einsum('pq,pq->', h_core, rdm1))
    e_two = 0.5 * float(np.einsum('pqrs,pqrs->', eri, rdm2))
    return float(nuclear_repulsion) + e_one + e_two


def spin_squared_from_statevector(statevector, n_orbitals: int) -> float:
    """``⟨Ŝ²⟩`` from a JW statevector via direct Pauli measurement.

    Builds the JW image of ``Ŝ² = Ŝ_z² + ½(Ŝ_+ Ŝ_- + Ŝ_- Ŝ_+)`` once and
    measures it. Unambiguous — no spin-RDM convention ambiguity. For a
    closed-shell singlet ``⟨Ŝ²⟩ = 0`` to numerical precision; deviation
    signals spin-symmetry breaking in the ansatz.

    JW convention: α at even qubits, β at odd qubits.
    """
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from kanad.core.operators.fermion_operator import FermionOperator
    from kanad.core.operators.jordan_wigner import jordan_wigner

    sv = statevector if isinstance(statevector, Statevector) else Statevector(statevector)
    n_qubits = 2 * n_orbitals

    # S² operator now lives in core.operators.spin_operators (the single
    # indigenous home; this was its original construction site). (reorg B4)
    from kanad.core.operators.spin_operators import build_spin_operators
    _, _, s_squared = build_spin_operators(n_orbitals, 'jordan_wigner')
    return float(sv.expectation_value(s_squared).real)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def extract_1rdm_from_statevector(
    statevector,
    n_orbitals: int,
    n_electrons: int,
    mapper: str = 'jordan_wigner',
    trace_tol: float = 1e-4,
) -> np.ndarray:
    """One-shot extraction: build the extractor and return the 1-RDM."""
    return QuantumRDMExtractor(
        n_orbitals=n_orbitals,
        n_electrons=n_electrons,
        mapper=mapper,
        trace_tol=trace_tol,
    ).extract_1rdm(statevector)


# ---------------------------------------------------------------------------
# M4 D3 — Natural orbitals + nonidempotency diagnostic (2026-05-28)
# Solver-agnostic: takes any 1-RDM, gives chemistry-relevant indicators.
# ---------------------------------------------------------------------------


def compute_natural_orbital_occupations(rdm1_mo: np.ndarray) -> np.ndarray:
    """Natural orbital occupation numbers from a spin-summed 1-RDM (MO basis).

    Eigenvalues of the spin-summed 1-RDM are the natural orbital occupation
    numbers ``n_i ∈ [0, 2]``. For single-reference closed-shell systems,
    the spectrum is ``{2, 2, ..., 2, 0, 0, ...}`` (integer occupations).
    Fractional occupations (away from 0 or 2) signal **static correlation**
    — the more spread the spectrum, the more multireference the state.

    Returns:
        np.ndarray sorted descending — ``n_i ∈ [0, 2]`` per orbital.
    """
    rdm1_mo = np.asarray(rdm1_mo, dtype=float)
    # Symmetrize before eigval to suppress numerical noise that would
    # produce small imaginary components
    rdm1_sym = 0.5 * (rdm1_mo + rdm1_mo.T)
    eigvals = np.linalg.eigvalsh(rdm1_sym)
    # Clip tiny numerical drift outside [0, 2]
    eigvals = np.clip(eigvals, 0.0, 2.0)
    # Sort descending (highest-occupation first)
    return np.sort(eigvals)[::-1]


def compute_nonidempotency_diagnostic(rdm1_mo: np.ndarray) -> float:
    """Nonidempotency diagnostic for multireference character.

    ``D = ½ Σ_i n_i · (2 - n_i)``  where ``n_i`` are natural-orbital
    occupation numbers from the spin-summed 1-RDM. This is a measure of how
    far the 1-RDM is from idempotent (single-determinant) form.

    NOTE: This is NOT Truhlar's M-diagnostic. Truhlar's M-diagnostic
    (Tishchenko/Truhlar 2008-2009) is a half-sum over frontier donor/acceptor
    natural-orbital occupations ``½(Σ(2 - n_donor) + Σ n_acceptor)``, which
    differs in form from this global nonidempotency measure; the 0.05/0.1
    thresholds quoted for the M-diagnostic do not transfer to this quantity.

    Interpretation:
        - ``D ≈ 0``  → single-reference (NOs near 0 or 2)
        - larger ``D`` → more static (multireference) correlation

    For an ideal closed-shell singlet with ``n_i ∈ {0, 2}``, ``D = 0``
    exactly (every term ``n_i(2-n_i) = 0``). The most multireference state
    has all ``n_i = 1`` giving the maximum ``D = ½ × n_orb``.

    Returns:
        Scalar nonidempotency value (always ≥ 0; monotonic MR indicator).
    """
    occs = compute_natural_orbital_occupations(rdm1_mo)
    return float(0.5 * np.sum(occs * (2.0 - occs)))


# Backward-compatible alias: prior name mislabeled this as Truhlar's
# M-diagnostic. Kept so existing callers/imports keep working.
compute_m_diagnostic = compute_nonidempotency_diagnostic


def compute_n_unpaired_electrons(rdm1_mo: np.ndarray) -> float:
    """Head-Gordon's number of effectively unpaired electrons.

    ``n_u = Σ_i min(n_i, 2 - n_i)``

    Counts how many NOs are far from 0 or 2. Closed-shell single-ref → ~0.
    Singlet biradical → ~2. Triplet → ~2 (high-spin), etc.
    Distinct from the nonidempotency diagnostic in shape; both useful as MR indicators.
    """
    occs = compute_natural_orbital_occupations(rdm1_mo)
    return float(np.sum(np.minimum(occs, 2.0 - occs)))
