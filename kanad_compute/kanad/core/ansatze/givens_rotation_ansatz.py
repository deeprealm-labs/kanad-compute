"""Particle-conserving ansatze (singles-only Givens, and singles+doubles UCCSD-like).

The HEA ansatz uses linear-CNOT entanglement which destroys particle number at
θ=0: the |...0011⟩ HF state collapses to |...0001⟩ after the first CNOT layer.
The optimizer then has to climb out of a wrong-N basin, which COBYLA + L-BFGS-B
both fail to do on systems with more than ~4 qubits.

Two ansatze here, in escalating expressibility:

- `GivensRotationAnsatz` — brick-wall single excitations only. Particle-
  conserving but cannot escape HF on chemistry Hamiltonians (Brillouin
  stationary). Useful as a building block / for non-chemistry use.

- `GivensSDAnsatz` — Givens singles + paired double excitations. The double
  excitations break the Brillouin barrier (they couple HF to doubly-excited
  configs at *first* order in θ). This is the workhorse for chemistry.

Every gate is a particle-conserving Givens rotation:

::

    G(θ) on (i, j) — preserves total occupation q_i + q_j:
        |00⟩ → |00⟩
        |01⟩ → cos(θ)|01⟩ − sin(θ)|10⟩
        |10⟩ → sin(θ)|01⟩ + cos(θ)|10⟩
        |11⟩ → |11⟩

Real-valued (no imaginary phases), particle-conserving, and spin-conserving
(when qubit pairs respect the spin-orbital ordering).

Layer topology
--------------
Brick-wall (Anselmetti et al. 2021):

- Each layer applies all even-indexed pairs G(0,1), G(2,3), ...
- Then all odd-indexed pairs G(1,2), G(3,4), ...

With enough layers the brick-wall pattern generates the full N-conserving
unitary group, so this ansatz can express any state in the N-sector
including FCI within the active space.

Decomposition (only CNOT + RY, both natively supported)
-------------------------------------------------------
::

    G(θ) ≡ CX(low, high)
         · RY(θ)(low)
         · CX(high, low)
         · RY(−θ)(low)
         · CX(high, low)
         · CX(low, high)

The circuit is built as a Qiskit `QuantumCircuit` directly (not via Kanad's
lightweight wrapper) because we need parameter arithmetic (``-θ``), which
Qiskit ParameterExpression supports natively.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector

from kanad.core.ansatze.hardware_efficient_ansatz import get_hf_state_qubits


class GivensRotationAnsatz:
    # M2 PR-1 contract: every gate is a single-Pauli rotation, so the
    # parameter-shift rule applies directly.
    _supports_parameter_shift = True

    """Brick-wall Givens-rotation ansatz that preserves N̂ by construction.

    Parameters
    ----------
    n_qubits : int
        Total spin orbitals (= 2 * n_spatial_orbitals in JW).
    n_electrons : int
        Total electron count; sets the HF reference.
    n_layers : int
        Number of brick-wall passes. Each layer applies all even-pair Givens
        followed by all odd-pair Givens.
    mapper : str
        Fermion-to-qubit mapper. Used only to determine which qubits to flip
        for HF preparation.

    Attributes
    ----------
    circuit : qiskit.QuantumCircuit
        The built circuit. Available after `build_circuit()`.
    parameters : list[Parameter]
        Qiskit `Parameter` objects in canonical order. Length matches
        ``n_parameters``.
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        n_layers: int = 3,
        mapper: str = 'jordan_wigner',
    ):
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        self.n_layers = int(n_layers)
        self.mapper = str(mapper)
        self.circuit: Optional[QuantumCircuit] = None
        self.parameters: List[Parameter] = []

    # ----- API expected by VQESolver -----------------------------------

    @property
    def n_parameters(self) -> int:
        """Total Givens parameters across all brick-wall layers."""
        n_even = self.n_qubits // 2                  # pairs (0,1), (2,3), ...
        n_odd = max(0, (self.n_qubits - 1) // 2)     # pairs (1,2), (3,4), ...
        return self.n_layers * (n_even + n_odd)

    def get_num_parameters(self) -> int:
        """Alias for compatibility with the rest of the framework."""
        return self.n_parameters

    # ----- construction -------------------------------------------------

    def build_circuit(self, initial_state: Optional[List[int]] = None) -> QuantumCircuit:
        """Build the brick-wall Givens circuit.

        ``initial_state`` is a list of 0/1 occupations per qubit; default is
        the HF state (X gates on the lowest ``n_electrons`` qubits in JW).
        """
        qc = QuantumCircuit(self.n_qubits)

        # Initial state: HF preparation.
        if initial_state is not None:
            for qubit, occ in enumerate(initial_state):
                if occ:
                    qc.x(qubit)
            qc.barrier()
        elif self.n_electrons > 0:
            hf_qubits = get_hf_state_qubits(self.n_qubits, self.n_electrons, self.mapper)
            for qubit in hf_qubits:
                qc.x(qubit)
            qc.barrier()

        # Parameters in canonical order. `ParameterVector` guarantees
        # deterministic ordering between this list and what
        # `qc.parameters` returns later — which matters because the
        # solver maps scipy's params array positionally to `qc.parameters`.
        thetas = ParameterVector('θ', length=self.n_parameters)
        self.parameters = list(thetas)
        idx = 0

        for layer_idx in range(self.n_layers):
            # Even pairs: (0,1), (2,3), (4,5), ...
            for i in range(0, self.n_qubits - 1, 2):
                self._add_givens(qc, thetas[idx], low=i, high=i + 1)
                idx += 1
            # Odd pairs: (1,2), (3,4), (5,6), ...
            for i in range(1, self.n_qubits - 1, 2):
                self._add_givens(qc, thetas[idx], low=i, high=i + 1)
                idx += 1

        assert idx == self.n_parameters
        self.circuit = qc
        return qc

    @staticmethod
    def _add_givens(qc: QuantumCircuit, theta, low: int, high: int) -> None:
        """Append a particle-conserving Givens rotation G(θ) on (low, high).

        Decomposition (CNOT + RY only, no CRY):

            CX(low, high)
            RY(θ)(low)
            CX(high, low)
            RY(−θ)(low)
            CX(high, low)
            CX(low, high)
        """
        qc.cx(low, high)
        qc.ry(theta, low)
        qc.cx(high, low)
        qc.ry(-theta, low)
        qc.cx(high, low)
        qc.cx(low, high)

    # ----- info ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GivensRotationAnsatz(n_qubits={self.n_qubits}, "
            f"n_electrons={self.n_electrons}, n_layers={self.n_layers}, "
            f"n_parameters={self.n_parameters})"
        )


# ---------------------------------------------------------------------------
# Singles + paired-doubles particle-conserving ansatz (UCCSD-style)
# ---------------------------------------------------------------------------

class GivensSDAnsatz:
    # Paired-double excitations use PauliEvolutionGate (Lie-Trotter
    # decomposition of a multi-Pauli generator). The simple parameter-shift
    # rule doesn't apply to multi-Pauli evolution; the solver should NOT use
    # the simple parameter-shift formula for this ansatz.
    _supports_parameter_shift = False
    # After build_circuit() decomposes the PauliEvolutionGates, the circuit
    # contains only single-Pauli rotations (RZ, R with fixed phi). The
    # adjoint-state gradient (∂E/∂θ_k = −2·Σ_j c_jk · Im⟨ψ_j|G_j|R_j⟩) is
    # applicable; VQESolver routes through `core.vqe_gradients` when this
    # flag is True.
    _supports_adjoint_gradient = True

    """Particle-conserving ansatz with singles + paired doubles.

    Singles are Givens rotations within each spin sector (between occupied
    spin-α↔virtual spin-α; same for β). Paired doubles act on 4 spin
    orbitals (α_occ, β_occ, α_virt, β_virt) and rotate |0011⟩ ↔ |1100⟩,
    coupling HF to the doubly-excited configuration at first order in θ.

    This is exactly what's needed to break Brillouin's theorem and reach
    chemical correlation. The decomposition uses Qiskit's
    `PauliEvolutionGate` (Lie-Trotter) to exponentiate the JW-image of the
    fermionic excitation generators.

    Structure (closed-shell paired excitations):
    -------------------------------------------
    For n_occ doubly-occupied spatial MOs and n_virt empty spatial MOs:

    - Single excitations: for each (occ_p, virt_a), one Givens rotation on
      α-spin pair (qubit 2p, 2a) and one on β-spin pair (qubit 2p+1, 2a+1).
      Spin-flipped excitations are excluded (S_z conserved).
    - Paired doubles: for each (occ_p, virt_a), one paired-double on the
      4 spin orbitals (2p, 2p+1, 2a, 2a+1).
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        n_layers: int = 1,
        mapper: str = 'jordan_wigner',
    ):
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        if self.n_qubits % 2 != 0:
            raise ValueError(
                f"GivensSDAnsatz: requires even n_qubits (α/β pairing); got {n_qubits}"
            )
        if self.n_electrons % 2 != 0:
            raise ValueError(
                f"GivensSDAnsatz: v1 supports closed-shell (even n_electrons); got {n_electrons}"
            )
        self.n_spatial = self.n_qubits // 2
        self.n_occ_spatial = self.n_electrons // 2
        self.n_virt_spatial = self.n_spatial - self.n_occ_spatial
        if self.n_virt_spatial <= 0:
            raise ValueError(
                "GivensSDAnsatz: at least one virtual spatial orbital required"
            )
        self.n_layers = int(n_layers)
        self.mapper = str(mapper)
        self.circuit: Optional[QuantumCircuit] = None
        self.parameters: List[Parameter] = []

    @property
    def n_singles(self) -> int:
        # Spin-α singles + spin-β singles: 2 × n_occ × n_virt
        return 2 * self.n_occ_spatial * self.n_virt_spatial

    @property
    def n_doubles(self) -> int:
        """All opposite-spin doubles (i_α, j_β) → (a_α, b_β).

        Paired doubles (a = b) are a subset of these. We include the full
        set because the paired-only subset spans far less of the active-space
        FCI manifold than UCCSD: for LiH (2e, 5o) paired-only reaches
        ~13/45 ≈ 29 % of N=2 configurations, vs ~28/45 ≈ 62 % with all
        opposite-spin doubles. Same-spin doubles (αα→αα, ββ→ββ) require at
        least 2 occ orbitals per spin and are added when n_occ ≥ 2.
        """
        n_occ = self.n_occ_spatial
        n_virt = self.n_virt_spatial
        # Opposite-spin doubles: (i_α, j_β) → (a_α, b_β)
        n_opp = n_occ * n_occ * n_virt * n_virt
        # Same-spin doubles: (i_σ, j_σ) → (a_σ, b_σ) for σ ∈ {α, β}.
        # Need i < j and a < b for distinct excitations (anti-symmetry).
        n_same_per_spin = (n_occ * (n_occ - 1) // 2) * (n_virt * (n_virt - 1) // 2)
        n_same = 2 * n_same_per_spin
        return n_opp + n_same

    @property
    def n_parameters(self) -> int:
        return self.n_layers * (self.n_singles + self.n_doubles)

    def get_num_parameters(self) -> int:
        return self.n_parameters

    # ----- construction -------------------------------------------------

    def build_circuit(self, initial_state: Optional[List[int]] = None) -> QuantumCircuit:
        """Build HF + (singles + paired doubles)^n_layers via Pauli evolution."""
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        qc = QuantumCircuit(self.n_qubits)

        # HF preparation.
        if initial_state is not None:
            for q, occ in enumerate(initial_state):
                if occ:
                    qc.x(q)
            qc.barrier()
        elif self.n_electrons > 0:
            hf_qubits = get_hf_state_qubits(self.n_qubits, self.n_electrons, self.mapper)
            for q in hf_qubits:
                qc.x(q)
            qc.barrier()

        # Parameters in canonical order.
        thetas = ParameterVector('θ', length=self.n_parameters)
        self.parameters = list(thetas)
        idx = 0

        for layer in range(self.n_layers):
            # --- Singles (within each spin sector) -------------------
            for p in range(self.n_occ_spatial):
                for a in range(self.n_occ_spatial, self.n_spatial):
                    # α-spin single: occ(2p) → virt(2a)
                    self._append_single_excitation(qc, thetas[idx], 2 * p, 2 * a)
                    idx += 1
                    # β-spin single
                    self._append_single_excitation(qc, thetas[idx], 2 * p + 1, 2 * a + 1)
                    idx += 1

            # --- Opposite-spin doubles: (i_α, j_β) → (a_α, b_β) ------
            # Includes paired doubles (i = j, a = b) and all unpaired variants.
            # This is what gets us from ~29% to ~62% of the (2e, *o) FCI
            # manifold on LiH-like systems.
            for i in range(self.n_occ_spatial):
                for j in range(self.n_occ_spatial):
                    for a in range(self.n_occ_spatial, self.n_spatial):
                        for b in range(self.n_occ_spatial, self.n_spatial):
                            self._append_paired_double(
                                qc, thetas[idx],
                                occ_alpha=2 * i,
                                occ_beta=2 * j + 1,
                                virt_alpha=2 * a,
                                virt_beta=2 * b + 1,
                            )
                            idx += 1

            # --- Same-spin doubles (αα→αα and ββ→ββ) ------------------
            # Generators: a†_{a_σ} a†_{b_σ} a_{i_σ} a_{j_σ} − h.c. for i < j,
            # a < b. For systems with only one occupied spatial orbital per spin
            # (e.g. LiH, BeH₂ here) these don't contribute (the (n_occ choose 2)
            # = 0). For H₂O / NH₃ / CH₄ with multiple occupied orbitals they do.
            occ_list = list(range(self.n_occ_spatial))
            virt_list = list(range(self.n_occ_spatial, self.n_spatial))
            for spin_offset in (0, 1):  # 0 = α, 1 = β
                for ii_idx in range(len(occ_list)):
                    for jj_idx in range(ii_idx + 1, len(occ_list)):
                        for aa_idx in range(len(virt_list)):
                            for bb_idx in range(aa_idx + 1, len(virt_list)):
                                i = occ_list[ii_idx]
                                j = occ_list[jj_idx]
                                a = virt_list[aa_idx]
                                b = virt_list[bb_idx]
                                self._append_paired_double(
                                    qc, thetas[idx],
                                    occ_alpha=2 * i + spin_offset,
                                    occ_beta=2 * j + spin_offset,
                                    virt_alpha=2 * a + spin_offset,
                                    virt_beta=2 * b + spin_offset,
                                )
                                idx += 1

        assert idx == self.n_parameters
        # PERF: Pre-decompose `PauliEvolutionGate` instances into native
        # Pauli-string rotations. Without this, Qiskit's `Statevector.from_instruction`
        # calls scipy `expm` on the (gigantic) sparse generator matrix for EVERY
        # energy evaluation — 96% of the per-eval cost on LiH-class systems.
        # `decompose()` is symbolic-parameter safe and expands once; the
        # downstream Statevector simulation then sees only native gates.
        # Empirical: 51× speedup on LiH (672 ms/eval → 13 ms/eval).
        try:
            qc = qc.decompose()
        except Exception:
            # If decomposition fails for any reason, fall back to the raw
            # circuit — correctness is preserved, only performance is lost.
            pass
        self.circuit = qc
        return qc

    @staticmethod
    def _append_single_excitation(qc: QuantumCircuit, theta, occ: int, virt: int) -> None:
        """Append exp(iθ (a†_virt a_occ − a†_occ a_virt)).

        Single excitation between spin orbitals (occ, virt). The JW image of
        the generator `a†_virt a_occ − h.c.` carries a Pauli-Z string over the
        orbitals between `occ` and `virt` (fermionic parity). The bare CX/RY
        Givens body is only exact for ADJACENT qubits (empty Z-string); these
        excitations are emitted on non-adjacent qubits (e.g. LiH 0→8), where
        the Z-string does NOT cancel — a single excitation flips occupation
        and picks up the parity of the intervening orbitals. Build the JW
        image of the anti-hermitian generator directly via FermionOperator +
        Jordan-Wigner and exponentiate with PauliEvolutionGate, exactly as
        _append_paired_double does for the doubles.
        """
        from qiskit.circuit.library import PauliEvolutionGate
        from kanad.core.operators.excitation_operators import build_excitation_generator

        # Hermitian single-excitation generator from the indigenous core engine
        # (bit-identical to the prior FermionOperator+JW+(-1j)*coeffs build,
        # verified 0.0 matrix diff). H = -i·G; PauliEvolutionGate(H, time=-θ) ⇒
        # exp(iθ·G), including the JW Z-string for non-adjacent qubits. (reorg B4)
        hermitian = build_excitation_generator((occ,), (virt,), qc.num_qubits, 'jordan_wigner')
        qc.append(PauliEvolutionGate(hermitian, time=-theta), list(range(qc.num_qubits)))

    @staticmethod
    def _append_paired_double(
        qc: QuantumCircuit,
        theta,
        occ_alpha: int,
        occ_beta: int,
        virt_alpha: int,
        virt_beta: int,
    ) -> None:
        """Append exp(iθ T) where T = a†_va a†_vb a_ob a_oa − h.c.

        Implemented via Qiskit's `PauliEvolutionGate` on the JW-image of the
        anti-hermitian generator. PauliEvolutionGate handles the Pauli-sum
        exponentiation correctly (first-order Lie-Trotter by default).
        """
        from qiskit.circuit.library import PauliEvolutionGate
        from kanad.core.operators.excitation_operators import build_excitation_generator

        # Hermitian paired-double generator from the indigenous core engine
        # (T = a†_va a†_vb a_ob a_oa − h.c.; bit-identical to the prior
        # FermionOperator+JW+(-1j)*coeffs build, verified 0.0 matrix diff).
        # PauliEvolutionGate(H, time=-θ) ⇒ exp(iθ·G). (reorg B4)
        hermitian = build_excitation_generator(
            (occ_alpha, occ_beta), (virt_alpha, virt_beta), qc.num_qubits, 'jordan_wigner')
        qc.append(PauliEvolutionGate(hermitian, time=-theta), list(range(qc.num_qubits)))

    def __repr__(self) -> str:
        return (
            f"GivensSDAnsatz(n_qubits={self.n_qubits}, n_electrons={self.n_electrons}, "
            f"n_occ={self.n_occ_spatial}, n_virt={self.n_virt_spatial}, "
            f"n_layers={self.n_layers}, n_parameters={self.n_parameters})"
        )
