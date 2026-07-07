"""LUCJ ansatz — Local Unitary Cluster Jastrow (M4-A).

Reference:
- Motta *et al.*, *Quantum-classical efficient computational chemistry on
  noisy quantum hardware*, NPJ Quantum Information (2023).
- Robledo-Moreno *et al.*, Nature 638, 87 (2025) — used LUCJ on 77-qubit
  Heron r3 for [2Fe-2S].

Why this exists (vs `GivensSDAnsatz`):
- `GivensSDAnsatz` depth scales as ``O(n_orb²)`` with paired-double
  PauliEvolution gates. For NH₃ cc-pVDZ (14 qubits) the transpiled circuit on
  IBM's heavy-hex connectivity is depth ~62 000 with ~18 000 two-qubit gates.
  At IBM Heron r3's ~10⁻³ 2q error rate, the per-shot success probability is
  ``(1 − 1e-3)^18000 ≈ 1.5e-8`` — i.e., < 1 in 10⁵ samples survives noise.
- LUCJ depth is ``O(n_orb)`` per layer. For the same NH₃ system: ~170
  two-qubit gates per layer → per-shot success ~85% with 1 layer. Sampling
  efficiency improves by **~10⁷×**.

The trade-off: LUCJ is less expressive per parameter than Givens-SD, but for
*sampling-based* SQD we don't need a VQE-optimal wavefunction — we need a
distribution that spreads weight across the dominant determinants. LUCJ
gives that with minimal noise budget.

Structure per layer:
1. Orbital rotation: same-spin nearest-neighbor Givens rotations
   ``exp(iθ_pq (a†_{pσ} a_{qσ} − h.c.))`` for σ ∈ {α, β} on adjacent
   spatial orbitals (p, p+1). N and S_z exactly preserved.
2. Number-number Jastrow: ``exp(iK_pq n_p n_q)`` on nearest-neighbor
   qubits. Diagonal in computational basis → N + S_z exactly preserved.
3. Inverse orbital rotation: same as step 1 with parameter signs flipped.

For closed-shell singlets (n_α = n_β), HF reference is the spin-paired
configuration ``|α₀ β₀ α₁ β₁ ...⟩`` (interleaved JW).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


class LUCJAnsatz:
    """Local Unitary Cluster Jastrow — depth ``O(n_orb)`` per layer.

    Particle (N) and spin-z (S_z) are exactly preserved by construction.
    Designed for hardware-efficient sampling rather than VQE optimization.

    Parameters
    ----------
    n_qubits : int
        2 × n_spatial. Must be even (interleaved α/β convention).
    n_electrons : int
        Total electron count. Must be even (closed-shell singlet).
    n_layers : int
        Number of LUCJ blocks. Each block = orbital → Jastrow → inverse
        orbital. More layers = broader sampling distribution at the cost
        of depth. v1 default = 1.
    mapper : str
        Only ``'jordan_wigner'`` is supported.
    """

    _supports_parameter_shift = False  # PauliEvolution → multi-Pauli generator
    _supports_adjoint_gradient = True

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        n_layers: int = 1,
        mapper: str = 'jordan_wigner',
        target_sz: float = 0.0,
    ):
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        # Per-spin occupation for HF init: closed-shell (Sz=0) gives
        # n_alpha = n_beta = n_e/2; triplet (Sz=1) gives n_alpha = n_e/2+1.
        # Sz can be half-integer for doublets etc.
        self.target_sz = float(target_sz)
        self.n_alpha = (self.n_electrons + int(round(2 * self.target_sz))) // 2
        self.n_beta = self.n_electrons - self.n_alpha
        if self.n_qubits % 2 != 0:
            raise ValueError(
                f"LUCJAnsatz: requires even n_qubits (α/β pairing); got {n_qubits}"
            )
        if self.n_alpha < 0 or self.n_beta < 0:
            raise ValueError(
                f"LUCJAnsatz: invalid (n_α, n_β) = ({self.n_alpha}, {self.n_beta}) "
                f"for n_e={n_electrons}, target_sz={target_sz}"
            )
        if mapper.lower() not in ('jordan_wigner', 'jw'):
            raise NotImplementedError(
                f"LUCJAnsatz v1 supports only Jordan-Wigner; got mapper={mapper!r}"
            )
        self.n_spatial = self.n_qubits // 2
        # Upper-bound guard: occupation cannot exceed the active orbital count.
        # Without this, an over-filled active space (e.g. a manual open-shell space
        # whose occupied orbitals below the window were not frozen, so n_electrons is
        # too large for n_spatial) reaches `qc.x(2*p)` with 2*p >= n_qubits and dies
        # with a cryptic qiskit `CircuitError: Index N out of range`. Fail clearly here.
        if self.n_alpha > self.n_spatial or self.n_beta > self.n_spatial:
            raise ValueError(
                f"LUCJAnsatz: occupation exceeds orbital count — (n_α, n_β) = "
                f"({self.n_alpha}, {self.n_beta}) but only n_spatial={self.n_spatial} "
                f"active orbitals ({self.n_qubits} qubits). The active space is "
                f"over-filled: n_active_electrons={self.n_electrons} is too large for "
                f"{self.n_spatial} orbitals. For a manual (especially open-shell) active "
                f"space, freeze ALL occupied orbitals below the active window so the "
                f"active electron count matches the active orbitals.")
        self.n_occ_spatial = self.n_electrons // 2
        self.n_layers = int(n_layers)
        self.mapper = 'jordan_wigner'
        self.circuit: Optional[QuantumCircuit] = None
        self.parameters: List[Parameter] = []

    # ----- circuit construction -----------------------------------------

    def _orbital_rotation_generator(self, p_sp: int, q_sp: int, spin: int) -> SparsePauliOp:
        """JW image of the Hermitian generator ``i(a†_pσ a_qσ − a†_qσ a_pσ)``.

        Used as the generator of an orbital-rotation Givens between two
        same-spin orbitals.
        """
        # Spin-orbital qubit indices in JW interleaved convention
        q_p = 2 * p_sp + spin
        q_q = 2 * q_sp + spin
        low, high = min(q_p, q_q), max(q_p, q_q)

        # The single-excitation generator i(a†_p a_q − h.c.) is the same Hermitian
        # Pauli operator the indigenous core builder produces for occ=(high,),
        # virt=(low,): G = (1/2)(X_low Z..Z Y_high − Y_low Z..Z X_high). Verified
        # bit-identical (to_matrix) across all (p,q,spin,n_qubits) — see B-audit #17.
        from kanad.core.operators.excitation_operators import build_excitation_generator
        return build_excitation_generator((high,), (low,), self.n_qubits, 'jordan_wigner')

    def _add_givens(self, qc: QuantumCircuit, theta: Parameter,
                    p_sp: int, q_sp: int, spin: int) -> None:
        """Apply a particle-conserving Givens rotation between two same-spin
        orbitals via the PauliEvolution of the orbital-rotation generator."""
        gen = self._orbital_rotation_generator(p_sp, q_sp, spin)
        qc.append(PauliEvolutionGate(gen, time=theta), range(self.n_qubits))

    def _add_jastrow_zz(self, qc: QuantumCircuit, theta: Parameter,
                        q_a: int, q_b: int) -> None:
        """Add a number-number Jastrow term ``exp(i θ n_a n_b)``.

        n_p = (I − Z_p) / 2 →  n_a n_b = (I − Z_a − Z_b + Z_a Z_b)/4

        exp(i θ n_a n_b) (up to global phase, which doesn't affect physics):
            Rz(θ/2) on q_a
            Rz(θ/2) on q_b
            Rzz(-θ/2) on (q_a, q_b)

        Audit H17: the ZZ term is emitted as the native CX·Rz·CX form rather
        than `rzz`, so the *only* parameterized gate in the Jastrow is `Rz`.
        `AdjointGradientCalculator` has no generator for `rzz` (it would
        silently zero every Jastrow gradient); `Rz` is a supported single-Pauli
        rotation, so the adjoint path differentiates K_jas correctly.
        Rzz(φ) = CX(a,b)·Rz(φ, b)·CX(a,b); here φ = −θ/2.
        """
        qc.rz(theta / 2.0, q_a)
        qc.rz(theta / 2.0, q_b)
        qc.cx(q_a, q_b)
        qc.rz(-theta / 2.0, q_b)
        qc.cx(q_a, q_b)

    def build_circuit(self, initial_state: Optional[List[int]] = None) -> QuantumCircuit:
        """Build the LUCJ circuit. Returns a Qiskit ``QuantumCircuit``."""
        qc = QuantumCircuit(self.n_qubits)
        self.parameters = []

        # 1. HF reference (interleaved JW: α at even q=2p, β at odd q=2p+1).
        # For closed-shell (n_α = n_β): lowest n_e spin orbitals occupied.
        # For open-shell (e.g. triplet n_α=4, n_β=2): set α orbitals at
        # qubits 0, 2, 4, 6 and β orbitals at qubits 1, 3.
        if initial_state is None:
            for p in range(self.n_alpha):
                qc.x(2 * p)         # α at even qubit
            for p in range(self.n_beta):
                qc.x(2 * p + 1)     # β at odd qubit
        else:
            for i, occ in enumerate(initial_state):
                if occ:
                    qc.x(i)

        for layer in range(self.n_layers):
            # Forward orbital rotations (nearest-neighbor spatial orbital pairs)
            orbital_params_forward = []
            for spin in (0, 1):  # α then β
                for p_sp in range(self.n_spatial - 1):
                    theta = Parameter(f'θ_orb_{layer}_{"α" if spin == 0 else "β"}_{p_sp}')
                    self.parameters.append(theta)
                    orbital_params_forward.append((theta, p_sp, spin))
                    self._add_givens(qc, theta, p_sp, p_sp + 1, spin)

            # Diagonal Jastrow on nearest-neighbor QUBIT pairs
            # (includes both same-spin and cross-spin nearest pairs)
            for q in range(self.n_qubits - 1):
                k = Parameter(f'K_jas_{layer}_{q}')
                self.parameters.append(k)
                self._add_jastrow_zz(qc, k, q, q + 1)

            # Inverse orbital rotations (same Givens with negated parameters,
            # in reverse order)
            for theta, p_sp, spin in reversed(orbital_params_forward):
                # exp(-iθ G) = exp(iθ (-G)); we just use -theta below
                # We don't introduce a new Parameter — reuse `theta` and negate.
                gen = self._orbital_rotation_generator(p_sp, p_sp + 1, spin)
                qc.append(PauliEvolutionGate(gen, time=-theta), range(self.n_qubits))

        # Audit H17: with _supports_adjoint_gradient=True, VQESolver routes the
        # gradient through AdjointGradientCalculator, which (a) crashes on a raw
        # PauliEvolutionGate and (b) cannot differentiate a multi-Pauli generator
        # anyway. Decompose ONLY the PauliEvolutionGates (orbital rotations) into
        # native single-Pauli rotations (Rz + Clifford), exactly as GivensSDAnsatz
        # does. We deliberately do NOT use a blanket qc.decompose(): that lowers
        # standalone Rz → P (phase gate), which the adjoint calculator also can't
        # differentiate. The Jastrow ZZ is already emitted as CX·Rz·CX (see
        # _add_jastrow_zz), so after this step every parameterized gate is an Rz.
        try:
            qc = qc.decompose(gates_to_decompose=["PauliEvolution"])
        except Exception:
            # Fall back to the raw circuit — correctness of the forward pass is
            # preserved; only the adjoint gradient path would be unavailable.
            pass

        self.circuit = qc
        return qc

    @property
    def num_parameters(self) -> int:
        if self.circuit is None:
            # Compute analytically
            orb_per_layer = 2 * (self.n_spatial - 1)  # α + β
            jas_per_layer = self.n_qubits - 1
            return self.n_layers * (orb_per_layer + jas_per_layer)
        return self.circuit.num_parameters

    def to_qiskit(self) -> QuantumCircuit:
        if self.circuit is None:
            self.build_circuit()
        return self.circuit
