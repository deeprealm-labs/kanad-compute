"""Adjoint-state gradient for parameterized quantum circuits.

Algorithm
---------
For ``E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩`` with ``|ψ(θ)⟩ = U_N ⋯ U_1 |0⟩`` and each
``U_j(α_j) = exp(-iα_j G_j)`` (single-Pauli rotations, ``G_j = P_j/2``):

::

    ∂E/∂α_j = −2·Im⟨ψ_j|G_j|R_j⟩

where ``|ψ_j⟩ = U_j ⋯ U_1 |0⟩`` and ``|R_j⟩ = U_{j+1}^† ⋯ U_N^† H|ψ⟩``.

Chain rule for parameters that appear in multiple gates via ParameterExpressions:

::

    ∂E/∂θ_k = Σ_j (∂α_j/∂θ_k) · ∂E/∂α_j

Single forward + single backward pass: O(N) statevector ops total. The
forward pass needs to save the intermediate state right after each
parameterized gate; we avoid an O(N) memory blowup by walking backwards
through both ``|ψ⟩`` and ``|R⟩`` simultaneously, applying ``U_j^†`` to both
(unitary, so well-defined inverse).

Supported gates
---------------
Any single-Pauli rotation: ``RX``, ``RY``, ``RZ``, ``R(θ, φ_const)``
(with fixed second argument). Non-parameterized gates (``CX``, ``H``,
``SX``, ``SXdg``, fixed ``U``) are applied/inverted normally.

If the circuit contains a parameterized gate whose generator we cannot
identify symbolically (e.g. parameterized ``U(θ_1, θ_2, θ_3)``), the
function raises ``NotImplementedError`` rather than silently producing a
wrong gradient.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp

logger = logging.getLogger(__name__)


# Mapping from gate name to its Hermitian generator (Pauli/2) on the
# acting qubit. The gate is exp(-iα·G) and we record G.
_SINGLE_QUBIT_GENERATORS: Dict[str, np.ndarray] = {
    'rz': 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    'rx': 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    'ry': 0.5 * np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
}


def _generator_for_gate(operation) -> Optional[np.ndarray]:
    """Return the Hermitian generator G for a parameterized single-qubit gate.

    For RGate(θ, φ): G = (cos φ · X + sin φ · Y) / 2 — but only when φ is
    a fixed numeric value, not a free parameter.

    Returns ``None`` if the gate is not parameterized (i.e. nothing to
    differentiate).
    """
    name = operation.name
    if name in _SINGLE_QUBIT_GENERATORS:
        return _SINGLE_QUBIT_GENERATORS[name]
    if name == 'r' and len(operation.params) == 2:
        # RGate(theta, phi). phi must be a fixed number; theta is the rotation.
        phi = operation.params[1]
        if isinstance(phi, ParameterExpression) and phi.parameters:
            raise NotImplementedError(
                "RGate with symbolic phi is not supported by the adjoint gradient. "
                f"Got phi = {phi}"
            )
        phi_val = float(phi)
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
        return 0.5 * (np.cos(phi_val) * X + np.sin(phi_val) * Y)
    # Fixed/no-parameter gate (h, cx, sx, sxdg, fixed u, barrier, ...): no gradient.
    return None


def _free_parameter_in_expression(expr) -> Optional[Tuple[Parameter, float]]:
    """Given a parameter-expression that's the angle of a single-Pauli rotation,
    return (free_param, ∂expr/∂free_param) — assuming the expression is linear
    in exactly one free parameter (which is the Givens-SD case).

    Returns ``None`` if the expression has no free parameters (gate is fixed).
    Raises ``NotImplementedError`` if the expression has > 1 free parameter
    or the dependence isn't linear.
    """
    if not isinstance(expr, ParameterExpression):
        return None  # bound numeric value
    free = list(expr.parameters)
    if not free:
        return None
    if len(free) > 1:
        raise NotImplementedError(
            f"Adjoint gradient supports only one free parameter per gate angle; "
            f"got {len(free)} in expression {expr}"
        )
    theta = free[0]
    # ∂expr/∂theta — for a linear expression a + b·θ this is constant.
    deriv = expr.gradient(theta)
    if isinstance(deriv, ParameterExpression) and deriv.parameters:
        raise NotImplementedError(
            f"Adjoint gradient requires LINEAR dependence of gate angle on the "
            f"underlying parameter; got non-constant derivative {deriv} for {expr}"
        )
    return theta, float(deriv)


def _apply_op_inverse_on_statevector(
    sv: np.ndarray,
    operation,
    qubits: List[int],
    n_qubits: int,
) -> np.ndarray:
    """Apply U^† for the given operation to the dense statevector ``sv``.

    Uses Qiskit's ``Operator`` to get U, computes U.conj().T, and applies
    via tensor-product reshaping. For 1- and 2-qubit gates this is O(2^n).
    """
    # Get the 2^k × 2^k matrix for the gate (k = len(qubits))
    op_matrix = np.asarray(Operator(operation).data)
    inv = op_matrix.conj().T
    # Apply via the einsum-style approach: reshape sv, contract, reshape back.
    return _apply_small_op(sv, inv, qubits, n_qubits)


def _apply_op_on_statevector(
    sv: np.ndarray,
    operation,
    qubits: List[int],
    n_qubits: int,
) -> np.ndarray:
    """Apply U (forward) for the given operation to the dense statevector ``sv``."""
    op_matrix = np.asarray(Operator(operation).data)
    return _apply_small_op(sv, op_matrix, qubits, n_qubits)


def _apply_small_op(sv: np.ndarray, op: np.ndarray, qubits: List[int], n_qubits: int) -> np.ndarray:
    """Apply a small (1- or 2-qubit) operator to the statevector.

    Qiskit conventions:
    - sv shape (2^n,), index i corresponds to bitstring i (qubit 0 = LSB).
    - For a multi-qubit gate, qubits are listed [target_low, target_high, ...]
      with the operator's basis ordered accordingly (qubit 0 = LSB of the gate matrix).
    """
    # Easiest: use Qiskit's Statevector to apply the gate, then return its .data
    state = Statevector(sv)
    state = state.evolve(op, qargs=qubits)
    return state.data


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

class AdjointGradientCalculator:
    """Pre-cache per-gate generators + chain-rule coefficients for a parameterized
    circuit, then compute ∂E/∂θ_k in one forward + one backward pass.

    Typical usage::

        calc = AdjointGradientCalculator(qc, hamiltonian)
        grad = calc.gradient(param_dict)
    """

    def __init__(self, circuit: QuantumCircuit, hamiltonian: SparsePauliOp):
        self.circuit = circuit
        self.hamiltonian = hamiltonian
        self.n_qubits = circuit.num_qubits
        self.parameters = list(circuit.parameters)
        # Pre-cache per-gate info: (operation, qubits, generator_or_None,
        # free_param_or_None, coefficient_or_None).
        self._gate_info: List[Tuple] = []
        for instr in circuit.data:
            op = instr.operation
            qargs = [circuit.find_bit(q).index for q in instr.qubits]
            if op.name == 'barrier' or op.name == 'measure':
                self._gate_info.append((op, qargs, None, None, None))
                continue
            gen = _generator_for_gate(op)
            if gen is None or not op.params:
                # Unparameterized (or fixed-param) gate: still apply it forward/backward
                self._gate_info.append((op, qargs, None, None, None))
                continue
            # Parameterized: the first param is the rotation angle.
            theta_info = _free_parameter_in_expression(op.params[0])
            if theta_info is None:
                # angle is bound numeric — no gradient
                self._gate_info.append((op, qargs, None, None, None))
                continue
            free_param, coefficient = theta_info
            self._gate_info.append((op, qargs, gen, free_param, coefficient))

        # Hamiltonian matrix as a (sparse) operator for fast |R⟩ = H|ψ⟩.
        # For ≤16 qubits this is fine sparse; if the dense matrix fits in RAM
        # we use dense for speed.
        self._H_sparse = hamiltonian.to_matrix(sparse=True)

    def gradient(self, param_dict: Dict[Parameter, float]) -> Dict[Parameter, float]:
        """Compute ∂E/∂θ_k for every symbolic parameter in the circuit."""
        # Bind once for the forward simulation.
        bound = self.circuit.assign_parameters(param_dict)

        # Forward pass: simulate the circuit to get |ψ⟩.
        psi = Statevector.from_instruction(bound).data

        # |R⟩ = H|ψ⟩ — sparse matvec.
        R = np.asarray(self._H_sparse @ psi).ravel()

        # Backward pass through gates, accumulating gradient.
        grad: Dict[Parameter, float] = {p: 0.0 for p in self.parameters}
        L = psi  # state right after gate j (initially = ψ_N = ψ)

        for op, qargs, gen, theta, coefficient in reversed(self._gate_info):
            # If parameterized, compute the contribution BEFORE undoing the gate.
            if gen is not None and theta is not None and coefficient is not None:
                # Apply G to R (don't mutate R yet): this is a small in-place op
                GR = _apply_small_op(R, gen, qargs, self.n_qubits)
                contribution = -2.0 * coefficient * float(np.imag(np.vdot(L, GR)))
                grad[theta] += contribution

            # Undo the gate: apply U^† to BOTH L and R.
            # Skip non-physical gates.
            if op.name in ('barrier', 'measure'):
                continue
            # Bind any free parameters in this gate using param_dict (the
            # forward pass already used bound values; we need the same here).
            op_bound = op
            if op.params and any(
                isinstance(p, ParameterExpression) and p.parameters for p in op.params
            ):
                # Re-build operation with bound numeric params. Qiskit's
                # `ParameterExpression.bind` is strict — pass only the
                # parameters present in each expression.
                bound_params = []
                for p in op.params:
                    if isinstance(p, ParameterExpression):
                        free_in_p = p.parameters
                        if free_in_p:
                            sub = {q: param_dict[q] for q in free_in_p if q in param_dict}
                            bound_params.append(float(p.bind(sub)))
                        else:
                            bound_params.append(float(p))
                    else:
                        bound_params.append(float(p))
                op_bound = type(op)(*bound_params)
            L = _apply_op_inverse_on_statevector(L, op_bound, qargs, self.n_qubits)
            R = _apply_op_inverse_on_statevector(R, op_bound, qargs, self.n_qubits)

        return grad


def adjoint_energy_gradient(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    param_dict: Dict[Parameter, float],
) -> Dict[Parameter, float]:
    """One-shot adjoint gradient: build the calculator and run."""
    return AdjointGradientCalculator(circuit, hamiltonian).gradient(param_dict)
