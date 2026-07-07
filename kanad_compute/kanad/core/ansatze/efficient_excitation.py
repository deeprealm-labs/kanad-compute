"""Hardware-Efficient Ansatz (HEA) circuit builders.

Builds the HF reference + layered RY / CNOT-entangler / RZ circuits used by
`HardwareVQE`. Circular entanglement is essential for expressibility on small
systems (linear cannot reach the ground state — see `core/ansatze/CLAUDE.md`).

Note (reorg, 2026-05-31): the FEB (Fermionic Excitation Based) decompositions
that previously lived here — `apply_single_excitation_FEB`,
`apply_double_excitation_FEB`, `_apply_double_exc_h2_optimal`,
`_apply_double_exc_jw` — were dead (zero callers repo-wide) and were removed.
The hardcoded 8-term Pauli table in `_apply_double_exc_h2_optimal` was a
correctness trap. The single canonical fermionic excitation generator now lives
in `core.operators.excitation_operators.build_excitation_generator`.
"""

import numpy as np
from typing import Tuple, List
from qiskit import QuantumCircuit


def build_hea_layer(
    circuit: QuantumCircuit,
    params: np.ndarray,
    layer: int = 0,
    entanglement: str = 'circular'
):
    """
    Build one layer of Hardware-Efficient Ansatz.

    Structure:
    1. RY rotation on all qubits
    2. CNOT entanglement (linear or circular)
    3. RZ rotation on all qubits

    Args:
        circuit: Quantum circuit to modify
        params: Parameters for this layer (2*n_qubits values)
        layer: Layer index (for parameter offset)
        entanglement: 'linear' (n-1 CNOTs) or 'circular' (n CNOTs, wraps around)
                     CRITICAL: 'circular' achieves chemical accuracy for H₂!

    Note:
        Circular entanglement is essential for expressibility.
        Linear entanglement (default before 2026-01-30) cannot reach ground state.
    """
    n_qubits = circuit.num_qubits

    # RY rotation layer
    for q in range(n_qubits):
        if layer * 2 * n_qubits + q < len(params):
            circuit.ry(params[layer * 2 * n_qubits + q], q)

    # CNOT entanglement
    if entanglement == 'circular':
        # Circular: connects last qubit back to first
        # This is CRITICAL for expressibility in small systems
        for q in range(n_qubits):
            circuit.cx(q, (q + 1) % n_qubits)
    else:
        # Linear: simpler but less expressive
        for q in range(n_qubits - 1):
            circuit.cx(q, q + 1)

    # RZ rotation layer
    for q in range(n_qubits):
        param_idx = layer * 2 * n_qubits + n_qubits + q
        if param_idx < len(params):
            circuit.rz(params[param_idx], q)


def build_hea_circuit(
    n_qubits: int,
    n_electrons: int,
    params: np.ndarray,
    n_layers: int = 2,
    entanglement: str = 'circular'
) -> QuantumCircuit:
    """
    Build complete Hardware-Efficient Ansatz circuit.

    Args:
        n_qubits: Number of qubits
        n_electrons: Number of electrons (for HF state)
        params: All variational parameters
        n_layers: Number of HEA layers (default 2 for hardware)
        entanglement: 'linear' or 'circular' (default: circular for accuracy)

    Returns:
        QuantumCircuit with HF state + HEA layers

    Note:
        With circular entanglement and 2 layers, achieves:
        - H₂: 0.00 mHa error with 8 CNOTs
        - LiH: ~0.5 mHa error with 20 CNOTs
    """
    circuit = QuantumCircuit(n_qubits)

    # Prepare HF reference state
    for i in range(n_electrons):
        circuit.x(i)

    # Apply HEA layers with specified entanglement
    for layer in range(n_layers):
        build_hea_layer(circuit, params, layer, entanglement=entanglement)

    return circuit


def count_cnots(circuit: QuantumCircuit) -> int:
    """Count CNOT gates in circuit."""
    ops = circuit.count_ops()
    return ops.get('cx', 0) + ops.get('cnot', 0)


def estimate_circuit_fidelity(n_cnots: int, cnot_error: float = 0.01) -> float:
    """
    Estimate circuit fidelity based on CNOT count.

    Args:
        n_cnots: Number of CNOT gates
        cnot_error: Single CNOT error rate (default 1% for IBM)

    Returns:
        Estimated fidelity (0 to 1)
    """
    return (1 - cnot_error) ** n_cnots
