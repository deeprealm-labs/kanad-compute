"""
Custom circuit ansatz for user-defined gate sequences.

Accepts per-qubit gate definitions from the frontend circuit builder.
"""

from typing import List, Dict, Optional
from kanad.core.ansatze.base_ansatz import BaseAnsatz, QuantumCircuit, Parameter
from kanad.core.ansatze.hardware_efficient_ansatz import get_hf_state_qubits


# Gates that take a variational parameter
PARAMETRIZED_SINGLE = {'rx', 'ry', 'rz'}
PARAMETRIZED_TWO = {'rxx', 'ryy', 'rzz'}
FIXED_SINGLE = {'h', 'x', 'y', 'z'}
FIXED_TWO = {'cx', 'cz', 'swap'}


class CustomCircuitAnsatz(BaseAnsatz):
    """
    User-defined circuit ansatz from explicit gate sequences.

    Accepts a list of gate layers where each layer specifies individual
    gates with their qubit targets. This enables the frontend circuit
    builder to send arbitrary gate arrangements to the VQE solver.

    Args:
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        gate_layers: List of layer dicts, each with 'gates' key containing
            list of gate dicts: {'type': str, 'qubit': int, 'target_qubit': int (optional)}
        include_hf_init: Whether to prepare HF initial state (default True)
        mapper: Fermion-to-qubit mapper for HF state prep
    """

    def __init__(
        self,
        n_qubits: int,
        n_electrons: int,
        gate_layers: List[Dict],
        include_hf_init: bool = True,
        mapper: str = 'jordan_wigner'
    ):
        super().__init__(n_qubits, n_electrons)
        self.gate_layers = gate_layers
        self.include_hf_init = include_hf_init
        self.mapper = mapper.lower()

    @property
    def n_parameters(self) -> int:
        """Count variational parameters from gate definitions."""
        count = 0
        for layer in self.gate_layers:
            for gate in layer.get('gates', []):
                gtype = gate.get('type', '').lower()
                qubit = gate.get('qubit', 0)
                target_qubit = gate.get('target_qubit')
                # Mirror build_circuit's range validation (lines 84-88) so the
                # parameter count never disagrees with the built circuit.
                if qubit >= self.n_qubits:
                    continue
                if target_qubit is not None and target_qubit >= self.n_qubits:
                    continue
                if gtype in PARAMETRIZED_SINGLE or gtype in PARAMETRIZED_TWO:
                    count += 1
        return count

    def build_circuit(self, **kwargs) -> QuantumCircuit:
        """
        Build circuit from user-defined gate layers.

        Returns:
            QuantumCircuit with HF init + user gates
        """
        circuit = QuantumCircuit(self.n_qubits)

        # 1. HF state preparation
        if self.include_hf_init and self.n_electrons > 0:
            hf_qubits = get_hf_state_qubits(self.n_qubits, self.n_electrons, self.mapper)
            for qubit in hf_qubits:
                circuit.x(qubit)
            circuit.barrier()

        # 2. Apply user-defined gate layers
        param_idx = 0
        for layer_idx, layer in enumerate(self.gate_layers):
            for gate in layer.get('gates', []):
                gtype = gate.get('type', '').lower()
                qubit = gate.get('qubit', 0)
                target_qubit = gate.get('target_qubit')

                # Validate qubit indices
                if qubit >= self.n_qubits:
                    continue
                if target_qubit is not None and target_qubit >= self.n_qubits:
                    continue

                if gtype in PARAMETRIZED_SINGLE:
                    param = Parameter(f'θ_{layer_idx}_{gtype}_{qubit}_{param_idx}')
                    param_idx += 1
                    if gtype == 'rx':
                        circuit.rx(param, qubit)
                    elif gtype == 'ry':
                        circuit.ry(param, qubit)
                    elif gtype == 'rz':
                        circuit.rz(param, qubit)

                elif gtype in FIXED_SINGLE:
                    if gtype == 'h':
                        circuit.h(qubit)
                    elif gtype == 'x':
                        circuit.x(qubit)
                    elif gtype == 'y':
                        circuit.y(qubit)
                    elif gtype == 'z':
                        circuit.z(qubit)

                elif gtype in FIXED_TWO:
                    tq = target_qubit if target_qubit is not None else (qubit + 1) % self.n_qubits
                    if gtype == 'cx':
                        circuit.cx(qubit, tq)
                    elif gtype == 'cz':
                        circuit.cz(qubit, tq)
                    elif gtype == 'swap':
                        circuit.swap(qubit, tq)

                elif gtype in PARAMETRIZED_TWO:
                    tq = target_qubit if target_qubit is not None else (qubit + 1) % self.n_qubits
                    param = Parameter(f'θ_{layer_idx}_{gtype}_{qubit}_{tq}_{param_idx}')
                    param_idx += 1
                    if gtype == 'rxx':
                        circuit.rxx(param, qubit, tq)
                    elif gtype == 'ryy':
                        circuit.ryy(param, qubit, tq)
                    elif gtype == 'rzz':
                        circuit.rzz(param, qubit, tq)

        self.circuit = circuit
        return circuit

    def __repr__(self) -> str:
        total_gates = sum(len(l.get('gates', [])) for l in self.gate_layers)
        return (f"CustomCircuitAnsatz(n_qubits={self.n_qubits}, "
                f"layers={len(self.gate_layers)}, gates={total_gates})")
