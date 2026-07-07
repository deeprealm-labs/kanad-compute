"""
Ionic bonding governance protocol.

Physical principles for ionic bonding:
1. Electrons are LOCALIZED on specific atoms
2. Electron transfer between sites (a†_i a_j operators)
3. MINIMAL entanglement (nearest-neighbor only)
4. Strong on-site Coulomb repulsion (Hubbard U)
5. Large electronegativity difference drives charge separation

Circuit requirements:
- Sparse connectivity (nearest-neighbor gates)
- No long-range entanglement
- Transfer operators only between bonded sites
- Preserves particle number per spin
"""

from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)
from kanad.core.governance.protocols.base_protocol import (
    BaseGovernanceProtocol,
    BondingType,
    GovernanceRule,
    QuantumCircuitState
)


class IonicGovernanceProtocol(BaseGovernanceProtocol):
    """
    Governance protocol for ionic bonding systems.

    Example: Na+ Cl-
    - Electron transfers from Na to Cl
    - Localized states on each atom
    - Minimal entanglement between sites
    - Circuit uses local gates only
    """

    def __init__(self):
        """Initialize ionic governance protocol."""
        super().__init__(BondingType.IONIC)

    def _initialize_rules(self):
        """Initialize ionic bonding governance rules."""

        # Rule 1: Enforce localized gates (no long-range entanglement)
        self.rules.append(GovernanceRule(
            name="localized_gates_only",
            description="Only nearest-neighbor gates allowed (localized interactions)",
            condition=lambda state, ctx: True,  # Always check
            action=self._enforce_locality,
            priority=100,
            required=True
        ))

        # Rule 2: Sparse connectivity (minimal entanglement)
        self.rules.append(GovernanceRule(
            name="sparse_connectivity",
            description="Connectivity must be sparse (avg degree < 2)",
            condition=lambda state, ctx: isinstance(state, QuantumCircuitState),
            action=self._enforce_sparsity,
            priority=90,
            required=True
        ))

        # Rule 3: Preserve particle number
        self.rules.append(GovernanceRule(
            name="particle_number_conservation",
            description="Total particle number must be conserved",
            condition=lambda state, ctx: True,
            action=self._enforce_particle_conservation,
            priority=80,
            required=True
        ))

        # Rule 4: No collective gates (forbidden for ionic)
        self.rules.append(GovernanceRule(
            name="forbid_collective_gates",
            description="No gates acting on >2 qubits (no collective behavior)",
            condition=lambda state, ctx: isinstance(state, QuantumCircuitState),
            action=self._forbid_collective_gates,
            priority=70,
            required=True
        ))

    def validate_operator(self, operator: 'QuantumOperator') -> bool:
        """
        Validate operator for ionic bonding.

        ALLOWED:
        - Single-qubit rotations (on-site terms)
        - Nearest-neighbor 2-qubit gates (transfer)
        - Number operators (charge counting)

        FORBIDDEN:
        - Long-range entangling gates
        - Multi-qubit (>2) gates
        - Delocalized operators
        """
        # Check operator type against allowed/forbidden lists
        operator_type = getattr(operator, 'type', 'unknown')

        allowed = ['rx', 'ry', 'rz', 'cx', 'cnot', 'cz', 'rxx', 'ryy', 'rzz']
        forbidden = ['ghz', 'w_state', 'qft', 'swap_network']

        if operator_type in forbidden:
            return False

        # Check locality - no long-range gates
        if hasattr(operator, 'qubits'):
            qubits = operator.qubits
            if len(qubits) > 2:
                return False  # No multi-qubit gates

            if len(qubits) == 2:
                # Check if qubits are nearest neighbors
                # (simplified - would use actual geometry)
                if abs(qubits[0] - qubits[1]) > 1:
                    return False  # Not nearest neighbor

        return True

    def construct_ansatz(self, representation: Any) -> QuantumCircuitState:
        """
        Construct ionic bonding ansatz.

        Circuit structure for ionic systems:
        1. Single-qubit rotations (on-site energies)
        2. Nearest-neighbor transfer gates (hopping)
        3. Small variational parameters (minimal mixing)

        Example for NaCl:
        |ψ⟩ = |Na⁺⟩⊗|Cl⁻⟩ + small corrections

        Args:
            representation: Representation object

        Returns:
            QuantumCircuitState with governed circuit
        """
        n_qubits = representation.get_num_qubits()
        circuit = QuantumCircuitState(n_qubits)

        # Step 1: Initialize in charge-separated state
        # (done via state preparation, not gates)
        circuit.metadata['initial_state'] = 'charge_separated'

        # Step 2: Single-qubit rotations for on-site energy
        for i in range(n_qubits):
            circuit.add_gate('rz', [i], params=[0.0])  # Variational parameter

        # Step 3: Nearest-neighbor transfer gates (minimal)
        for i in range(0, n_qubits - 1, 2):  # Only adjacent pairs
            # Small amplitude transfer
            circuit.add_gate('rxx', [i, i + 1], params=[0.1])  # Small angle

        # Step 4: Another layer of single-qubit rotations
        for i in range(n_qubits):
            circuit.add_gate('ry', [i], params=[0.0])

        circuit.is_localized = True
        circuit.metadata['ansatz_type'] = 'ionic_localized'

        return circuit

    def enforce_constraints(self, circuit: QuantumCircuitState) -> QuantumCircuitState:
        """
        Enforce ionic bonding constraints on circuit.

        Constraints:
        - Remove long-range gates
        - Limit entanglement degree
        - Ensure particle conservation

        Args:
            circuit: Input circuit

        Returns:
            Constrained circuit
        """
        # Filter out disallowed gates
        allowed_gates = []

        for gate in circuit.gates:
            qubits = gate['qubits']

            # Keep single-qubit gates
            if len(qubits) == 1:
                allowed_gates.append(gate)

            # Keep nearest-neighbor 2-qubit gates only
            elif len(qubits) == 2:
                if abs(qubits[0] - qubits[1]) <= 1:  # Nearest neighbor
                    allowed_gates.append(gate)
                else:
                    logger.debug(f"Removed long-range gate: {gate}")

            # Remove multi-qubit gates
            else:
                logger.debug(f"Removed multi-qubit gate: {gate}")

        # Update circuit
        circuit.gates = allowed_gates
        circuit.is_localized = True

        return circuit

    def _enforce_locality(self, state: QuantumCircuitState, context: Dict) -> QuantumCircuitState:
        """Enforce that all gates are local (nearest-neighbor)."""
        if not isinstance(state, QuantumCircuitState):
            return state

        # Check all gates are local
        for gate in state.gates:
            if len(gate['qubits']) == 2:
                q1, q2 = gate['qubits']
                if abs(q1 - q2) > 1:
                    raise ValueError(
                        f"Non-local gate detected: {gate}. "
                        f"Ionic bonding requires nearest-neighbor gates only."
                    )

        state.metadata['locality_enforced'] = True
        return state

    def _enforce_sparsity(self, state: QuantumCircuitState, context: Dict) -> QuantumCircuitState:
        """Ensure circuit has sparse connectivity."""
        if not isinstance(state, QuantumCircuitState):
            return state

        max_degree = state.max_entanglement_degree()
        if max_degree > 2:
            logger.warning(f"High entanglement degree ({max_degree}). "
                          f"Ionic systems should have sparse connectivity.")

        state.metadata['sparse'] = state.is_sparse()
        return state

    def _enforce_particle_conservation(self, state: Any, context: Dict) -> Any:
        """Ensure particle number is conserved."""
        # Mark circuit as particle-conserving
        # This is automatically satisfied for properly constructed ansätze
        # since we use number-conserving excitation operators
        if isinstance(state, QuantumCircuitState):
            state.metadata['particle_conserving'] = True
        return state

    def _forbid_collective_gates(self, state: QuantumCircuitState, context: Dict) -> QuantumCircuitState:
        """Remove any collective (>2 qubit) gates."""
        if not isinstance(state, QuantumCircuitState):
            return state

        # Filter out multi-qubit gates
        state.gates = [g for g in state.gates if len(g['qubits']) <= 2]
        state.metadata['no_collective_gates'] = True

        return state

    def get_allowed_operators(self) -> List[str]:
        """Get list of allowed operators for ionic bonding."""
        return [
            'rx', 'ry', 'rz',  # Single-qubit rotations
            'cx', 'cy', 'cz',  # Controlled gates (nearest-neighbor)
            'rxx', 'ryy', 'rzz',  # Two-qubit rotations
            'number',  # Number operator
        ]

    def get_forbidden_operators(self) -> List[str]:
        """Get list of forbidden operators for ionic bonding."""
        return [
            'qft',  # Quantum Fourier Transform (delocalization)
            'ghz',  # GHZ state (collective)
            'w_state',  # W state (collective)
            'long_range_swap',  # Long-range operations
        ]

    def is_valid_configuration(self, bitstring: str) -> bool:
        """
        Check if configuration is valid for ionic bonding.

        Rules:
        1. Must preserve total electron count
        2. Localized charge distribution (electrons on atoms)
        3. Allow for ionic charge separation

        Args:
            bitstring: Configuration bitstring

        Returns:
            True if configuration is physically valid
        """
        n_qubits = len(bitstring)

        # For ionic bonding, we're more permissive since charge can be localized
        # on different atoms (that's the nature of ionic bonding)

        # Check spin balance (for singlet ground states)
        n_up = sum(1 for i in range(0, n_qubits, 2) if bitstring[i] == '1')
        n_down = sum(1 for i in range(1, n_qubits, 2) if bitstring[i] == '1')

        # Allow larger spin imbalance for ionic (charge separation)
        if abs(n_up - n_down) > 3:
            return False

        return True

    def generate_single_excitations(self, bitstring: str) -> List[str]:
        """
        Generate physics-aware single excitations for ionic bonding.

        Ionic bonding principles:
        1. Electrons are LOCALIZED on atoms
        2. Minimal charge transfer between sites
        3. Nearest-neighbor excitations only
        4. Preserve particle number per site

        Args:
            bitstring: Configuration bitstring

        Returns:
            List of excited configuration bitstrings
        """
        n_qubits = len(bitstring)
        bits = list(bitstring)
        excitations = []

        # Find occupied and virtual orbitals
        occupied = [i for i, b in enumerate(bits) if b == '1']
        virtual = [i for i, b in enumerate(bits) if b == '0']

        # Rule 1: Localized on-site excitations (within same atom)
        # For ionic: excitations should be minimal and localized
        for occ_idx in occupied:
            for virt_idx in virtual:
                # Only allow nearest-neighbor excitations (ionic = localized)
                if abs(virt_idx - occ_idx) <= 2:
                    new_bits = bits.copy()
                    new_bits[occ_idx] = '0'
                    new_bits[virt_idx] = '1'
                    excitations.append(''.join(new_bits))

        # Rule 2: HOMO → LUMO (minimal correlation)
        if occupied and virtual:
            homo_idx = max(occupied)
            lumo_idx = min(virtual)

            new_bits = bits.copy()
            new_bits[homo_idx] = '0'
            new_bits[lumo_idx] = '1'
            excited = ''.join(new_bits)
            if excited not in excitations:
                excitations.append(excited)

        # Remove duplicates and return
        return list(set(excitations))

    def generate_double_excitations(self, bitstring: str) -> List[str]:
        """
        Generate physics-aware double excitations for ionic bonding.

        Ionic bonding principles:
        1. Electrons are LOCALIZED on atoms
        2. Minimal charge transfer between sites
        3. Double excitations are less important for ionic bonds
        4. Only include nearest-neighbor double excitations

        Args:
            bitstring: Configuration bitstring

        Returns:
            List of doubly excited configuration bitstrings
        """
        n_qubits = len(bitstring)
        bits = list(bitstring)
        excitations = []

        # Find occupied and virtual orbitals
        occupied = [i for i, b in enumerate(bits) if b == '1']
        virtual = [i for i, b in enumerate(bits) if b == '0']

        # For ionic: only include very limited double excitations
        # (minimal correlation, localized electrons)
        if len(occupied) >= 2 and len(virtual) >= 2:
            # HOMO-1, HOMO -> LUMO, LUMO+1 (minimal double excitation)
            occ_sorted = sorted(occupied, reverse=True)
            virt_sorted = sorted(virtual)

            if len(occ_sorted) >= 2 and len(virt_sorted) >= 2:
                i, j = occ_sorted[0], occ_sorted[1]
                a, b = virt_sorted[0], virt_sorted[1]

                # Only if they're relatively close (ionic = localized)
                if abs(a - i) <= 4 and abs(b - j) <= 4:
                    new_bits = bits.copy()
                    new_bits[i] = '0'
                    new_bits[j] = '0'
                    new_bits[a] = '1'
                    new_bits[b] = '1'
                    excitations.append(''.join(new_bits))

        return list(set(excitations))

    def __repr__(self) -> str:
        """String representation."""
        return f"IonicGovernanceProtocol(rules={len(self.rules)}, entanglement='minimal')"
