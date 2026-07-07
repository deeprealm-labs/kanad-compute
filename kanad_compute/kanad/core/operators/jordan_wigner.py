"""
Native Jordan-Wigner transformation for Kanad.

This replaces OpenFermion's jordan_wigner with a governance-aware implementation
that is optimized for molecular Hamiltonian construction.

References:
- Jordan & Wigner, Z. Phys. 47, 631 (1928)
- Seeley, Richard, Love, J. Chem. Phys. 137, 224109 (2012)
"""

import numpy as np
from typing import Dict, Tuple, List
from qiskit.quantum_info import SparsePauliOp

from kanad.core.operators.fermion_operator import FermionOperator


def jordan_wigner(fermion_op: FermionOperator, n_qubits: int = None) -> Dict[str, complex]:
    """
    Apply Jordan-Wigner transformation to convert fermionic operator to qubit operator.

    The transformation is:
        a†_j → ½(X_j - iY_j) ⊗ Z₀⊗Z₁⊗...⊗Z_{j-1}
        a_j  → ½(X_j + iY_j) ⊗ Z₀⊗Z₁⊗...⊗Z_{j-1}

    Args:
        fermion_op: FermionOperator to transform
        n_qubits: Number of qubits (if None, inferred from operator)

    Returns:
        Dictionary mapping Pauli strings to coefficients
    """
    # Determine number of qubits
    if n_qubits is None:
        n_qubits = _get_max_orbital(fermion_op) + 1

    result = {}

    for term, coeff in fermion_op.terms.items():
        if len(term) == 0:
            # Identity term
            identity = 'I' * n_qubits
            if identity in result:
                result[identity] += coeff
            else:
                result[identity] = coeff
            continue

        # Transform each fermionic term to Pauli operators
        pauli_terms = _transform_term(term, n_qubits)

        for pauli_str, pauli_coeff in pauli_terms.items():
            full_coeff = coeff * pauli_coeff
            if pauli_str in result:
                result[pauli_str] += full_coeff
            else:
                result[pauli_str] = full_coeff

    # Remove negligible terms
    result = {k: v for k, v in result.items() if abs(v) > 1e-15}

    return result


def _get_max_orbital(fermion_op: FermionOperator) -> int:
    """Get maximum orbital index in the operator."""
    max_orb = -1
    for term in fermion_op.terms.keys():
        for orb, _ in term:
            max_orb = max(max_orb, orb)
    return max_orb


def _transform_term(term: Tuple, n_qubits: int) -> Dict[str, complex]:
    """
    Transform a single fermionic term to Pauli operators.

    For a product of fermionic operators, we transform each one and
    multiply the resulting Pauli operators together.
    """
    # Start with identity
    result = {'I' * n_qubits: 1.0}

    for orbital, action in term:
        # Get Pauli representation of this operator
        if action == 1:  # Creation
            op_paulis = _creation_to_pauli(orbital, n_qubits)
        else:  # Annihilation
            op_paulis = _annihilation_to_pauli(orbital, n_qubits)

        # Multiply with current result
        result = _multiply_pauli_dicts(result, op_paulis)

    return result


def _creation_to_pauli(orbital: int, n_qubits: int) -> Dict[str, complex]:
    """
    Convert creation operator a†_j to Pauli operators.

    a†_j = ½(X_j - iY_j) ⊗ Z₀⊗Z₁⊗...⊗Z_{j-1}

    IMPORTANT: Qiskit uses little-endian bit ordering where the rightmost
    character in a Pauli string corresponds to qubit 0. We build strings
    accordingly by reversing the final result.
    """
    # Build X and Y terms with Z-string
    x_term = ['I'] * n_qubits
    y_term = ['I'] * n_qubits

    # Z-string for all qubits before orbital
    for i in range(orbital):
        x_term[i] = 'Z'
        y_term[i] = 'Z'

    # X and Y at the orbital position
    x_term[orbital] = 'X'
    y_term[orbital] = 'Y'

    # CRITICAL: Reverse for Qiskit's little-endian convention
    # Qiskit expects rightmost character = qubit 0
    return {
        ''.join(x_term[::-1]): 0.5,
        ''.join(y_term[::-1]): -0.5j
    }


def _annihilation_to_pauli(orbital: int, n_qubits: int) -> Dict[str, complex]:
    """
    Convert annihilation operator a_j to Pauli operators.

    a_j = ½(X_j + iY_j) ⊗ Z₀⊗Z₁⊗...⊗Z_{j-1}

    IMPORTANT: Qiskit uses little-endian bit ordering where the rightmost
    character in a Pauli string corresponds to qubit 0. We build strings
    accordingly by reversing the final result.
    """
    # Build X and Y terms with Z-string
    x_term = ['I'] * n_qubits
    y_term = ['I'] * n_qubits

    # Z-string for all qubits before orbital
    for i in range(orbital):
        x_term[i] = 'Z'
        y_term[i] = 'Z'

    # X and Y at the orbital position
    x_term[orbital] = 'X'
    y_term[orbital] = 'Y'

    # CRITICAL: Reverse for Qiskit's little-endian convention
    # Qiskit expects rightmost character = qubit 0
    return {
        ''.join(x_term[::-1]): 0.5,
        ''.join(y_term[::-1]): 0.5j
    }


def _multiply_pauli_dicts(dict1: Dict[str, complex], dict2: Dict[str, complex]) -> Dict[str, complex]:
    """
    Multiply two Pauli operator dictionaries.

    Uses Pauli multiplication rules:
    - XX = YY = ZZ = I
    - XY = iZ, YX = -iZ
    - YZ = iX, ZY = -iX
    - ZX = iY, XZ = -iY
    """
    result = {}

    for p1, c1 in dict1.items():
        for p2, c2 in dict2.items():
            # Multiply Pauli strings
            new_pauli, phase = _multiply_pauli_strings(p1, p2)

            coeff = c1 * c2 * phase
            if new_pauli in result:
                result[new_pauli] += coeff
            else:
                result[new_pauli] = coeff

    # Remove negligible terms
    result = {k: v for k, v in result.items() if abs(v) > 1e-15}

    return result


def _multiply_pauli_strings(p1: str, p2: str) -> Tuple[str, complex]:
    """
    Multiply two Pauli strings and return result with phase.

    Returns:
        (result_pauli_string, phase)
    """
    if len(p1) != len(p2):
        raise ValueError(f"Pauli strings must have same length: {len(p1)} vs {len(p2)}")

    result = []
    phase = 1.0

    for c1, c2 in zip(p1, p2):
        r, p = _multiply_single_pauli(c1, c2)
        result.append(r)
        phase *= p

    return ''.join(result), phase


# Pauli multiplication table
_PAULI_MULT = {
    ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
    ('X', 'I'): ('X', 1), ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
    ('Y', 'I'): ('Y', 1), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
    ('Z', 'I'): ('Z', 1), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
}


def _multiply_single_pauli(p1: str, p2: str) -> Tuple[str, complex]:
    """Multiply two single-qubit Paulis."""
    return _PAULI_MULT[(p1, p2)]


def jordan_wigner_sparse_pauli_op(fermion_op: FermionOperator, n_qubits: int = None) -> SparsePauliOp:
    """
    Apply Jordan-Wigner transformation and return Qiskit SparsePauliOp.

    Args:
        fermion_op: FermionOperator to transform
        n_qubits: Number of qubits

    Returns:
        qiskit.quantum_info.SparsePauliOp
    """
    pauli_dict = jordan_wigner(fermion_op, n_qubits)

    if len(pauli_dict) == 0:
        n = n_qubits or 1
        return SparsePauliOp(['I' * n], [0.0])

    pauli_strings = list(pauli_dict.keys())
    coefficients = list(pauli_dict.values())

    return SparsePauliOp(pauli_strings, coefficients)


def build_molecular_hamiltonian_jw(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    nuclear_repulsion: float,
    n_electrons: int = None  # Not needed, kept for API compatibility
) -> SparsePauliOp:
    """
    Build molecular Hamiltonian using native Jordan-Wigner transformation.

    This is the main entry point that replaces OpenFermion's jordan_wigner
    for molecular Hamiltonians.

    Args:
        h_mo: One-electron integrals in MO basis (n_orbitals x n_orbitals)
        eri_mo: Two-electron integrals in MO basis (chemist notation ⟨pq|rs⟩)
        nuclear_repulsion: Nuclear repulsion energy
        n_electrons: Not used, kept for API compatibility

    Returns:
        Qiskit SparsePauliOp
    """
    n_orbitals = h_mo.shape[0]
    n_qubits = 2 * n_orbitals  # Spin orbitals

    # Build fermionic Hamiltonian
    fermion_ham = FermionOperator((), nuclear_repulsion)  # Start with nuclear repulsion

    # One-body terms: Σ_{pq} h_{pq} a†_p a_q
    # For both spin-up (even indices) and spin-down (odd indices)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            if abs(h_mo[p, q]) < 1e-12:
                continue

            # Spin-up: indices 2*p, 2*q
            fermion_ham += FermionOperator(((2*p, 1), (2*q, 0)), h_mo[p, q])

            # Spin-down: indices 2*p+1, 2*q+1
            fermion_ham += FermionOperator(((2*p+1, 1), (2*q+1, 0)), h_mo[p, q])

    # Two-body terms: (1/2) Σ_{pqrs} ⟨pq|rs⟩ a†_p a†_r a_s a_q
    # Using chemist notation: g[p,q,r,s] = ⟨pq|rs⟩
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    if abs(eri_mo[p, q, r, s]) < 1e-12:
                        continue

                    coeff = 0.5 * eri_mo[p, q, r, s]

                    # Four spin combinations:
                    # ↑↑: a†_{p↑} a†_{r↑} a_{s↑} a_{q↑}
                    fermion_ham += FermionOperator(
                        ((2*p, 1), (2*r, 1), (2*s, 0), (2*q, 0)),
                        coeff
                    )

                    # ↑↓: a†_{p↑} a†_{r↓} a_{s↓} a_{q↑}
                    fermion_ham += FermionOperator(
                        ((2*p, 1), (2*r+1, 1), (2*s+1, 0), (2*q, 0)),
                        coeff
                    )

                    # ↓↑: a†_{p↓} a†_{r↑} a_{s↑} a_{q↓}
                    fermion_ham += FermionOperator(
                        ((2*p+1, 1), (2*r, 1), (2*s, 0), (2*q+1, 0)),
                        coeff
                    )

                    # ↓↓: a†_{p↓} a†_{r↓} a_{s↓} a_{q↓}
                    fermion_ham += FermionOperator(
                        ((2*p+1, 1), (2*r+1, 1), (2*s+1, 0), (2*q+1, 0)),
                        coeff
                    )

    # Apply Jordan-Wigner transformation
    pauli_op = jordan_wigner_sparse_pauli_op(fermion_ham, n_qubits)

    return pauli_op
