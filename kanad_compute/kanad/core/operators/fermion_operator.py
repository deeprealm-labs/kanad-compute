"""
Native FermionOperator implementation for Kanad.

This replaces OpenFermion's FermionOperator with a governance-aware implementation
that is optimized for molecular Hamiltonian construction.

References:
- Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure Theory" (2000)
- Nielsen & Chuang, "Quantum Computation" (2010)
"""

import numpy as np
from typing import Dict, Tuple, Union, List
from collections import defaultdict


class FermionOperator:
    """
    Fermionic operator in second quantization.

    Represents operators of the form:
        Σ c_i (a†_{p1} a_{q1} a†_{p2} ... )

    where a† is creation and a is annihilation operator.

    Terms are stored as:
        {term: coefficient}

    where term is a tuple of (orbital_index, action) pairs:
        - action = 1 for creation (a†)
        - action = 0 for annihilation (a)

    Example:
        a†_0 a_1 → ((0, 1), (1, 0))
        a†_0 a†_1 a_2 a_3 → ((0, 1), (1, 1), (2, 0), (3, 0))

    Attributes:
        terms: Dictionary mapping term tuples to coefficients
    """

    def __init__(self, term: Union[str, Tuple, None] = None, coefficient: complex = 1.0):
        """
        Initialize FermionOperator.

        Args:
            term: Operator term, can be:
                - None or (): Identity operator
                - String: "0^ 1" means a†_0 a_1, "2^ 3^ 0 1" means a†_2 a†_3 a_0 a_1
                - Tuple: ((0, 1), (1, 0)) means a†_0 a_1
            coefficient: Complex coefficient for the term
        """
        self.terms: Dict[Tuple, complex] = defaultdict(complex)

        if term is None:
            # Zero operator
            pass
        elif term == () or term == '':
            # Identity operator
            self.terms[()] = coefficient
        elif isinstance(term, str):
            # Parse string like "0^ 1" or "2^ 3^ 0 1"
            parsed_term = self._parse_string(term)
            if parsed_term is not None:
                self.terms[parsed_term] = coefficient
        elif isinstance(term, tuple):
            # Direct tuple specification
            self.terms[term] = coefficient
        else:
            raise ValueError(f"Unknown term type: {type(term)}")

    def _parse_string(self, term_str: str) -> Tuple:
        """
        Parse string representation to term tuple.

        Format: "0^ 1" means a†_0 a_1
                "2^ 3^ 0 1" means a†_2 a†_3 a_0 a_1
        """
        if not term_str.strip():
            return ()

        ops = []
        for token in term_str.split():
            token = token.strip()
            if token.endswith('^'):
                # Creation operator
                orbital = int(token[:-1])
                ops.append((orbital, 1))
            else:
                # Annihilation operator
                orbital = int(token)
                ops.append((orbital, 0))

        return tuple(ops)

    def __add__(self, other: 'FermionOperator') -> 'FermionOperator':
        """Add two FermionOperators."""
        result = FermionOperator()
        result.terms = defaultdict(complex, self.terms)

        for term, coeff in other.terms.items():
            result.terms[term] += coeff

        # Remove zero terms
        result.terms = defaultdict(complex, {k: v for k, v in result.terms.items() if abs(v) > 1e-15})

        return result

    def __radd__(self, other):
        """Right add (for sum())."""
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: 'FermionOperator') -> 'FermionOperator':
        """Subtract two FermionOperators."""
        result = FermionOperator()
        result.terms = defaultdict(complex, self.terms)

        for term, coeff in other.terms.items():
            result.terms[term] -= coeff

        # Remove zero terms
        result.terms = defaultdict(complex, {k: v for k, v in result.terms.items() if abs(v) > 1e-15})

        return result

    def __mul__(self, other: Union['FermionOperator', complex, float, int]) -> 'FermionOperator':
        """Multiply FermionOperator by scalar or another operator."""
        result = FermionOperator()

        if isinstance(other, (complex, float, int)):
            # Scalar multiplication
            for term, coeff in self.terms.items():
                result.terms[term] = coeff * other
        elif isinstance(other, FermionOperator):
            # Operator multiplication
            for term1, coeff1 in self.terms.items():
                for term2, coeff2 in other.terms.items():
                    # Concatenate terms
                    new_term = term1 + term2
                    result.terms[new_term] += coeff1 * coeff2
        else:
            raise TypeError(f"Cannot multiply FermionOperator by {type(other)}")

        # Remove zero terms
        result.terms = defaultdict(complex, {k: v for k, v in result.terms.items() if abs(v) > 1e-15})

        return result

    def __rmul__(self, other: Union[complex, float, int]) -> 'FermionOperator':
        """Right multiplication by scalar."""
        return self.__mul__(other)

    def __neg__(self) -> 'FermionOperator':
        """Negate operator."""
        return self * (-1)

    def __iadd__(self, other: 'FermionOperator') -> 'FermionOperator':
        """In-place addition."""
        for term, coeff in other.terms.items():
            self.terms[term] += coeff
        # Remove zero terms
        self.terms = defaultdict(complex, {k: v for k, v in self.terms.items() if abs(v) > 1e-15})
        return self

    def normal_order(self) -> 'FermionOperator':
        """
        Return normal-ordered form of the operator.

        Normal ordering: all creation operators to the left of annihilation operators.
        Accounts for anticommutation relations: {a†_i, a_j} = δ_ij, {a_i, a_j} = 0

        Returns:
            Normal-ordered FermionOperator
        """
        result = FermionOperator()

        for term, coeff in self.terms.items():
            # Normal order each term
            normal_terms = self._normal_order_term(term)
            for normal_term, sign in normal_terms:
                result.terms[normal_term] += coeff * sign

        # Remove zero terms
        result.terms = defaultdict(complex, {k: v for k, v in result.terms.items() if abs(v) > 1e-15})

        return result

    def _normal_order_term(self, term: Tuple) -> List[Tuple[Tuple, complex]]:
        """
        Normal order a single term.

        Uses bubble sort with anticommutation rules:
        - a_i a†_j → -a†_j a_i + δ_ij
        - a†_i a†_j → -a†_j a†_i
        - a_i a_j → -a_j a_i

        Returns:
            List of (term, coefficient) pairs
        """
        if len(term) <= 1:
            return [(term, 1.0)]

        # Convert to list for manipulation
        ops = list(term)
        result = []

        # Bubble sort to normal order
        self._normal_order_recursive(ops, 1.0, result)

        return result

    def _normal_order_recursive(self, ops: List, coeff: complex, result: List):
        """Recursively normal order using anticommutation."""
        n = len(ops)
        if n <= 1:
            result.append((tuple(ops), coeff))
            return

        # Find first out-of-order pair
        swapped = False
        for i in range(n - 1):
            orb_i, action_i = ops[i]
            orb_j, action_j = ops[i + 1]

            # Normal order: creation (1) before annihilation (0)
            # Within same type: higher orbital index first (arbitrary convention)
            if action_i < action_j:  # annihilation before creation
                # Anticommute: a_i a†_j = δ_ij - a†_j a_i
                if orb_i == orb_j:
                    # {a_i, a†_i} = 1, so a_i a†_i = 1 - a†_i a_i
                    # Add identity term
                    remaining = ops[:i] + ops[i+2:]
                    if remaining:
                        self._normal_order_recursive(remaining, coeff, result)
                    else:
                        result.append(((), coeff))  # Identity

                # Swap with sign change
                new_ops = ops.copy()
                new_ops[i], new_ops[i + 1] = new_ops[i + 1], new_ops[i]
                self._normal_order_recursive(new_ops, -coeff, result)
                swapped = True
                break
            elif action_i == action_j == 1:  # two creations
                # a†_i a†_j = -a†_j a†_i if i ≠ j
                if orb_i < orb_j:  # Swap to get higher index first
                    new_ops = ops.copy()
                    new_ops[i], new_ops[i + 1] = new_ops[i + 1], new_ops[i]
                    self._normal_order_recursive(new_ops, -coeff, result)
                    swapped = True
                    break
                elif orb_i == orb_j:
                    # a†_i a†_i = 0 (Pauli exclusion)
                    return
            elif action_i == action_j == 0:  # two annihilations
                # a_i a_j = -a_j a_i if i ≠ j
                if orb_i > orb_j:  # Swap to get lower index first
                    new_ops = ops.copy()
                    new_ops[i], new_ops[i + 1] = new_ops[i + 1], new_ops[i]
                    self._normal_order_recursive(new_ops, -coeff, result)
                    swapped = True
                    break
                elif orb_i == orb_j:
                    # a_i a_i = 0 (Pauli exclusion)
                    return

        if not swapped:
            result.append((tuple(ops), coeff))

    def is_hermitian(self) -> bool:
        """Check if operator is Hermitian (H = H†)."""
        adjoint = self.hermitian_conjugate()

        for term, coeff in self.terms.items():
            if term not in adjoint.terms:
                return False
            if abs(coeff - adjoint.terms[term]) > 1e-10:
                return False

        return len(self.terms) == len(adjoint.terms)

    def hermitian_conjugate(self) -> 'FermionOperator':
        """
        Return Hermitian conjugate (adjoint) of operator.

        (a†_i a_j)† = a†_j a_i
        """
        result = FermionOperator()

        for term, coeff in self.terms.items():
            # Reverse order and flip creation/annihilation
            adj_term = tuple((orb, 1 - action) for orb, action in reversed(term))
            result.terms[adj_term] = np.conj(coeff)

        return result

    def __repr__(self) -> str:
        """String representation."""
        if not self.terms:
            return "0"

        parts = []
        for term, coeff in sorted(self.terms.items()):
            if abs(coeff) < 1e-15:
                continue

            # Format coefficient
            if abs(coeff.imag) < 1e-15:
                coeff_str = f"{coeff.real:.6g}"
            elif abs(coeff.real) < 1e-15:
                coeff_str = f"{coeff.imag:.6g}j"
            else:
                coeff_str = f"({coeff.real:.6g}+{coeff.imag:.6g}j)"

            # Format term
            if len(term) == 0:
                term_str = "I"
            else:
                ops = []
                for orb, action in term:
                    if action == 1:
                        ops.append(f"a†_{orb}")
                    else:
                        ops.append(f"a_{orb}")
                term_str = " ".join(ops)

            parts.append(f"{coeff_str} [{term_str}]")

        return " + ".join(parts) if parts else "0"

    def __len__(self) -> int:
        """Number of terms."""
        return len(self.terms)

    def __iter__(self):
        """Iterate over terms."""
        return iter(self.terms.items())

    def copy(self) -> 'FermionOperator':
        """Return a copy of this operator."""
        result = FermionOperator()
        result.terms = defaultdict(complex, self.terms)
        return result


def creation(orbital: int) -> FermionOperator:
    """Create a†_orbital operator."""
    return FermionOperator(((orbital, 1),), 1.0)


def annihilation(orbital: int) -> FermionOperator:
    """Create a_orbital operator."""
    return FermionOperator(((orbital, 0),), 1.0)


def number_op(orbital: int) -> FermionOperator:
    """Create number operator n_orbital = a†_orbital a_orbital."""
    return creation(orbital) * annihilation(orbital)


def excitation(p: int, q: int) -> FermionOperator:
    """Create excitation operator a†_p a_q."""
    return creation(p) * annihilation(q)


def double_excitation(p: int, q: int, r: int, s: int) -> FermionOperator:
    """Create double excitation operator a†_p a†_q a_r a_s."""
    return creation(p) * creation(q) * annihilation(r) * annihilation(s)
