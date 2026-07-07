"""
Bravyi-Kitaev fermion-to-qubit mapper.

Thin wrapper around OpenFermion's verified BK transform. The previous in-house
mapper was structurally broken (`_get_update_set` / `_get_flip_set` were endpoint
placeholders, not Seeley-Richard-Love tree sets) and was replaced in M1 (2026-05-26).

The mapper conforms to the `BaseMapper` interface used by
`kanad.core.hamiltonians.PauliConverter`, with the same Qiskit little-endian
Pauli-string convention as `JordanWignerMapper`.

JW and BK produce isospectral Hamiltonians on the same fermionic operator.
BK's advantage is `O(log n)` Pauli weight on n-qubit operators, useful for
gate-count reductions on >12-qubit problems.
"""

from typing import Dict
from kanad.core.mappers.base_mapper import BaseMapper
from kanad.core.operators.bravyi_kitaev import bravyi_kitaev as _bravyi_kitaev_transform
from kanad.core.operators.fermion_operator import FermionOperator


class BravyiKitaevMapper(BaseMapper):
    """Bravyi-Kitaev mapper.

    Uses OpenFermion's BK transform under the hood (already a Kanad dependency).
    """

    def n_qubits(self, n_spin_orbitals: int) -> int:
        """One qubit per spin orbital (same as JW)."""
        return n_spin_orbitals

    def map_creation_operator(self, orbital: int, n_orbitals: int) -> Dict[str, complex]:
        """Map ``a†_orbital`` to a Pauli-string dict."""
        return _bravyi_kitaev_transform(FermionOperator(((orbital, 1),)), n_qubits=n_orbitals)

    def map_annihilation_operator(self, orbital: int, n_orbitals: int) -> Dict[str, complex]:
        """Map ``a_orbital`` to a Pauli-string dict."""
        return _bravyi_kitaev_transform(FermionOperator(((orbital, 0),)), n_qubits=n_orbitals)

    def map_number_operator(self, orbital: int, n_orbitals: int) -> Dict[str, complex]:
        """Map number operator ``n_i = a†_i a_i`` via BK."""
        return _bravyi_kitaev_transform(
            FermionOperator(((orbital, 1), (orbital, 0))),
            n_qubits=n_orbitals,
        )

    def map_excitation_operator(
        self,
        orbital_from: int,
        orbital_to: int,
        n_orbitals: int,
    ) -> Dict[str, complex]:
        """Map single excitation ``a†_{orbital_to} a_{orbital_from}`` via BK."""
        if orbital_from == orbital_to:
            return self.map_number_operator(orbital_from, n_orbitals)
        return _bravyi_kitaev_transform(
            FermionOperator(((orbital_to, 1), (orbital_from, 0))),
            n_qubits=n_orbitals,
        )

    def map_double_excitation(
        self,
        orb_from_1: int,
        orb_from_2: int,
        orb_to_1: int,
        orb_to_2: int,
        n_orbitals: int,
    ) -> Dict[str, complex]:
        """Map ``a†_{to_1} a†_{to_2} a_{from_2} a_{from_1}`` via BK.

        Signature kept compatible with `JordanWignerMapper.map_double_excitation`
        for use by `PauliConverter`.
        """
        return _bravyi_kitaev_transform(
            FermionOperator((
                (orb_to_1, 1),
                (orb_to_2, 1),
                (orb_from_2, 0),
                (orb_from_1, 0),
            )),
            n_qubits=n_orbitals,
        )

    def __repr__(self) -> str:
        return "BravyiKitaevMapper(backend='openfermion', weight='O(log n)')"
