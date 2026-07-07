"""
Native quantum operators for Kanad.

This module provides custom implementations of fermionic and qubit operators,
replacing external dependencies like OpenFermion.
"""

from kanad.core.operators.fermion_operator import (
    FermionOperator,
    creation,
    annihilation,
    number_op,
    excitation,
    double_excitation,
)

from kanad.core.operators.jordan_wigner import (
    jordan_wigner,
    jordan_wigner_sparse_pauli_op,
    build_molecular_hamiltonian_jw,
)

from kanad.core.operators.bravyi_kitaev import (
    bravyi_kitaev,
    bravyi_kitaev_sparse_pauli_op,
    build_molecular_hamiltonian_bk,
)

from kanad.core.operators.spin_operators import build_spin_operators
from kanad.core.operators.excitation_operators import build_excitation_generator

__all__ = [
    'FermionOperator',
    'creation',
    'annihilation',
    'number_op',
    'excitation',
    'double_excitation',
    'jordan_wigner',
    'jordan_wigner_sparse_pauli_op',
    'build_molecular_hamiltonian_jw',
    'bravyi_kitaev',
    'bravyi_kitaev_sparse_pauli_op',
    'build_molecular_hamiltonian_bk',
    'build_spin_operators',
    'build_excitation_generator',
]
