"""
Fast direct construction of Pauli operators from molecular integrals.

Supports multiple fermion-to-qubit mappings:
- Jordan-Wigner (default)
- Bravyi-Kitaev

Works for ALL bonding types with ZERO accuracy loss.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
import logging

logger = logging.getLogger(__name__)


def build_molecular_hamiltonian_pauli(
    h_core: np.ndarray,
    eri: np.ndarray,
    nuclear_repulsion: float,
    n_orbitals: int,
    mapper: str = 'jordan_wigner'
) -> SparsePauliOp:
    """
    Build molecular Hamiltonian as Pauli operators using native Kanad transformations.

    Supports multiple fermion-to-qubit mappings:
    - jordan_wigner: Standard JW transformation (default, recommended)
    - bravyi_kitaev: BK transformation (O(log n) Pauli weight)

    Args:
        h_core: One-electron integrals (MO basis)
        eri: Two-electron integrals (MO basis)
        nuclear_repulsion: Nuclear repulsion energy
        n_orbitals: Number of spatial orbitals
        mapper: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')

    Returns:
        SparsePauliOp representing the Hamiltonian
    """
    mapper_lower = mapper.lower()

    if mapper_lower in ['jordan_wigner', 'jw']:
        # Use native Jordan-Wigner transformation
        from kanad.core.operators.jordan_wigner import build_molecular_hamiltonian_jw
        logger.info("Using native Jordan-Wigner transformation")

        return build_molecular_hamiltonian_jw(
            h_mo=h_core,
            eri_mo=eri,
            nuclear_repulsion=nuclear_repulsion,
            n_electrons=0  # Not needed for Hamiltonian construction
        )

    elif mapper_lower in ['bravyi_kitaev', 'bk']:
        # Use native Bravyi-Kitaev transformation
        from kanad.core.operators.bravyi_kitaev import build_molecular_hamiltonian_bk
        logger.info("Using native Bravyi-Kitaev transformation")

        return build_molecular_hamiltonian_bk(
            h_mo=h_core,
            eri_mo=eri,
            nuclear_repulsion=nuclear_repulsion,
            n_electrons=0  # Not needed for Hamiltonian construction
        )

    else:
        raise ValueError(f"Unknown mapper: {mapper}. Supported: 'jordan_wigner', 'bravyi_kitaev'")
