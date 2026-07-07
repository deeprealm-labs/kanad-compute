"""
Core quantum chemistry components.

This module provides foundational classes for:
- Molecular integrals (one-electron, two-electron)
- Hamiltonians (molecular, ionic, covalent, metallic)
- Fermionic-to-qubit mappers (Jordan-Wigner, Bravyi-Kitaev)
- Quantum representations for different bond types
- Physical constants and atomic data
"""

# Constants and atomic data
from kanad.core.constants.physical_constants import CONSTANTS, PhysicalConstants
from kanad.core.constants.atomic_data import PeriodicTable, AtomicProperties

# Core classes
from kanad.core.atom import Atom
from kanad.core.molecule import Molecule, MolecularHamiltonian
from kanad.core.correlation import MP2Solver
from kanad.core.gradients import GradientCalculator
from kanad.core.lattice import Lattice

# Submodules (expose for easy access)
from kanad.core import hamiltonians
from kanad.core import mappers
from kanad.core import operators
from kanad.core import integrals
from kanad.core import representations

__all__ = [
    # Constants
    'CONSTANTS',
    'PhysicalConstants',
    'PeriodicTable',
    'AtomicProperties',

    # Core classes
    'Atom',
    'Molecule',
    'MolecularHamiltonian',
    'MP2Solver',
    'GradientCalculator',
    'Lattice',

    # Submodules
    'hamiltonians',
    'mappers',
    'operators',
    'integrals',
    'representations',
]
