"""
Bond creation and management.

Provides user-friendly interface for creating and analyzing chemical bonds.

This module serves as the PRIMARY INTERFACE for the Kanad framework.
It exposes all core components needed for quantum chemistry calculations:
- Bond creation and management
- Analysis tools (energy, bonding, properties)
- Optimization tools (geometry, circuits, orbitals)
- All necessary core framework components
"""

# Bond Classes (Primary Interface)
from kanad.core.bonds.bond_factory import BondFactory, BondType
from kanad.core.bonds.base_bond import BaseBond
from kanad.core.bonds.ionic_bond import IonicBond
from kanad.core.bonds.covalent_bond import CovalentBond
from kanad.core.bonds.metallic_bond import MetallicBond

# Core Framework Components (for solver access)
from kanad.core.atom import Atom
from kanad.core.representations.base_representation import Molecule

# Hamiltonians
from kanad.core.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from kanad.core.hamiltonians.ionic_hamiltonian import IonicHamiltonian
from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian

# Ansatze — only verified-working ones (governance/UCC ansatze removed 2026-05-12; see CLEANUP.md)
from kanad.core.ansatze.base_ansatz import BaseAnsatz
from kanad.core.ansatze.hardware_efficient_ansatz import HardwareEfficientAnsatz
from kanad.core.ansatze.physics_driven_ansatz import PhysicsDrivenAnsatz

# Mappers
from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper
from kanad.core.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
# Note: ParityMapper not implemented, using available mappers

# Analysis Module (full exposure)
from kanad.analysis import (
    EnergyAnalyzer,
    BondingAnalyzer,
    CorrelationAnalyzer,
    PropertyCalculator,
    BondLengthScanner,
    ThermochemistryCalculator,
    FrequencyCalculator,
    UVVisCalculator,
    ExcitedStateSolver,
    DOSCalculator,
    UncertaintyAnalyzer
)

# Optimization Module (full exposure)
# QuantumOptimizer / AdaptiveOptimizer removed in M0 truth pass — both required
# the deleted active_space module and raised TypeError on construction.
from kanad.core.optimization import (
    OrbitalOptimizer,
    GeometryOptimizer,
)

__all__ = [
    # Primary Bond Interface
    'BondFactory',
    'BondType',
    'BaseBond',
    'IonicBond',
    'CovalentBond',
    'MetallicBond',

    # Core Components
    'Atom',
    'Molecule',

    # Hamiltonians
    'MolecularHamiltonian',
    'IonicHamiltonian',
    'CovalentHamiltonian',
    'MetallicHamiltonian',

    # Ansatze (verified-working only)
    'BaseAnsatz',
    'HardwareEfficientAnsatz',
    'PhysicsDrivenAnsatz',

    # Mappers
    'JordanWignerMapper',
    'BravyiKitaevMapper',

    # Analysis Tools
    'EnergyAnalyzer',
    'BondingAnalyzer',
    'CorrelationAnalyzer',
    'PropertyCalculator',
    'BondLengthScanner',
    'ThermochemistryCalculator',
    'FrequencyCalculator',
    'UVVisCalculator',
    'ExcitedStateSolver',
    'DOSCalculator',
    'UncertaintyAnalyzer',

    # Optimization Tools
    'OrbitalOptimizer',
    'GeometryOptimizer',
]
