"""
Kanad — wavefunction-first quantum chemistry framework.

Public API last truth-passed 2026-05-25 (M0). See PHILOSOPHY.md for guiding
principles and CLEANUP.md for what was removed in the 2026-05-12 cleanup.
"""

__version__ = "0.1.2"
__author__ = "DeepRealm Labs"

# === Core ===
from kanad.core.molecule import Molecule
# `MolecularHamiltonian` at the top level is the *concrete* multi-atom Hamiltonian
# from core.molecule (works for user code like `MolecularHamiltonian(atoms)`).
# The abstract base lives at kanad.core.hamiltonians.molecular_hamiltonian and
# is no longer exposed at the top level (calling it directly raises TypeError —
# abstract classes don't construct).
from kanad.core.molecule import MolecularHamiltonian as MultiAtomHamiltonian
MolecularHamiltonian = MultiAtomHamiltonian

# === Bonds ===
from kanad.core.bonds.bond_factory import BondFactory
from kanad.core.bonds.ionic_bond import IonicBond
from kanad.core.bonds.covalent_bond import CovalentBond
from kanad.core.bonds.metallic_bond import MetallicBond

# === Hamiltonians ===
from kanad.core.hamiltonians.ionic_hamiltonian import IonicHamiltonian
from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
from kanad.core.hamiltonians.metallic_hamiltonian import MetallicHamiltonian

# === Ansatze (verified-working only) ===
from kanad.core.ansatze.base_ansatz import BaseAnsatz
from kanad.core.ansatze.hardware_efficient_ansatz import HardwareEfficientAnsatz
from kanad.core.ansatze.physics_driven_ansatz import PhysicsDrivenAnsatz

# === Solvers ===
from kanad.solvers.vqe_solver import VQESolver
from kanad.solvers.ci_solver import CISolver
# DeterministicCI: legacy deterministic HF+S+D classical CI (was misnamed
# `SQDSolver`; the alias is preserved). SamplingSQDSolver: the real
# circuit-sampling SQD (M4).
from kanad.solvers.deterministic_ci import DeterministicCI, SQDSolver
from kanad.solvers.sampling_sqd import SamplingSQDSolver
try:
    from kanad.solvers.excited_states_solver import ExcitedStatesSolver
except ImportError:
    ExcitedStatesSolver = None

# === Molecular Builder (unified workflow entry point) ===
from kanad.builder import MolecularBuilder, QuantumSystem

# === Backends (optional — may not be installed) ===
try:
    from kanad.backends.qiskit_backend import QiskitBackend
except ImportError:
    QiskitBackend = None

try:
    from kanad.backends.ibm import IBMBackend as IBMRuntimeBackend
except ImportError:
    IBMRuntimeBackend = None

try:
    from kanad.backends.bluequbit import BlueQubitBackend
except ImportError:
    BlueQubitBackend = None

# === Governance (protocol framework — abstract base, ionic/covalent/metallic) ===
# Note: governance *ansatze* were removed (returned HF, no correlation). The
# governance *protocol* framework is preserved and reusable. New verified-
# correct governance ansatze can be re-added in the future.
from kanad.core.governance.protocols.base_protocol import (
    BaseGovernanceProtocol,
    BondingType,
    GovernanceRule,
)
from kanad.core.governance.protocols.ionic_protocol import IonicGovernanceProtocol
from kanad.core.governance.protocols.covalent_protocol import CovalentGovernanceProtocol
from kanad.core.governance.protocols.metallic_protocol import MetallicGovernanceProtocol

__all__ = [
    # Core
    'Molecule',
    # Bonds
    'BondFactory',
    'IonicBond',
    'CovalentBond',
    'MetallicBond',
    # Hamiltonians
    'MolecularHamiltonian',     # alias of MultiAtomHamiltonian (concrete)
    'MultiAtomHamiltonian',
    'IonicHamiltonian',
    'CovalentHamiltonian',
    'MetallicHamiltonian',
    # Ansatze
    'BaseAnsatz',
    'HardwareEfficientAnsatz',
    'PhysicsDrivenAnsatz',
    # Solvers
    'VQESolver',
    'CISolver',
    'DeterministicCI',
    'SQDSolver',
    'SamplingSQDSolver',
    'ExcitedStatesSolver',
    # Builder
    'MolecularBuilder',
    'QuantumSystem',
    # Backends
    'QiskitBackend',
    'IBMRuntimeBackend',
    'BlueQubitBackend',
    # Governance protocols
    'BaseGovernanceProtocol',
    'BondingType',
    'GovernanceRule',
    'IonicGovernanceProtocol',
    'CovalentGovernanceProtocol',
    'MetallicGovernanceProtocol',
]
