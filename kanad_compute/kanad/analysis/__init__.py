"""
Kanad Analysis Module
=====================

Comprehensive molecular analysis tools for quantum chemistry calculations.
Provides post-processing utilities for VQE/SQD results and molecular properties.

Categories
----------

Core Analysis:
    - EnergyAnalyzer: Energy decomposition and convergence analysis
    - BondingAnalyzer: Bond order and bonding type classification
    - CorrelationAnalyzer: Electron correlation analysis
    - PropertyCalculator: Dipole moment, polarizability, molecular properties

Spectroscopy:
    - UVVisCalculator: UV-Vis absorption spectra
    - ExcitedStateSolver: Excited state calculations
    - VibronicCalculator: Vibronic coupling analysis
    - NMRCalculator: NMR chemical shifts
    - RamanIRCalculator: Raman and IR spectroscopy

Thermodynamics:
    - ThermochemistryCalculator: Thermochemistry (ΔH, ΔG, ΔS)
    - FrequencyCalculator: Vibrational frequencies and normal modes

Medicinal-chemistry descriptors:
    - physicochemical_from_smiles: validated RDKit 2D descriptors (logP, TPSA, ...)
    - quantum_reactivity: conceptual-DFT reactivity indices (χ, η, S, ω) from the wavefunction
    - druglikeness_rules: Lipinski / Veber / Ghose rule-filter violations

Configuration Space:
    - BondLengthScanner: Potential energy surface scans
    - ConfigurationExplorer: Reaction pathway exploration
    - ReactionPath: Reaction pathway representation

Materials Science:
    - DOSCalculator: Density of states calculations

Uncertainty:
    - UncertaintyAnalyzer: Error and uncertainty quantification

Example
-------
>>> from kanad.analysis import PropertyCalculator, EnergyAnalyzer
>>> calc = PropertyCalculator(hamiltonian)
>>> dipole = calc.compute_dipole_moment()
>>> print(f"Dipole: {dipole['dipole_magnitude']:.4f} D")

See CLAUDE.md for comprehensive documentation.
"""

# Core Analysis
from kanad.analysis.energy_analysis import (
    EnergyAnalyzer,
    BondingAnalyzer,
    CorrelationAnalyzer
)
from kanad.analysis.property_calculator import PropertyCalculator

# Spectroscopy
from kanad.analysis.spectroscopy import UVVisCalculator, ExcitedStateSolver, VibronicCalculator
from kanad.analysis.nmr_calculator import NMRCalculator
from kanad.analysis.raman_calculator import RamanIRCalculator

# Thermodynamics
from kanad.analysis.thermochemistry import ThermochemistryCalculator
from kanad.analysis.vibrational_analysis import FrequencyCalculator

# Medicinal-chemistry descriptors (honest provenance: RDKit physchem + conceptual-DFT reactivity)
from kanad.analysis.molecular_descriptors import (
    PhysicochemicalDescriptors,
    QuantumReactivityDescriptors,
    DrugLikenessRules,
    physicochemical_from_smiles,
    quantum_reactivity,
    druglikeness_rules,
)

# Configuration Space
from kanad.analysis.bond_scanner import BondLengthScanner
from kanad.analysis.configuration_explorer import ConfigurationExplorer, ConfigurationSnapshot, ReactionPath

# Materials Science
from kanad.analysis.dos_calculator import DOSCalculator

# Uncertainty
from kanad.analysis.uncertainty import UncertaintyAnalyzer

# Dynamics and Reactions Analysis (basic)
from kanad.analysis.dynamics_analysis import (
    TrajectoryAnalyzer,
    ReactionAnalyzer,
    NAMDAnalyzer,
    EnvironmentEffectsAnalyzer,
    TrajectoryAnalysisResult,
    ReactionAnalysisResult,
    NAMDAnalysisResult,
    analyze_trajectory,
    analyze_reaction,
    analyze_namd,
    compute_rate_constant
)

__all__ = [
    # Core Analysis
    'EnergyAnalyzer',
    'BondingAnalyzer',
    'CorrelationAnalyzer',
    'PropertyCalculator',

    # Spectroscopy
    'UVVisCalculator',
    'ExcitedStateSolver',
    'VibronicCalculator',
    'NMRCalculator',
    'RamanIRCalculator',

    # Thermodynamics
    'ThermochemistryCalculator',
    'FrequencyCalculator',

    # Medicinal-chemistry descriptors
    'PhysicochemicalDescriptors',
    'QuantumReactivityDescriptors',
    'DrugLikenessRules',
    'physicochemical_from_smiles',
    'quantum_reactivity',
    'druglikeness_rules',

    # Configuration Space
    'BondLengthScanner',
    'ConfigurationExplorer',
    'ConfigurationSnapshot',
    'ReactionPath',

    # Materials Science
    'DOSCalculator',

    # Uncertainty
    'UncertaintyAnalyzer',

    # Dynamics and Reactions Analysis (basic)
    'TrajectoryAnalyzer',
    'ReactionAnalyzer',
    'NAMDAnalyzer',
    'EnvironmentEffectsAnalyzer',
    'TrajectoryAnalysisResult',
    'ReactionAnalysisResult',
    'NAMDAnalysisResult',
    'analyze_trajectory',
    'analyze_reaction',
    'analyze_namd',
    'compute_rate_constant',
]
