"""
Environmental Effects Module

Modulates molecular Hamiltonians based on external conditions:
- Temperature: Thermal effects on bonding
- Pressure: Compression and phase transitions
- pH: Protonation states
- Solvent: Solvation and dielectric effects
- Electric fields: External perturbations

These effects integrate with Kanad's governance system to provide
realistic molecular simulations under various conditions.
"""

from kanad.core.environment.temperature import TemperatureModulator
from kanad.core.environment.solvent import SolventModulator, SOLVENT_DATABASE
from kanad.core.environment.ph_effects import pHModulator, ProtonationSite
from kanad.core.environment.pressure import PressureModulator

# Environment Integration (unified interface)
from kanad.core.environment.integration import (
    EnvironmentIntegration,
    EnvironmentConditions,
    EnvironmentCorrectedEnergy,
    create_environment,
    compute_energy_in_environment,
    get_solvent_screening,
    estimate_rate_enhancement
)

__all__ = [
    # Core Modulators
    'TemperatureModulator',
    'SolventModulator',
    'pHModulator',
    'ProtonationSite',
    'PressureModulator',

    # Data
    'SOLVENT_DATABASE',

    # Unified Integration
    'EnvironmentIntegration',
    'EnvironmentConditions',
    'EnvironmentCorrectedEnergy',
    'create_environment',
    'compute_energy_in_environment',
    'get_solvent_screening',
    'estimate_rate_enhancement',
]
