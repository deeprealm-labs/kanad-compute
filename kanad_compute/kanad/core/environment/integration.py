"""
Environment Integration Module

Unified interface for integrating environmental effects with:
- Hamiltonian construction
- Molecular dynamics simulations
- Chemical reactions
- Property calculations

This module provides the critical connection between environmental conditions
(temperature, pressure, solvent, pH) and quantum chemistry calculations.

Theory:
------
Environmental effects modify molecular properties through:

1. Temperature (T):
   - Thermal population of vibrational states
   - Free energy: G = H - TS
   - Rate constants: k ∝ exp(-ΔG‡/RT)

2. Pressure (P):
   - Bond compression: r(P) = r₀(1 - α*P)
   - Activation volume: ΔV‡ affects rate
   - Phase transitions at high P

3. Solvent (ε):
   - Dielectric screening: V_solv = V_vac/ε
   - Solvation free energy: ΔG_solv
   - Reaction field effects on dipoles

4. pH:
   - Protonation equilibria: pKa
   - Charge state populations
   - Mechanism switching

References:
----------
1. Tomasi et al. Chem. Rev. (2005) - PCM solvation
2. Eyring J. Chem. Phys. (1935) - TST with environment
3. Evans & Polanyi Trans. Faraday Soc. (1935) - Activation energy
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Physical constants
K_BOLTZMANN = 3.166811563e-6  # Ha/K
GAS_CONSTANT = K_BOLTZMANN  # Ha/(mol*K)
HARTREE_TO_KCAL = 627.509


@dataclass
class EnvironmentConditions:
    """
    Container for environmental conditions.

    Attributes:
        temperature: Temperature in Kelvin
        pressure: Pressure in atm
        solvent: Solvent name or 'vacuum'
        pH: pH value (for protonation effects)
        ionic_strength: Ionic strength in M (for Debye-Huckel)
        electric_field: External electric field in V/Å
    """
    temperature: float = 298.15
    pressure: float = 1.0
    solvent: str = 'vacuum'
    pH: Optional[float] = None
    ionic_strength: float = 0.0
    electric_field: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'temperature': self.temperature,
            'pressure': self.pressure,
            'solvent': self.solvent,
            'pH': self.pH,
            'ionic_strength': self.ionic_strength,
            'electric_field': self.electric_field.tolist() if self.electric_field is not None else None
        }


@dataclass
class EnvironmentCorrectedEnergy:
    """
    Energy with environmental corrections.

    Attributes:
        gas_phase_energy: Energy in vacuum (Ha)
        total_energy: Environment-corrected energy (Ha)
        temperature_correction: Thermal correction (Ha)
        pressure_correction: Pressure correction (Ha)
        solvation_energy: Solvation free energy (Ha)
        ph_correction: pH-dependent correction (Ha)
        environment: EnvironmentConditions used
    """
    gas_phase_energy: float
    total_energy: float
    temperature_correction: float = 0.0
    pressure_correction: float = 0.0
    solvation_energy: float = 0.0
    ph_correction: float = 0.0
    environment: Optional[EnvironmentConditions] = None

    @property
    def corrections_summary(self) -> Dict[str, float]:
        """Summary of all corrections."""
        return {
            'temperature': self.temperature_correction,
            'pressure': self.pressure_correction,
            'solvation': self.solvation_energy,
            'pH': self.ph_correction,
            'total_correction': self.total_energy - self.gas_phase_energy
        }


class EnvironmentIntegration:
    """
    Unified environment integration for Kanad calculations.

    Provides methods to:
    1. Apply environment to Hamiltonian construction
    2. Modify dynamics with environmental forces
    3. Compute environment-corrected reaction rates
    4. Analyze environmental effects on properties

    Example:
    -------
    >>> from kanad import BondFactory
    >>> from kanad.core.environment.integration import (
    ...     EnvironmentIntegration, EnvironmentConditions
    ... )
    >>>
    >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
    >>> env = EnvironmentConditions(
    ...     temperature=300,
    ...     pressure=1.0,
    ...     solvent='water'
    ... )
    >>>
    >>> integrator = EnvironmentIntegration(env)
    >>> result = integrator.compute_energy(bond)
    >>> print(f"Gas phase: {result.gas_phase_energy:.6f} Ha")
    >>> print(f"In water: {result.total_energy:.6f} Ha")
    >>> print(f"Solvation: {result.solvation_energy:.6f} Ha")
    """

    def __init__(
        self,
        environment: Union[EnvironmentConditions, Dict] = None
    ):
        """
        Initialize environment integration.

        Args:
            environment: EnvironmentConditions or dict with conditions
        """
        if environment is None:
            self.environment = EnvironmentConditions()
        elif isinstance(environment, dict):
            self.environment = EnvironmentConditions(**environment)
        else:
            self.environment = environment

        # Initialize modulators lazily
        self._temp_mod = None
        self._solvent_mod = None
        self._ph_mod = None
        self._pressure_mod = None

        logger.info(
            f"EnvironmentIntegration: T={self.environment.temperature}K, "
            f"P={self.environment.pressure}atm, solvent={self.environment.solvent}"
        )

    @property
    def temperature_modulator(self):
        """Lazy-load temperature modulator."""
        if self._temp_mod is None:
            from kanad.core.environment.temperature import TemperatureModulator
            self._temp_mod = TemperatureModulator()
        return self._temp_mod

    @property
    def solvent_modulator(self):
        """Lazy-load solvent modulator."""
        if self._solvent_mod is None:
            from kanad.core.environment.solvent import SolventModulator
            self._solvent_mod = SolventModulator()
        return self._solvent_mod

    @property
    def ph_modulator(self):
        """Lazy-load pH modulator."""
        if self._ph_mod is None:
            from kanad.core.environment.ph_effects import pHModulator
            self._ph_mod = pHModulator()
        return self._ph_mod

    @property
    def pressure_modulator(self):
        """Lazy-load pressure modulator."""
        if self._pressure_mod is None:
            from kanad.core.environment.pressure import PressureModulator
            self._pressure_mod = PressureModulator()
        return self._pressure_mod

    def compute_energy(self, bond_or_molecule) -> EnvironmentCorrectedEnergy:
        """
        Compute environment-corrected energy.

        Applies all environmental effects to get the total energy
        under specified conditions.

        Args:
            bond_or_molecule: Kanad Bond or Molecule

        Returns:
            EnvironmentCorrectedEnergy with all corrections
        """
        env = self.environment

        # 1. Get gas-phase energy
        E_gas = self._get_gas_phase_energy(bond_or_molecule)

        # 2. Temperature correction (thermal contribution)
        T_corr = 0.0
        if env.temperature != 298.15:
            T_corr = self._compute_temperature_correction(
                bond_or_molecule, env.temperature
            )

        # 3. Pressure correction
        P_corr = 0.0
        if env.pressure != 1.0:
            P_corr = self._compute_pressure_correction(
                bond_or_molecule, env.pressure
            )

        # 4. Solvation energy
        E_solv = 0.0
        if env.solvent != 'vacuum':
            E_solv = self._compute_solvation_energy(
                bond_or_molecule, env.solvent, env.temperature
            )

        # 5. pH correction
        pH_corr = 0.0
        if env.pH is not None:
            pH_corr = self._compute_ph_correction(
                bond_or_molecule, env.pH, env.temperature
            )

        # Total energy
        E_total = E_gas + T_corr + P_corr + E_solv + pH_corr

        return EnvironmentCorrectedEnergy(
            gas_phase_energy=E_gas,
            total_energy=E_total,
            temperature_correction=T_corr,
            pressure_correction=P_corr,
            solvation_energy=E_solv,
            ph_correction=pH_corr,
            environment=env
        )

    def modify_hamiltonian_parameters(
        self,
        hamiltonian
    ) -> Dict[str, Any]:
        """
        Compute Hamiltonian modifications from environment.

        Instead of adding scalar corrections, this method provides
        modifications to Hamiltonian integrals that incorporate
        environmental effects directly.

        Args:
            hamiltonian: Kanad Hamiltonian object

        Returns:
            Dict with:
                h_core_correction: One-electron integral correction
                eri_screening: Two-electron integral screening factor
                nuclear_correction: Nuclear repulsion correction
                effective_charge: Environment-modified atomic charges
        """
        env = self.environment
        corrections = {
            'h_core_correction': 0.0,
            'eri_screening': 1.0,
            'nuclear_correction': 0.0,
            'effective_charge': {}
        }

        # Solvent dielectric screening for two-electron integrals
        if env.solvent != 'vacuum':
            epsilon = self._get_dielectric(env.solvent)
            # In dielectric medium, bulk Coulomb is screened by the static
            # dielectric constant: g_ijkl -> g_ijkl / ε. The previous Onsager
            # cavity-field factor (3ε)/(2ε+1) saturates at 1.5, giving a
            # screening floor of ~0.67 that never approaches the physical 1/ε.
            corrections['eri_screening'] = 1.0 / epsilon

        # Pressure effect on integrals (bond compression)
        if env.pressure > 1.0:
            # Scaled coordinates affect all integrals
            compression = self._compute_compression_factor(env.pressure)
            # One-electron integrals scale roughly as 1/r
            corrections['h_core_correction'] = (1 - compression) * 0.01  # Small effect

        return corrections

    def get_dynamics_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for environment-aware dynamics.

        Returns parameters that should be passed to MDSimulator
        for proper environmental modeling.

        Returns:
            Dict with dynamics parameters:
                friction_coefficient: For Langevin dynamics (1/fs)
                temperature: Thermostat target (K)
                pressure: Barostat target (atm)
                dielectric_screening: For force calculation
        """
        env = self.environment
        params = {
            'temperature': env.temperature,
            'pressure': env.pressure
        }

        # Solvent viscosity affects friction in Langevin
        if env.solvent != 'vacuum':
            viscosity = self._get_viscosity(env.solvent)
            # Friction γ ∝ viscosity / (particle radius)
            # Typical: γ ~ 5 ps⁻¹ for water, 2 ps⁻¹ for acetonitrile
            params['friction_coefficient'] = self._compute_friction(
                viscosity, env.solvent
            )

            # Dielectric screening for Coulomb forces
            params['dielectric_screening'] = self._get_dielectric(env.solvent)

        return params

    def get_reaction_parameters(
        self,
        barrier: float,
        reaction_energy: float = 0.0
    ) -> Dict[str, Any]:
        """
        Get environment-modified reaction parameters.

        Args:
            barrier: Gas-phase barrier height (Ha)
            reaction_energy: Gas-phase ΔE (Ha)

        Returns:
            Dict with:
                effective_barrier: Environment-modified barrier (Ha)
                effective_delta_E: Environment-modified ΔE (Ha)
                rate_enhancement: Ratio k_env / k_gas
                solvation_effect: Barrier change from solvation (Ha)
        """
        env = self.environment

        # Start with gas-phase values
        barrier_eff = barrier
        delta_E_eff = reaction_energy
        solvation_effect = 0.0

        # 1. Solvent effect on barrier (Hammond postulate)
        # If TS is more polar than reactant, barrier is lowered
        if env.solvent != 'vacuum':
            epsilon = self._get_dielectric(env.solvent)
            # Simplified: polar solvents stabilize TS ~10% more than reactant
            # Better model would use actual dipole moments
            if epsilon > 10:  # Polar solvent
                ts_stabilization = -barrier * 0.05 * (1 - 1/epsilon)
                solvation_effect = ts_stabilization
                barrier_eff = barrier + ts_stabilization

        # 2. Pressure effect on activation volume
        if env.pressure > 1.0:
            # ΔV‡ ≈ -10 cm³/mol for typical reactions
            delta_V = -10e-6  # L/mol = m³/mol / 1000
            R = 8.314  # J/(mol*K)
            T = env.temperature
            # ΔG‡(P) = ΔG‡(1atm) + ΔV‡ * (P - 1)
            # Convert: atm to Pa, then to Ha
            P_Pa = (env.pressure - 1.0) * 101325
            pressure_correction = delta_V * P_Pa / (R * T) * K_BOLTZMANN * T
            barrier_eff += pressure_correction

        # 3. Rate enhancement
        kT = K_BOLTZMANN * env.temperature
        rate_enhancement = np.exp(-(barrier_eff - barrier) / kT)

        return {
            'gas_phase_barrier': barrier,
            'effective_barrier': barrier_eff,
            'gas_phase_delta_E': reaction_energy,
            'effective_delta_E': delta_E_eff,
            'rate_enhancement': rate_enhancement,
            'solvation_effect': solvation_effect
        }

    def compute_reaction_rate_correction(
        self,
        barrier: float,
        temperature: float
    ) -> Dict[str, Any]:
        """
        Compute environment corrections for reaction rate constant.

        This method returns all parameters needed for rate constant corrections
        including Kramers friction, solvent effects, and rate enhancement.

        Args:
            barrier: Gas-phase barrier height in Hartree
            temperature: Temperature in Kelvin

        Returns:
            Dict with:
                rate_enhancement: Ratio k_env / k_gas
                friction_coefficient: Langevin friction (ps⁻¹)
                effective_barrier: Environment-modified barrier (Ha)
                solvation_stabilization: TS stabilization from solvent (Ha)
                dielectric: Solvent dielectric constant
        """
        env = self.environment

        # Get reaction parameters (barrier modification, enhancement)
        reaction_params = self.get_reaction_parameters(barrier, 0.0)

        # Get dynamics parameters (friction, dielectric)
        dynamics_params = self.get_dynamics_parameters()

        result = {
            'rate_enhancement': reaction_params['rate_enhancement'],
            'effective_barrier': reaction_params['effective_barrier'],
            'solvation_stabilization': reaction_params['solvation_effect'],
            'friction_coefficient': dynamics_params.get('friction_coefficient', 0.0),
            'dielectric': dynamics_params.get('dielectric_screening', 1.0),
            'temperature': temperature,
            'solvent': env.solvent,
            'pressure': env.pressure
        }

        # Log for debugging
        logger.debug(f"Reaction rate corrections: enhancement={result['rate_enhancement']:.3f}, "
                    f"friction={result['friction_coefficient']:.2f} ps⁻¹")

        return result

    def apply_to_md_result(
        self,
        md_result,
        bond_or_molecule
    ) -> Dict[str, Any]:
        """
        Post-process MD result with environmental corrections.

        Args:
            md_result: MDResult from simulation
            bond_or_molecule: System being simulated

        Returns:
            Dict with environment-corrected analysis
        """
        env = self.environment

        analysis = {
            'environment': env.to_dict(),
            'original_avg_temperature': getattr(md_result, 'avg_temperature', env.temperature)
        }

        # If trajectory is available, compute environment-corrected energies
        if hasattr(md_result, 'energies') and md_result.energies is not None:
            gas_energies = np.array(md_result.energies)

            # Add solvation correction
            if env.solvent != 'vacuum':
                E_solv = self._compute_solvation_energy(
                    bond_or_molecule, env.solvent, env.temperature
                )
                corrected_energies = gas_energies + E_solv
                analysis['solvation_energy'] = E_solv
                analysis['corrected_energies'] = corrected_energies.tolist()

        return analysis

    # === Private helper methods ===

    def _get_gas_phase_energy(self, bond_or_molecule) -> float:
        """Get gas-phase energy from bond/molecule."""
        if hasattr(bond_or_molecule, 'energy'):
            return bond_or_molecule.energy
        elif hasattr(bond_or_molecule, '_cached_energy'):
            return bond_or_molecule._cached_energy
        elif hasattr(bond_or_molecule, 'hamiltonian'):
            _, E = bond_or_molecule.hamiltonian.solve_scf()
            return E
        else:
            logger.warning("Cannot determine gas phase energy, returning 0")
            return 0.0

    def _compute_temperature_correction(
        self,
        bond_or_molecule,
        temperature: float
    ) -> float:
        """Compute thermal correction to energy."""
        try:
            result = self.temperature_modulator.apply_temperature(
                bond_or_molecule, temperature
            )
            # Return free energy difference from 298.15 K
            return result.get('free_energy', 0.0) - result.get('energy', 0.0)
        except Exception as e:
            logger.debug(f"Temperature correction failed: {e}")
            return 0.0

    def _compute_pressure_correction(
        self,
        bond_or_molecule,
        pressure: float
    ) -> float:
        """Compute pressure correction to energy."""
        try:
            result = self.pressure_modulator.apply_pressure(
                bond_or_molecule, pressure
            )
            # apply_pressure returns E_pressure = E_base + pV_work + E_strain;
            # the correction = E_pressure - E_base = pV_work + strain_energy.
            return result.get('pV_work', 0.0) + result.get('strain_energy', 0.0)
        except Exception as e:
            logger.debug(f"Pressure correction failed: {e}")
            return 0.0

    def _compute_solvation_energy(
        self,
        bond_or_molecule,
        solvent: str,
        temperature: float
    ) -> float:
        """Compute solvation free energy."""
        try:
            result = self.solvent_modulator.apply_solvent(
                bond_or_molecule,
                solvent,
                model='pcm',
                temperature=temperature
            )
            return result.get('solvation_energy', 0.0)
        except Exception as e:
            logger.debug(f"Solvation calculation failed: {e}")
            return 0.0

    def _compute_ph_correction(
        self,
        bond_or_molecule,
        pH: float,
        temperature: float
    ) -> float:
        """Compute pH-dependent energy correction."""
        try:
            result = self.ph_modulator.apply_pH(bond_or_molecule, pH)
            # apply_pH returns free_energy = E_base + protonation_free_energy +
            # charging_energy; the correction = free_energy - E_base.
            return result.get('protonation_free_energy', 0.0) + result.get('charging_energy', 0.0)
        except Exception as e:
            logger.debug(f"pH correction failed: {e}")
            return 0.0

    def _get_dielectric(self, solvent: str) -> float:
        """Get dielectric constant for solvent."""
        dielectrics = {
            'vacuum': 1.0,
            'water': 78.4,
            'acetonitrile': 37.5,
            'dmso': 46.7,
            'methanol': 32.7,
            'ethanol': 24.5,
            'chloroform': 4.8,
            'hexane': 1.9,
            'toluene': 2.4,
            'benzene': 2.3,
            'thf': 7.4,
            'dichloromethane': 8.9
        }
        return dielectrics.get(solvent.lower(), 1.0)

    def _get_viscosity(self, solvent: str) -> float:
        """Get dynamic viscosity in mPa·s (cP)."""
        viscosities = {
            'vacuum': 0.0,
            'water': 0.89,
            'acetonitrile': 0.37,
            'dmso': 2.0,
            'methanol': 0.54,
            'ethanol': 1.07,
            'chloroform': 0.54,
            'hexane': 0.30,
            'toluene': 0.56,
            'benzene': 0.60,
            'thf': 0.46,
            'dichloromethane': 0.41
        }
        return viscosities.get(solvent.lower(), 0.5)

    def _compute_friction(self, viscosity: float, solvent: str) -> float:
        """Compute Langevin friction coefficient from viscosity."""
        # Stokes friction: γ = 6πηr / m
        # For molecular dynamics, typical values: 1-10 ps⁻¹
        # Simplified scaling: γ ≈ 5 * viscosity (in ps⁻¹)
        if viscosity < 0.01:
            return 0.0  # No friction in vacuum
        return 5.0 * viscosity  # ps⁻¹

    def _compute_compression_factor(self, pressure: float) -> float:
        """Compute bond compression factor at given pressure."""
        # Typical compressibility: κ ~ 0.01 GPa⁻¹
        # Compression: ΔV/V = -κ * ΔP
        kappa = 0.01  # GPa⁻¹
        P_GPa = pressure * 101325e-9  # atm to GPa
        compression = kappa * P_GPa
        return min(compression, 0.1)  # Cap at 10%


# Convenience functions

def create_environment(
    temperature: float = 298.15,
    pressure: float = 1.0,
    solvent: str = 'vacuum',
    pH: float = None
) -> EnvironmentConditions:
    """
    Create EnvironmentConditions with specified parameters.

    Args:
        temperature: Temperature in K
        pressure: Pressure in atm
        solvent: Solvent name
        pH: pH value

    Returns:
        EnvironmentConditions object
    """
    return EnvironmentConditions(
        temperature=temperature,
        pressure=pressure,
        solvent=solvent,
        pH=pH
    )


def compute_energy_in_environment(
    bond_or_molecule,
    temperature: float = 298.15,
    pressure: float = 1.0,
    solvent: str = 'vacuum',
    pH: float = None
) -> EnvironmentCorrectedEnergy:
    """
    Compute environment-corrected energy in one call.

    Args:
        bond_or_molecule: Kanad Bond or Molecule
        temperature: Temperature in K
        pressure: Pressure in atm
        solvent: Solvent name
        pH: pH value

    Returns:
        EnvironmentCorrectedEnergy
    """
    env = create_environment(temperature, pressure, solvent, pH)
    integrator = EnvironmentIntegration(env)
    return integrator.compute_energy(bond_or_molecule)


def get_solvent_screening(solvent: str) -> float:
    """
    Get dielectric screening factor for solvent.

    Args:
        solvent: Solvent name

    Returns:
        Screening factor (1/ε_eff)
    """
    integrator = EnvironmentIntegration()
    epsilon = integrator._get_dielectric(solvent)
    # Full bulk dielectric screening 1/ε (not the saturating Onsager factor)
    return 1.0 / epsilon


def estimate_rate_enhancement(
    barrier: float,
    solvent: str = 'water',
    temperature: float = 298.15
) -> float:
    """
    Estimate rate enhancement in solvent vs vacuum.

    Args:
        barrier: Gas-phase barrier in Ha
        solvent: Solvent name
        temperature: Temperature in K

    Returns:
        Rate enhancement factor (k_solv / k_gas)
    """
    env = create_environment(temperature=temperature, solvent=solvent)
    integrator = EnvironmentIntegration(env)
    result = integrator.get_reaction_parameters(barrier)
    return result['rate_enhancement']
