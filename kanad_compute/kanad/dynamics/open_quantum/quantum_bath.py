"""
Quantum Bath Models for Open Quantum Systems

Provides bath models that connect molecular systems to their environment
(solvent, phonon modes, etc.) using physically motivated spectral densities.

Bath Models:
-----------
1. Spin-Boson: System coupled to harmonic oscillator bath
2. Caldeira-Leggett: Ohmic spectral density
3. Drude-Lorentz: Condensed phase with memory (common for solvents)

Spectral Density J(ω):
    Describes bath coupling strength vs frequency.
    Determines:
    - Reorganization energy λ = ∫ J(ω)/ω dω
    - Correlation time τ_c
    - Decoherence rates T1, T2

Common Forms:
- Ohmic: J(ω) = η·ω·exp(-ω/ω_c)
- Drude-Lorentz: J(ω) = 2λγω/(ω² + γ²)
- Super-Ohmic: J(ω) = η·ω³·exp(-ω/ω_c)

References:
----------
1. Leggett et al. (1987) Rev. Mod. Phys. 59, 1 - Spin-boson model
2. Caldeira & Leggett (1983) Ann. Phys. 149, 374 - System-bath theory
3. Ishizaki & Fleming (2009) J. Chem. Phys. 130, 234111 - HEOM for chemistry
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Solvent properties (reorganization energy in Ha, cutoff frequency in Ha)
SOLVENT_PROPERTIES = {
    'water': {
        'reorganization_energy': 0.0037,  # ~800 cm⁻¹ = 0.0037 Ha
        'cutoff_frequency': 0.00091,       # ~200 cm⁻¹
        'dielectric': 80.0,
        'refractive_index': 1.33,
        'temperature': 300.0,
    },
    'methanol': {
        'reorganization_energy': 0.0032,
        'cutoff_frequency': 0.00091,
        'dielectric': 33.0,
        'refractive_index': 1.33,
        'temperature': 300.0,
    },
    'acetonitrile': {
        'reorganization_energy': 0.0023,
        'cutoff_frequency': 0.00091,
        'dielectric': 37.0,
        'refractive_index': 1.34,
        'temperature': 300.0,
    },
    'dmso': {
        'reorganization_energy': 0.0028,
        'cutoff_frequency': 0.00068,
        'dielectric': 47.0,
        'refractive_index': 1.48,
        'temperature': 300.0,
    },
    'benzene': {
        'reorganization_energy': 0.0014,
        'cutoff_frequency': 0.00046,
        'dielectric': 2.3,
        'refractive_index': 1.50,
        'temperature': 300.0,
    },
    'vacuum': {
        'reorganization_energy': 0.0,
        'cutoff_frequency': 0.0001,
        'dielectric': 1.0,
        'refractive_index': 1.0,
        'temperature': 300.0,
    }
}


class QuantumBath:
    """
    Base class for quantum bath models.

    A bath represents the environment (solvent, phonons, etc.) that
    interacts with and decoheres the molecular system.

    Key quantities:
    - Spectral density J(ω): coupling strength vs frequency
    - Reorganization energy λ: total bath coupling
    - Correlation time τ_c: memory timescale
    - Temperature T: thermal occupation
    """

    def __init__(
        self,
        reorganization_energy: float,
        cutoff_frequency: float,
        temperature: float = 300.0
    ):
        """
        Initialize bath.

        Args:
            reorganization_energy: λ in Hartree (~0.001-0.01 Ha for solvents)
            cutoff_frequency: ω_c in Hartree (~0.001 Ha typical)
            temperature: Temperature in Kelvin
        """
        self.lambda_reorg = reorganization_energy
        self.omega_c = cutoff_frequency
        self.temperature = temperature

        # Thermal energy in Hartree
        self.kT = 0.000316682 * temperature / 100.0  # kT at 300K ≈ 0.00095 Ha

        logger.debug(f"QuantumBath initialized")
        logger.debug(f"  λ = {self.lambda_reorg:.6f} Ha")
        logger.debug(f"  ω_c = {self.omega_c:.6f} Ha")
        logger.debug(f"  T = {self.temperature} K")

    def spectral_density(self, omega: float) -> float:
        """
        Compute spectral density J(ω).

        Override in subclasses for specific bath types.

        Args:
            omega: Frequency in Hartree

        Returns:
            J(ω): Spectral density
        """
        raise NotImplementedError("Subclass must implement spectral_density")

    def correlation_function(self, t: float) -> complex:
        """
        Compute bath correlation function C(t) = ⟨B(t)B(0)⟩.

        Related to spectral density via:
        C(t) = ∫ J(ω)[coth(βω/2)cos(ωt) - i·sin(ωt)] dω / π

        Args:
            t: Time in Hartree⁻¹

        Returns:
            C(t): Correlation function
        """
        # Numerical integration
        omega_max = 10 * self.omega_c
        omega = np.linspace(0.001, omega_max, 1000)
        dw = omega[1] - omega[0]

        J = np.array([self.spectral_density(w) for w in omega])

        # Bose factor
        if self.kT > 0:
            n_bose = 1.0 / (np.exp(omega / self.kT) - 1)
            n_bose = np.where(omega < 1e-10, self.kT / omega, n_bose)
        else:
            n_bose = np.zeros_like(omega)

        # Real part: ∫ J(ω)(2n+1)cos(ωt) dω
        real_part = np.sum(J * (2 * n_bose + 1) * np.cos(omega * t)) * dw / np.pi

        # Imaginary part: -∫ J(ω)sin(ωt) dω
        imag_part = -np.sum(J * np.sin(omega * t)) * dw / np.pi

        return real_part + 1j * imag_part

    def get_correlation_time(self) -> float:
        """
        Estimate correlation time τ_c from bath parameters.

        Returns:
            τ_c: Correlation time in Hartree⁻¹
        """
        # For Drude-Lorentz: τ_c = 1/γ = 1/ω_c
        return 1.0 / self.omega_c if self.omega_c > 0 else float('inf')

    def get_lindblad_operators(self, system_H: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Derive Lindblad operators from bath properties.

        Uses secular approximation (valid when ω_ab >> γ).

        Args:
            system_H: System Hamiltonian

        Returns:
            (operators, rates): Lindblad operators and rates
        """
        # Diagonalize system Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(system_H)
        dim = len(eigenvalues)

        operators = []
        rates = []

        # Transition operators between energy eigenstates
        for i in range(dim):
            for j in range(i + 1, dim):
                omega_ij = eigenvalues[j] - eigenvalues[i]

                if omega_ij < 1e-10:
                    continue

                # Jump operator: |i⟩⟨j|
                jump_down = np.outer(eigenvectors[:, i], eigenvectors[:, j].conj())
                jump_up = jump_down.conj().T

                # Rate from spectral density
                J_omega = self.spectral_density(omega_ij)

                # Thermal factors
                if self.kT > 0:
                    n_th = 1.0 / (np.exp(omega_ij / self.kT) - 1)
                else:
                    n_th = 0

                # Emission (down) rate
                # Breuer-Petruccione golden rule: gamma(w) = 2*J(w)*(n+1); no extra pi
                # (the 1/pi normalization already lives in correlation_function, lines 169/172)
                rate_down = 2 * J_omega * (n_th + 1)
                # Absorption (up) rate
                rate_up = 2 * J_omega * n_th

                operators.extend([jump_down, jump_up])
                rates.extend([rate_down, rate_up])

        return operators, rates


class SpinBosonBath(QuantumBath):
    """
    Spin-Boson model bath with Ohmic spectral density.

    J(ω) = η · ω · exp(-ω/ω_c)

    Common for:
    - Qubit decoherence in solid-state systems
    - Electronic transitions with weak coupling
    """

    def __init__(
        self,
        coupling_strength: float = 0.1,
        cutoff_frequency: float = 0.001,
        temperature: float = 300.0
    ):
        """
        Initialize Ohmic spin-boson bath.

        Args:
            coupling_strength: η (dimensionless)
            cutoff_frequency: ω_c in Hartree
            temperature: Temperature in Kelvin
        """
        # Reorganization energy for Ohmic: λ = η·ω_c
        lambda_reorg = coupling_strength * cutoff_frequency
        super().__init__(lambda_reorg, cutoff_frequency, temperature)
        self.eta = coupling_strength

    def spectral_density(self, omega: float) -> float:
        """Ohmic spectral density: J(ω) = η·ω·exp(-ω/ω_c)."""
        if omega <= 0:
            return 0.0
        return self.eta * omega * np.exp(-omega / self.omega_c)


class DruideLorenzBath(QuantumBath):
    """
    Drude-Lorentz bath (common for solvents).

    J(ω) = 2λγω / (ω² + γ²)

    Where:
    - λ: Reorganization energy
    - γ: Cutoff frequency (inverse correlation time)

    This spectral density gives exponential correlation:
    C(t) ∝ exp(-γ|t|)

    Common for:
    - Solvated molecules
    - Protein environments
    - Energy transfer in photosynthesis
    """

    def __init__(
        self,
        reorganization_energy: float,
        cutoff_frequency: float,
        temperature: float = 300.0
    ):
        super().__init__(reorganization_energy, cutoff_frequency, temperature)

    def spectral_density(self, omega: float) -> float:
        """Drude-Lorentz spectral density: J(ω) = 2λγω/(ω² + γ²)."""
        gamma = self.omega_c
        return 2 * self.lambda_reorg * gamma * omega / (omega**2 + gamma**2)


def create_bath_from_solvent(solvent_name: str) -> DruideLorenzBath:
    """
    Create a bath model from solvent name.

    Uses tabulated solvent properties.

    Args:
        solvent_name: One of 'water', 'methanol', 'acetonitrile', 'dmso', 'benzene'

    Returns:
        DruideLorenzBath configured for the solvent
    """
    if solvent_name.lower() not in SOLVENT_PROPERTIES:
        logger.warning(f"Unknown solvent '{solvent_name}', using water properties")
        solvent_name = 'water'

    props = SOLVENT_PROPERTIES[solvent_name.lower()]

    bath = DruideLorenzBath(
        reorganization_energy=props['reorganization_energy'],
        cutoff_frequency=props['cutoff_frequency'],
        temperature=props['temperature']
    )

    logger.info(f"Created bath for {solvent_name}")
    logger.info(f"  λ = {bath.lambda_reorg:.6f} Ha ({bath.lambda_reorg * 219474:.0f} cm⁻¹)")
    logger.info(f"  ω_c = {bath.omega_c:.6f} Ha ({bath.omega_c * 219474:.0f} cm⁻¹)")

    return bath
