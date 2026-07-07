"""
Decoherence Models for Molecular Quantum Systems

Provides bond-type specific decoherence rates integrating with
Kanad's governance protocols.

Key Timescales:
--------------
- T1 (energy relaxation): Population decay to ground state
- T2 (dephasing): Loss of phase coherence
- T2* (pure dephasing): Inhomogeneous broadening

Relation: 1/T2 = 1/(2T1) + 1/T2*

Bond-Type Dependencies:
----------------------
- Ionic bonds: Fast dephasing (localized charges, strong coupling)
- Covalent bonds: Moderate (shared electrons)
- Metallic bonds: Slow (delocalized, weak local coupling)

References:
----------
1. Nitzan "Chemical Dynamics in Condensed Phases" (2006)
2. May & Kühn "Charge and Energy Transfer Dynamics" (2011)
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Typical decoherence timescales for different bond types (in femtoseconds)
BOND_TYPE_TIMESCALES = {
    'ionic': {
        'T1': 100.0,    # Fast energy relaxation (fs)
        'T2_star': 10.0,  # Very fast dephasing
        'description': 'Localized charges couple strongly to polar environments'
    },
    'covalent': {
        'T1': 500.0,    # Moderate relaxation
        'T2_star': 50.0,   # Moderate dephasing
        'description': 'Shared electrons have intermediate coupling'
    },
    'metallic': {
        'T1': 1000.0,   # Slow relaxation (delocalized)
        'T2_star': 200.0,  # Slow dephasing
        'description': 'Delocalized band electrons weakly coupled locally'
    },
    'hydrogen_bond': {
        'T1': 200.0,    # Fast due to proton motion
        'T2_star': 20.0,   # Fast dephasing
        'description': 'Light proton mass leads to rapid fluctuations'
    },
    'van_der_waals': {
        'T1': 2000.0,   # Weak coupling, slow relaxation
        'T2_star': 500.0,  # Slow dephasing
        'description': 'Weak dispersion interactions, slow dynamics'
    },
    'default': {
        'T1': 500.0,
        'T2_star': 50.0,
        'description': 'Generic molecular system'
    }
}


@dataclass
class DecoherenceRates:
    """Container for decoherence rates."""
    T1: float           # Energy relaxation time (fs)
    T2_star: float      # Pure dephasing time (fs)
    T2: float           # Total dephasing time (fs)
    gamma_1: float      # T1 rate (fs⁻¹)
    gamma_2: float      # T2 rate (fs⁻¹)
    gamma_phi: float    # Pure dephasing rate (fs⁻¹)


class DecoherenceModel:
    """
    Bond-type specific decoherence model.

    Integrates with Kanad's governance protocols to provide
    physically motivated decoherence rates based on bond character.

    Example:
    -------
    >>> from kanad import BondFactory
    >>> from kanad.dynamics.open_quantum import DecoherenceModel

    >>> bond = BondFactory.create_bond('Na', 'Cl', distance=2.36)
    >>> model = DecoherenceModel(bond)
    >>> rates = model.get_rates()
    >>> print(f"T1 = {rates.T1:.1f} fs, T2 = {rates.T2:.1f} fs")
    """

    def __init__(
        self,
        bond_or_molecule,
        temperature: float = 300.0,
        solvent: Optional[str] = None
    ):
        """
        Initialize decoherence model.

        Args:
            bond_or_molecule: Kanad Bond or Molecule
            temperature: Temperature in Kelvin
            solvent: Solvent name (affects rates)
        """
        self.bond = bond_or_molecule
        self.temperature = temperature
        self.solvent = solvent

        # Determine bond type from governance
        self.bond_type = self._detect_bond_type()

        logger.info(f"DecoherenceModel for {self.bond_type} bond")
        logger.info(f"  Temperature: {temperature} K")
        logger.info(f"  Solvent: {solvent or 'vacuum'}")

    def _detect_bond_type(self) -> str:
        """Detect bond type from governance protocol."""
        # Check if bond has governance protocol
        if hasattr(self.bond, 'governance_protocol'):
            protocol = self.bond.governance_protocol
            if hasattr(protocol, 'bond_type'):
                return protocol.bond_type.lower()

        # Check bond type attribute
        if hasattr(self.bond, 'bond_type'):
            return self.bond.bond_type.lower()

        # Infer from electronegativity difference
        if hasattr(self.bond, 'atom_1') and hasattr(self.bond, 'atom_2'):
            delta_en = self._get_electronegativity_difference()
            if delta_en > 1.7:
                return 'ionic'
            elif delta_en > 0.4:
                return 'covalent'
            else:
                return 'metallic'

        return 'default'

    def _get_electronegativity_difference(self) -> float:
        """Get electronegativity difference between bonded atoms."""
        # Pauling electronegativities
        EN = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55,
            'N': 3.04, 'O': 3.44, 'F': 3.98, 'Na': 0.93, 'Mg': 1.31,
            'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
            'K': 0.82, 'Ca': 1.00, 'Br': 2.96, 'I': 2.66
        }

        if hasattr(self.bond, 'atom_1') and hasattr(self.bond, 'atom_2'):
            sym1 = self.bond.atom_1.symbol
            sym2 = self.bond.atom_2.symbol
            en1 = EN.get(sym1, 2.5)
            en2 = EN.get(sym2, 2.5)
            return abs(en1 - en2)

        return 0.0

    def get_rates(self) -> DecoherenceRates:
        """
        Get decoherence rates for this bond type.

        Returns:
            DecoherenceRates with T1, T2, and rates
        """
        timescales = BOND_TYPE_TIMESCALES.get(self.bond_type, BOND_TYPE_TIMESCALES['default'])

        T1 = timescales['T1']
        T2_star = timescales['T2_star']

        # Apply temperature scaling (faster decoherence at higher T)
        T_ratio = 300.0 / self.temperature
        T1 *= T_ratio
        T2_star *= T_ratio

        # Apply solvent effects
        if self.solvent:
            T1, T2_star = self._apply_solvent_effects(T1, T2_star)

        # Total T2: 1/T2 = 1/(2T1) + 1/T2*
        gamma_1 = 1.0 / T1
        gamma_phi = 1.0 / T2_star
        gamma_2 = gamma_1 / 2 + gamma_phi
        T2 = 1.0 / gamma_2

        return DecoherenceRates(
            T1=T1,
            T2_star=T2_star,
            T2=T2,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            gamma_phi=gamma_phi
        )

    def _apply_solvent_effects(self, T1: float, T2_star: float) -> Tuple[float, float]:
        """Apply solvent-specific modifications to rates."""
        # Solvent polarizability effects
        SOLVENT_FACTORS = {
            'water': (0.5, 0.3),      # Fast relaxation in water
            'methanol': (0.6, 0.4),
            'acetonitrile': (0.7, 0.5),
            'dmso': (0.6, 0.4),
            'benzene': (1.5, 2.0),    # Slow in nonpolar
            'vacuum': (10.0, 10.0),   # Very slow in vacuum
        }

        factors = SOLVENT_FACTORS.get(self.solvent.lower(), (1.0, 1.0))

        return T1 * factors[0], T2_star * factors[1]

    def get_lindblad_rates_in_hartree(self) -> Dict[str, float]:
        """
        Get Lindblad rates in Hartree units (for quantum simulation).

        Returns:
            Dict with 'dephasing_rate' and 'relaxation_rate' in Ha⁻¹
        """
        rates = self.get_rates()

        # Convert rate from fs⁻¹ to Hartree (a.u. energy): gamma[Ha] = gamma[fs⁻¹] * AU_TIME_TO_FS
        FS_TO_HARTREE = 0.0241888432651  # AU_TIME_TO_FS (matches quantum_forces.py:47)

        return {
            'relaxation_rate': rates.gamma_1 * FS_TO_HARTREE,  # T1 rate
            'dephasing_rate': rates.gamma_phi * FS_TO_HARTREE,  # Pure dephasing
            'total_dephasing': rates.gamma_2 * FS_TO_HARTREE   # Total T2 rate
        }


def get_decoherence_rates(bond_type: str, temperature: float = 300.0) -> DecoherenceRates:
    """
    Convenience function to get decoherence rates for a bond type.

    Args:
        bond_type: 'ionic', 'covalent', 'metallic', 'hydrogen_bond', 'van_der_waals'
        temperature: Temperature in Kelvin

    Returns:
        DecoherenceRates
    """
    class DummyBond:
        def __init__(self, btype):
            self.bond_type = btype

    model = DecoherenceModel(DummyBond(bond_type), temperature=temperature)
    return model.get_rates()


def estimate_T1_T2(
    energy_gap: float,
    reorganization_energy: float,
    temperature: float = 300.0
) -> Tuple[float, float]:
    """
    Estimate T1 and T2 from physical parameters.

    Uses Marcus theory for T1 and fluctuation analysis for T2.

    Args:
        energy_gap: Electronic energy gap in Hartree
        reorganization_energy: λ in Hartree
        temperature: Temperature in Kelvin

    Returns:
        (T1, T2) in femtoseconds
    """
    # Thermal energy
    kT = 0.00095 * (temperature / 300.0)  # Ha

    # Marcus rate for T1
    # k = (2π/ℏ)|V|² / √(4πλkT) * exp(-(ΔG + λ)²/(4λkT))
    # Approximate |V| ~ 0.001 Ha for typical coupling
    V = 0.001  # Ha

    if reorganization_energy > 0 and kT > 0:
        prefactor = 2 * np.pi * V**2 / np.sqrt(4 * np.pi * reorganization_energy * kT)
        exponent = -(energy_gap + reorganization_energy)**2 / (4 * reorganization_energy * kT)
        k_T1 = prefactor * np.exp(exponent)  # Ha

        # Convert a.u. time (1/rate) to fs: T[fs] = T[a.u.] * AU_TIME_TO_FS
        T1 = (1.0 / k_T1) * 0.0241888432651  # fs
    else:
        T1 = 1000.0  # Default 1 ps

    # T2* from energy gap fluctuations
    # δE ~ √(2λkT) for Gaussian fluctuations
    # T2* ~ ℏ/δE
    if reorganization_energy > 0 and kT > 0:
        delta_E = np.sqrt(2 * reorganization_energy * kT)
        # Convert a.u. time (1/delta_E) to fs: T[fs] = T[a.u.] * AU_TIME_TO_FS
        T2_star = (1.0 / delta_E) * 0.0241888432651  # fs
    else:
        T2_star = 100.0  # Default 100 fs

    # Bound to reasonable values
    T1 = max(10.0, min(10000.0, T1))
    T2_star = max(1.0, min(1000.0, T2_star))

    # T2 from relation
    T2 = 1.0 / (0.5 / T1 + 1.0 / T2_star)

    return T1, T2
