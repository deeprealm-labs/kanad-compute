"""
Temperature Modulator

Modifies molecular Hamiltonians based on temperature, accounting for:
- Thermal population of excited states
- Temperature-dependent bond strengths
- Vibrational zero-point and thermal energy
- Boltzmann-weighted configuration mixing

Physical Basis:
    At temperature T, molecular properties are averaged over thermal
    populations:  ⟨A⟩_T = Σ_i p_i A_i  where p_i ∝ exp(-E_i/kT)

References:
    - McQuarrie "Statistical Mechanics" (2000)
    - Atkins "Physical Chemistry" (2018)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemperatureModulator:
    """
    Apply temperature effects to molecular Hamiltonians.

    Temperature affects:
    1. Bond dissociation: Bonds weaken at high T (Morse potential)
    2. Configuration entropy: More configurations accessible at high T
    3. Vibrational energy: ZPE + thermal contributions
    4. Electronic excitations: Population of excited states

    Example:
        >>> from kanad.bonds import BondFactory
        >>> from kanad.core.environment import TemperatureModulator
        >>>
        >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
        >>> temp_mod = TemperatureModulator()
        >>>
        >>> # Room temperature
        >>> result_298 = temp_mod.apply_temperature(bond, 298.15)
        >>> print(f"Energy at 298K: {result_298['energy']:.6f} Ha")
        >>>
        >>> # High temperature
        >>> result_1000 = temp_mod.apply_temperature(bond, 1000.0)
        >>> print(f"Energy at 1000K: {result_1000['energy']:.6f} Ha")
        >>> print(f"Bond weakening: {result_1000['bond_strength_factor']:.3f}")
    """

    # Physical constants (CODATA 2018)
    k_B_Ha = 3.1668115634556076e-6  # Boltzmann constant in Ha/K
    k_B_eV = 8.617333262e-5          # Boltzmann constant in eV/K
    Ha_to_kcal = 627.509474          # Hartree to kcal/mol

    def __init__(self, temperature: float = 298.15):
        """
        Initialize temperature modulator.

        Args:
            temperature: Temperature in Kelvin (default 298.15 K = 25°C)
        """
        self.temperature = temperature
        self.reference_temp = 298.15  # K (standard conditions)
        self._bond_or_molecule = None  # Set for MD simulations
        logger.info(f"TemperatureModulator initialized: T={temperature:.2f}K")

    def set_molecule(self, bond_or_molecule):
        """Set the molecule/bond for temperature calculations during MD."""
        self._bond_or_molecule = bond_or_molecule

    # NOTE: compute_thermal_correction was removed (fix): it fabricated a
    # thermal "correction" from hardcoded fudge constants (omega=0.0046,
    # anharmonic prefactor -0.01, gap Delta_E=0.1, 0.001*occupation) that
    # never read the actual molecule. MD temperature effects are now routed
    # exclusively through apply_temperature (see MDSimulator), which uses the
    # real per-mode frequencies / electronic gap of self._bond_or_molecule.

    def apply_temperature(
        self,
        bond_or_molecule,
        temperature: float,
        include_vibrational: bool = True,
        include_electronic: bool = True,
        n_excited_states: int = 5
    ) -> Dict[str, Any]:
        """
        Apply temperature effects to molecular system.

        Args:
            bond_or_molecule: Bond or Molecule object
            temperature: Temperature in Kelvin
            include_vibrational: Include vibrational thermal energy
            include_electronic: Include electronic excited state populations
            n_excited_states: Number of excited states for Boltzmann averaging

        Returns:
            Dictionary with:
                energy: Temperature-corrected energy (Ha)
                free_energy: Helmholtz free energy A = E - TS (Ha)
                entropy: Entropy S (Ha/K)
                heat_capacity: Cv (Ha/K)
                bond_strength_factor: Multiplicative factor for bond strength
                thermal_population: Population distribution over states
                vibrational_energy: Thermal vibrational contribution (Ha)
        """
        logger.info(f"Applying temperature effects: T = {temperature:.2f} K")

        # Get base energy and Hamiltonian
        base_energy = self._get_base_energy(bond_or_molecule)

        # 1. Compute thermal bond weakening
        bond_strength_factor = self._compute_bond_strength_factor(temperature)

        # 2. Vibrational thermal energy
        if include_vibrational:
            # Resolve per-mode frequencies once so the vibrational energy and
            # the entropy share the same omega array (needed for a correct
            # per-mode harmonic entropy).
            omega_Ha = self._resolve_vibrational_frequencies_Ha(bond_or_molecule)
            E_vib = self._compute_vibrational_energy(
                bond_or_molecule, temperature, omega_Ha=omega_Ha
            )
        else:
            omega_Ha = np.array([])
            E_vib = 0.0

        # 3. Electronic thermal population
        if include_electronic and hasattr(bond_or_molecule, 'compute_excited_states'):
            electronic_correction, populations = self._compute_electronic_thermal_correction(
                bond_or_molecule, temperature, n_excited_states
            )
        else:
            electronic_correction = 0.0
            populations = [1.0]  # Only ground state

        # 4. Total thermal energy
        E_thermal = base_energy + E_vib + electronic_correction

        # 5. Entropy (per-mode harmonic + electronic from real populations)
        S = self._compute_entropy(
            temperature, omega_Ha, populations
        )

        # 6. Free energy: A = E - TS
        A = E_thermal - temperature * S

        # 7. Heat capacity: Cv = dE_vib/dT via finite difference (reuse omega_Ha)
        Cv = self._compute_heat_capacity(temperature, bond_or_molecule, omega_Ha=omega_Ha)

        return {
            'energy': E_thermal,
            'free_energy': A,
            'entropy': S,
            'heat_capacity': Cv,
            'bond_strength_factor': bond_strength_factor,
            'thermal_population': populations,
            'vibrational_energy': E_vib,
            'electronic_correction': electronic_correction,
            'temperature': temperature
        }

    def scan_temperature(
        self,
        bond_or_molecule,
        temp_range: Tuple[float, float] = (100, 1000),
        n_points: int = 20,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Scan temperature and compute properties at each point.

        Args:
            bond_or_molecule: Bond or Molecule object
            temp_range: (T_min, T_max) in Kelvin
            n_points: Number of temperature points
            **kwargs: Additional arguments for apply_temperature

        Returns:
            Dictionary with arrays:
                temperatures: Temperature values (K)
                energies: Total energy vs T (Ha)
                free_energies: Helmholtz free energy vs T (Ha)
                entropies: Entropy vs T (Ha/K)
                heat_capacities: Cv vs T (Ha/K)
                bond_strengths: Bond strength factor vs T
        """
        T_min, T_max = temp_range
        temperatures = np.linspace(T_min, T_max, n_points)

        energies = []
        free_energies = []
        entropies = []
        heat_capacities = []
        bond_strengths = []

        for T in temperatures:
            result = self.apply_temperature(bond_or_molecule, T, **kwargs)
            energies.append(result['energy'])
            free_energies.append(result['free_energy'])
            entropies.append(result['entropy'])
            heat_capacities.append(result['heat_capacity'])
            bond_strengths.append(result['bond_strength_factor'])

        return {
            'temperatures': temperatures,
            'energies': np.array(energies),
            'free_energies': np.array(free_energies),
            'entropies': np.array(entropies),
            'heat_capacities': np.array(heat_capacities),
            'bond_strengths': np.array(bond_strengths)
        }

    def _get_base_energy(self, bond_or_molecule) -> float:
        """Get base ground state energy."""
        if hasattr(bond_or_molecule, 'energy'):
            return bond_or_molecule.energy
        elif hasattr(bond_or_molecule, 'hamiltonian'):
            # Compute ground state energy
            # For bonds, use cached energy if available
            if hasattr(bond_or_molecule, '_cached_energy'):
                return bond_or_molecule._cached_energy
            else:
                # Compute HF energy as baseline (fast and reliable)
                logger.info("Computing HF energy for temperature analysis")
                try:
                    _, hf_energy = bond_or_molecule.hamiltonian.solve_scf(
                        max_iterations=100,
                        conv_tol=1e-8
                    )
                    # Cache for future use
                    bond_or_molecule._cached_energy = hf_energy
                    return hf_energy
                except Exception as e:
                    logger.error(f"Failed to compute HF energy: {e}")
                    raise ValueError(f"Cannot compute energy for temperature analysis: {e}")
        else:
            raise ValueError("Cannot extract energy from object")

    def _compute_bond_strength_factor(self, temperature: float) -> float:
        """
        Compute temperature-dependent bond strength factor.

        Physical basis: Morse potential depth decreases with T
        Empirical: D(T) ≈ D(0) × exp(-αT/T_ref)

        Args:
            temperature: Temperature in K

        Returns:
            Factor to multiply bond dissociation energy (0-1)
        """
        # Empirical temperature scaling (typical: α ~ 0.0003)
        alpha = 0.0003  # Adjustable parameter
        T_ref = self.reference_temp

        factor = np.exp(-alpha * (temperature - T_ref))

        # Constrain to reasonable range
        factor = np.clip(factor, 0.5, 1.2)

        return factor

    def _estimate_vibrational_frequencies(self, bond_or_molecule) -> np.ndarray:
        """
        Estimate vibrational frequencies from bond/molecule properties.

        Uses empirical correlations based on:
        - Bond type (H-H, C-C, C-H, etc.)
        - Reduced mass
        - Bond order

        Args:
            bond_or_molecule: Bond or molecule object

        Returns:
            Estimated frequencies in cm^-1
        """
        # Try to identify bond type (check for atom_1/atom_2 with underscores)
        if hasattr(bond_or_molecule, 'atom_1') and hasattr(bond_or_molecule, 'atom_2'):
            # It's a bond - use empirical frequency table
            elem1 = bond_or_molecule.atom_1.symbol
            elem2 = bond_or_molecule.atom_2.symbol

            # Empirical bond frequencies (cm^-1) from spectroscopy
            freq_table = {
                ('H', 'H'): 4400.0,   # H2 stretch
                ('C', 'C'): 1000.0,   # C-C stretch
                ('C', 'H'): 3000.0,   # C-H stretch
                ('C', 'O'): 1700.0,   # C=O stretch (carbonyl)
                ('C', 'N'): 1650.0,   # C=N stretch
                ('N', 'H'): 3300.0,   # N-H stretch
                ('O', 'H'): 3600.0,   # O-H stretch
                ('N', 'N'): 1600.0,   # N=N stretch
                ('O', 'O'): 1550.0,   # O=O stretch
                ('Li', 'H'): 1400.0,  # LiH stretch
                ('Na', 'H'): 1172.0,  # NaH stretch
            }

            # Look up frequency (order-independent)
            key1 = (elem1, elem2)
            key2 = (elem2, elem1)

            if key1 in freq_table:
                freq = freq_table[key1]
            elif key2 in freq_table:
                freq = freq_table[key2]
            else:
                # Estimate from reduced mass (lighter = higher frequency)
                mass1 = bond_or_molecule.atom_1.atomic_mass
                mass2 = bond_or_molecule.atom_2.atomic_mass
                reduced_mass = (mass1 * mass2) / (mass1 + mass2)

                # Empirical: ω ∝ 1/sqrt(μ) with k~500 N/m typical
                freq = 1000.0 * np.sqrt(10.0 / reduced_mass)  # Rough estimate
                logger.info(f"Estimated frequency for {elem1}-{elem2}: {freq:.0f} cm^-1 (from reduced mass)")

            return np.array([freq])

        elif hasattr(bond_or_molecule, 'molecule'):
            # It's a molecule - estimate fundamental mode
            # Use average of all bond types present
            logger.info("Estimating molecular vibrational modes from structure")
            # Conservative estimate: average frequency ~1500 cm^-1
            return np.array([1500.0])

        else:
            # Unknown - use conservative estimate
            logger.warning("Cannot determine bond/molecule type - using conservative estimate")
            return np.array([1500.0])

    def _resolve_vibrational_frequencies_Ha(self, bond_or_molecule) -> np.ndarray:
        """
        Resolve per-mode vibrational frequencies and return them in Hartree.

        Uses explicit frequency data if the object provides it, otherwise
        falls back to the empirical estimate. Centralizing this lets the
        vibrational energy and the entropy use the *same* per-mode omega
        array, which is required for a physically correct per-mode harmonic
        entropy.

        Args:
            bond_or_molecule: Molecular system

        Returns:
            Frequencies in Hartree (omega_Ha), as a numpy array.
        """
        # Get vibrational frequencies (if available)
        if hasattr(bond_or_molecule, 'frequencies'):
            frequencies = bond_or_molecule.frequencies  # cm^-1
        elif hasattr(bond_or_molecule, 'get_frequencies'):
            frequencies = bond_or_molecule.get_frequencies()
        else:
            # Estimate from bond type and atomic composition
            logger.info("No vibrational data - estimating from bond/molecule properties")
            frequencies = self._estimate_vibrational_frequencies(bond_or_molecule)

        # Convert cm^-1 to Ha
        cm_to_Ha = 4.556335e-6  # conversion factor
        return np.asarray(frequencies, dtype=float) * cm_to_Ha

    def _compute_vibrational_energy(
        self,
        bond_or_molecule,
        temperature: float,
        omega_Ha: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute vibrational thermal energy.

        For harmonic oscillator:
            E_vib = Σ_i [½ℏω_i + ℏω_i/(exp(ℏω_i/kT) - 1)]
                  = ZPE + thermal excitation

        Args:
            bond_or_molecule: Molecular system
            temperature: Temperature in K
            omega_Ha: Optional pre-resolved per-mode frequencies in Hartree.
                If None, they are resolved from bond_or_molecule.

        Returns:
            Vibrational thermal energy in Ha
        """
        if omega_Ha is None:
            omega_Ha = self._resolve_vibrational_frequencies_Ha(bond_or_molecule)

        # Compute thermal energy for each mode
        E_vib = 0.0
        for omega in omega_Ha:
            # Zero-point energy
            E_ZPE = 0.5 * omega

            # Thermal excitation
            x = omega / (self.k_B_Ha * temperature)
            if x < 50:  # Avoid overflow
                E_thermal = omega / (np.exp(x) - 1)
            else:
                E_thermal = 0.0  # Frozen out

            E_vib += E_ZPE + E_thermal

        return E_vib

    def compute_thermal_population(
        self,
        energies: np.ndarray,
        temperature: float = 298.15
    ) -> np.ndarray:
        """
        Compute Boltzmann populations at temperature T.

        Uses the Boltzmann distribution:
            P_i = exp(-E_i/kT) / Σ_j exp(-E_j/kT)

        Args:
            energies: Array of state energies in Hartree (Ha). k_B is applied
                in Ha/K (self.k_B_Ha), so callers MUST pass Hartree; convert
                eV->Ha (E_Ha = E_eV / 27.2114) before calling.
            temperature: Temperature in Kelvin (default: 298.15K)

        Returns:
            Array of populations (sum to 1.0)

        Example:
            >>> temp_mod = TemperatureModulator()
            >>> energies = np.array([0.0, 0.001, 0.002])  # Ha (~0.027 eV spacing)
            >>> pops = temp_mod.compute_thermal_population(energies, 298.15)
            >>> print(pops)  # ~[0.55, 0.30, 0.16] - ground state favored,
            ...              # but thermally relevant gaps populate excited states
        """
        energies = np.asarray(energies)

        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature} K")

        # Compute inverse temperature (beta = 1/kT)
        beta = 1.0 / (self.k_B_Ha * temperature)

        # Shift energies to avoid numerical overflow
        # Use lowest energy as reference (E_0 = 0)
        E_min = np.min(energies)
        Delta_E = energies - E_min

        # Boltzmann factors: exp(-E_i/kT)
        exp_factors = np.exp(-beta * Delta_E)

        # Partition function: Z = Σ_i exp(-E_i/kT)
        Z = np.sum(exp_factors)

        # Populations: P_i = exp(-E_i/kT) / Z
        populations = exp_factors / Z

        logger.debug(f"Thermal populations at T={temperature:.1f}K: {populations}")
        logger.debug(f"Partition function Z = {Z:.4f}")

        return populations

    def _compute_electronic_thermal_correction(
        self,
        molecule,
        temperature: float,
        n_states: int = 5
    ) -> Tuple[float, np.ndarray]:
        """
        Compute thermal population correction from excited states.

        Boltzmann-weighted energy:
            E = Σ_i p_i E_i  where p_i = exp(-E_i/kT) / Z

        Args:
            molecule: Molecule with excited state data
            temperature: Temperature in K
            n_states: Number of states to include

        Returns:
            (energy_correction, populations)
        """
        # Get excited state energies. If the cached attribute is missing but
        # the object can compute excited states, populate it first so the
        # branch does not silently no-op (fix: caller only checks for the
        # method, not the cached attribute).
        if not hasattr(molecule, 'excited_energies') and hasattr(molecule, 'compute_excited_states'):
            molecule.compute_excited_states(n_states)

        if hasattr(molecule, 'excited_energies') and molecule.excited_energies is not None:
            energies = molecule.excited_energies[:n_states]
        else:
            logger.warning("No excited state data - skipping thermal correction")
            return 0.0, np.array([1.0])

        # Use public method for Boltzmann calculation
        populations = self.compute_thermal_population(energies, temperature)

        # Thermal average energy
        E_avg = np.sum(populations * energies)

        # Correction = thermal average - ground state
        E_0 = energies[0]
        correction = E_avg - E_0

        return correction, populations

    def _compute_entropy(
        self,
        temperature: float,
        omega_Ha: np.ndarray,
        populations: np.ndarray
    ) -> float:
        """
        Compute entropy from the harmonic-oscillator partition function.

        Per-mode harmonic-oscillator entropy:
            S_i = k_B [ x/(e^x - 1) - ln(1 - e^-x) ],   x = ℏω_i / (k_B T)

        Electronic contribution (Gibbs / Shannon entropy of the thermal
        populations):
            S_elec = -k_B Σ_i p_i ln p_i

        Unlike the previous S ≈ E_thermal/T approximation, this form keeps the
        zero-point energy in A = E_thermal - T·S (since S only captures the
        thermal-excitation part, A_vib = ZPE + k_B T ln(1 - e^-x)) and goes to
        zero as T → 0. The hard-coded k_B·ln(2) electronic degeneracy term has
        been removed; electronic entropy is now derived from the actual
        Boltzmann populations.

        Args:
            temperature: Temperature in K
            omega_Ha: Per-mode vibrational frequencies in Hartree
            populations: Boltzmann populations over electronic states (sum to 1)

        Returns:
            Entropy in Ha/K
        """
        if temperature <= 0:
            return 0.0

        S = 0.0

        # Vibrational (harmonic-oscillator) entropy, summed over modes
        for omega in np.atleast_1d(omega_Ha):
            if omega <= 0:
                continue
            x = omega / (self.k_B_Ha * temperature)
            if x < 50:  # higher x => mode frozen out, contributes ~0
                S += self.k_B_Ha * (
                    x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))
                )

        # Electronic entropy from the real thermal populations:
        #   S_elec = -k_B Σ_i p_i ln p_i
        p = np.asarray(populations, dtype=float)
        nonzero = p > 0
        if np.any(nonzero):
            S += -self.k_B_Ha * np.sum(p[nonzero] * np.log(p[nonzero]))

        return S

    def _compute_heat_capacity(
        self,
        temperature: float,
        bond_or_molecule,
        omega_Ha: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute heat capacity Cv from vibrational energy.

        Cv = dE_vib/dT (at constant volume), evaluated by central finite
        difference on the harmonic vibrational energy. This is the real
        Einstein/harmonic-oscillator heat capacity (T- and mode-dependent),
        replacing the previous hardcoded classical limit Cv = 3*k_B.

        Args:
            temperature: Temperature in K
            bond_or_molecule: Molecular system (for vibrational frequencies)
            omega_Ha: Optional pre-resolved per-mode frequencies in Hartree
                (reused so the derivative shares the same modes as E_vib).

        Returns:
            Heat capacity in Ha/K
        """
        if temperature <= 0:
            return 0.0

        # Central finite difference of E_vib(T) -> dE_vib/dT
        dT = 1.0  # K
        T1 = temperature - dT / 2
        T2 = temperature + dT / 2

        E_vib_T1 = self._compute_vibrational_energy(
            bond_or_molecule, T1, omega_Ha=omega_Ha
        )
        E_vib_T2 = self._compute_vibrational_energy(
            bond_or_molecule, T2, omega_Ha=omega_Ha
        )

        Cv = (E_vib_T2 - E_vib_T1) / dT

        return Cv

    def modify_hamiltonian_with_temperature(
        self,
        hamiltonian,
        temperature: float,
        bond_strength_factor: Optional[float] = None
    ):
        """
        Create temperature-modified Hamiltonian.

        Modifies:
        - Bond dissociation terms (weaker at high T)
        - Adds thermal fluctuations (optional)

        Args:
            hamiltonian: Original Hamiltonian
            temperature: Temperature in K
            bond_strength_factor: Pre-computed factor (or compute if None)

        Returns:
            Modified Hamiltonian object
        """
        if bond_strength_factor is None:
            bond_strength_factor = self._compute_bond_strength_factor(temperature)

        # For SparsePauliOp, scale coupling terms
        # This is a simplified approach - in practice would need
        # to identify bond-specific terms

        logger.info(f"Applying bond strength factor: {bond_strength_factor:.4f}")

        # Scale Hamiltonian (affects all interactions equally)
        # More sophisticated: identify and scale only bond terms
        H_thermal = hamiltonian * bond_strength_factor

        return H_thermal

    def plot_temperature_scan(
        self,
        scan_result: Dict[str, np.ndarray],
        properties: list = ['energies', 'free_energies', 'entropies'],
        save_path: Optional[str] = None
    ):
        """
        Plot temperature-dependent properties.

        Args:
            scan_result: Output from scan_temperature()
            properties: List of properties to plot
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed - cannot plot")
            return

        fig, axes = plt.subplots(len(properties), 1, figsize=(10, 4*len(properties)))
        if len(properties) == 1:
            axes = [axes]

        T = scan_result['temperatures']

        for ax, prop in zip(axes, properties):
            values = scan_result[prop]

            ax.plot(T, values, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Temperature (K)', fontsize=12)
            ax.set_ylabel(prop.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(T[0], T[-1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()
