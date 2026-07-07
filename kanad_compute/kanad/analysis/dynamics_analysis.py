"""
Dynamics and Reactions Analysis Module

Comprehensive tools for analyzing molecular dynamics trajectories and
chemical reactions at different steps of the process.

Features:
---------
1. Trajectory Analysis
   - Energy profiles (kinetic, potential, total)
   - Temperature evolution
   - Structural metrics (bond lengths, angles, RMSDs)
   - Velocity distributions

2. Reaction Analysis
   - Energy barrier analysis
   - Reaction coordinate projection
   - Rate constant calculation (Eyring, Arrhenius)
   - Transition state characterization

3. Environment-Aware Analysis
   - Solvent effects on dynamics
   - Temperature-dependent kinetics
   - Pressure effects on barriers
   - pH-dependent reaction pathways

4. Non-Adiabatic Dynamics Analysis
   - State population evolution
   - Surface hopping statistics
   - Coherence/decoherence analysis
   - Conical intersection detection

References:
----------
1. Eyring (1935) J. Chem. Phys. 3, 107 - Transition State Theory
2. Tully (1990) J. Chem. Phys. 93, 1061 - Surface Hopping
3. Marx & Hutter (2009) - Ab Initio Molecular Dynamics
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Physical constants
K_BOLTZMANN = 3.166811563e-6  # Ha/K
PLANCK_CONSTANT = 1.519829846e-16  # Ha*s
GAS_CONSTANT = 3.166811563e-6  # Ha/(K*mol) = k_B
HARTREE_TO_KCAL = 627.509  # kcal/mol per Ha
HARTREE_TO_EV = 27.2114  # eV per Ha


class AnalysisTimeScale(Enum):
    """Time scales for dynamics analysis."""
    FEMTOSECOND = "fs"
    PICOSECOND = "ps"
    NANOSECOND = "ns"


@dataclass
class TrajectoryAnalysisResult:
    """Result from trajectory analysis."""
    n_steps: int
    total_time: float  # fs
    timestep: float  # fs

    # Energy analysis
    energies: Dict[str, np.ndarray]  # kinetic, potential, total
    energy_drift: float  # Ha/ps
    energy_fluctuation: float  # Ha

    # Temperature analysis
    temperatures: np.ndarray
    avg_temperature: float
    temperature_fluctuation: float

    # Structural analysis
    bond_lengths: Optional[Dict[str, np.ndarray]] = None
    rmsd: Optional[np.ndarray] = None

    # Velocity analysis
    velocity_distribution: Optional[Dict[str, Any]] = None
    diffusion_coefficient: Optional[float] = None

    # Environment effects
    environment_info: Optional[Dict[str, Any]] = None


@dataclass
class ReactionAnalysisResult:
    """Result from reaction analysis."""
    # Energetics
    barrier_height: float  # Ha
    reaction_energy: float  # Ha (ΔE)
    activation_energy: float  # Ha (Ea)
    activation_free_energy: float  # Ha (ΔG‡)

    # Kinetics
    rate_constant: float  # s⁻¹
    half_life: float  # s
    transmission_coefficient: float

    # Transition state
    ts_geometry: Optional[np.ndarray] = None
    imaginary_frequency: Optional[float] = None  # cm⁻¹
    ts_verified: bool = False

    # Pathway analysis
    reaction_coordinate: Optional[np.ndarray] = None
    energy_profile: Optional[np.ndarray] = None

    # Environment effects
    environment_corrections: Optional[Dict[str, float]] = None


@dataclass
class NAMDAnalysisResult:
    """Result from non-adiabatic dynamics analysis."""
    n_trajectories: int
    total_time: float
    n_states: int

    # Population analysis
    state_populations: np.ndarray  # (n_steps, n_states)
    final_state_distribution: Dict[int, float]

    # Hopping statistics
    total_hops: int
    hop_times: List[float]
    hop_states: List[Tuple[int, int]]  # (from, to)
    hop_rate: float  # hops/fs

    # Coherence analysis
    coherence_times: Optional[Dict[str, float]] = None
    decoherence_rate: Optional[float] = None

    # CI detection
    ci_events: Optional[List[Dict]] = None


class TrajectoryAnalyzer:
    """
    Analyzer for molecular dynamics trajectories.

    Provides comprehensive analysis of MD trajectories including:
    - Energy conservation and fluctuations
    - Temperature control quality
    - Structural evolution
    - Diffusion and transport properties

    Example:
    -------
    >>> from kanad import BondFactory
    >>> from kanad.dynamics import MDSimulator
    >>> from kanad.analysis import TrajectoryAnalyzer

    >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
    >>> md = MDSimulator(bond, temperature=300, timestep=0.5)
    >>> result = md.run(n_steps=1000)
    >>>
    >>> analyzer = TrajectoryAnalyzer(result)
    >>> analysis = analyzer.analyze()
    >>> print(f"Energy drift: {analysis.energy_drift:.6f} Ha/ps")
    >>> print(f"Avg temperature: {analysis.avg_temperature:.1f} K")
    """

    def __init__(self, md_result, bond=None, environment=None):
        """
        Initialize trajectory analyzer.

        Args:
            md_result: MDResult from MDSimulator.run()
            bond: Optional Kanad bond for additional analysis
            environment: Optional environment conditions dict
        """
        self.md_result = md_result
        self.bond = bond
        self.environment = environment or {}

        # Extract trajectory data
        self._extract_trajectory_data()

    def _extract_trajectory_data(self):
        """Extract data from MD result."""
        result = self.md_result

        # Time information
        self.timestep = getattr(result, 'timestep', 0.5)  # fs

        # Energy arrays
        if hasattr(result, 'energies') and result.energies is not None:
            self.energies = np.array(result.energies)
        elif hasattr(result, 'trajectory') and result.trajectory is not None:
            # Extract from trajectory frames
            self.energies = np.array([f.energy for f in result.trajectory])
        else:
            self.energies = np.array([getattr(result, 'initial_energy', 0)])

        # Temperature array
        if hasattr(result, 'temperatures') and result.temperatures is not None:
            self.temperatures = np.array(result.temperatures)
        else:
            self.temperatures = np.array([getattr(result, 'avg_temperature', 300)])

        # Trajectory (positions over time)
        if hasattr(result, 'trajectory') and result.trajectory is not None:
            self.trajectory = result.trajectory
        else:
            self.trajectory = None

        self.n_steps = len(self.energies)
        self.total_time = self.n_steps * self.timestep

    def analyze(self) -> TrajectoryAnalysisResult:
        """
        Perform comprehensive trajectory analysis.

        Returns:
            TrajectoryAnalysisResult with all computed metrics
        """
        # Energy analysis
        energy_analysis = self._analyze_energy()

        # Temperature analysis
        temp_analysis = self._analyze_temperature()

        # Structural analysis (if trajectory available)
        struct_analysis = self._analyze_structure() if self.trajectory else None

        # Velocity analysis (if available)
        velocity_analysis = self._analyze_velocities() if self.trajectory else None

        return TrajectoryAnalysisResult(
            n_steps=self.n_steps,
            total_time=self.total_time,
            timestep=self.timestep,
            energies=energy_analysis['energies'],
            energy_drift=energy_analysis['drift'],
            energy_fluctuation=energy_analysis['fluctuation'],
            temperatures=self.temperatures,
            avg_temperature=temp_analysis['avg'],
            temperature_fluctuation=temp_analysis['fluctuation'],
            bond_lengths=struct_analysis.get('bond_lengths') if struct_analysis else None,
            rmsd=struct_analysis.get('rmsd') if struct_analysis else None,
            velocity_distribution=velocity_analysis,
            environment_info=self.environment
        )

    def _analyze_energy(self) -> Dict[str, Any]:
        """Analyze energy conservation and fluctuations."""
        E = self.energies

        # Energy drift (linear fit)
        if len(E) > 1:
            time_ps = np.arange(len(E)) * self.timestep / 1000  # ps
            slope, _ = np.polyfit(time_ps, E, 1)
            drift = slope  # Ha/ps
        else:
            drift = 0.0

        # Energy fluctuation
        fluctuation = np.std(E) if len(E) > 1 else 0.0

        return {
            'energies': {'total': E},
            'drift': drift,
            'fluctuation': fluctuation,
            'mean': np.mean(E),
            'min': np.min(E),
            'max': np.max(E)
        }

    def _analyze_temperature(self) -> Dict[str, Any]:
        """Analyze temperature control quality."""
        T = self.temperatures

        return {
            'avg': np.mean(T),
            'fluctuation': np.std(T) if len(T) > 1 else 0.0,
            'min': np.min(T),
            'max': np.max(T)
        }

    def _analyze_structure(self) -> Dict[str, Any]:
        """Analyze structural evolution."""
        result = {}

        if self.trajectory is None:
            return result

        # Extract positions
        positions = []
        for frame in self.trajectory:
            if hasattr(frame, 'positions'):
                positions.append(frame.positions)

        if not positions:
            return result

        positions = np.array(positions)

        # RMSD from initial structure
        ref = positions[0]
        rmsd = np.sqrt(np.mean(np.sum((positions - ref) ** 2, axis=2), axis=1))
        result['rmsd'] = rmsd

        # Bond length tracking (for diatomic)
        if positions.shape[1] >= 2:
            bond_lengths = np.linalg.norm(
                positions[:, 1] - positions[:, 0], axis=1
            )
            result['bond_lengths'] = {'0-1': bond_lengths}

        return result

    def _analyze_velocities(self) -> Optional[Dict[str, Any]]:
        """Analyze velocity distribution."""
        if self.trajectory is None:
            return None

        velocities = []
        for frame in self.trajectory:
            if hasattr(frame, 'velocities') and frame.velocities is not None:
                velocities.append(frame.velocities)

        if not velocities:
            return None

        velocities = np.array(velocities)

        # Speed distribution
        speeds = np.linalg.norm(velocities, axis=-1)

        return {
            'mean_speed': np.mean(speeds),
            'speed_std': np.std(speeds),
            'maxwell_boltzmann_fit': None  # TODO: fit to MB distribution
        }

    def compute_diffusion_coefficient(self) -> Optional[float]:
        """
        Compute diffusion coefficient from mean-squared displacement.

        Returns:
            D: Diffusion coefficient in Å²/fs, or None if not computable
        """
        if self.trajectory is None:
            return None

        positions = []
        for frame in self.trajectory:
            if hasattr(frame, 'positions'):
                positions.append(frame.positions)

        if len(positions) < 10:
            return None

        positions = np.array(positions)

        # Mean-squared displacement
        ref = positions[0]
        msd = np.mean(np.sum((positions - ref) ** 2, axis=-1), axis=-1)

        # Linear fit: MSD = 6Dt (3D diffusion)
        time = np.arange(len(msd)) * self.timestep
        if len(time) > 1:
            slope, _ = np.polyfit(time, msd, 1)
            D = slope / 6  # Å²/fs
            return D

        return None


class ReactionAnalyzer:
    """
    Analyzer for chemical reactions.

    Provides comprehensive analysis of reaction pathways including:
    - Energy barrier determination
    - Rate constant calculation (Eyring, Arrhenius)
    - Transition state characterization
    - Environment effects on reaction kinetics

    Example:
    -------
    >>> from kanad import BondFactory
    >>> from kanad.reactions import ReactionSimulator
    >>> from kanad.analysis import ReactionAnalyzer

    >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
    >>> sim = ReactionSimulator([bond])
    >>> ts = sim.find_transition_state()
    >>>
    >>> analyzer = ReactionAnalyzer(sim)
    >>> analysis = analyzer.analyze(temperature=298.15)
    >>> print(f"Barrier: {analysis.barrier_height * 627.5:.2f} kcal/mol")
    >>> print(f"Rate: {analysis.rate_constant:.2e} s⁻¹")
    """

    def __init__(
        self,
        reaction_simulator=None,
        reactant_energy: float = None,
        product_energy: float = None,
        ts_energy: float = None,
        environment: Dict = None
    ):
        """
        Initialize reaction analyzer.

        Args:
            reaction_simulator: Optional ReactionSimulator instance
            reactant_energy: Reactant energy in Ha
            product_energy: Product energy in Ha
            ts_energy: Transition state energy in Ha
            environment: Environment conditions dict
        """
        self.sim = reaction_simulator
        self.environment = environment or {}

        # Energies
        self.E_reactant = reactant_energy
        self.E_product = product_energy
        self.E_ts = ts_energy

        # Extract from simulator if available
        if self.sim is not None:
            self._extract_from_simulator()

    def _extract_from_simulator(self):
        """Extract energies from reaction simulator."""
        if self.sim is None:
            return

        # Try to get energies from simulator
        if hasattr(self.sim, 'reactant_energy') and self.E_reactant is None:
            self.E_reactant = self.sim.reactant_energy

        if hasattr(self.sim, 'product_energy') and self.E_product is None:
            self.E_product = self.sim.product_energy

        if hasattr(self.sim, 'ts_energy') and self.E_ts is None:
            self.E_ts = self.sim.ts_energy

    def analyze(
        self,
        temperature: float = 298.15,
        pressure: float = 1.0,
        include_tunneling: bool = True
    ) -> ReactionAnalysisResult:
        """
        Perform comprehensive reaction analysis.

        Args:
            temperature: Temperature in K
            pressure: Pressure in atm
            include_tunneling: Include Wigner tunneling correction

        Returns:
            ReactionAnalysisResult with all computed metrics
        """
        # Compute barrier and reaction energy
        barrier = self._compute_barrier()
        delta_E = self._compute_reaction_energy()

        # Get TS info from simulator BEFORE the rate constant so the Wigner
        # tunneling correction can use the imaginary frequency.
        ts_geometry = None
        imag_freq = None
        ts_verified = False

        if self.sim is not None and hasattr(self.sim, '_ts_result'):
            ts_result = self.sim._ts_result
            if ts_result is not None:
                ts_geometry = getattr(ts_result, 'geometry', None)
                imag_freq = getattr(ts_result, 'imaginary_frequency', None)
                ts_verified = getattr(ts_result, 'verified', False)

        # Fall back to a manually-set private attr (module-level
        # compute_rate_constant() path sets self._imaginary_frequency).
        if imag_freq is None:
            imag_freq = getattr(self, '_imaginary_frequency', None)

        # Compute rate constant
        k, kappa = self._compute_rate_constant(
            barrier, temperature, include_tunneling, imag_freq
        )

        # Half-life from rate constant
        half_life = np.log(2) / k if k > 0 else np.inf

        # Free energy of activation
        delta_G_barrier = self._compute_activation_free_energy(
            barrier, temperature
        )

        # Environment corrections
        env_corrections = self._compute_environment_corrections(temperature)

        return ReactionAnalysisResult(
            barrier_height=barrier,
            reaction_energy=delta_E,
            activation_energy=barrier,  # Ea ≈ ΔE‡ at low T
            activation_free_energy=delta_G_barrier,
            rate_constant=k,
            half_life=half_life,
            transmission_coefficient=kappa,
            ts_geometry=ts_geometry,
            imaginary_frequency=imag_freq,
            ts_verified=ts_verified,
            environment_corrections=env_corrections
        )

    def _compute_barrier(self) -> float:
        """Compute forward reaction barrier."""
        if self.E_ts is not None and self.E_reactant is not None:
            return self.E_ts - self.E_reactant
        return 0.0

    def _compute_reaction_energy(self) -> float:
        """Compute reaction energy (ΔE)."""
        if self.E_product is not None and self.E_reactant is not None:
            return self.E_product - self.E_reactant
        return 0.0

    def _compute_rate_constant(
        self,
        barrier: float,
        temperature: float,
        include_tunneling: bool,
        imag_freq: float = None
    ) -> Tuple[float, float]:
        """
        Compute rate constant using Eyring equation.

        k = (k_B T / h) * κ * exp(-ΔG‡ / RT)

        Args:
            barrier: Barrier height in Ha
            temperature: Temperature in K
            include_tunneling: Include Wigner tunneling correction
            imag_freq: Transition-state imaginary frequency in cm⁻¹ (for tunneling)

        Returns:
            Tuple of (rate_constant, transmission_coefficient)
        """
        kT = K_BOLTZMANN * temperature  # Ha

        # Pre-exponential factor: k_B T / h
        # Convert to s⁻¹ (h in Ha*s)
        h_Ha_s = 1.519829846e-16  # h in Ha*s (= 6.62607015e-34 J*s / 4.3597447e-18 J/Ha)
        A = kT / h_Ha_s  # s⁻¹

        # Tunneling correction (Wigner)
        kappa = 1.0
        if include_tunneling and imag_freq is not None and imag_freq > 0:
            # ν_i in cm⁻¹ to Ha
            nu_i = imag_freq * 4.5563352529e-6  # cm⁻¹ to Ha
            # Wigner correction: κ = 1 + (1/24) * (hν_i / kT)²
            u = nu_i / kT
            kappa = 1 + (1/24) * u**2

        # Rate constant
        k = A * kappa * np.exp(-barrier / kT)

        return k, kappa

    def _compute_activation_free_energy(
        self,
        barrier: float,
        temperature: float
    ) -> float:
        """
        Compute Gibbs free energy of activation.

        ΔG‡ = ΔH‡ - TΔS‡

        Approximation: ΔG‡ ≈ ΔE‡ (neglecting entropy at low T)
        """
        # For now, use barrier as approximation
        # TODO: Compute vibrational entropy contribution
        return barrier

    def _compute_environment_corrections(
        self,
        temperature: float
    ) -> Dict[str, float]:
        """Compute environment corrections to kinetics."""
        corrections = {}

        # Solvent correction
        if 'solvent' in self.environment:
            solvent = self.environment['solvent']
            # Solvent reorganization energy correction
            # λ_solv ≈ q² * (1/ε_opt - 1/ε_s) / (2 * a)
            # Simplified: just note the solvent
            corrections['solvent'] = solvent

        # Temperature effect (already in rate constant)
        corrections['temperature'] = temperature

        # Pressure effect on activation volume
        if 'pressure' in self.environment:
            P = self.environment['pressure']
            # ΔV‡ ≈ -10 cm³/mol for typical reactions
            # ΔG‡(P) = ΔG‡(1atm) + ΔV‡ * (P - 1)
            delta_V = -10  # cm³/mol
            # cm³·atm/mol → Ha/molecule: (0.101325 J per cm³·atm/mol) / (2625500 J/mol per Ha) ≈ 3.8593e-8
            P_correction = delta_V * (P - 1) * (0.101325 / 2625500.0)  # to Ha
            corrections['pressure_correction'] = P_correction

        return corrections

    def compute_arrhenius_parameters(
        self,
        temperatures: List[float] = None
    ) -> Dict[str, float]:
        """
        Compute Arrhenius parameters from rate constants at multiple temperatures.

        k = A * exp(-Ea / RT)
        ln(k) = ln(A) - Ea / RT

        Args:
            temperatures: List of temperatures in K

        Returns:
            Dict with 'A' (pre-exponential) and 'Ea' (activation energy)
        """
        if temperatures is None:
            temperatures = [250, 275, 298.15, 325, 350]

        barrier = self._compute_barrier()

        # Compute rate constants at each temperature
        ln_k = []
        inv_T = []

        for T in temperatures:
            k, _ = self._compute_rate_constant(barrier, T, include_tunneling=False)
            if k > 0:
                ln_k.append(np.log(k))
                inv_T.append(1 / T)

        if len(ln_k) < 2:
            return {'A': 0, 'Ea': 0}

        # Linear fit: ln(k) = ln(A) - Ea / R * (1/T)
        slope, intercept = np.polyfit(inv_T, ln_k, 1)

        Ea = -slope * K_BOLTZMANN  # Ha
        A = np.exp(intercept)  # s⁻¹

        return {
            'A': A,
            'Ea': Ea,
            'Ea_kcal': Ea * HARTREE_TO_KCAL
        }


class NAMDAnalyzer:
    """
    Analyzer for non-adiabatic molecular dynamics.

    Provides analysis of surface hopping trajectories including:
    - State population evolution
    - Hopping statistics
    - Coherence/decoherence analysis
    - Conical intersection detection

    Example:
    -------
    >>> from kanad.dynamics import NonAdiabaticMD
    >>> from kanad.analysis import NAMDAnalyzer

    >>> namd = NonAdiabaticMD(bond, n_states=3)
    >>> result = namd.run(n_steps=1000, initial_state=1)
    >>>
    >>> analyzer = NAMDAnalyzer(result)
    >>> analysis = analyzer.analyze()
    >>> print(f"Total hops: {analysis.total_hops}")
    >>> print(f"Final populations: {analysis.final_state_distribution}")
    """

    def __init__(self, namd_result):
        """
        Initialize NAMD analyzer.

        Args:
            namd_result: NAMDTrajectory from NonAdiabaticMD.run()
        """
        self.result = namd_result
        self._extract_data()

    def _extract_data(self):
        """Extract data from NAMD result."""
        result = self.result

        self.n_steps = getattr(result, 'n_steps', 0)
        self.n_states = getattr(result, 'n_states', 2)
        self.timestep = getattr(result, 'timestep', 0.5)
        self.total_time = self.n_steps * self.timestep

        # State populations over time
        if hasattr(result, 'state_populations'):
            self.populations = np.array(result.state_populations)
        elif hasattr(result, 'trajectory'):
            # Extract from trajectory frames
            self.populations = self._extract_populations_from_trajectory()
        else:
            self.populations = None

        # Hopping events
        if hasattr(result, 'hop_events'):
            self.hop_events = result.hop_events
        else:
            self.hop_events = []

    def _extract_populations_from_trajectory(self) -> Optional[np.ndarray]:
        """Extract populations from trajectory frames."""
        if not hasattr(self.result, 'trajectory') or self.result.trajectory is None:
            return None

        populations = []
        for frame in self.result.trajectory:
            if hasattr(frame, 'state_populations'):
                populations.append(frame.state_populations)
            elif hasattr(frame, 'current_state'):
                # Binary population from active state
                pop = np.zeros(self.n_states)
                pop[frame.current_state] = 1.0
                populations.append(pop)

        return np.array(populations) if populations else None

    def analyze(self) -> NAMDAnalysisResult:
        """
        Perform comprehensive NAMD analysis.

        Returns:
            NAMDAnalysisResult with all computed metrics
        """
        # Population analysis
        final_pop = self._compute_final_populations()

        # Hopping statistics
        hop_stats = self._analyze_hopping()

        # Coherence analysis
        coherence = self._analyze_coherence()

        return NAMDAnalysisResult(
            n_trajectories=1,  # Single trajectory
            total_time=self.total_time,
            n_states=self.n_states,
            state_populations=self.populations if self.populations is not None else np.array([]),
            final_state_distribution=final_pop,
            total_hops=hop_stats['total'],
            hop_times=hop_stats['times'],
            hop_states=hop_stats['states'],
            hop_rate=hop_stats['rate'],
            coherence_times=coherence.get('times'),
            decoherence_rate=coherence.get('rate')
        )

    def _compute_final_populations(self) -> Dict[int, float]:
        """Compute final state population distribution."""
        if self.populations is None or len(self.populations) == 0:
            return {i: 1.0/self.n_states for i in range(self.n_states)}

        # Average over last 10% of trajectory
        n_avg = max(1, len(self.populations) // 10)
        final_pop = np.mean(self.populations[-n_avg:], axis=0)

        return {i: float(final_pop[i]) for i in range(len(final_pop))}

    def _analyze_hopping(self) -> Dict[str, Any]:
        """Analyze surface hopping events."""
        if not self.hop_events:
            return {
                'total': 0,
                'times': [],
                'states': [],
                'rate': 0.0
            }

        times = [event.get('time', 0) for event in self.hop_events]
        states = [(event.get('from_state', 0), event.get('to_state', 0))
                  for event in self.hop_events]

        total_hops = len(self.hop_events)
        rate = total_hops / self.total_time if self.total_time > 0 else 0.0

        return {
            'total': total_hops,
            'times': times,
            'states': states,
            'rate': rate
        }

    def _analyze_coherence(self) -> Dict[str, Any]:
        """Analyze electronic coherence and decoherence."""
        if self.populations is None or len(self.populations) < 10:
            return {}

        # Simple coherence analysis: look at population oscillations
        # True coherence would require off-diagonal density matrix elements

        # Population fluctuation as proxy for coherence
        pop_std = np.std(self.populations, axis=0)

        # Estimate decoherence time from decay of fluctuations
        # TODO: Proper autocorrelation analysis

        return {
            'population_fluctuation': pop_std.tolist(),
            'times': None,
            'rate': None
        }


class EnvironmentEffectsAnalyzer:
    """
    Analyzer for environmental effects on reactions and dynamics.

    Provides analysis of how environmental conditions affect:
    - Reaction barriers and rates
    - Dynamical properties
    - Equilibrium distributions

    Example:
    -------
    >>> from kanad.analysis import EnvironmentEffectsAnalyzer

    >>> analyzer = EnvironmentEffectsAnalyzer(reaction_simulator)
    >>> effects = analyzer.analyze_temperature_effects(
    ...     temperatures=[250, 275, 300, 325, 350]
    ... )
    >>> print(f"Arrhenius Ea: {effects['Ea_kcal']:.2f} kcal/mol")
    """

    def __init__(
        self,
        system=None,
        reaction_analyzer: ReactionAnalyzer = None,
        trajectory_analyzer: TrajectoryAnalyzer = None
    ):
        """
        Initialize environment effects analyzer.

        Args:
            system: Bond, molecule, or reaction simulator
            reaction_analyzer: Pre-configured ReactionAnalyzer
            trajectory_analyzer: Pre-configured TrajectoryAnalyzer
        """
        self.system = system
        self.reaction_analyzer = reaction_analyzer
        self.trajectory_analyzer = trajectory_analyzer

    def analyze_temperature_effects(
        self,
        temperatures: List[float] = None,
        property_type: str = 'rate'
    ) -> Dict[str, Any]:
        """
        Analyze temperature effects on reaction/dynamics.

        Args:
            temperatures: List of temperatures in K
            property_type: 'rate', 'equilibrium', or 'diffusion'

        Returns:
            Dict with temperature-dependent properties
        """
        if temperatures is None:
            temperatures = [250, 275, 298.15, 325, 350, 400]

        results = {
            'temperatures': temperatures,
            'property_type': property_type
        }

        if property_type == 'rate' and self.reaction_analyzer:
            rates = []
            for T in temperatures:
                analysis = self.reaction_analyzer.analyze(temperature=T)
                rates.append(analysis.rate_constant)

            results['rates'] = rates
            results['arrhenius'] = self.reaction_analyzer.compute_arrhenius_parameters(temperatures)

        return results

    def analyze_solvent_effects(
        self,
        solvents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze solvent effects on reaction/dynamics.

        Args:
            solvents: List of solvent names

        Returns:
            Dict with solvent-dependent properties
        """
        if solvents is None:
            solvents = ['vacuum', 'water', 'acetonitrile', 'dmso', 'chloroform']

        results = {
            'solvents': solvents,
            'effects': {}
        }

        # Solvent dielectric constants (from literature)
        dielectrics = {
            'vacuum': 1.0,
            'water': 78.4,
            'acetonitrile': 37.5,
            'dmso': 46.7,
            'chloroform': 4.8,
            'hexane': 1.9,
            'methanol': 32.7,
            'ethanol': 24.5,
            'benzene': 2.3
        }

        for solvent in solvents:
            eps = dielectrics.get(solvent, 1.0)

            # Onsager reaction field correction
            # ΔG_solv ≈ -μ² * (ε-1)/(2ε+1) / (a³)
            # Simplified: relative stabilization factor
            solvent_factor = (eps - 1) / (2 * eps + 1)

            results['effects'][solvent] = {
                'dielectric': eps,
                'reaction_field_factor': solvent_factor
            }

        return results

    def analyze_pressure_effects(
        self,
        pressures: List[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze pressure effects on reaction.

        Args:
            pressures: List of pressures in atm

        Returns:
            Dict with pressure-dependent properties
        """
        if pressures is None:
            pressures = [1, 10, 100, 1000, 10000]  # atm

        results = {
            'pressures': pressures,
            'effects': {}
        }

        # Activation volume (ΔV‡) determines pressure effect
        # k(P) = k(1atm) * exp(-ΔV‡ * (P-1) / RT)
        # Typical ΔV‡: -10 to -20 cm³/mol (accelerates)
        # or +10 to +20 cm³/mol (decelerates)

        delta_V_typical = -10.0e-6   # m³/mol (-10 cm³/mol)
        R_SI = 8.314462618           # J/(mol*K)
        T = 298.15                   # K
        PA_PER_ATM = 101325.0        # Pa/atm

        for P in pressures:
            dP_pa = (P - 1) * PA_PER_ATM                       # Pa
            exponent = -delta_V_typical * dP_pa / (R_SI * T)   # dimensionless
            pressure_factor = np.exp(exponent)
            results['effects'][P] = {
                'rate_factor': pressure_factor
            }

        return results

    def analyze_ph_effects(
        self,
        ph_values: List[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze pH effects on reaction.

        Args:
            ph_values: List of pH values

        Returns:
            Dict with pH-dependent properties
        """
        if ph_values is None:
            ph_values = [2, 4, 5, 6, 7, 8, 9, 10, 12]

        results = {
            'ph_values': ph_values,
            'effects': {}
        }

        # Henderson-Hasselbalch for protonation state
        # For acid with pKa: f_prot = 1 / (1 + 10^(pH - pKa))
        # This affects reaction mechanism and rate

        pKa_generic = 7.0  # Example

        for pH in ph_values:
            f_protonated = 1 / (1 + 10**(pH - pKa_generic))
            f_deprotonated = 1 - f_protonated

            results['effects'][pH] = {
                'f_protonated': f_protonated,
                'f_deprotonated': f_deprotonated
            }

        return results


# Convenience functions

def analyze_trajectory(md_result, bond=None) -> TrajectoryAnalysisResult:
    """
    Convenience function to analyze MD trajectory.

    Args:
        md_result: MDResult from MDSimulator.run()
        bond: Optional bond for additional analysis

    Returns:
        TrajectoryAnalysisResult
    """
    analyzer = TrajectoryAnalyzer(md_result, bond)
    return analyzer.analyze()


def analyze_reaction(
    reaction_simulator=None,
    E_reactant: float = None,
    E_product: float = None,
    E_ts: float = None,
    temperature: float = 298.15
) -> ReactionAnalysisResult:
    """
    Convenience function to analyze reaction.

    Args:
        reaction_simulator: Optional ReactionSimulator
        E_reactant: Reactant energy in Ha
        E_product: Product energy in Ha
        E_ts: TS energy in Ha
        temperature: Temperature in K

    Returns:
        ReactionAnalysisResult
    """
    analyzer = ReactionAnalyzer(
        reaction_simulator,
        reactant_energy=E_reactant,
        product_energy=E_product,
        ts_energy=E_ts
    )
    return analyzer.analyze(temperature=temperature)


def analyze_namd(namd_result) -> NAMDAnalysisResult:
    """
    Convenience function to analyze NAMD trajectory.

    Args:
        namd_result: NAMDTrajectory from NonAdiabaticMD.run()

    Returns:
        NAMDAnalysisResult
    """
    analyzer = NAMDAnalyzer(namd_result)
    return analyzer.analyze()


def compute_rate_constant(
    barrier: float,
    temperature: float = 298.15,
    imaginary_frequency: float = None
) -> float:
    """
    Compute rate constant using Eyring equation.

    Args:
        barrier: Barrier height in Ha
        temperature: Temperature in K
        imaginary_frequency: Imaginary frequency in cm⁻¹ for tunneling

    Returns:
        Rate constant in s⁻¹
    """
    analyzer = ReactionAnalyzer(
        ts_energy=barrier,
        reactant_energy=0.0
    )

    if imaginary_frequency:
        analyzer._imaginary_frequency = imaginary_frequency

    result = analyzer.analyze(temperature=temperature)
    return result.rate_constant
