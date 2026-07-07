"""
Main Molecular Dynamics Simulator

Orchestrates all MD components to run simulations with:
- Multiple force methods (HF, MP2, VQE, SQD)
- Various integrators (Velocity Verlet, Leapfrog, RK4)
- Temperature control (Berendsen, Nose-Hoover, Langevin)
- Trajectory storage and analysis
- Energy monitoring and conservation checks

This is the primary user interface for running MD simulations in Kanad.

Example Usage:
-------------
```python
from kanad.bonds import BondFactory
from kanad.dynamics import MDSimulator

# Create molecule
bond = BondFactory.create_bond('H', 'H', distance=0.74)

# Run classical MD with HF forces
md = MDSimulator(
    bond_or_molecule=bond,
    temperature=300.0,  # K
    timestep=0.5,  # fs
    integrator='velocity_verlet',
    thermostat='berendsen',
    force_method='hf'
)

result = md.run(n_steps=1000, save_trajectory=True)

# Access results
print(f"Final energy: {result.final_energy:.6f} Ha")
print(f"Average temperature: {result.avg_temperature:.2f} K")

# Or run quantum MD with VQE
md_quantum = MDSimulator(
    bond,
    temperature=300.0,
    timestep=0.5,
    force_method='vqe',
    use_governance=True
)

result = md_quantum.run(n_steps=100)
```

References:
----------
- Allen & Tildesley (2017) Computer Simulation of Liquids
- Frenkel & Smit (2002) Understanding Molecular Simulation
- Car & Parrinello (1985) Ab initio MD: Phys. Rev. Lett. 55, 2471
"""

import numpy as np
import logging
from typing import Optional, Callable, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time as time_module

from kanad.dynamics.integrators import create_integrator, IntegratorState
from kanad.dynamics.thermostats import create_thermostat
from kanad.dynamics.trajectory import Trajectory, TrajectoryWriter
from kanad.dynamics.initialization import generate_initial_conditions

# Import environment integration (lazy to avoid circular imports)
_EnvironmentIntegration = None
_EnvironmentConditions = None

def _get_environment_classes():
    """Lazy import environment classes."""
    global _EnvironmentIntegration, _EnvironmentConditions
    if _EnvironmentIntegration is None:
        from kanad.core.environment.integration import EnvironmentIntegration, EnvironmentConditions
        _EnvironmentIntegration = EnvironmentIntegration
        _EnvironmentConditions = EnvironmentConditions
    return _EnvironmentIntegration, _EnvironmentConditions

logger = logging.getLogger(__name__)


@dataclass
class MDResult:
    """
    Result from MD simulation.

    Attributes:
        trajectory: Full trajectory (if saved)
        final_positions: Final atomic positions (N_atoms, 3) Bohr
        final_velocities: Final velocities (N_atoms, 3) Bohr/fs
        final_energy: Final total energy (Hartree)
        avg_temperature: Average temperature (K)
        avg_kinetic_energy: Average KE (Hartree)
        avg_potential_energy: Average PE (Hartree)
        avg_total_energy: Average total energy (Hartree)
        energy_drift: Energy drift (Hartree)
        temperature_std: Temperature std deviation (K)
        n_steps_completed: Number of steps completed
        wall_time: Wall clock time (seconds)
        steps_per_second: Performance metric
        converged: Whether simulation completed successfully
        metadata: Additional simulation information
        energies: List of total energies at each step
        temperatures: List of temperatures at each step
        environment_effects: Dict with environment correction details
    """
    trajectory: Optional[Trajectory] = None
    final_positions: Optional[np.ndarray] = None
    final_velocities: Optional[np.ndarray] = None
    final_energy: float = 0.0
    avg_temperature: float = 0.0
    avg_kinetic_energy: float = 0.0
    avg_potential_energy: float = 0.0
    avg_total_energy: float = 0.0
    energy_drift: float = 0.0
    temperature_std: float = 0.0
    n_steps_completed: int = 0
    wall_time: float = 0.0
    steps_per_second: float = 0.0
    converged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    energies: Optional[list] = None
    temperatures: Optional[list] = None
    environment_effects: Optional[Dict[str, Any]] = None
    timestep: float = 0.5


class MDSimulator:
    """
    Main molecular dynamics simulator.

    Coordinates integrator, thermostat, force computation, and trajectory storage
    to run MD simulations with various force methods (classical or quantum).
    """

    def __init__(
        self,
        bond_or_molecule,
        temperature: Optional[float] = None,
        timestep: float = 0.5,
        integrator: str = 'velocity_verlet',
        thermostat: Optional[str] = 'berendsen',
        force_method: str = 'hf',
        use_governance: bool = False,
        backend: str = 'statevector',
        initial_velocities: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
        environment: Optional[Dict[str, Any]] = None,
        quantum_system: Optional[Any] = None,
        *,
        temperature_K: Optional[float] = None,
        n_steps: Optional[int] = None,
        solver: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize MD simulator.

        Args:
            bond_or_molecule: Bond or Molecule object
            temperature: Target temperature in K
            timestep: Integration timestep in fs
            integrator: Integrator type ('velocity_verlet', 'leapfrog', 'rk4')
            thermostat: Thermostat type ('berendsen', 'nose_hoover', 'langevin', None for NVE)
            force_method: Force computation method ('hf', 'mp2', 'hivqe', 'vqe', 'sqd')
                         Note: 'hivqe' is recommended for quantum MD (more efficient than 'vqe')
            use_governance: Use governance protocols (for VQE/SQD)
            backend: Quantum backend ('statevector', 'aer', 'ibm')
            initial_velocities: Initial velocities (None = Maxwell-Boltzmann)
            random_seed: Random seed for initialization
            environment: Optional environmental effects dict with keys:
                         - 'solvent': Solvent name (e.g., 'water', 'ethanol')
                         - 'temperature': Environment temperature in K
                         - 'pressure': Pressure in atm
                         - 'pH': pH value for protonation states
            **kwargs: Additional parameters for integrator/thermostat/forces

        Example with environment:
        -------------------------
        ```python
        md = MDSimulator(
            bond,
            temperature=300,
            force_method='hivqe',
            environment={
                'solvent': 'water',
                'temperature': 300,
                'pressure': 1.0
            }
        )
        ```
        """
        # Accept temperature= or temperature_K= (alias); exactly one required.
        if temperature is None and temperature_K is None:
            raise TypeError("MDSimulator requires temperature= (or its alias temperature_K=) in Kelvin")
        if temperature is not None and temperature_K is not None and temperature != temperature_K:
            raise TypeError(
                f"MDSimulator: temperature ({temperature}) and temperature_K ({temperature_K}) both given with different values"
            )
        temperature = temperature if temperature is not None else temperature_K

        # `n_steps=` is accepted at construction for compat with old callsites that
        # passed it through; the actual integration count is the n_steps= argument
        # to .run(). If both are given they must agree.
        if n_steps is not None:
            self._init_n_steps = n_steps

        self.bond_or_molecule = bond_or_molecule
        self.temperature = temperature

        # Setup environment modulators
        self.environment_config = environment or {}
        self.modulators = self._setup_environment(environment)

        # Initialize solver cache for quantum forces (CRITICAL for performance!)
        self.solver_cache = {}
        self.timestep = timestep
        self.force_method = force_method.lower()
        self.use_governance = use_governance
        self.backend = backend
        self.random_seed = random_seed
        # Builder QuantumSystem → drives quantum forces through its energy_fn()
        # (re-solves + warm-starts at each geometry → correct off-equilibrium).
        self.quantum_system = quantum_system
        # Capability path: any ForceProvider solver instance can drive MD forces
        # directly via force_method='solver' (see _setup_forces / compute_forces).
        self.solver = solver

        # Extract atomic information
        self._setup_system()

        # Create integrator
        self.integrator = create_integrator(integrator, timestep)
        logger.info(f"Created integrator: {integrator}")

        # Create thermostat (if specified)
        if thermostat is not None:
            thermostat_kwargs = {}
            if 'coupling_time' in kwargs:
                thermostat_kwargs['coupling_time'] = kwargs['coupling_time']

            # For Langevin thermostat, use friction from solvent if available
            if 'friction_coefficient' in kwargs:
                thermostat_kwargs['friction_coefficient'] = kwargs['friction_coefficient']
            elif thermostat.lower() == 'langevin' and hasattr(self, 'langevin_friction') and self.langevin_friction:
                # Use solvent-derived friction for Langevin dynamics
                thermostat_kwargs['friction_coefficient'] = self.langevin_friction
                logger.info(f"  Using solvent-derived Langevin friction: {self.langevin_friction:.2f} ps⁻¹")

            self.thermostat = create_thermostat(thermostat, temperature, **thermostat_kwargs)
            self.ensemble = 'NVT'
            logger.info(f"Created thermostat: {thermostat} (NVT ensemble)")
        else:
            self.thermostat = None
            self.ensemble = 'NVE'
            logger.info("No thermostat (NVE ensemble - energy conservation)")

        # Setup force computation
        self._setup_forces(**kwargs)

        # Initialize velocities
        if initial_velocities is not None:
            self.velocities = initial_velocities.copy()
            logger.info("Using provided initial velocities")
        else:
            # Will generate Maxwell-Boltzmann in run()
            self.velocities = None
            logger.info("Will generate Maxwell-Boltzmann velocities")

        # Statistics
        self.n_steps_run = 0

        logger.info(f"MDSimulator initialized:")
        logger.info(f"  System: {self.n_atoms} atoms")
        logger.info(f"  Temperature: {temperature:.1f} K")
        logger.info(f"  Timestep: {timestep:.3f} fs")
        logger.info(f"  Force method: {force_method}")
        logger.info(f"  Ensemble: {self.ensemble}")

    def _setup_system(self):
        """Extract atomic positions, masses, symbols from bond/molecule."""
        # Check if it's a bond or molecule
        if hasattr(self.bond_or_molecule, 'atom_1') and hasattr(self.bond_or_molecule, 'atom_2'):
            # It's a bond
            atoms = [self.bond_or_molecule.atom_1, self.bond_or_molecule.atom_2]
            self.is_bond = True
        elif hasattr(self.bond_or_molecule, 'atoms'):
            # It's a molecule
            atoms = self.bond_or_molecule.atoms
            self.is_bond = False
        else:
            raise ValueError("Input must be Bond or Molecule object")

        self.atoms = atoms
        self.n_atoms = len(atoms)

        # Extract positions, masses, symbols
        # Note: Kanad stores positions in Angstroms, but MD uses Bohr internally
        ANGSTROM_TO_BOHR = 1.8897259886
        positions_angstrom = np.array([atom.position for atom in atoms])
        self.positions = positions_angstrom * ANGSTROM_TO_BOHR  # Convert to Bohr
        self.masses = np.array([atom.atomic_mass for atom in atoms])
        self.symbols = [atom.symbol for atom in atoms]

        logger.debug(f"System setup: {self.n_atoms} atoms, total mass = {np.sum(self.masses):.2f} amu")

    def _setup_environment(self, environment: Optional[Dict[str, Any]]) -> list:
        """
        Setup environment modulators based on configuration.

        Uses EnvironmentIntegration for unified environment handling with:
        - Dielectric screening for Coulomb interactions
        - Langevin friction from solvent viscosity
        - Pressure-dependent compression corrections
        - Temperature-dependent thermal effects

        Args:
            environment: Dictionary with environment settings

        Returns:
            List of modulator objects
        """
        if environment is None:
            self.env_integration = None
            return []

        modulators = []

        try:
            # Create unified EnvironmentIntegration
            EnvironmentIntegration, EnvironmentConditions = _get_environment_classes()

            env_conditions = EnvironmentConditions(
                temperature=environment.get('temperature', self.temperature),
                pressure=environment.get('pressure', 1.0),
                solvent=environment.get('solvent', 'vacuum'),
                pH=environment.get('pH', None)
            )

            self.env_integration = EnvironmentIntegration(env_conditions)
            logger.info(f"  Environment Integration: ENABLED")
            logger.info(f"    Temperature: {env_conditions.temperature} K")
            logger.info(f"    Pressure: {env_conditions.pressure} atm")
            logger.info(f"    Solvent: {env_conditions.solvent}")
            if env_conditions.pH is not None:
                logger.info(f"    pH: {env_conditions.pH}")

            # Get dynamics parameters from environment integration
            dynamics_params = self.env_integration.get_dynamics_parameters()

            # If solvent specified, add solvent modulator and get Langevin friction
            if environment.get('solvent') and environment['solvent'] != 'vacuum':
                from kanad.core.environment.solvent import SolventModulator
                solvent_mod = SolventModulator(
                    solvent_name=environment['solvent']
                )
                modulators.append(('solvent', solvent_mod))

                # Store dielectric screening factor
                self.dielectric_screening = dynamics_params.get('dielectric_screening', 1.0)

                # Get friction coefficient for Langevin dynamics
                if 'friction_coefficient' in dynamics_params:
                    self.langevin_friction = dynamics_params['friction_coefficient']
                    logger.info(f"    Langevin friction: {self.langevin_friction:.2f} ps⁻¹")
                else:
                    self.langevin_friction = None

            # Temperature modulator
            if 'temperature' in environment and environment['temperature'] != 298.15:
                from kanad.core.environment.temperature import TemperatureModulator
                temp_mod = TemperatureModulator(
                    temperature=environment['temperature']
                )
                modulators.append(('temperature', temp_mod))

            # Pressure modulator
            if environment.get('pressure', 1.0) > 1.0:
                from kanad.core.environment.pressure import PressureModulator
                pressure_mod = PressureModulator(
                    pressure=environment['pressure']
                )
                modulators.append(('pressure', pressure_mod))

            # pH modulator (pH itself is carried through EnvironmentConditions/env_integration)
            if environment.get('pH') is not None:
                from kanad.core.environment.ph_effects import pHModulator
                ph_mod = pHModulator()  # constructor takes no args; pH handled via env_integration
                modulators.append(('pH', ph_mod))

        except ImportError as e:
            logger.warning(f"Environment modulator import failed: {e}")
            self.env_integration = None
        except Exception as e:
            logger.warning(f"Environment setup failed: {e}")
            self.env_integration = None

        return modulators

    def apply_environment_effects(
        self,
        positions: np.ndarray,
        forces: np.ndarray,
        potential_energy: float
    ) -> tuple:
        """
        Apply environmental effects to forces and energy.

        This method modifies forces and potential energy based on
        active environment modulators (solvent, pressure, temperature, pH).

        Environment effects applied:
        1. Dielectric screening of Coulomb interactions
        2. Solvation energy and cavity forces
        3. Pressure-dependent compression
        4. Temperature-dependent anharmonic corrections

        Args:
            positions: Atomic positions (N_atoms, 3) in Bohr
            forces: Current forces (N_atoms, 3) in Ha/Bohr
            potential_energy: Current potential energy in Hartree

        Returns:
            (modified_forces, modified_potential_energy)
        """
        if not self.modulators and getattr(self, 'env_integration', None) is None:
            return forces, potential_energy

        modified_forces = forces.copy()
        modified_energy = potential_energy

        # Apply dielectric screening to forces if in solvent
        if hasattr(self, 'dielectric_screening') and self.dielectric_screening != 1.0:
            # The electrostatic part of the force is screened by dielectric
            # For simple model: F_screened ≈ F_gas * screening_factor
            # More sophisticated: separate ionic/covalent force contributions
            screening = 1.0 / self.dielectric_screening
            # Apply partial screening (mostly electrostatic in nature)
            # Don't screen covalent bond forces, only partial charges
            electrostatic_fraction = 0.3  # Assume 30% of force is electrostatic
            modified_forces *= (1.0 - electrostatic_fraction * (1.0 - screening))

        for mod_type, modulator in self.modulators:
            try:
                if mod_type == 'solvent':
                    # Apply implicit solvent effects (reaction field)
                    # Use apply_solvent method if available
                    if hasattr(modulator, 'apply_solvent'):
                        BOHR_TO_ANGSTROM = 0.529177
                        geometry = positions * BOHR_TO_ANGSTROM

                        # Try to get solvation energy from modulator
                        try:
                            solv_result = modulator.apply_solvent(
                                self.bond_or_molecule,
                                modulator.solvent_name if hasattr(modulator, 'solvent_name') else 'water',
                                model='pcm'
                            )
                            if 'solvation_energy' in solv_result:
                                modified_energy += solv_result['solvation_energy']
                        except Exception:
                            pass

                    # Fallback to compute_solvation_energy
                    elif hasattr(modulator, 'compute_solvation_energy'):
                        BOHR_TO_ANGSTROM = 0.529177
                        geometry = positions * BOHR_TO_ANGSTROM

                        solv_result = modulator.compute_solvation_energy(geometry)
                        if 'solvation_energy' in solv_result:
                            modified_energy += solv_result['solvation_energy']

                        if 'solvation_force' in solv_result:
                            modified_forces += solv_result['solvation_force'] / BOHR_TO_ANGSTROM

                elif mod_type == 'pressure':
                    # Apply pressure correction using apply_pressure if available
                    if hasattr(modulator, 'apply_pressure'):
                        try:
                            pressure_val = self.environment_config.get('pressure', 1.0)
                            pressure_result = modulator.apply_pressure(
                                self.bond_or_molecule, pressure_val
                            )
                            if 'energy_correction' in pressure_result:
                                modified_energy += pressure_result['energy_correction']
                        except Exception:
                            pass
                    elif hasattr(modulator, 'compute_pressure_correction'):
                        pressure_result = modulator.compute_pressure_correction(positions)
                        if 'energy_correction' in pressure_result:
                            modified_energy += pressure_result['energy_correction']
                        if 'force_correction' in pressure_result:
                            modified_forces += pressure_result['force_correction']

                elif mod_type == 'temperature':
                    # Temperature effects on potential (anharmonic corrections)
                    if hasattr(modulator, 'apply_temperature'):
                        try:
                            temp_val = self.environment_config.get('temperature', 298.15)
                            thermal_result = modulator.apply_temperature(
                                self.bond_or_molecule, temp_val
                            )
                            # Free energy correction from thermal effects
                            if 'free_energy' in thermal_result and 'energy' in thermal_result:
                                # G - E = -TS is the thermal correction
                                thermal_corr = thermal_result['free_energy'] - thermal_result['energy']
                                modified_energy += thermal_corr
                        except Exception:
                            pass
                    elif hasattr(modulator, 'compute_thermal_correction'):
                        thermal_result = modulator.compute_thermal_correction(
                            potential_energy
                        )
                        if 'energy_correction' in thermal_result:
                            modified_energy += thermal_result['energy_correction']

                elif mod_type == 'pH':
                    # pH effects on protonation states
                    # Typically handled at initialization, not during dynamics
                    pass

            except Exception as e:
                logger.debug(f"Environment effect {mod_type} failed: {e}")

        return modified_forces, modified_energy

    def _setup_forces(self, **kwargs):
        """Setup force computation based on force_method."""
        if self.force_method in ['hf', 'mp2']:
            # Classical ab initio forces using PySCF gradients
            from kanad.core.gradients import GradientCalculator

            # Get hamiltonian
            if hasattr(self.bond_or_molecule, 'hamiltonian'):
                hamiltonian = self.bond_or_molecule.hamiltonian
            else:
                raise ValueError("Bond/Molecule must have hamiltonian for force calculation")

            # Create gradient calculator
            self.gradient_calc = GradientCalculator(
                self.bond_or_molecule,
                method=self.force_method.upper()
            )

            logger.info(f"Using {self.force_method.upper()} forces via PySCF gradients")

        elif self.force_method in ['hivqe', 'vqe', 'sqd']:
            # Quantum forces with solver caching for performance
            logger.info(f"Using quantum {self.force_method.upper()} forces")
            logger.info(f"  Backend: {self.backend}")
            logger.info(f"  Governance: {self.use_governance}")
            logger.info(f"  Solver caching: ENABLED (critical for performance!)")

            # Store parameters for the legacy quantum force computation
            self.quantum_params = {
                'backend': self.backend,
                'use_governance': self.use_governance,
                'solver_cache': self.solver_cache,  # Pass cache for reuse!
                **kwargs
            }

            # Validated path: if a builder QuantumSystem is supplied, forces come
            # from its energy_fn() via central finite differences that RE-SOLVE the
            # electronic structure at each displaced geometry (warm-started). That
            # is correct off-equilibrium. Without it we fall back to the legacy
            # quantum_md path, which freezes the VQE parameters at the reference
            # geometry (Pulay term ignored) and is therefore wrong anywhere but the
            # minimum — so we warn loudly.
            self._force_energy_fn = None
            self._force_warm_state = None
            if self.quantum_system is not None:
                self._force_energy_fn = self.quantum_system.energy_fn()
                logger.info("  Quantum forces via builder energy_fn (validated: "
                            "re-solves + warm-starts per geometry).")
            else:
                logger.warning(
                    "  No quantum_system supplied — using the LEGACY frozen-θ "
                    "quantum_md forces. These are correct only AT the equilibrium "
                    "geometry (Pulay term ignored); MD/IRC trajectories away from "
                    "the minimum will be WRONG. Pass quantum_system=<builder "
                    "QuantumSystem> to use the validated re-solving force path.")

        elif self.force_method == 'solver':
            # Capability path: consume ANY ForceProvider solver instance directly
            # (PhysicsVQE, SamplingSQDSolver, or a user-defined solver). Forces come
            # from solver.nuclear_gradient() — the same central-FD floor over the
            # solver's energy_fn() that the validated builder path uses.
            from kanad.solvers.capabilities import ForceProvider
            if self.solver is None:
                raise ValueError(
                    "force_method='solver' requires a solver= ForceProvider instance "
                    "(e.g. PhysicsVQE or SamplingSQDSolver)."
                )
            if not isinstance(self.solver, ForceProvider):
                raise TypeError(
                    f"solver={type(self.solver).__name__} is not a ForceProvider "
                    f"(it must expose energy_fn + nuclear_gradient); cannot drive MD forces."
                )
            self._force_warm_state = None
            logger.info(
                f"Quantum forces via capability protocol: "
                f"{type(self.solver).__name__}.nuclear_gradient()"
            )

        else:
            raise ValueError(f"Unknown force_method: {self.force_method}")

    def compute_forces(self, positions: np.ndarray) -> tuple:
        """
        Compute forces on atoms at given positions.

        Args:
            positions: Atomic positions (N_atoms, 3) in Bohr

        Returns:
            (forces, potential_energy):
                forces: (N_atoms, 3) in Ha/Bohr
                potential_energy: Potential energy in Hartree
        """
        if self.force_method in ['hf', 'mp2']:
            # Update positions (convert from Bohr to Angstrom for gradient calculator)
            BOHR_TO_ANGSTROM = 0.529177
            for i, atom in enumerate(self.atoms):
                atom.position = positions[i] * BOHR_TO_ANGSTROM

            # Compute gradient
            result = self.gradient_calc.compute_gradient()

            forces = result['forces']  # Already -gradient, in Ha/Bohr
            potential_energy = result['energy']

        elif self.force_method in ['hivqe', 'vqe', 'sqd']:
            if self._force_energy_fn is not None:
                # Validated path: numerical forces from the builder energy_fn,
                # re-solving + warm-starting at each geometry (correct off-eq).
                from kanad.dynamics.quantum_forces import compute_numerical_forces
                fr = compute_numerical_forces(
                    self._force_energy_fn,
                    np.asarray(positions, dtype=float),
                    warm_state=self._force_warm_state,
                )
                forces = fr.forces
                potential_energy = fr.energy
                self._force_warm_state = fr.warm_state
            else:
                # Legacy frozen-θ path (wrong off-equilibrium; warned at setup).
                from kanad.dynamics.quantum_md import compute_quantum_forces
                forces, potential_energy = compute_quantum_forces(
                    positions,
                    self.bond_or_molecule,
                    method=self.force_method,
                    **self.quantum_params,
                )

        elif self.force_method == 'solver':
            # Capability path: forces from the solver's nuclear_gradient. positions
            # are already in Bohr (the protocol contract), so no unit conversion.
            gr = self.solver.nuclear_gradient(
                np.asarray(positions, dtype=float),
                warm_state=self._force_warm_state,
            )
            forces = gr.forces          # = -gradient (Ha/Bohr)
            potential_energy = gr.energy
            self._force_warm_state = gr.warm_state

        else:
            raise ValueError(f"Unknown force_method: {self.force_method}")

        # Apply environment effects (solvent, pressure, temperature, pH)
        if self.modulators:
            forces, potential_energy = self.apply_environment_effects(
                positions, forces, potential_energy
            )

        return forces, potential_energy

    def _n_dof(self) -> int:
        """Degrees of freedom, matching thermostats.compute_temperature convention.

        Monatomic -> 3, diatomic -> 3N-5, else -> 3N-6. This keeps NVE-branch
        temperatures consistent with the thermostat path for the same KE.
        """
        if self.n_atoms == 1:
            return 3  # monatomic
        elif self.n_atoms == 2:
            return 3 * self.n_atoms - 5  # diatomic (one rotation degenerate)
        return 3 * self.n_atoms - 6

    def _nve_temperature(self, kinetic_energy: float) -> float:
        """NVE instantaneous temperature: T = 2*KE / (k_B * N_dof)."""
        return (2.0 * kinetic_energy) / (3.1668105e-6 * self._n_dof())

    def run(
        self,
        n_steps: int,
        save_frequency: int = 10,
        save_trajectory: bool = True,
        output_file: Optional[str] = None,
        equilibrate: bool = False,
        n_equil_steps: int = 100,
        verbose: bool = True,
        check_energy: bool = True,
        on_frame: Optional[callable] = None,
        on_equil_frame: Optional[callable] = None,
    ) -> MDResult:
        """
        Run molecular dynamics simulation.

        Args:
            n_steps: Number of MD steps
            save_frequency: Save trajectory every N steps
            save_trajectory: Store trajectory in memory
            output_file: Save trajectory to file (HDF5 or XYZ)
            equilibrate: Run equilibration before production
            n_equil_steps: Equilibration steps
            verbose: Print progress
            check_energy: Monitor energy conservation (NVE only)

        Returns:
            MDResult with simulation results and trajectory
        """
        logger.info("=" * 70)
        logger.info("STARTING MOLECULAR DYNAMICS SIMULATION")
        logger.info("=" * 70)

        start_time = time_module.time()

        # Reset per-trajectory integrator state so reusing this MDSimulator
        # across multiple run() calls starts each run fresh (e.g. Leapfrog's
        # first-step init / half-step velocity), not from stale state.
        self.integrator.reset()

        # Initialize velocities if not provided
        if self.velocities is None:
            if verbose:
                logger.info("Generating Maxwell-Boltzmann initial velocities...")

            init_result = generate_initial_conditions(
                self.positions,
                self.masses,
                self.temperature,
                force_function=self.compute_forces if equilibrate else None,
                equilibrate=equilibrate,
                n_equil_steps=n_equil_steps,
                seed=self.random_seed,
                on_equil_step=on_equil_frame,
            )

            self.positions = init_result.positions
            self.velocities = init_result.velocities

            if verbose:
                logger.info(f"Initial temperature: {init_result.temperature:.2f} K")
                logger.info(f"Initial KE: {init_result.kinetic_energy:.6f} Ha")

        # Initialize trajectory
        if save_trajectory:
            trajectory = Trajectory()
            trajectory.n_atoms = self.n_atoms
            trajectory.atom_symbols = self.symbols
            trajectory.atom_masses = self.masses
        else:
            trajectory = None

        # Compute initial forces
        forces, pot_energy = self.compute_forces(self.positions)

        # Compute initial kinetic energy
        ke = self.integrator.compute_kinetic_energy(self.velocities, self.masses)

        # Create initial state
        state = IntegratorState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            forces=forces,
            masses=self.masses,
            time=0.0,
            kinetic_energy=ke,
            potential_energy=pot_energy
        )

        # Save initial frame
        if save_trajectory and (0 % save_frequency == 0):
            if self.thermostat:
                T = self.thermostat.compute_temperature(state.velocities, state.masses)
            else:
                # NVE: report real T from initial (Maxwell-Boltzmann) KE, not 0.0
                T = self._nve_temperature(state.kinetic_energy)

            trajectory.add_frame(
                state.positions,
                state.velocities,
                state.forces,
                state.kinetic_energy,
                state.potential_energy,
                T,
                state.time
            )

        # Statistics
        initial_energy = state.kinetic_energy + state.potential_energy
        energies = []
        temperatures = []
        kinetic_energies = []
        potential_energies = []

        # MD loop
        if verbose:
            logger.info(f"\nRunning {n_steps} MD steps...")
            logger.info(f"  Timestep: {self.timestep:.3f} fs")
            logger.info(f"  Total time: {n_steps * self.timestep:.2f} fs")
            logger.info(f"  Ensemble: {self.ensemble}")

        for step in range(1, n_steps + 1):
            # Integration step
            state = self.integrator.step(state, self.compute_forces)

            # Apply thermostat (if NVT)
            if self.thermostat is not None:
                state.velocities = self.thermostat.apply(
                    state.velocities,
                    state.masses,
                    self.timestep,
                    current_ke=state.kinetic_energy
                )

                # Recompute kinetic energy after thermostat
                state.kinetic_energy = self.integrator.compute_kinetic_energy(
                    state.velocities,
                    state.masses
                )

            # Compute temperature
            if self.thermostat:
                T = self.thermostat.compute_temperature(state.velocities, state.masses)
            else:
                # Use shared dof convention so NVE T matches the thermostat path
                T = self._nve_temperature(state.kinetic_energy)

            # Statistics
            total_energy = state.kinetic_energy + state.potential_energy
            energies.append(total_energy)
            temperatures.append(T)
            kinetic_energies.append(state.kinetic_energy)
            potential_energies.append(state.potential_energy)

            # Save frame
            if save_trajectory and (step % save_frequency == 0):
                trajectory.add_frame(
                    state.positions,
                    state.velocities,
                    state.forces,
                    state.kinetic_energy,
                    state.potential_energy,
                    T,
                    state.time
                )
                # Live callback for streaming frames to API
                if on_frame is not None:
                    try:
                        on_frame(step, n_steps, state.positions.tolist(), T, total_energy,
                                 state.kinetic_energy, state.potential_energy,
                                 state.forces.tolist() if state.forces is not None else None)
                    except Exception:
                        pass  # Don't let callback errors break simulation

            # Progress (avoid division by zero for small n_steps)
            progress_interval = max(1, n_steps // 10)
            if verbose and (step % progress_interval == 0 or step == n_steps):
                energy_drift = total_energy - initial_energy
                logger.info(f"  Step {step}/{n_steps}: "
                          f"T={T:.1f}K, E_tot={total_energy:.6f}Ha, "
                          f"ΔE={energy_drift:.6e}Ha")

        # End timing
        wall_time = time_module.time() - start_time
        steps_per_second = n_steps / wall_time

        # Compute final statistics
        final_energy = energies[-1]
        avg_energy = np.mean(energies)
        energy_drift = final_energy - initial_energy

        avg_temp = np.mean(temperatures)
        temp_std = np.std(temperatures)

        # Compute environment effects summary if applicable
        env_effects = None
        if hasattr(self, 'env_integration') and self.env_integration is not None:
            env_effects = {
                'conditions': self.env_integration.environment.to_dict(),
                'dielectric_screening': getattr(self, 'dielectric_screening', 1.0),
                'langevin_friction': getattr(self, 'langevin_friction', None),
                'modulators_active': [m[0] for m in self.modulators]
            }

        # Create result
        result = MDResult(
            trajectory=trajectory,
            final_positions=state.positions,
            final_velocities=state.velocities,
            final_energy=final_energy,
            avg_temperature=avg_temp,
            avg_kinetic_energy=np.mean(kinetic_energies) if kinetic_energies else 0.0,
            avg_potential_energy=np.mean(potential_energies) if potential_energies else 0.0,
            avg_total_energy=avg_energy,
            energy_drift=energy_drift,
            temperature_std=temp_std,
            n_steps_completed=n_steps,
            wall_time=wall_time,
            steps_per_second=steps_per_second,
            converged=True,
            metadata={
                'force_method': self.force_method,
                'integrator': type(self.integrator).__name__,
                'thermostat': type(self.thermostat).__name__ if self.thermostat else 'None',
                'ensemble': self.ensemble,
                'timestep': self.timestep,
                'temperature': self.temperature,
            },
            energies=energies,
            temperatures=temperatures,
            environment_effects=env_effects,
            timestep=self.timestep
        )

        # Save trajectory to file
        if output_file is not None and trajectory is not None:
            output_path = Path(output_file)
            if output_path.suffix in ['.h5', '.hdf5']:
                writer = TrajectoryWriter('hdf5')
            elif output_path.suffix == '.xyz':
                writer = TrajectoryWriter('xyz')
            else:
                logger.warning(f"Unknown file extension: {output_path.suffix}, using HDF5")
                writer = TrajectoryWriter('hdf5')

            writer.write(trajectory, output_file)
            logger.info(f"Trajectory saved to {output_file}")

        # Final summary
        logger.info("=" * 70)
        logger.info("MD SIMULATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Final results:")
        logger.info(f"  Steps completed: {n_steps}")
        logger.info(f"  Final energy: {final_energy:.6f} Ha")
        logger.info(f"  Average energy: {avg_energy:.6f} Ha")
        logger.info(f"  Energy drift: {energy_drift:.6e} Ha ({abs(energy_drift/initial_energy)*100:.3f}%)")
        logger.info(f"  Average temperature: {avg_temp:.2f} ± {temp_std:.2f} K")
        logger.info(f"  Wall time: {wall_time:.2f} s")
        logger.info(f"  Performance: {steps_per_second:.1f} steps/s")

        if check_energy and self.ensemble == 'NVE':
            energy_conservation = abs(energy_drift / initial_energy)
            if energy_conservation < 1e-4:
                logger.info(f"  ✅ Excellent energy conservation (<0.01%)")
            elif energy_conservation < 1e-3:
                logger.info(f"  ✅ Good energy conservation (<0.1%)")
            elif energy_conservation < 1e-2:
                logger.warning(f"  ⚠️  Moderate energy drift (>0.1%)")
            else:
                logger.warning(f"  ⚠️  Poor energy conservation (>1%) - reduce timestep!")

        return result

    def analyze_environment_effects(self, md_result: MDResult = None) -> Dict[str, Any]:
        """
        Analyze how environment affects the MD simulation results.

        Computes environment-corrected energies and provides analysis
        of how temperature, solvent, and pressure modify the dynamics.

        Args:
            md_result: MDResult to analyze. If None, requires previous run.

        Returns:
            Dict with environment analysis:
                - 'gas_phase_energies': Energies without environment corrections
                - 'environment_corrected_energies': Energies with all corrections
                - 'solvation_energy': Average solvation energy (if solvent present)
                - 'thermal_correction': Temperature-dependent correction
                - 'dielectric_screening': Coulomb screening factor
                - 'dynamics_parameters': Parameters used for dynamics
        """
        if md_result is None:
            raise ValueError("Must provide md_result or run simulation first")

        analysis = {
            'gas_phase_energies': md_result.energies,
            'n_steps': md_result.n_steps_completed,
            'timestep': md_result.timestep,
        }

        if not hasattr(self, 'env_integration') or self.env_integration is None:
            analysis['environment'] = 'vacuum'
            analysis['environment_corrected_energies'] = md_result.energies
            return analysis

        # Get environment conditions
        analysis['environment'] = self.env_integration.environment.to_dict()

        # Apply environment corrections to energies
        if md_result.energies is not None:
            # Get solvation energy contribution
            try:
                solvation_energy = self.env_integration._compute_solvation_energy(
                    self.bond_or_molecule,
                    self.env_integration.environment.solvent,
                    self.env_integration.environment.temperature
                )
                analysis['solvation_energy'] = solvation_energy
            except Exception:
                solvation_energy = 0.0
                analysis['solvation_energy'] = 0.0

            # Compute corrected energies
            corrected = np.array(md_result.energies) + solvation_energy
            analysis['environment_corrected_energies'] = corrected.tolist()
            analysis['avg_solvation_correction'] = solvation_energy

        # Get dynamics parameters
        analysis['dynamics_parameters'] = self.env_integration.get_dynamics_parameters()

        # Dielectric screening
        analysis['dielectric_screening'] = getattr(self, 'dielectric_screening', 1.0)

        # Langevin friction
        if hasattr(self, 'langevin_friction') and self.langevin_friction:
            analysis['langevin_friction'] = self.langevin_friction

        return analysis
