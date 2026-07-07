"""
Chemical Reaction Dynamics with Governance

Simulates chemical reactions using quantum methods with governance protocol
integration for accurate modeling of bond breaking/forming processes.

Key Features:
- Transition state (TS) finding using NEB/Dimer methods
- Intrinsic Reaction Coordinate (IRC) following
- Eyring-Polanyi rate constant calculations
- Reactive molecular dynamics with bond detection
- Governance-aware reaction mechanisms

Theory:
------
For a chemical reaction A → B via transition state TS:
- ΔG‡ = G(TS) - G(reactants)  [activation free energy]
- k = (kT/h) * exp(-ΔG‡/RT)   [Eyring equation]
- IRC connects TS to reactants and products

Governance Integration:
---------------------
- CovalentProtocol: Orbital hybridization changes
- IonicProtocol: Electron transfer reactions
- MetallicProtocol: Catalytic surfaces

References:
----------
- Eyring (1935) J. Chem. Phys. 3, 107
- Fukui (1970) J. Phys. Chem. 74, 4161 - IRC
- Henkelman (2000) J. Chem. Phys. 113, 9978 - NEB
- Mills & Jónsson (1994) Phys. Rev. Lett. 72, 1124 - Dimer
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from kanad.reactions._irc import mass_weighted_irc_step

logger = logging.getLogger(__name__)


# Physical constants
K_BOLTZMANN = 3.1668105e-6  # Hartree/K
PLANCK_CONSTANT = 1.0  # In atomic units (ℏ = 1)
HARTREE_TO_KCAL = 627.5095


class ReactionType(Enum):
    """Types of chemical reactions."""
    BOND_BREAKING = "bond_breaking"
    BOND_FORMING = "bond_forming"
    DISSOCIATION = "bond_breaking"  # Alias for bond breaking (e.g., H2 → H + H)
    ASSOCIATION = "bond_forming"    # Alias for bond forming (e.g., H + H → H2)
    SUBSTITUTION = "substitution"
    ADDITION = "addition"
    ELIMINATION = "elimination"
    REARRANGEMENT = "rearrangement"
    ELECTRON_TRANSFER = "electron_transfer"
    PROTON_TRANSFER = "proton_transfer"


@dataclass
class TransitionState:
    """
    Transition state structure and properties.

    Attributes:
        geometry: Atomic positions at TS (N_atoms, 3)
        energy: Total energy at TS in Hartree
        gradient: Energy gradient at TS
        hessian: Hessian matrix at TS
        imaginary_frequency: Single imaginary frequency (cm⁻¹)
        reaction_coordinate: Eigenvector of imaginary mode
        bond_orders: Bond orders at TS
        verified: Whether TS is verified (one imaginary freq)
    """
    geometry: np.ndarray
    energy: float
    gradient: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None
    imaginary_frequency: Optional[float] = None
    reaction_coordinate: Optional[np.ndarray] = None
    bond_orders: Optional[Dict[Tuple[int, int], float]] = None
    verified: bool = False


@dataclass
class ReactionPath:
    """
    Reaction path from reactants to products via TS.

    Attributes:
        geometries: (n_images, n_atoms, 3) geometries along path
        energies: (n_images,) energies along path
        reaction_coordinate: (n_images,) s-values along IRC
        ts_index: Index of transition state in path
        barrier_height: Forward activation energy (Hartree)
        reaction_energy: Reaction enthalpy (Hartree)
        path_length: Total path length in a₀
    """
    geometries: np.ndarray
    energies: np.ndarray
    reaction_coordinate: np.ndarray
    ts_index: int
    barrier_height: float
    reaction_energy: float
    path_length: float


@dataclass
class ReactionResult:
    """
    Complete results from reaction simulation.

    Attributes:
        reaction_type: Type of reaction
        reactants: Reactant bond/molecule
        products: Product bond/molecule (if known)
        transition_state: TS structure and properties
        forward_path: IRC path from reactants to TS
        reverse_path: IRC path from TS to products
        rate_constant: Eyring rate constant (1/s)
        half_life: Reaction half-life (s)
        activation_energy: Activation energy (kcal/mol)
        reaction_enthalpy: Reaction enthalpy (kcal/mol)
        governance_protocol: Governance used for reaction
    """
    reaction_type: ReactionType
    reactants: Any
    products: Optional[Any]
    transition_state: TransitionState
    forward_path: Optional[ReactionPath]
    reverse_path: Optional[ReactionPath]
    rate_constant: float
    half_life: float
    activation_energy: float
    reaction_enthalpy: float
    governance_protocol: str


class ReactionSimulator:
    """
    Simulate chemical reactions with governance-aware quantum methods.

    This class provides tools for:
    1. Finding transition states
    2. Following IRC to reactants/products
    3. Computing reaction rates
    4. Running reactive MD

    Example:
    --------
    ```python
    from kanad.bonds import BondFactory
    from kanad.reactions import ReactionSimulator

    # Create H2 molecule
    h2 = BondFactory.create_bond('H', 'H', distance=0.74)

    # Set up H2 dissociation reaction
    reaction = ReactionSimulator(reactants=[h2])

    # Find transition state
    ts = reaction.find_transition_state()
    print(f"TS energy: {ts.energy:.4f} Ha")

    # Compute rate constant at 1000 K
    rate = reaction.compute_rate_constant(temperature=1000)
    print(f"Rate constant: {rate:.2e} 1/s")
    ```
    """

    def __init__(
        self,
        reactants: List,
        products: Optional[List] = None,
        reaction_type: Optional[ReactionType] = None,
        solver_method: str = 'vqe',
        use_governance: bool = True,
        max_excitations: int = 5,
        environment: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize reaction simulator.

        Args:
            reactants: List of Bond/Molecule objects
            products: Optional list of product structures
            reaction_type: Type of reaction (auto-detected if None)
            solver_method: Quantum solver ('vqe', 'physics_vqe', 'sqd')
            use_governance: Use governance protocols
            max_excitations: Max excitations for PhysicsVQE (affects accuracy vs cost)
            environment: Environment conditions dict with keys:
                - temperature: Temperature in K (default 298.15)
                - pressure: Pressure in atm (default 1.0)
                - solvent: Solvent name or 'vacuum' (default 'vacuum')
                - pH: pH value (optional)
        """
        self.reactants = reactants
        self.products = products
        self.solver_method = solver_method
        self.use_governance = use_governance
        self.max_excitations = max_excitations
        self.environment = environment or {}

        # Auto-detect reaction type
        if reaction_type is None:
            self.reaction_type = self._detect_reaction_type()
        else:
            self.reaction_type = reaction_type

        # Select governance protocol
        self.governance = self._select_governance()

        # Cache for energy calculations
        self._energy_cache = {}

        # Setup environment integration
        self.env_integration = None
        self._setup_environment()

        # Extract atoms from reactants
        self.atoms = []
        self.masses = []
        for reactant in reactants:
            if hasattr(reactant, 'atom_1'):
                self.atoms.extend([reactant.atom_1, reactant.atom_2])
                self.masses.extend([reactant.atom_1.atomic_mass, reactant.atom_2.atomic_mass])
            elif hasattr(reactant, 'atoms'):
                self.atoms.extend(reactant.atoms)
                self.masses.extend([a.atomic_mass for a in reactant.atoms])

        self.n_atoms = len(self.atoms)
        self.masses = np.array(self.masses)

        logger.info(f"ReactionSimulator initialized:")
        logger.info(f"  Type: {self.reaction_type.value}")
        logger.info(f"  Atoms: {self.n_atoms}")
        logger.info(f"  Governance: {self.governance}")
        if self.env_integration:
            logger.info(f"  Environment: {self.environment}")

    def _setup_environment(self):
        """Setup environment integration for reaction rate corrections."""
        if not self.environment:
            return

        try:
            from kanad.core.environment.integration import EnvironmentIntegration, EnvironmentConditions

            conditions = EnvironmentConditions(
                temperature=self.environment.get('temperature', 298.15),
                pressure=self.environment.get('pressure', 1.0),
                solvent=self.environment.get('solvent', 'vacuum'),
                pH=self.environment.get('pH', None)
            )

            self.env_integration = EnvironmentIntegration(conditions)
            logger.info("Environment integration enabled for reaction rates")

        except ImportError:
            logger.warning("Environment module not available - using gas phase rates")
            self.env_integration = None
        except Exception as e:
            logger.warning(f"Environment setup failed: {e}")
            self.env_integration = None

    def _detect_reaction_type(self) -> ReactionType:
        """Auto-detect reaction type from reactants."""
        # Simple heuristics
        if len(self.reactants) == 1:
            # Single molecule - likely dissociation or rearrangement
            return ReactionType.BOND_BREAKING
        elif len(self.reactants) == 2:
            # Two reactants - likely addition or substitution
            return ReactionType.ADDITION
        else:
            return ReactionType.REARRANGEMENT

    def _select_governance(self) -> str:
        """Select appropriate governance protocol."""
        if not self.use_governance:
            return 'none'

        # Check bond types
        bond_types = []
        for reactant in self.reactants:
            if hasattr(reactant, 'bond_type'):
                bond_types.append(reactant.bond_type)

        if 'ionic' in str(bond_types).lower():
            return 'ionic'
        elif 'metallic' in str(bond_types).lower():
            return 'metallic'
        else:
            return 'covalent'

    def compute_energy(
        self,
        positions: np.ndarray,
        use_cache: bool = True
    ) -> float:
        """
        Compute potential energy at given geometry.

        Uses VQE/PhysicsVQE for quantum energy if solver_method='vqe',
        otherwise uses classical Morse/LJ potentials.

        Args:
            positions: Atomic positions (N_atoms, 3)
            use_cache: Use cached values if available

        Returns:
            Energy in Hartree
        """
        # Check cache
        if use_cache:
            pos_key = tuple(positions.flatten().round(6))
            if pos_key in self._energy_cache:
                return self._energy_cache[pos_key]

        # Update geometry and compute energy
        try:
            # QUANTUM PATH: Use VQE/PhysicsVQE for ab initio energy
            if self.solver_method in ['vqe', 'physics_vqe', 'sqd']:
                energy = self._compute_quantum_energy(positions)

            # CLASSICAL PATH: Use classical force fields (fast but approximate)
            else:
                energy = self._compute_classical_energy(positions)

        except Exception as e:
            # Do NOT silently substitute a classical force-field energy for a FAILED
            # quantum solve — that returns a plausible-but-wrong number (a Morse/LJ
            # well, D_e~0.17 Ha) with no signal that the ab-initio path broke. Surface
            # the failure on the quantum path; only fall back when classical was chosen.
            # (Audit H11.)
            if self.solver_method in ['vqe', 'physics_vqe', 'sqd']:
                raise RuntimeError(
                    f"Quantum energy ({self.solver_method}) failed at this geometry: {e}. "
                    f"Refusing to silently substitute a classical force-field energy."
                ) from e
            logger.warning(f"Classical energy calculation issue: {e}")
            energy = self._compute_classical_energy(positions)

        # Cache result
        if use_cache:
            pos_key = tuple(positions.flatten().round(6))
            self._energy_cache[pos_key] = energy

        return energy

    def _compute_quantum_energy(self, positions: np.ndarray) -> float:
        """
        Compute energy using VQE/PhysicsVQE.

        This is the REAL quantum computation path.
        """
        from pyscf import gto

        # Build PySCF molecule at current geometry
        atom_str = ""
        for i, atom in enumerate(self.atoms):
            symbol = atom.symbol
            x, y, z = positions[i]
            atom_str += f"{symbol} {x} {y} {z}; "

        # positions come from atom.position, which kanad stores in ANGSTROM
        # (verified). Passing unit='bohr' compressed every geometry by ~1.889x, so
        # the quantum reaction energy was computed at the WRONG geometry. (Audit H10.)
        mol = gto.M(atom=atom_str.strip('; '), basis='sto-3g', unit='angstrom')

        # Registry-resolved (no concrete-class import) + EnergyProvider consumption.
        # PhysicsVQE is used for best accuracy; note all quantum solver_method values
        # ('vqe'/'sqd'/'physics_vqe') route here to PhysicsVQE today (unchanged behavior).
        from kanad.solvers import get_solver
        from kanad.solvers.capabilities import EnergyProvider

        solver = get_solver('physics_vqe')(pyscf_mol=mol, max_excitations=self.max_excitations)
        if not isinstance(solver, EnergyProvider):
            raise TypeError(f"solver {type(solver).__name__} is not an EnergyProvider")
        result = solver.solve()

        logger.debug(f"VQE energy at R={np.linalg.norm(positions[1]-positions[0]):.2f}: {result.energy:.6f} Ha")

        return result.energy

    def _compute_classical_energy(self, positions: np.ndarray) -> float:
        """
        Compute energy using classical potentials (Morse/LJ).

        Fast but no quantum correlation effects.
        """
        # For diatomic: Morse potential
        if self.n_atoms == 2:
            r = np.linalg.norm(positions[1] - positions[0])
            # Morse potential: E(r) = D_e * (1 - exp(-a(r-r_e)))^2 - D_e
            D_e = 0.17  # ~4.7 eV for H2
            r_e = 1.4   # ~0.74 Angstrom in Bohr
            a = 1.0     # Width parameter

            energy = D_e * (1 - np.exp(-a * (r - r_e)))**2 - D_e

            # Add repulsive core
            if r < 0.5:
                energy += 10.0 / r

        else:
            # For polyatomic: sum of pairwise LJ-like interactions
            energy = 0.0
            for i in range(self.n_atoms):
                for j in range(i + 1, self.n_atoms):
                    r = np.linalg.norm(positions[j] - positions[i])
                    if r > 0.5:
                        energy += -1.0 / r + 0.1 / r**6
                    else:
                        energy += 10.0 / r

        return energy

    def compute_gradient(
        self,
        positions: np.ndarray,
        delta: float = 0.001
    ) -> np.ndarray:
        """
        Compute energy gradient using finite differences.

        Args:
            positions: Atomic positions (N_atoms, 3)
            delta: Finite difference step in Bohr

        Returns:
            Gradient (N_atoms, 3) in Ha/Bohr
        """
        gradient = np.zeros_like(positions)

        for i in range(self.n_atoms):
            for j in range(3):
                pos_plus = positions.copy()
                pos_plus[i, j] += delta

                pos_minus = positions.copy()
                pos_minus[i, j] -= delta

                E_plus = self.compute_energy(pos_plus, use_cache=False)
                E_minus = self.compute_energy(pos_minus, use_cache=False)

                gradient[i, j] = (E_plus - E_minus) / (2 * delta)

        return gradient

    def find_transition_state(
        self,
        initial_guess: Optional[np.ndarray] = None,
        method: str = 'dimer',
        max_iterations: int = 100,
        convergence: float = 1e-4
    ) -> TransitionState:
        """
        Find transition state for the reaction.

        Args:
            initial_guess: Initial geometry guess
            method: TS search method ('dimer', 'neb', 'qst')
            max_iterations: Maximum optimization iterations
            convergence: Gradient convergence threshold

        Returns:
            TransitionState object
        """
        logger.info(f"Finding transition state using {method} method")

        # Get initial positions
        if initial_guess is None:
            positions = self._get_initial_guess()
        else:
            positions = initial_guess.copy()

        if method == 'dimer':
            ts_positions, ts_energy = self._dimer_search(
                positions, max_iterations, convergence
            )
        elif method == 'neb':
            ts_positions, ts_energy = self._neb_search(
                positions, max_iterations, convergence
            )
        else:
            # Simple gradient search towards saddle point
            ts_positions, ts_energy = self._simple_ts_search(
                positions, max_iterations, convergence
            )

        # Compute gradient at TS
        gradient = self.compute_gradient(ts_positions)

        # Estimate imaginary frequency BEFORE verifying: a true TS is a
        # first-order saddle (low gradient AND exactly one negative curvature
        # mode). Checking gradient norm alone would flag minima as verified.
        imag_freq = self._estimate_imaginary_frequency(ts_positions)

        # Create TS object
        ts = TransitionState(
            geometry=ts_positions,
            energy=ts_energy,
            gradient=gradient,
            verified=(np.linalg.norm(gradient) < convergence)
                     and (imag_freq is not None and imag_freq < 0)
        )

        ts.imaginary_frequency = imag_freq

        logger.info(f"TS found: E={ts_energy:.6f} Ha, |g|={np.linalg.norm(gradient):.6f}")

        return ts

    def _get_initial_guess(self) -> np.ndarray:
        """Get initial guess for TS search."""
        positions = []
        for atom in self.atoms:
            positions.append(atom.position.copy())

        positions = np.array(positions)

        # For dissociation: stretch bond
        if self.reaction_type == ReactionType.BOND_BREAKING and self.n_atoms == 2:
            # Stretch to 2x equilibrium
            center = (positions[0] + positions[1]) / 2
            direction = positions[1] - positions[0]
            direction /= np.linalg.norm(direction)

            positions[0] = center - direction * 2.0  # Stretched
            positions[1] = center + direction * 2.0

        return positions

    def _dimer_search(
        self,
        positions: np.ndarray,
        max_iterations: int,
        convergence: float
    ) -> Tuple[np.ndarray, float]:
        """
        Dimer method for transition state search.

        The dimer method uses two images separated by a small distance
        to estimate the lowest curvature direction and climb uphill
        along it while minimizing in perpendicular directions.
        """
        # Dimer parameters
        dimer_length = 0.01  # Bohr
        step_size = 0.1

        # Initialize dimer direction (along bond)
        if self.n_atoms == 2:
            bond_vec = positions[1] - positions[0]
            bond_vec /= np.linalg.norm(bond_vec)
            direction = np.zeros_like(positions)
            direction[0] = -bond_vec
            direction[1] = bond_vec
            direction /= np.linalg.norm(direction)
        else:
            direction = np.random.randn(*positions.shape)
            direction /= np.linalg.norm(direction)

        for iteration in range(max_iterations):
            # Current energy and gradient
            E0 = self.compute_energy(positions)
            g0 = self.compute_gradient(positions)

            # Check convergence
            if np.linalg.norm(g0) < convergence:
                break

            # Create dimer images
            pos1 = positions + dimer_length * direction
            pos2 = positions - dimer_length * direction

            E1 = self.compute_energy(pos1)
            E2 = self.compute_energy(pos2)

            # Estimate curvature along dimer
            curvature = (E1 + E2 - 2 * E0) / dimer_length**2

            # Rotate dimer to minimize curvature
            g1 = self.compute_gradient(pos1)
            g2 = self.compute_gradient(pos2)
            torque = g1 - g2
            torque_perp = torque - np.sum(torque * direction) * direction

            # Update direction
            if np.linalg.norm(torque_perp) > 1e-10:
                direction -= 0.1 * torque_perp / np.linalg.norm(torque_perp)
                direction /= np.linalg.norm(direction)

            # Move: uphill along dimer, downhill perpendicular
            g_parallel = np.sum(g0 * direction) * direction
            g_perp = g0 - g_parallel

            # Invert parallel component (climb uphill)
            step = -step_size * (g_perp - g_parallel)
            positions = positions + step

        energy = self.compute_energy(positions)
        return positions, energy

    def _neb_search(
        self,
        positions: np.ndarray,
        max_iterations: int,
        convergence: float,
        n_images: int = 7,
        spring_constant: float = 0.1,
        climbing_image: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        Nudged Elastic Band (NEB) method for minimum energy path.

        Creates a chain of images between reactants and products
        and optimizes to find the MEP using the NEB algorithm.

        Algorithm:
        1. Generate initial path by linear interpolation
        2. Optimize images with spring forces (parallel) + true forces (perpendicular)
        3. Apply climbing image for TS refinement (optional)
        4. Return highest energy image as TS

        Args:
            positions: Initial guess (used if no products defined)
            max_iterations: Maximum optimization iterations
            convergence: Force convergence threshold (Ha/Bohr)
            n_images: Number of images along path (including endpoints)
            spring_constant: Spring constant for image spacing (Ha/Bohr²)
            climbing_image: Use CI-NEB for better TS refinement

        Returns:
            (ts_positions, ts_energy)
        """
        logger.info(f"NEB search with {n_images} images, CI={climbing_image}")

        # Get endpoint positions
        reactant_pos = np.array([a.position for a in self.atoms])

        # Product positions: use given products or estimate from stretched geometry
        if self.products is not None:
            try:
                product_atoms = self.products[0].atoms if hasattr(self.products[0], 'atoms') else [self.products[0].atom_1, self.products[0].atom_2]
                product_pos = np.array([a.position for a in product_atoms])
            except Exception:
                # Fallback: stretch the bond
                product_pos = self._get_stretched_geometry(reactant_pos, factor=3.0)
        else:
            # For dissociation, use stretched geometry
            product_pos = self._get_stretched_geometry(reactant_pos, factor=3.0)

        # Generate initial path by linear interpolation
        images = self._interpolate_images(reactant_pos, product_pos, n_images)
        energies = np.array([self.compute_energy(img) for img in images])

        logger.debug(f"Initial NEB energies: {energies}")

        # Optimization loop
        step_size = 0.01  # Ha/Bohr step
        best_ts_energy = np.max(energies)
        best_ts_pos = images[np.argmax(energies)].copy()

        for iteration in range(max_iterations):
            # Compute forces on each image
            forces = np.array([self.compute_gradient(img) for img in images])

            # Compute NEB forces (perpendicular true + parallel spring)
            neb_forces = self._compute_neb_forces(
                images, forces, energies, spring_constant
            )

            # Apply climbing image modification for highest energy image
            if climbing_image and iteration > max_iterations // 4:
                i_max = np.argmax(energies[1:-1]) + 1  # Exclude endpoints
                neb_forces[i_max] = self._climbing_image_force(
                    images, forces, i_max
                )

            # Update images (keep endpoints fixed)
            max_force = 0.0
            for i in range(1, n_images - 1):
                images[i] = images[i] - step_size * neb_forces[i]
                max_force = max(max_force, np.linalg.norm(neb_forces[i]))

            # Recompute energies
            energies = np.array([self.compute_energy(img) for img in images])

            # Track best TS
            ts_idx = np.argmax(energies[1:-1]) + 1
            if energies[ts_idx] > best_ts_energy - 0.001:  # Allow small fluctuation
                best_ts_energy = energies[ts_idx]
                best_ts_pos = images[ts_idx].copy()

            # Check convergence
            if max_force < convergence:
                logger.info(f"NEB converged at iteration {iteration}, max force = {max_force:.6f}")
                break

            if iteration % 10 == 0:
                logger.debug(f"NEB iter {iteration}: max_force={max_force:.4f}, TS_E={energies[ts_idx]:.4f}")

        logger.info(f"NEB TS energy: {best_ts_energy:.6f} Ha")

        return best_ts_pos, best_ts_energy

    def _get_stretched_geometry(
        self,
        positions: np.ndarray,
        factor: float = 3.0
    ) -> np.ndarray:
        """Get geometry with bond stretched by given factor."""
        if self.n_atoms == 2:
            center = (positions[0] + positions[1]) / 2
            direction = positions[1] - positions[0]
            r_eq = np.linalg.norm(direction)
            direction = direction / r_eq

            stretched = np.zeros_like(positions)
            stretched[0] = center - (factor * r_eq / 2) * direction
            stretched[1] = center + (factor * r_eq / 2) * direction
            return stretched
        else:
            # For polyatomics, just scale from center
            center = np.mean(positions, axis=0)
            return center + factor * (positions - center)

    def _interpolate_images(
        self,
        start: np.ndarray,
        end: np.ndarray,
        n_images: int
    ) -> List[np.ndarray]:
        """Linear interpolation between start and end geometries."""
        images = []
        for i in range(n_images):
            t = i / (n_images - 1)
            image = (1 - t) * start + t * end
            images.append(image.copy())
        return images

    def _compute_neb_forces(
        self,
        images: List[np.ndarray],
        forces: np.ndarray,
        energies: np.ndarray,
        spring_constant: float
    ) -> np.ndarray:
        """
        Compute NEB forces: perpendicular true force + parallel spring force.

        F_NEB = F_perp + F_spring_parallel
        """
        n_images = len(images)
        neb_forces = np.zeros_like(forces)

        for i in range(1, n_images - 1):  # Skip endpoints
            # Tangent vector (normalized)
            tau = self._compute_tangent(images, energies, i)

            # True force perpendicular to path
            F_true = forces[i]
            F_parallel = np.sum(F_true * tau) * tau
            F_perp = F_true - F_parallel

            # Spring force parallel to path.
            # neb_forces are used in gradient-descent updates (x -= step*neb_force),
            # so all terms must be gradient-encoded (point uphill). F_true here is
            # already a gradient (compute_gradient points uphill); the spring must
            # therefore carry a leading minus so it enters with the same convention.
            r_plus = images[i + 1] - images[i]
            r_minus = images[i] - images[i - 1]
            F_spring = -spring_constant * (np.linalg.norm(r_plus) - np.linalg.norm(r_minus))
            F_spring_parallel = F_spring * tau

            # NEB force
            neb_forces[i] = F_perp + F_spring_parallel

        return neb_forces

    def _compute_tangent(
        self,
        images: List[np.ndarray],
        energies: np.ndarray,
        i: int
    ) -> np.ndarray:
        """
        Compute tangent vector at image i using improved tangent estimator.

        Uses energy-weighted bisection for smoother paths near TS.
        """
        tau_plus = (images[i + 1] - images[i]).flatten()
        tau_minus = (images[i] - images[i - 1]).flatten()

        E_plus = energies[i + 1]
        E_i = energies[i]
        E_minus = energies[i - 1]

        # Energy-weighted tangent (smoother near TS)
        if E_plus > E_i > E_minus:
            tau = tau_plus
        elif E_plus < E_i < E_minus:
            tau = tau_minus
        else:
            # At extremum, use weighted average
            dE_plus = abs(E_plus - E_i)
            dE_minus = abs(E_i - E_minus)
            if E_plus > E_minus:
                tau = dE_plus * tau_plus + dE_minus * tau_minus
            else:
                tau = dE_minus * tau_plus + dE_plus * tau_minus

        # Normalize
        tau_norm = np.linalg.norm(tau)
        if tau_norm > 1e-10:
            tau = tau / tau_norm

        return tau.reshape(images[i].shape)

    def _climbing_image_force(
        self,
        images: List[np.ndarray],
        forces: np.ndarray,
        i_climb: int
    ) -> np.ndarray:
        """
        Compute force for climbing image (CI-NEB).

        For the highest energy image, reverse the parallel component
        so it climbs uphill along the path towards the true saddle point.

        Returned gradient-encoded (matches neb_forces / gradient-descent update):
        F_CI = F_true - 2 * (F_true · tau) * tau
        """
        # Get tangent at climbing image
        tau = self._compute_tangent(
            images,
            np.array([self.compute_energy(img) for img in images]),
            i_climb
        ).flatten()

        # Climbing image force: invert the parallel component while keeping the
        # gradient-descent convention (F_true = gradient points uphill). This is
        # subtracted from the image position downstream, so it must NOT carry the
        # force-encoded leading minus that previously inverted the climb.
        F_true = forces[i_climb].flatten()
        F_parallel = np.dot(F_true, tau) * tau
        F_ci = F_true - 2 * F_parallel

        return F_ci.reshape(forces[i_climb].shape)

    def _simple_ts_search(
        self,
        positions: np.ndarray,
        max_iterations: int,
        convergence: float
    ) -> Tuple[np.ndarray, float]:
        """Simple gradient-based TS search."""
        step_size = 0.05

        for iteration in range(max_iterations):
            gradient = self.compute_gradient(positions)

            if np.linalg.norm(gradient) < convergence:
                break

            # Move along gradient (simplified)
            positions = positions - step_size * gradient

        energy = self.compute_energy(positions)
        return positions, energy

    def _estimate_imaginary_frequency(
        self,
        positions: np.ndarray
    ) -> float:
        """Estimate imaginary frequency at geometry."""
        # For diatomic: use force constant
        if self.n_atoms == 2:
            delta = 0.01
            E0 = self.compute_energy(positions)

            # Stretch
            direction = positions[1] - positions[0]
            direction /= np.linalg.norm(direction)

            pos_plus = positions.copy()
            pos_plus[0] -= delta * direction
            pos_plus[1] += delta * direction

            pos_minus = positions.copy()
            pos_minus[0] += delta * direction
            pos_minus[1] -= delta * direction

            E_plus = self.compute_energy(pos_plus)
            E_minus = self.compute_energy(pos_minus)

            # Curvature
            k = (E_plus + E_minus - 2 * E0) / delta**2

            if k < 0:
                # Imaginary frequency
                mu = self.masses[0] * self.masses[1] / (self.masses[0] + self.masses[1])
                mu_kg = mu * 1.66054e-27  # amu to kg
                k_SI = abs(k) * 4.3597e-18 / (5.29177e-11)**2  # Ha/Bohr^2 to J/m^2

                omega = np.sqrt(k_SI / mu_kg)  # rad/s
                freq_cm = omega / (2 * np.pi * 2.998e10)  # cm^-1

                return -freq_cm  # Negative for imaginary
            else:
                return 0.0

        return 0.0

    def compute_reaction_coordinate(
        self,
        ts: TransitionState,
        n_points: int = 21,
        step_size: float = 0.1
    ) -> ReactionPath:
        """
        Follow intrinsic reaction coordinate (IRC) from TS.

        Args:
            ts: Transition state
            n_points: Number of points along path
            step_size: Step size for IRC following

        Returns:
            ReactionPath object
        """
        logger.info("Computing IRC...")

        # Storage
        geometries = np.zeros((n_points, self.n_atoms, 3))
        energies = np.zeros(n_points)
        s_values = np.zeros(n_points)

        # Start at TS
        ts_index = n_points // 2
        geometries[ts_index] = ts.geometry
        energies[ts_index] = ts.energy
        s_values[ts_index] = 0.0

        # Get reaction coordinate direction
        if ts.reaction_coordinate is not None:
            direction = ts.reaction_coordinate.reshape((self.n_atoms, 3))
        else:
            # Use gradient direction
            direction = self.compute_gradient(ts.geometry)
            direction /= np.linalg.norm(direction)

        # Follow forward (towards products). Kick off the TS along the reaction
        # coordinate first: at the TS the gradient is ~0, so without this initial
        # displacement steepest descent never leaves the saddle point.
        pos = ts.geometry + step_size * direction
        s = 0.0
        for i in range(ts_index + 1, n_points):
            # Mass-weighted steepest descent (Fukui IRC)
            grad = self.compute_gradient(pos)
            step = mass_weighted_irc_step(grad, self.masses, step_size, descend=True)

            pos = pos + step
            s += step_size

            geometries[i] = pos
            energies[i] = self.compute_energy(pos)
            s_values[i] = s

        # Follow backward (towards reactants): opposite initial kick along the
        # reaction coordinate, then steepest DESCENT (downhill). The IRC goes
        # downhill on BOTH branches from the TS — descend=False made the reactant
        # branch climb uphill, giving non-physical (negative) barriers.
        pos = ts.geometry - step_size * direction
        s = 0.0
        for i in range(ts_index - 1, -1, -1):
            # Mass-weighted steepest descent (Fukui IRC)
            grad = self.compute_gradient(pos)
            step = mass_weighted_irc_step(grad, self.masses, step_size, descend=True)

            pos = pos + step
            s -= step_size

            geometries[i] = pos
            energies[i] = self.compute_energy(pos)
            s_values[i] = s

        # Compute path properties
        barrier = ts.energy - energies[0]
        reaction_energy = energies[-1] - energies[0]

        path_length = 0.0
        for i in range(1, n_points):
            path_length += np.linalg.norm(geometries[i] - geometries[i-1])

        path = ReactionPath(
            geometries=geometries,
            energies=energies,
            reaction_coordinate=s_values,
            ts_index=ts_index,
            barrier_height=barrier,
            reaction_energy=reaction_energy,
            path_length=path_length
        )

        logger.info(f"IRC complete: barrier={barrier*HARTREE_TO_KCAL:.2f} kcal/mol")

        return path

    def compute_rate_constant(
        self,
        temperature: float,
        ts: Optional[TransitionState] = None,
        include_environment: bool = True
    ) -> float:
        """
        Compute reaction rate constant using the Eyring-Polanyi equation.

        k = (k_B·T/h) · κ · exp(−ΔE‡ / k_B·T)

        IMPORTANT — what ΔE‡ actually is here: the **electronic (potential-energy)
        barrier** ``ts.energy − E_reactant``, NOT the Gibbs free energy of
        activation ΔG‡. The activation entropy and ZPE are NOT included; a proper
        ΔG‡ would need TS-vs-reactant thermochemistry (frequencies at both states),
        which is not wired in. So this is a *potential-barrier* Eyring estimate —
        treat it as such, not as a free-energy rate. Its accuracy is further gated
        by the TS energy from ``find_transition_state()``, which is not yet
        value-validated against a reference barrier (see the trust map / NEB H+H₂
        9.7 kcal/mol target). κ is the transmission coefficient (Kramers friction +
        Wigner/Eckart tunneling), applied only when an environment is attached.

        Args:
            temperature: Temperature in Kelvin
            ts: Transition state (finds if not provided)
            include_environment: Apply environment corrections

        Returns:
            Rate constant in 1/s
        """
        if ts is None:
            ts = self.find_transition_state()

        # Get reactant energy
        reactant_positions = np.array([a.position for a in self.atoms])
        E_reactant = self.compute_energy(reactant_positions)

        # Activation energy
        E_act = ts.energy - E_reactant

        # Apply environment corrections to activation energy
        if include_environment and self.env_integration:
            env_corrections = self._compute_environment_corrections(
                E_act, temperature, ts
            )
            E_act += env_corrections.get('activation_correction', 0.0)

        # Eyring-Polanyi rate constant
        # k = (kT/h) * exp(-E_act/(kT))

        kT = K_BOLTZMANN * temperature  # in Hartree
        h = PLANCK_CONSTANT  # 1 in a.u.

        # Convert to SI units for rate constant
        # k_B = 1.380649e-23 J/K
        # h = 6.62607e-34 J·s
        # 1 Hartree = 4.3597e-18 J

        E_act_J = E_act * 4.3597e-18  # Hartree to J
        kT_J = 1.380649e-23 * temperature  # J
        h_J = 6.62607e-34  # J·s

        rate_constant = (kT_J / h_J) * np.exp(-E_act_J / kT_J)

        # Apply transmission coefficient (Kramers, tunneling)
        if include_environment and self.env_integration:
            transmission_coeff = env_corrections.get('transmission_coefficient', 1.0)
            rate_constant *= transmission_coeff

        logger.info(f"Rate constant at {temperature}K: {rate_constant:.2e} 1/s")
        if include_environment and self.env_integration:
            logger.info(f"  Environment corrections applied: "
                       f"κ={env_corrections.get('transmission_coefficient', 1.0):.3f}")

        return rate_constant

    def _compute_environment_corrections(
        self,
        E_act: float,
        temperature: float,
        ts: TransitionState
    ) -> Dict[str, float]:
        """
        Compute environment corrections to rate constant.

        Includes:
        1. Solvent dielectric stabilization of TS
        2. Kramers friction correction
        3. Pressure-volume work
        4. Tunneling corrections (Wigner)

        Args:
            E_act: Gas-phase activation energy in Hartree
            temperature: Temperature in K
            ts: Transition state

        Returns:
            Dictionary with corrections
        """
        corrections = {
            'activation_correction': 0.0,
            'transmission_coefficient': 1.0,
            'kramers_factor': 1.0,
            'tunneling_factor': 1.0,
            'solvent_stabilization': 0.0
        }

        if not self.env_integration:
            return corrections

        try:
            # Get reaction rate parameters from environment
            rate_params = self.env_integration.compute_reaction_rate_correction(
                barrier=E_act,
                temperature=temperature
            )

            # Solvent stabilization of TS
            # Polar solvents can stabilize polar TS
            if hasattr(self.env_integration, 'conditions'):
                solvent = self.env_integration.conditions.solvent
                if solvent != 'vacuum':
                    # Use environment's rate enhancement estimate
                    enhancement = rate_params.get('rate_enhancement', 1.0)
                    # Convert enhancement to effective barrier change
                    # k_solv/k_gas = exp(-ΔΔG‡/RT)
                    # ΔΔG‡ = -RT * ln(enhancement)
                    kT = K_BOLTZMANN * temperature
                    if enhancement > 0:
                        corrections['activation_correction'] = -kT * np.log(max(enhancement, 1e-10))
                        corrections['solvent_stabilization'] = corrections['activation_correction']

            # Kramers friction correction
            # κ_Kramers = ω‡/2πγ * [√(1 + (2πγ/ω‡)²) - 1]
            # For high friction: κ ≈ ω‡/γ
            friction = rate_params.get('friction_coefficient', 0.0)
            if friction > 0 and ts.imaginary_frequency:
                # Convert imaginary frequency to angular frequency
                # ω‡ = 2π * ν (in 1/s)
                omega_ts = abs(ts.imaginary_frequency) * 2.998e10 * 2 * np.pi  # cm⁻¹ to rad/s
                gamma = friction * 1e12  # ps⁻¹ to s⁻¹

                if gamma > 0:
                    ratio = 2 * np.pi * gamma / omega_ts
                    kramers = (omega_ts / (2 * np.pi * gamma)) * (np.sqrt(1 + ratio**2) - 1)
                    corrections['kramers_factor'] = min(kramers, 1.0)

            # Tunneling correction (Wigner)
            # κ_tunnel = 1 + (1/24) * (hν‡/kT)²
            if ts.imaginary_frequency:
                h_nu = abs(ts.imaginary_frequency) * 4.556e-6  # cm⁻¹ to Hartree
                kT = K_BOLTZMANN * temperature
                corrections['tunneling_factor'] = 1.0 + (1.0/24.0) * (h_nu / kT)**2

            # Total transmission coefficient
            corrections['transmission_coefficient'] = (
                corrections['kramers_factor'] *
                corrections['tunneling_factor']
            )

        except Exception as e:
            logger.debug(f"Environment correction failed: {e}")

        return corrections

    def compute_half_life(
        self,
        temperature: float,
        ts: Optional[TransitionState] = None
    ) -> float:
        """
        Compute reaction half-life.

        t_1/2 = ln(2) / k

        Args:
            temperature: Temperature in Kelvin
            ts: Transition state

        Returns:
            Half-life in seconds
        """
        k = self.compute_rate_constant(temperature, ts)
        if k > 0:
            return np.log(2) / k
        else:
            return float('inf')

    def get_reaction_result(
        self,
        temperature: float = 298.15
    ) -> ReactionResult:
        """
        Get complete reaction analysis.

        Args:
            temperature: Temperature for rate calculation

        Returns:
            ReactionResult with all reaction data
        """
        # Find TS
        ts = self.find_transition_state()

        # Compute IRC
        path = self.compute_reaction_coordinate(ts)

        # Compute rate
        rate = self.compute_rate_constant(temperature, ts)
        half_life = self.compute_half_life(temperature, ts)

        # Activation energy in kcal/mol
        E_act_kcal = path.barrier_height * HARTREE_TO_KCAL

        # Reaction enthalpy in kcal/mol
        delta_H_kcal = path.reaction_energy * HARTREE_TO_KCAL

        result = ReactionResult(
            reaction_type=self.reaction_type,
            reactants=self.reactants,
            products=self.products,
            transition_state=ts,
            forward_path=path,
            reverse_path=None,
            rate_constant=rate,
            half_life=half_life,
            activation_energy=E_act_kcal,
            reaction_enthalpy=delta_H_kcal,
            governance_protocol=self.governance
        )

        return result

    def can_react(
        self,
        energy_available: float
    ) -> bool:
        """
        Check if reaction can occur given available energy.

        Compares the available energy against the activation barrier
        (TS energy minus reactant energy).

        Args:
            energy_available: Available energy in Hartree

        Returns:
            True if reaction can proceed
        """
        # Compute barrier
        ts = self.find_transition_state()
        barrier = ts.energy - self.compute_energy(
            np.array([a.position for a in self.atoms])
        )

        # Energy check
        # NOTE: the former governance branch here was dead code — self.governance
        # is a string from _select_governance(), so hasattr(protocol, ...) was
        # always False and the branch never executed. Removed for honesty.
        if energy_available < barrier:
            return False

        return True


# Factory function
def create_reaction_simulator(
    reactants: List,
    products: Optional[List] = None,
    **kwargs
) -> ReactionSimulator:
    """
    Factory function to create ReactionSimulator.

    Args:
        reactants: List of reactant structures
        products: Optional list of product structures
        **kwargs: Additional arguments

    Returns:
        ReactionSimulator instance
    """
    return ReactionSimulator(
        reactants=reactants,
        products=products,
        **kwargs
    )
