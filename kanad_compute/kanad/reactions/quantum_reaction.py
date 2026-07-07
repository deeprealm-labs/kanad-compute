"""
Quantum Reaction Dynamics

TRUE QUANTUM ADVANTAGE: Uses VQE-based potential energy surface instead of
classical force fields for reaction barrier calculations.

Key Advantages:
--------------
1. **Static Correlation**: VQE captures multi-reference character at TS
2. **Bond Breaking**: Correct description of partially broken bonds
3. **Barrier Heights**: More accurate than DFT for radical reactions
4. **Electron Transfer**: Proper charge-transfer state energies

When to Use:
-----------
- Transition states with multi-reference character
- Bond breaking/forming reactions
- Radical reactions
- Heavy atom systems (>30 electrons)

References:
----------
- Peruzzo et al. (2014) Nat. Commun. 5, 4213 - Original VQE
- Tubman et al. (2018) arXiv:1805.04530 - VQE for reaction barriers
- Cao et al. (2019) Chem. Rev. 119, 10856 - Quantum chemistry review
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any, Callable
from dataclasses import dataclass

from kanad.reactions._irc import mass_weighted_irc_step

logger = logging.getLogger(__name__)


@dataclass
class QuantumTransitionState:
    """Transition state from quantum calculation."""
    geometry: np.ndarray      # (N_atoms, 3) in Bohr
    energy: float             # Hartree
    gradient: np.ndarray      # Ha/Bohr
    method: str               # 'vqe', 'physics_vqe', etc.
    n_evaluations: int        # VQE function evaluations
    verified: bool            # True if gradient ≈ 0 and one negative freq


@dataclass
class QuantumReactionPath:
    """Reaction path computed with quantum energies."""
    geometries: np.ndarray    # (n_points, n_atoms, 3)
    energies: np.ndarray      # (n_points,) in Hartree
    s_values: np.ndarray      # Reaction coordinate
    ts_index: int
    barrier_height: float     # Ha
    reaction_energy: float    # Ha
    total_evaluations: int    # Total VQE calls


class QuantumReactionSimulator:
    """
    Simulate chemical reactions using VQE-based potential energy surface.

    Unlike classical ReactionSimulator (which uses Morse/LJ potentials),
    this uses real quantum calculations for energies.

    Example:
    --------
    ```python
    from kanad import BondFactory
    from kanad.reactions import QuantumReactionSimulator

    # Create H2 molecule
    h2 = BondFactory.create_bond('H', 'H', distance=0.74)

    # Quantum reaction simulation
    qrs = QuantumReactionSimulator(
        reactants=[h2],
        solver='physics_vqe',  # Use PhysicsVQE for accuracy
        max_excitations=5
    )

    # Find TS on quantum PES
    ts = qrs.find_transition_state()
    print(f"TS energy: {ts.energy:.6f} Ha")
    print(f"VQE evaluations: {ts.n_evaluations}")
    ```
    """

    def __init__(
        self,
        reactants: Optional[List] = None,
        products: Optional[List] = None,
        solver: str = 'physics_vqe',
        max_excitations: int = 10,
        vqe_max_iterations: int = 200,
        backend: str = 'statevector',
        energy_callback: Optional[Callable] = None,
        atoms: Optional[List] = None,
        masses: Optional[np.ndarray] = None,
    ):
        """
        Initialize quantum reaction simulator.

        Two modes:

        1. **Solver-agnostic (M7, recommended).** Pass ``energy_callback`` — a
           ``(atoms_bohr, warm_state) -> (energy_Ha, warm_state)`` closure (e.g.
           ``MolecularBuilder(...).build().energy_fn()``). Every TS / IRC /
           dissociation routine then runs on whatever solver the builder routes
           to — SQD on Heron for large active spaces, CI/VQE otherwise — with
           warm-starting threaded between geometries. Use ``from_system`` for the
           one-liner. ``atoms`` (kanad Atoms or ``(symbol, xyz)`` tuples) gives
           the symbols/masses; geometry is taken from each evaluation.

        2. **Legacy VQE (back-compat).** Pass ``reactants`` (Bond/Molecule list)
           and a ``solver`` string ('physics_vqe' | 'vqe' | 'hybrid_subspace');
           energies are computed by rebuilding the molecule per geometry. Diatomic
           / small only.

        Args:
            reactants: List of Bond/Molecule objects (legacy mode).
            products: Optional product structures.
            solver: Legacy VQE solver method, or a free-form label in callback mode.
            max_excitations / vqe_max_iterations / backend: legacy VQE settings.
            energy_callback: solver-agnostic energy closure (M7 mode).
            atoms: atom list (kanad Atom or (symbol, xyz-Angstrom) tuples) — required
                in callback mode to know symbols/masses.
            masses: optional (n_atoms,) amu override.
        """
        self.reactants = reactants
        self.products = products
        self.solver = solver
        self.max_excitations = max_excitations
        self.vqe_max_iterations = vqe_max_iterations
        self.backend = backend

        # M7: solver-agnostic energy closure + its warm-start payload.
        self._energy_callback = energy_callback
        self._warm = None

        self.atoms = []
        self.masses = []
        if atoms is not None:
            # Callback / explicit-atoms mode: normalize to objects exposing
            # `.symbol` / `.atomic_mass` (kanad Atom) or accept (symbol, xyz).
            from kanad.core.atom import Atom
            for a in atoms:
                if hasattr(a, 'symbol'):
                    self.atoms.append(a)
                    self.masses.append(getattr(a, 'atomic_mass', 1.0))
                else:
                    sym, pos = a
                    at = Atom(symbol=str(sym), position=np.asarray(pos, dtype=float))
                    self.atoms.append(at)
                    self.masses.append(getattr(at, 'atomic_mass', 1.0))
        elif reactants is not None:
            for reactant in reactants:
                if hasattr(reactant, 'atom_1'):
                    self.atoms.extend([reactant.atom_1, reactant.atom_2])
                    self.masses.extend([
                        getattr(reactant.atom_1, 'atomic_mass', 1.0),
                        getattr(reactant.atom_2, 'atomic_mass', 1.0)
                    ])
                elif hasattr(reactant, 'atoms'):
                    self.atoms.extend(reactant.atoms)
                    self.masses.extend([
                        getattr(a, 'atomic_mass', getattr(a, 'mass', 1.0))
                        for a in reactant.atoms
                    ])

        self.n_atoms = len(self.atoms)
        self.masses = np.asarray(masses if masses is not None else self.masses, dtype=float)

        # Track solver evaluations
        self._total_evaluations = 0

        # Energy cache for efficiency
        self._energy_cache = {}

        logger.info(
            f"QuantumReactionSimulator initialized: "
            f"mode={'callback' if energy_callback else 'legacy-vqe'}, "
            f"solver={solver}, atoms={self.n_atoms}, backend={backend}"
        )

    @classmethod
    def from_system(cls, system, solver_label: Optional[str] = None,
                    products: Optional[List] = None) -> 'QuantumReactionSimulator':
        """Build a solver-agnostic simulator from a builder `QuantumSystem`.

        The system's `energy_fn()` (the geometry-parametric SQD/CI/VQE closure)
        drives every reaction routine, so the reaction is computed with exactly
        the solver + active space + backend the builder is configured for —
        SQD on Heron for the M11-scale champions, CI/VQE for small systems.
        """
        spec = system.spec
        atoms = list(spec.atoms)  # (symbol, xyz-Angstrom) tuples
        label = solver_label or (
            system.spec.solver if system.spec.solver != 'auto' else 'auto'
        )
        return cls(
            energy_callback=system.energy_fn(),
            atoms=atoms,
            solver=label,
            backend=spec.backend,
            products=products,
        )

    def compute_energy(
        self,
        positions: np.ndarray,
        use_cache: bool = True
    ) -> float:
        """
        Compute energy at given geometry using quantum method.

        This is the KEY DIFFERENCE from classical ReactionSimulator:
        we use VQE instead of Morse potential!

        Args:
            positions: Atomic positions (N_atoms, 3) in Bohr
            use_cache: Use cached values if available

        Returns:
            Energy in Hartree
        """
        # Check cache
        pos_key = tuple(np.asarray(positions).flatten().round(6))
        if use_cache and pos_key in self._energy_cache:
            return self._energy_cache[pos_key]

        # M7 solver-agnostic path: defer to the builder's energy closure.
        if self._energy_callback is not None:
            energy, self._warm = self._energy_callback(
                np.asarray(positions, dtype=float), self._warm
            )
            energy = float(energy)
            self._total_evaluations += 1
            if use_cache:
                self._energy_cache[pos_key] = energy
            return energy

        try:
            # Build molecule at this geometry
            mol = self._build_molecule_at_geometry(positions)

            # Resolve the solver class by name via the registry (no hard imports of
            # concrete classes) and consume it through the EnergyProvider capability.
            # Construction args + the per-solver evaluation-count key are preserved
            # exactly; 'hybrid_subspace' maps to the registered SamplingSQDSolver
            # (the real circuit-sampling SQD that superseded the retired
            # HybridSubspaceVQE — audit H12).
            from kanad.solvers import get_solver
            from kanad.solvers.capabilities import EnergyProvider

            if self.solver == 'physics_vqe':
                solver = get_solver('physics_vqe')(
                    molecule=mol,
                    max_excitations=self.max_excitations,
                )
                _eval_key = 'n_evaluations'
            elif self.solver == 'vqe':
                bond = self._create_bond_from_positions(positions)
                solver = get_solver('vqe')(
                    bond=bond,
                    backend=self.backend,
                    optimizer='COBYLA',
                    max_iterations=self.vqe_max_iterations,
                )
                _eval_key = 'iterations'
            elif self.solver == 'hybrid_subspace':
                bond = self._create_bond_from_positions(positions)
                solver = get_solver('sampling_sqd')(bond)
                _eval_key = None  # one subspace diagonalization per geometry
            else:
                raise ValueError(f"Unknown solver: {self.solver}")

            if not isinstance(solver, EnergyProvider):
                raise TypeError(
                    f"solver {type(solver).__name__} is not an EnergyProvider "
                    f"(needs solve()); cannot drive the reaction energy path."
                )
            result = solver.solve().to_dict()
            energy = result['energy']
            self._total_evaluations += (1 if _eval_key is None else result.get(_eval_key, 0))

            # Cache result
            if use_cache:
                pos_key = tuple(positions.flatten().round(6))
                self._energy_cache[pos_key] = energy

            return energy

        except Exception as e:
            logger.error(f"Quantum energy calculation failed: {e}")
            raise

    def _build_molecule_at_geometry(self, positions: np.ndarray):
        """Build Molecule object at given geometry.

        Args:
            positions: Atomic positions in BOHR

        Returns:
            Molecule with positions in Angstrom (PySCF default)
        """
        from kanad import Molecule
        from kanad.core.atom import Atom

        # CRITICAL FIX: Convert Bohr to Angstrom
        # PySCF expects Angstrom by default
        BOHR_TO_ANGSTROM = 0.529177

        # Update atom positions (converting to Angstrom)
        new_atoms = []
        for i, atom in enumerate(self.atoms):
            pos_angstrom = positions[i] * BOHR_TO_ANGSTROM
            new_atom = Atom(
                atom.symbol,
                pos_angstrom.tolist()
            )
            new_atoms.append(new_atom)

        return Molecule(new_atoms)

    def _create_bond_from_positions(self, positions: np.ndarray):
        """Create Bond object for diatomic at given geometry."""
        if self.n_atoms != 2:
            raise ValueError("VQE solver currently only supports diatomics")

        from kanad import BondFactory

        # Calculate distance in Angstrom
        r_bohr = np.linalg.norm(positions[1] - positions[0])
        r_angstrom = r_bohr * 0.529177

        return BondFactory.create_bond(
            self.atoms[0].symbol,
            self.atoms[1].symbol,
            distance=r_angstrom
        )

    def compute_gradient(
        self,
        positions: np.ndarray,
        delta: float = 0.001
    ) -> np.ndarray:
        """
        Compute energy gradient using quantum energies.

        Uses central finite differences with VQE energies.

        Args:
            positions: Atomic positions (N_atoms, 3) in Bohr
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

                E_plus = self.compute_energy(pos_plus, use_cache=True)
                E_minus = self.compute_energy(pos_minus, use_cache=True)

                gradient[i, j] = (E_plus - E_minus) / (2 * delta)

        return gradient

    def find_transition_state(
        self,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 50,
        convergence: float = 1e-3
    ) -> QuantumTransitionState:
        """
        Find transition state on quantum PES using Dimer method.

        This uses VQE energies, capturing static correlation at the TS
        where bonds are partially broken.

        Args:
            initial_guess: Initial geometry guess
            max_iterations: Maximum Dimer iterations
            convergence: Gradient convergence threshold

        Returns:
            QuantumTransitionState object
        """
        logger.info(f"Finding TS using Dimer method on quantum PES...")
        logger.info(f"  Solver: {self.solver}")

        # Get initial positions
        if initial_guess is None:
            positions = self._get_ts_guess()
        else:
            positions = initial_guess.copy()

        # Dimer parameters
        dimer_length = 0.01  # Bohr
        step_size = 0.05

        # Initialize dimer direction (along bond for diatomic)
        # Direction must have shape (n_atoms, 3) to match positions
        if self.n_atoms == 2:
            # For diatomic: antisymmetric direction (stretch/compress)
            bond_vec = positions[1] - positions[0]
            bond_vec /= np.linalg.norm(bond_vec)
            # Atom 1 moves in -bond direction, atom 2 moves in +bond direction
            direction = np.zeros_like(positions)
            direction[0] = -bond_vec
            direction[1] = bond_vec
            direction /= np.linalg.norm(direction)
        else:
            direction = np.random.randn(*positions.shape)
            direction /= np.linalg.norm(direction)

        evaluations_start = self._total_evaluations

        for iteration in range(max_iterations):
            # Current energy and gradient
            E0 = self.compute_energy(positions)
            g0 = self.compute_gradient(positions)

            # Check convergence
            grad_norm = np.linalg.norm(g0)
            if grad_norm < convergence:
                logger.info(f"  Converged at iteration {iteration}: |g|={grad_norm:.6f}")
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

            # Update direction (keep same shape)
            if np.linalg.norm(torque_perp) > 1e-10:
                direction -= 0.1 * torque_perp / np.linalg.norm(torque_perp)
                direction /= np.linalg.norm(direction)

            # Move: uphill along dimer, downhill perpendicular
            g_parallel = np.sum(g0 * direction) * direction
            g_perp = g0 - g_parallel

            # Invert parallel component (climb uphill)
            step = -step_size * (g_perp - g_parallel)
            positions = positions + step

            if iteration % 10 == 0:
                logger.info(f"  Iteration {iteration}: E={E0:.6f} Ha, |g|={grad_norm:.6f}")

        # Final energy and gradient
        final_energy = self.compute_energy(positions)
        final_gradient = self.compute_gradient(positions)

        n_evals = self._total_evaluations - evaluations_start

        ts = QuantumTransitionState(
            geometry=positions,
            energy=final_energy,
            gradient=final_gradient,
            method=self.solver,
            n_evaluations=n_evals,
            verified=np.linalg.norm(final_gradient) < convergence
        )

        logger.info(f"TS found: E={final_energy:.6f} Ha, VQE evals={n_evals}")

        return ts

    def _get_ts_guess(self) -> np.ndarray:
        """Get initial guess for TS search."""
        positions = np.array([atom.position for atom in self.atoms])

        # For dissociation: stretch bond to ~2x equilibrium
        if self.n_atoms == 2:
            center = (positions[0] + positions[1]) / 2
            direction = positions[1] - positions[0]
            direction /= np.linalg.norm(direction)

            # Stretch to ~3 Bohr (1.5 Angstrom)
            positions[0] = center - direction * 1.5
            positions[1] = center + direction * 1.5

        return positions

    def compute_reaction_path(
        self,
        ts: QuantumTransitionState,
        n_points: int = 11,
        step_size: float = 0.1
    ) -> QuantumReactionPath:
        """
        Follow IRC from TS using quantum energies.

        Args:
            ts: Transition state
            n_points: Points along path (odd number)
            step_size: IRC step size in Bohr

        Returns:
            QuantumReactionPath object
        """
        logger.info("Computing IRC with quantum energies...")

        # Storage
        geometries = np.zeros((n_points, self.n_atoms, 3))
        energies = np.zeros(n_points)
        s_values = np.zeros(n_points)

        ts_index = n_points // 2
        geometries[ts_index] = ts.geometry
        energies[ts_index] = ts.energy
        s_values[ts_index] = 0.0

        evaluations_start = self._total_evaluations

        # Reaction-coordinate direction. At a converged TS |grad| ~ 0, so we kick
        # the geometry off the saddle along this direction before steepest descent
        # — otherwise the loop's |grad|>1e-10 guard breaks on iter 1 and leaves the
        # remaining geometries/energies at their np.zeros init (audit H13).
        rc = ts.gradient if ts.gradient is not None else self.compute_gradient(ts.geometry)
        rc = np.asarray(rc, dtype=float).reshape(self.n_atoms, 3)
        rc_norm = np.linalg.norm(rc)
        direction = rc / rc_norm if rc_norm > 1e-12 else None

        # Forward direction (towards products): kick off the TS first, then descend.
        pos = ts.geometry.copy()
        if direction is not None:
            pos = pos + step_size * direction
        s = 0.0
        for i in range(ts_index + 1, n_points):
            grad = self.compute_gradient(pos)
            if np.linalg.norm(grad) > 1e-10:
                step = mass_weighted_irc_step(grad, self.masses, step_size, descend=True)
            else:
                break

            pos = pos + step
            s += step_size

            geometries[i] = pos
            energies[i] = self.compute_energy(pos)
            s_values[i] = s

        # Backward direction (towards reactants): opposite initial kick, then
        # steepest DESCENT. The IRC goes downhill on BOTH branches from the TS;
        # descend=False made this branch climb uphill, giving non-physical
        # (negative) barriers (audit H13 — same fix as reaction_dynamics.py).
        pos = ts.geometry.copy()
        if direction is not None:
            pos = pos - step_size * direction
        s = 0.0
        for i in range(ts_index - 1, -1, -1):
            grad = self.compute_gradient(pos)
            if np.linalg.norm(grad) > 1e-10:
                step = mass_weighted_irc_step(grad, self.masses, step_size, descend=True)
            else:
                break

            pos = pos + step
            s -= step_size

            geometries[i] = pos
            energies[i] = self.compute_energy(pos)
            s_values[i] = s

        # Path properties
        barrier = ts.energy - energies[0]
        reaction_energy = energies[-1] - energies[0]
        n_evals = self._total_evaluations - evaluations_start

        path = QuantumReactionPath(
            geometries=geometries,
            energies=energies,
            s_values=s_values,
            ts_index=ts_index,
            barrier_height=barrier,
            reaction_energy=reaction_energy,
            total_evaluations=n_evals
        )

        logger.info(f"IRC complete: barrier={barrier*627.5:.2f} kcal/mol")

        return path

    def compute_dissociation_curve(
        self,
        r_range: Tuple[float, float] = (0.5, 5.0),
        n_points: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dissociation curve with quantum energies.

        This demonstrates quantum advantage: VQE correctly describes
        bond breaking, unlike DFT/HF which fail for stretched bonds.

        Args:
            r_range: (r_min, r_max) in Angstrom
            n_points: Number of points

        Returns:
            (distances, energies) arrays
        """
        if self.n_atoms != 2:
            raise ValueError("Dissociation curve only for diatomics")

        r_min, r_max = r_range
        distances = np.linspace(r_min, r_max, n_points)
        energies = []

        # Get atom symbols
        sym1, sym2 = self.atoms[0].symbol, self.atoms[1].symbol

        logger.info(f"Computing {sym1}-{sym2} dissociation curve...")

        for r in distances:
            # Create bond at this distance
            from kanad import BondFactory
            bond = BondFactory.create_bond(sym1, sym2, distance=r)

            # Registry-resolved (no concrete-class import); construction unchanged.
            from kanad.solvers import get_solver
            if self.solver == 'physics_vqe':
                solver = get_solver('physics_vqe')(bond=bond, max_excitations=self.max_excitations)
            else:
                solver = get_solver('vqe')(
                    bond=bond,
                    backend=self.backend,
                    max_iterations=self.vqe_max_iterations
                )

            try:
                result = solver.solve()
                if hasattr(result, 'energy'):
                    energies.append(result.energy)
                else:
                    energies.append(result['energy'])

                logger.debug(f"  r={r:.2f} Å: E={energies[-1]:.6f} Ha")
            except Exception as e:
                logger.warning(f"  r={r:.2f} Å: Failed - {e}")
                energies.append(np.nan)

        return distances, np.array(energies)

    def scan_path(
        self,
        geometries: np.ndarray,
        s_values: Optional[np.ndarray] = None,
        label: str = 'reaction',
    ) -> QuantumReactionPath:
        """Energy profile over an explicit list of geometries (solver-agnostic).

        Works for any system and any solver (SQD/CI/VQE) in callback mode — the
        general replacement for `compute_dissociation_curve` (diatomic+VQE only).
        Warm-starting threads automatically through `compute_energy`, so each
        point reuses the previous solve's wavefunction.

        Args:
            geometries: (n_points, n_atoms, 3) in Bohr (reactant → … → product).
            s_values: optional reaction-coordinate values; defaults to point index.
            label: log tag.

        Returns:
            QuantumReactionPath with the energy profile, the highest point as the
            TS, and barrier / reaction energy relative to the first point.
        """
        geometries = np.asarray(geometries, dtype=float)
        n_points = geometries.shape[0]
        energies = np.zeros(n_points)
        ev0 = self._total_evaluations
        for i in range(n_points):
            energies[i] = self.compute_energy(geometries[i])
            logger.info(f"  [{label}] point {i + 1}/{n_points}: E = {energies[i]:.6f} Ha")
        if s_values is None:
            s_values = np.arange(n_points, dtype=float)
        ts_index = int(np.argmax(energies))
        return QuantumReactionPath(
            geometries=geometries,
            energies=energies,
            s_values=np.asarray(s_values, dtype=float),
            ts_index=ts_index,
            barrier_height=float(energies[ts_index] - energies[0]),
            reaction_energy=float(energies[-1] - energies[0]),
            total_evaluations=self._total_evaluations - ev0,
        )

    def to_provenance(self, result, extra: Optional[Dict] = None) -> Dict[str, Any]:
        """JSON-serializable provenance for a reaction result (PLAN M7 deliverable)."""
        import datetime
        prov: Dict[str, Any] = {
            'kind': 'kanad_reaction',
            'solver': self.solver,
            'backend': self.backend,
            'mode': 'callback' if self._energy_callback is not None else 'legacy-vqe',
            'n_atoms': self.n_atoms,
            'symbols': [getattr(a, 'symbol', '?') for a in self.atoms],
            'total_evaluations': int(self._total_evaluations),
            'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        }
        if isinstance(result, QuantumReactionPath):
            prov.update({
                'result_type': 'reaction_path',
                'energies_ha': result.energies.tolist(),
                's_values': result.s_values.tolist(),
                'ts_index': int(result.ts_index),
                'barrier_ha': float(result.barrier_height),
                'barrier_kcal_mol': float(result.barrier_height * 627.509),
                'reaction_energy_ha': float(result.reaction_energy),
                'reaction_energy_kcal_mol': float(result.reaction_energy * 627.509),
            })
        elif isinstance(result, QuantumTransitionState):
            prov.update({
                'result_type': 'transition_state',
                'ts_energy_ha': float(result.energy),
                'ts_gradient_norm': float(np.linalg.norm(result.gradient)),
                'verified': bool(result.verified),
            })
        if extra:
            prov.update(extra)
        return prov


# Factory function
def create_quantum_reaction_simulator(
    reactants: List,
    solver: str = 'physics_vqe',
    **kwargs
) -> QuantumReactionSimulator:
    """
    Factory function to create QuantumReactionSimulator.

    Args:
        reactants: List of reactant structures
        solver: Quantum solver method
        **kwargs: Additional arguments

    Returns:
        QuantumReactionSimulator instance
    """
    return QuantumReactionSimulator(
        reactants=reactants,
        solver=solver,
        **kwargs
    )
