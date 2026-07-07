"""
Variational Quantum Eigensolver (VQE) - Rebuilt with Bonds Module Integration.

New Design:
- Takes bond as input (not raw Hamiltonian)
- Automatic analysis integration
- Automatic circuit optimization
- Rich, comprehensive results
- User-friendly interface

================================================================================
THEORETICAL FOUNDATION
================================================================================

The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm
for finding ground state energies of molecular systems.

VARIATIONAL PRINCIPLE:
----------------------
For any trial wavefunction |ψ(θ)⟩:
    E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ ≥ E_0

where E_0 is the exact ground state energy. This guarantees that minimizing E(θ)
approaches E_0 from above.

ALGORITHM:
----------
1. PREPARE: Apply parametrized unitary U(θ) to reference state |ψ_ref⟩
   |ψ(θ)⟩ = U(θ)|ψ_ref⟩

   Reference state is typically Hartree-Fock: |HF⟩ = |1100...⟩

2. MEASURE: Estimate expectation value
   E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ = Σ_k c_k ⟨ψ(θ)|P_k|ψ(θ)⟩

   where H = Σ_k c_k P_k is the Pauli decomposition of the Hamiltonian.

3. OPTIMIZE: Use classical optimizer to minimize E(θ)
   θ* = argmin_θ E(θ)

4. RETURN: Ground state energy E(θ*) and optimal parameters θ*

MEASUREMENT STRATEGIES:
-----------------------
- Statevector: Exact, no sampling noise (for simulators)
- Shot-based: Finite sampling, statistical errors (for real hardware)
- Grouping: Measure commuting Pauli terms together

OPTIMIZERS:
-----------
- SLSQP: Sequential Least Squares (gradient-based, fast convergence)
- COBYLA: Constrained Optimization BY Linear Approximations (gradient-free)
- Powell: Derivative-free conjugate direction method
- L-BFGS-B: Limited-memory BFGS for bounded problems

CONVERGENCE:
------------
VQE converges when:
- Energy change < threshold (e.g., 10^-6 Ha)
- Gradient norm < threshold
- Maximum iterations reached

ACCURACY FACTORS:
-----------------
1. Ansatz expressivity: Can |ψ(θ)⟩ represent the ground state?
2. Optimization landscape: Are there local minima?
3. Measurement noise: Statistical fluctuations (shot-based)
4. Basis set: Limited by electronic structure approximation

HI-VQE MODE:
------------
Hi-VQE (Hamiltonian-Informed VQE) uses active space reduction:
- Identify important configurations from classical calculation
- Focus quantum resources on correlation-critical orbitals
- 1000x reduction in measurements possible

================================================================================
DEVELOPMENT NOTES FOR DEVELOPERS
================================================================================

API DESIGN:
- High-level: VQESolver(bond) - automatic everything
- Low-level: VQESolver(hamiltonian=ham, ansatz=ans) - full control

BACKEND SUPPORT:
- statevector: Fast, exact (default)
- qasm: Shot-based simulation
- aer_simulator: Noise models
- bluequbit: GPU cloud acceleration
- ibm: Real quantum hardware

ADDING NEW OPTIMIZERS:
1. Add case in _setup_optimizer() method
2. Handle any optimizer-specific options
3. Test convergence on H2 benchmark

PERFORMANCE PROFILING:
- Most time in expectation value estimation
- Circuit transpilation cached after first call
- Parameter binding is O(n_parameters)

KNOWN ISSUES:
- UCC ansatz deprecated, use governance ansatze
- Ionic governance may have dimension mismatches
- Large systems (>20 qubits) need active space reduction
================================================================================
"""

from typing import Dict, Any, Optional, Callable
import numpy as np
import logging
from scipy.optimize import minimize

from kanad.solvers.base_solver import BaseSolver
from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper
from kanad.core.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper
from kanad.solvers._hivqe_mixin import HiVQESolverMixin

logger = logging.getLogger(__name__)


class VQESolver(HiVQESolverMixin, BaseSolver):
    """
    Variational Quantum Eigensolver for ground state energy.

    VQE is a hybrid quantum-classical algorithm:
    1. Prepare parametrized quantum state |ψ(θ)⟩
    2. Measure energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩
    3. Classically optimize parameters θ
    4. Return minimum energy found

    Usage:
        from kanad.bonds import BondFactory
        from kanad.solvers import VQESolver

        bond = BondFactory.create_bond('H', 'H', distance=0.74)
        solver = VQESolver(bond)
        result = solver.solve()  # -> SolverResult

        print(f"Energy: {result.energy:.6f} Hartree")
        solver.print_summary()
    """

    def __init__(
        self,
        system=None,
        *,
        bond: Optional['BaseBond'] = None,
        # High-level API (bond-based)
        # Default ansatz is hardware_efficient; 'ucc' was removed in the 2026-05-12 cleanup
        # because the implementation produced incorrect energies on every test molecule.
        ansatz_type: str = 'hardware_efficient',
        mapper_type: str = 'jordan_wigner',
        # Low-level API (component-based, for testing)
        hamiltonian: Optional[Any] = None,
        ansatz: Optional[Any] = None,
        mapper: Optional[Any] = None,
        molecule: Optional[Any] = None,  # Molecule for hamiltonian-based API
        # Common parameters
        # M2 PR-1: default changed from COBYLA → L-BFGS-B (gradient-based via
        # parameter-shift rule). COBYLA still available for gradient-free /
        # noisy backends. The change is backwards-compatible: callers passing
        # optimizer='COBYLA' explicitly keep COBYLA behavior.
        optimizer: str = 'L-BFGS-B',
        max_iterations: int = 200,  # outer-iter count for gradient methods;
                                    # function-eval count for gradient-free
        conv_threshold: float = 1e-6,
        backend: str = 'statevector',
        shots: Optional[int] = None,
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        experiment_id: Optional[str] = None,  # For WebSocket broadcasting and cancellation
        job_id: Optional[str] = None,  # For cancellation checking
        callback: Optional[Callable] = None,  # Progress callback
        # Solver mode parameters
        # Default is real variational quantum eigensolver. The classical-CI
        # subspace-diagonalization path (formerly mode='hivqe') is still callable
        # for backwards compatibility but emits a deprecation warning — prefer
        # CISolver from kanad.solvers, which is what that path actually is.
        mode: str = 'standard',
        hivqe_max_iterations: int = 10,  # Hi-VQE (deprecated) subspace expansion iterations
        hivqe_subspace_threshold: float = 0.05,  # Hi-VQE (deprecated) amplitude threshold
        # M1 D4 — symmetry penalties (McClean et al. 2018 constrained VQE).
        # Adds λ_N⟨(N̂ - N_target)²⟩ + λ_Sz⟨(Ŝ_z - S_z_target)²⟩ to the VQE loss.
        # Default λ_N = 1.0 (on) — required for clean observables (e.g. H₂ dipole
        # has < 1e-8 D leak instead of 4.65e-4 D). Default λ_Sz = 0.0 (off);
        # COBYLA + λ_Sz≥1 + HEA traps in the triplet S_z=0 basin on H₂ (proven
        # empirically). Set λ_Sz > 0 (~0.1) for open-shell workflows; M2 will
        # revisit defaults once a gradient-based optimizer can navigate the
        # augmented loss landscape.
        lambda_N: float = 1.0,
        lambda_Sz: float = 0.0,
        lambda_S2: float = 0.0,
        sz_target: Optional[float] = None,  # If None, inferred from molecule (default 0 for closed-shell).
        # M2 PR-3 — parameter cache for warm-start. Cache stores converged θ*
        # keyed by (geometry, ansatz config, mapper, basis). On `solve()`, if a
        # cached run exists for this exact system, the optimizer warm-starts
        # from it. `find_similar` looks up nearby geometries for scan workflows.
        use_cache: bool = True,
        # M2.5 — ansatz depth for particle-conserving ansatze.
        # `None` (default) picks a sensible value per ansatz type:
        #   - hardware_efficient: 3 (always)
        #   - givens / givens_rotation: max(3, n_qubits // 2)
        #   - givens_sd: 1 (1 Trotter step ≈ UCCSD; bump to 2+ for tighter chemistry)
        ansatz_n_layers: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize VQE solver.

        Supports two APIs:
        1. High-level (bond-based): solver = VQESolver(bond, ansatz_type='hardware_efficient')
        2. Low-level (component-based): solver = VQESolver(hamiltonian=ham, ansatz=ans, mapper=map)

        Args:
            bond: Bond object from BondFactory (high-level API)
            ansatz_type: Type of ansatz ('hardware_efficient' [default], 'physics_driven')
            mapper_type: Fermionic-to-qubit mapping ('jordan_wigner', 'parity', 'bravyi_kitaev')
            hamiltonian: Hamiltonian object (low-level API, for testing)
            ansatz: Ansatz object (low-level API, for testing)
            mapper: Mapper object (low-level API, for testing)
            optimizer: Classical optimizer ('SLSQP', 'COBYLA', 'L-BFGS-B')
            max_iterations: Maximum optimization iterations
            conv_threshold: Convergence threshold (Hartree)
            backend: Quantum backend ('statevector', 'qasm', 'bluequbit', 'ibm')
            shots: Number of shots for sampling backends
            enable_analysis: Enable automatic analysis
            enable_optimization: Enable automatic circuit optimization
            **kwargs: Additional backend-specific options
        """
        # Store ansatz/mapper types FIRST (needed by _init methods)
        self._ansatz_type_param = ansatz_type
        self._mapper_type_param = mapper_type
        # M2.5: ansatz layer-depth override must be set before _init_ansatz
        # is called (which happens during the api_mode dispatch below).
        self._ansatz_n_layers_override = ansatz_n_layers

        # Unified solver protocol: the positional `system` is the high-level input
        # (Bond / QuantumSystem / Molecule / bare Hamiltonian). Map it onto the
        # legacy `bond` slot unless an explicit low-level kwarg was given.
        if system is not None and bond is None and hamiltonian is None:
            bond = system

        # Normalize + remember the requested backend name; the BaseBackend object
        # is built by BaseSolver.__init__ (bond mode) or by _init_backend (others).
        backend = 'statevector' if backend == 'classical' else backend
        self._requested_backend = backend

        # Detect which API is being used
        if bond is not None and hamiltonian is None:
            # High-level API: Initialize from bond / system
            super().__init__(
                bond,
                backend=backend,
                enable_analysis=enable_analysis,
                enable_optimization=enable_optimization,
                **kwargs,
            )
            self._api_mode = 'bond'
        elif hamiltonian is not None and bond is None:
            if ansatz is None and mapper is None:
                # Hamiltonian-based API with types (for polyatomic molecules from API)
                # Use provided molecule or extract from hamiltonian
                if molecule is not None:
                    molecule_obj = molecule
                elif hasattr(hamiltonian, 'molecule'):
                    molecule_obj = hamiltonian.molecule
                else:
                    # Hamiltonian might not have molecule reference, try to get it
                    molecule_obj = getattr(hamiltonian, '_molecule', None)

                if molecule_obj is None:
                    raise ValueError("Must provide 'molecule' parameter or hamiltonian with molecule reference for type-based initialization")

                # Initialize with molecule and hamiltonian
                self.molecule = molecule_obj
                self.hamiltonian = hamiltonian
                self.bond = None
                # Expose atoms for print_summary / analysis (this branch skips
                # BaseSolver.__init__, which would otherwise set self.atoms).
                self.atoms = getattr(molecule_obj, 'atoms', [])
                self._enable_analysis = enable_analysis
                self._enable_optimization = enable_optimization
                self.enable_analysis = enable_analysis  # Public attribute for _build_circuit
                self.enable_optimization = enable_optimization  # Public attribute for _build_circuit
                self._api_mode = 'hamiltonian_types'
            else:
                # Low-level API: Initialize from components (for testing)
                self._init_from_components_mode(hamiltonian, ansatz, mapper, molecule, enable_analysis, enable_optimization)
                self._api_mode = 'components'
        elif bond is not None and hamiltonian is not None:
            raise ValueError("Cannot use both 'bond' and 'hamiltonian' parameters. Choose one API.")
        else:
            raise ValueError("Must provide either 'bond' (high-level API) or 'hamiltonian' (low-level API)")

        # Store common parameters
        self.optimizer_method = optimizer  # Support both 'optimizer' and 'optimizer_method' kwargs
        # The UI/API sends SCREAMING_SNAKE optimizer names ('NELDER_MEAD', 'L_BFGS_B');
        # scipy.optimize.minimize matches case-insensitively but on HYPHENS, so an
        # underscore name fails as 'Unknown solver'. Normalize _→- (SPSA is a custom
        # branch handled before the scipy call and must stay literal).
        if isinstance(self.optimizer_method, str) and self.optimizer_method.upper() != 'SPSA':
            self.optimizer_method = self.optimizer_method.replace('_', '-')
        self.max_iterations = max_iterations
        self.conv_threshold = conv_threshold
        # backend name already normalized above; the BaseBackend *object* lives on
        # self.backend (set by BaseSolver.__init__ / _init_backend). self.backend_name
        # is the string form.
        self.backend_name = backend
        self.shots = shots if shots is not None else 1024

        # Store experiment_id and job_id for WebSocket broadcasting and cancellation
        self.experiment_id = experiment_id
        self.job_id = job_id

        # Track cloud provider job information
        self.cloud_job_ids = []  # List of cloud job IDs from IBM/BlueQubit
        self.cloud_provider = None  # 'ibm' or 'bluequbit'
        self.execution_mode = None  # 'batch', 'session', 'instance' for IBM

        # Store callback (don't pass to backend via kwargs)
        self._callback = callback

        # This is a correlated method
        self._is_correlated = True

        # Solver mode + Hi-VQE (deprecated) parameters
        self.mode = mode.lower()
        self.hivqe_max_iterations = hivqe_max_iterations
        self.hivqe_subspace_threshold = hivqe_subspace_threshold

        if self.mode not in ['standard', 'hivqe']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'standard' (real VQE) or 'hivqe' (classical CI, deprecated — use CISolver).")

        if self.mode == 'hivqe':
            import warnings
            warnings.warn(
                "VQESolver(mode='hivqe') runs classical Configuration Interaction in a "
                "sampled subspace — not a variational quantum eigensolver. Use "
                "kanad.solvers.CISolver to call this algorithm under its true name. "
                "The mode='hivqe' alias on VQESolver is retained for backwards "
                "compatibility and may be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.info(f"Hi-VQE (classical CI) mode: {hivqe_max_iterations} iterations, threshold={hivqe_subspace_threshold}")

        # Initialize based on API mode
        if self._api_mode == 'bond':
            # AUDIT H16: honor a directly-passed ansatz object in bond mode.
            # Previously _init_ansatz() ran unconditionally and rebuilt the ansatz
            # from ansatz_type (default 'hardware_efficient'), silently ignoring a
            # passed ansatz= object (e.g. VQESolver(bond=H2, ansatz=LUCJ(...)) ran
            # HEA with the wrong parameter count). Use the object when given; raise
            # if the caller also set a non-default ansatz_type (ambiguous intent).
            if ansatz is not None:
                if self._ansatz_type_param != 'hardware_efficient':
                    raise ValueError(
                        "Cannot pass both an ansatz object and a non-default "
                        f"ansatz_type='{self._ansatz_type_param}'. Pass exactly one."
                    )
                self.ansatz = ansatz
                self.ansatz_type = type(ansatz).__name__
            else:
                # Initialize ansatz from ansatz_type
                self._init_ansatz()
            self._init_mapper()
            # Build quantum circuit
            self._build_circuit()
            # Initialize backend
            self._init_backend(**kwargs)
        elif self._api_mode == 'hamiltonian_types':
            # Hamiltonian-based API with types - initialize like bond mode
            self._init_ansatz()
            self._init_mapper()
            # Build quantum circuit
            self._build_circuit()
            # Initialize backend
            self._init_backend(**kwargs)
        else:
            # Components mode - ansatz and mapper already set
            # Initialize backend (this will set _use_statevector correctly)
            self._hamiltonian_matrix = None
            self._init_backend(**kwargs)
            logger.info(f"VQE initialized in components mode, backend={self.backend_name}, use_statevector={self._use_statevector}")

        # Optimization tracking
        self.energy_history = []
        self.parameter_history = []
        self.iteration_count = 0  # Function evaluation counter
        self.optimizer_iteration = 0  # Real optimizer iteration counter
        self._last_energy = None  # Track energy changes to detect optimizer iterations

        # Performance optimization: Cache sparse Pauli operator
        # CRITICAL: Cache must be invalidated if mapper changes!
        self._sparse_pauli_op = None
        self._cached_mapper = None  # Track which mapper was used for cache
        self._use_sparse = False

        # M1 D4 — symmetry penalty configuration.
        # Lambda weights and target values; the actual penalty SparsePauliOp is
        # built lazily on first use because it depends on the post-init n_qubits.
        self.lambda_N = float(lambda_N)
        self.lambda_Sz = float(lambda_Sz)
        self.lambda_S2 = float(lambda_S2)
        self._sz_target_override = sz_target
        # Cached SparsePauliOps for penalty operators
        self._n_penalty_op = None
        self._sz_penalty_op = None
        self._s2_penalty_op = None
        self._penalty_n_qubits = None  # n_qubits used to build the cached ops
        # Tracking
        self.penalty_history = []  # raw penalty (λ_N·var_N + λ_Sz·var_Sz) per evaluation
        self.loss_history = []     # energy + penalty per evaluation

        # M2 PR-3 — parameter cache
        self.use_cache = bool(use_cache)
        self._init_strategy = 'unknown'  # set in _solve_standard_vqe: cached | cached_similar | mp2 | random | user
        self._cache_hit = False
        # Note: self._ansatz_n_layers_override is set earlier (above the
        # api_mode dispatch) because _init_ansatz needs it.

        logger.info(f"VQE Solver initialized: {self.ansatz_type} ansatz, {self.mapper_type} mapping, {self.backend_name} backend")
        if self.lambda_N > 0 or self.lambda_Sz > 0 or self.lambda_S2 > 0:
            logger.info(
                f"Symmetry penalty enabled: λ_N={self.lambda_N}, λ_Sz={self.lambda_Sz}, λ_S²={self.lambda_S2}"
            )

    def _init_from_components_mode(self, hamiltonian, ansatz, mapper, molecule, enable_analysis, enable_optimization):
        """Initialize VQE from individual components (low-level API for testing)."""
        # Store components directly
        self.hamiltonian = hamiltonian
        self.mapper = mapper if mapper is not None else JordanWignerMapper()

        # Try to get molecule from parameter, then hamiltonian, then None
        if molecule is not None:
            self.molecule = molecule
        else:
            self.molecule = getattr(hamiltonian, 'molecule', None)

        self.bond = None

        # Enable analysis features if molecule is available
        self.enable_analysis = enable_analysis if self.molecule is not None else False
        self.enable_optimization = enable_optimization

        # Initialize analysis tools if enabled and molecule available
        if self.enable_analysis and self.molecule is not None:
            try:
                from kanad.analysis import EnergyAnalyzer, BondingAnalyzer, PropertyCalculator
                self.energy_analyzer = EnergyAnalyzer(self.hamiltonian)
                self.bonding_analyzer = BondingAnalyzer(self.hamiltonian)
                self.property_calculator = PropertyCalculator(self.hamiltonian)
                self.atoms = self.molecule.atoms
                logger.info("✅ Analysis tools initialized in components mode")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize analysis tools: {e}")
                import traceback
                traceback.print_exc()
                self.enable_analysis = False

        # Initialize ansatz (from object or type string)
        if ansatz is not None:
            # Ansatz object provided directly
            self.ansatz = ansatz
            self.ansatz_type = type(self.ansatz).__name__
        elif self._ansatz_type_param is not None:
            # Ansatz type string provided - create ansatz
            self.ansatz_type = self._ansatz_type_param
            self._init_ansatz()
        else:
            self.ansatz = None
            self.ansatz_type = 'None'

        # Set mapper type string for logging
        self.mapper_type = type(self.mapper).__name__ if self.mapper else 'None'

        # Build circuit to get n_parameters
        if self.ansatz is not None:
            if self.ansatz.circuit is None:
                self.ansatz.build_circuit()

            # Get n_parameters from various possible sources
            if hasattr(self.ansatz, 'n_parameters'):
                self.n_parameters = self.ansatz.n_parameters
            elif hasattr(self.ansatz.circuit, 'get_num_parameters'):
                self.n_parameters = self.ansatz.circuit.get_num_parameters()
            elif hasattr(self.ansatz, 'num_parameters'):
                self.n_parameters = self.ansatz.num_parameters
            else:
                self.n_parameters = 0
        else:
            self.n_parameters = 0

        logger.info("VQE solver initialized from components (testing mode)")

    def _init_ansatz(self):
        """Initialize ansatz from ansatze module.

        Currently supported ansatz_type values (verified-working):

        - 'hardware_efficient' (default; 0.016 mHa on H2 + standard mode).
        - 'givens' / 'givens_rotation' (M2.5 — particle-conserving brick-wall
          Givens rotations; the right choice for ≥10-qubit systems where HEA
          gets trapped by its non-N-conserving CNOT entanglement).
        - 'physics_driven' (use with PhysicsVQE solver for best results).

        Previously supported but REMOVED in 2026-05-12 cleanup (see CLEANUP.md):

        - 'ucc': parameters did not affect energy (1493 mHa error on H2)
        - 'governance' / 'adaptive_governance': returned HF energy, no correlation
        """
        from kanad.core.ansatze import (
            HardwareEfficientAnsatz,
            GivensRotationAnsatz,
            GivensSDAnsatz,
            PhysicsDrivenAnsatz,
        )

        # Derive n_qubits defensively: prefer n_orbitals (molecular Hamiltonians),
        # fall back to num_qubits (e.g. a directly-supplied SparsePauliOp, whose
        # num_qubits is already the spin-orbital count).
        if getattr(self.hamiltonian, 'n_orbitals', None) is not None:
            n_qubits = 2 * self.hamiltonian.n_orbitals
        elif hasattr(self.hamiltonian, 'num_qubits'):
            n_qubits = self.hamiltonian.num_qubits
        else:
            raise ValueError(
                "Cannot determine n_qubits: Hamiltonian lacks both 'n_orbitals' "
                "and 'num_qubits'."
            )

        # HEA needs n_electrons for its reference state. Prefer the molecule, but
        # fall back to the Hamiltonian (molecular Hamiltonians carry n_electrons),
        # so `from_hamiltonian(...)` works without an explicit molecule. Only error
        # when neither source can supply it (e.g. a bare SparsePauliOp).
        if self.molecule is not None and hasattr(self.molecule, 'n_electrons'):
            n_electrons = self.molecule.n_electrons
        elif getattr(self.hamiltonian, 'n_electrons', None) is not None:
            n_electrons = self.hamiltonian.n_electrons
        else:
            raise ValueError(
                "VQESolver with ansatz_type requires n_electrons, which could not "
                "be derived from the molecule or the Hamiltonian. Pass molecule=... "
                "or a Hamiltonian exposing n_electrons."
            )

        ansatz_type = self._ansatz_type_param.lower()

        # Resolve n_layers: user override > sensible per-type default
        layers_override = self._ansatz_n_layers_override

        if ansatz_type == 'hardware_efficient':
            n_layers = layers_override if layers_override is not None else 3
            self.ansatz = HardwareEfficientAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=n_layers,
                entanglement='linear',
                mapper=self._mapper_type_param,
            )
            logger.info(f"Hardware-efficient ansatz: {n_layers} layers, linear, {self._mapper_type_param}")

        elif ansatz_type in ('givens', 'givens_rotation'):
            # M2.5: brick-wall Givens singles. Particle-conserving but cannot
            # escape HF on chemistry Hamiltonians (Brillouin stationary). Use
            # 'givens_sd' for chemistry — that ansatz adds paired doubles.
            n_layers = layers_override if layers_override is not None else max(3, n_qubits // 2)
            self.ansatz = GivensRotationAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=n_layers,
                mapper=self._mapper_type_param,
            )
            logger.info(
                f"Givens-rotation ansatz (singles only): {n_layers} brick-wall layers, "
                f"{self.ansatz.n_parameters} params, particle-conserving. "
                "WARNING: cannot escape HF on chemistry — use 'givens_sd' instead."
            )

        elif ansatz_type in ('givens_sd', 'particle_conserving', 'uccsd_like'):
            # M2.5: Givens singles + paired doubles (UCCSD-style).
            # Doubles break Brillouin's theorem at first order in θ — the
            # right choice for chemistry on ≥4-qubit closed-shell systems.
            # Each Trotter step costs O(n_occ × n_virt) parameters; default 1
            # is the right baseline for small molecules, bump for tighter
            # chemistry on larger active spaces.
            n_layers = layers_override if layers_override is not None else 1
            self.ansatz = GivensSDAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=n_layers,
                mapper=self._mapper_type_param,
            )
            logger.info(
                f"Givens-SD ansatz: {n_layers} Trotter layer(s), {self.ansatz.n_singles} singles + "
                f"{self.ansatz.n_doubles} doubles per layer = {self.ansatz.n_parameters} params, "
                "particle-conserving and breaks Brillouin via paired doubles."
            )

        elif ansatz_type == 'physics_driven':
            raise NotImplementedError(
                "ansatz_type='physics_driven' is not supported by VQESolver: "
                "PhysicsDrivenAnsatz.build_circuit() only emits gates for numeric "
                "parameter values, not the symbolic Qiskit Parameters that VQESolver "
                "uses, so the resulting circuit has zero parameters and VQE has "
                "nothing to optimize. Use kanad.solvers.PhysicsVQE — it owns the "
                "parameter loop itself and feeds numeric values into the ansatz."
            )

        elif ansatz_type in ('ucc', 'governance', 'adaptive_governance'):
            raise ValueError(
                f"Ansatz '{ansatz_type}' was removed in the 2026-05-12 cleanup "
                f"because it produced incorrect energies. Use "
                f"ansatz_type='hardware_efficient' instead. See CLEANUP.md."
            )

        else:
            raise ValueError(
                f"Unknown ansatz type: {ansatz_type}. "
                f"Supported: 'hardware_efficient', 'givens', 'physics_driven'."
            )

        # CRITICAL FIX: Filter ansatz excitations using governance protocol validation
        # This ensures VQE uses same excitations as SQD (consistency!)
        if hasattr(self.hamiltonian, 'governance_protocol') and self.hamiltonian.governance_protocol:
            protocol = self.hamiltonian.governance_protocol

            # Only filter if ansatz has excitations attribute (UCC, not hardware-efficient)
            if hasattr(self.ansatz, 'excitations') and hasattr(self.ansatz, 'get_excitation_list'):
                from kanad.core.configuration import Configuration

                original_excitations = self.ansatz.get_excitation_list()
                n_original = len(original_excitations)

                # HF reference bitstring
                hf_bitstring = '1' * n_electrons + '0' * (n_qubits - n_electrons)

                valid_excitations = []
                for exc in original_excitations:
                    # Convert excitation to configuration
                    occ, virt = exc

                    # Build bitstring by applying excitation to HF reference
                    bitlist = list(hf_bitstring)
                    for i in occ:
                        bitlist[i] = '0'  # Remove electron
                    for a in virt:
                        bitlist[a] = '1'  # Add electron

                    excited_bitstring = ''.join(bitlist)

                    # Check if valid according to governance
                    if protocol.is_valid_configuration(excited_bitstring):
                        valid_excitations.append(exc)

                # Update ansatz excitations
                self.ansatz.excitations = valid_excitations

                logger.info(f"✅ Governance filtering: {n_original} → {len(valid_excitations)} valid excitations")
                logger.info(f"   Protocol: {type(protocol).__name__}")
            else:
                logger.debug(f"Ansatz {ansatz_type} does not have excitations - skipping governance filtering")

        # Store ansatz type after creation
        self.ansatz_type = ansatz_type

    def _init_mapper(self):
        """Initialize fermionic-to-qubit mapper from bonds module."""
        from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper; from kanad.core.mappers.bravyi_kitaev_mapper import BravyiKitaevMapper

        # Use stored parameter
        mapper_type = self._mapper_type_param

        if mapper_type.lower() == 'jordan_wigner':
            self.mapper = JordanWignerMapper()
        elif mapper_type.lower() == 'bravyi_kitaev':
            self.mapper = BravyiKitaevMapper()
        elif mapper_type.lower() == 'parity':
            # 'parity' was half-implemented: no parity Hamiltonian builder exists, so
            # the operator fell back to Jordan-Wigner while the ansatz still prepared a
            # PARITY-encoded HF reference (get_hf_state_qubits) — an inconsistent start
            # with the wrong particle number / reference energy. Reject loudly rather
            # than silently optimize from a wrong state. (CORE_BUGS B11.)
            raise NotImplementedError(
                "mapper_type='parity' is not supported: there is no parity Hamiltonian "
                "builder, so the qubit operator would be Jordan-Wigner while the HF "
                "reference is parity-encoded (wrong particle number). Use "
                "'jordan_wigner' or 'bravyi_kitaev'.")
        else:
            # Default to Jordan-Wigner
            logger.warning(f"Unknown mapper type {mapper_type}, using Jordan-Wigner")
            self.mapper = JordanWignerMapper()

        # Store mapper type after creation
        self.mapper_type = mapper_type
        logger.debug(f"Mapper initialized: {self.mapper_type}")

    def _build_circuit(self):
        """Build parametrized quantum circuit."""
        self.circuit = self.ansatz.build_circuit()
        # Different ansatze return either a Kanad circuit (.get_num_parameters())
        # or a raw Qiskit QuantumCircuit (.num_parameters property). Probe both.
        if hasattr(self.circuit, 'get_num_parameters') and callable(self.circuit.get_num_parameters):
            self.n_parameters = self.circuit.get_num_parameters()
        elif hasattr(self.circuit, 'num_parameters'):
            np_attr = self.circuit.num_parameters
            self.n_parameters = np_attr() if callable(np_attr) else np_attr
        elif hasattr(self.ansatz, 'n_parameters'):
            self.n_parameters = self.ansatz.n_parameters
        else:
            self.n_parameters = 0

        logger.info(f"Circuit built: {self.n_parameters} parameters")

        # Store pre-optimization stats
        if self.enable_optimization:
            try:
                self._gates_before_opt = self.circuit.count_ops().get('total', 0) if hasattr(self.circuit, 'count_ops') else None
                self._depth_before_opt = self.circuit.depth if isinstance(self.circuit.depth, int) else (self.circuit.depth() if callable(self.circuit.depth) else None)
            except (AttributeError, TypeError):
                self._gates_before_opt = None
                self._depth_before_opt = None

    def _init_backend(self, **kwargs):
        """Ensure the BaseBackend object exists and set the legacy cloud-path flags.

        Bond mode already built ``self.backend`` in ``BaseSolver.__init__``; the
        hamiltonian_types / components modes skip that, so build it here. The
        ``_use_statevector`` + ``_ibm_backend`` / ``_bluequbit_backend`` /
        ``_ionq_backend`` aliases preserve the existing cloud execution paths,
        which submit jobs directly via those backend objects' ``run_*`` methods.
        """
        from kanad.backends.base_backend import BaseBackend
        from kanad.backends.statevector_backend import StatevectorBackend
        from kanad.backends.factory import make_backend

        if not isinstance(getattr(self, 'backend', None), BaseBackend):
            self.backend = make_backend(self._requested_backend, **kwargs)
        self.backend_name = self.backend.name

        # Route energy through the LOCAL-statevector path for any backend that
        # produces a full |ψ⟩ we contract with H: StatevectorBackend (CPU) and
        # PlanckBackend (GPU; _compute_energy_statevector has a planck branch that
        # builds |ψ⟩ on-GPU and contracts via _planck_expect). Only true cloud/shot
        # backends (ibm/bluequbit/ionq) use the sampling path. Without the planck
        # clause, isinstance(StatevectorBackend) is False for PlanckBackend and the
        # energy mis-routes into _compute_energy_quantum's cloud branches (none of
        # which match planck), silently breaking planck energy. (planck audit #2)
        self._use_statevector = (
            isinstance(self.backend, StatevectorBackend)
            or getattr(self.backend, 'name', None) == 'planck'
        )
        self._ibm_backend = self.backend if self.backend.name == 'ibm' else None
        self._bluequbit_backend = self.backend if self.backend.name == 'bluequbit' else None
        self._ionq_backend = self.backend if self.backend.name == 'ionq' else None
        self._pauli_hamiltonian = getattr(self, '_pauli_hamiltonian', None)
        self._hamiltonian_matrix = getattr(self, '_hamiltonian_matrix', None)

    def _compute_energy(self, parameters: np.ndarray) -> float:
        """
        Compute energy expectation value for given parameters.

        E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩

        Args:
            parameters: Circuit parameters

        Returns:
            Energy expectation value (Hartree)
        """
        if self._use_statevector:
            # Classical statevector simulation
            return self._compute_energy_statevector(parameters)
        else:
            # Quantum backend
            return self._compute_energy_quantum(parameters)

    # ===== M1 D4: symmetry penalty (constrained VQE) =====================

    def _build_penalty_operators(self, n_qubits: int):
        """Lazily build N̂, Ŝ_z, and Ŝ² penalty SparsePauliOps for the given n_qubits.

        N̂ = Σ_q (I - Z_q) / 2 — particle number (Jordan-Wigner: qubit = spin orbital).
        Ŝ_z = ½ Σ_p (n_α(p) - n_β(p)) — spin projection, with the framework's
              JW convention of spin-α at even qubits, spin-β at odd qubits.
        Ŝ² — total spin (from core.operators.spin_operators); penalizes toward the
              target multiplicity S(S+1).

        We cache `(N̂ - N_target·I)²`, `(Ŝ_z - S_z_target·I)²`, and
        `(Ŝ² - S(S+1)·I)²` for fast per-evaluation expectation values.
        """
        from qiskit.quantum_info import SparsePauliOp

        if self._penalty_n_qubits == n_qubits and (
            self._n_penalty_op is not None or self.lambda_N <= 0
        ) and (
            self._sz_penalty_op is not None or self.lambda_Sz <= 0
        ) and (
            self._s2_penalty_op is not None or self.lambda_S2 <= 0
        ):
            return

        n_target = float(getattr(self.molecule, 'n_electrons', 0) or 0)

        # Sz target — closed-shell singlet by default; user can override.
        if self._sz_target_override is not None:
            sz_target = float(self._sz_target_override)
        else:
            # Try molecule.spin; PySCF stores 2S = (n_α - n_β) on `Mole.spin`.
            spin_2s = getattr(self.molecule, 'spin', None)
            if spin_2s is None:
                spin_2s = 0
            sz_target = float(spin_2s) / 2.0

        identity = SparsePauliOp(['I' * n_qubits], [1.0])

        if self.lambda_N > 0:
            # N̂ = n/2 · I − ½ Σ Z_q
            terms = [('', [], n_qubits / 2.0)]
            for q in range(n_qubits):
                terms.append(('Z', [q], -0.5))
            N_op = SparsePauliOp.from_sparse_list(terms, num_qubits=n_qubits)
            shifted = (N_op - n_target * identity).simplify()
            self._n_penalty_op = (shifted @ shifted).simplify()
            logger.debug(f"Built (N̂ - {n_target})² penalty op: {len(self._n_penalty_op)} Pauli terms")

        if self.lambda_Sz > 0:
            if n_qubits % 2 != 0:
                raise ValueError(
                    f"Ŝ_z penalty requires even n_qubits (alpha/beta pairing); got {n_qubits}. "
                    "Disable with lambda_Sz=0.0."
                )
            # Ŝ_z from the indigenous core.operators.spin_operators (single home;
            # bit-identical to the prior interleaved-JW Pauli-Z literals,
            # verified 0.0 diff). (reorg B4)
            from kanad.core.operators.spin_operators import build_spin_operators
            _, Sz_op, _ = build_spin_operators(n_qubits // 2, 'jordan_wigner')
            shifted = (Sz_op - sz_target * identity).simplify()
            self._sz_penalty_op = (shifted @ shifted).simplify()
            logger.debug(f"Built (Ŝ_z - {sz_target})² penalty op: {len(self._sz_penalty_op)} Pauli terms")

        if self.lambda_S2 > 0:
            if n_qubits % 2 != 0:
                raise ValueError(
                    f"Ŝ² penalty requires even n_qubits (alpha/beta pairing); got {n_qubits}. "
                    "Disable with lambda_S2=0.0."
                )
            # Ŝ² from the indigenous core.operators.spin_operators (3rd return).
            # Target S(S+1): S is taken as the spin from sz_target (S = M_s for the
            # ground multiplicity; closed-shell sz_target=0 → singlet target 0). The
            # penalty (Ŝ² − S(S+1)·I)² drives the state onto the target multiplicity.
            # Previously lambda_S2 was accepted/stored/logged but NEVER built or
            # summed — a silent no-op that misled users. (reorg B-audit #11)
            from kanad.core.operators.spin_operators import build_spin_operators
            _, _, S2_op = build_spin_operators(n_qubits // 2, 'jordan_wigner')
            s2_value = sz_target * (sz_target + 1.0)
            shifted = (S2_op - s2_value * identity).simplify()
            self._s2_penalty_op = (shifted @ shifted).simplify()
            logger.debug(f"Built (Ŝ² - {s2_value})² penalty op: {len(self._s2_penalty_op)} Pauli terms")

        self._penalty_n_qubits = n_qubits

    def _compute_penalty(self, parameters: np.ndarray) -> float:
        """Compute symmetry-penalty contribution for the given parameters.

        Returns λ_N · ⟨(N̂ - N)²⟩ + λ_Sz · ⟨(Ŝ_z - S_z)²⟩ + λ_S² · ⟨(Ŝ² - S(S+1))²⟩.
        Zero (and no statevector recomputation) when all penalty weights are 0.
        """
        if self.lambda_N <= 0 and self.lambda_Sz <= 0 and self.lambda_S2 <= 0:
            return 0.0
        if not self._use_statevector:
            # Penalty path requires a statevector for the expectation values.
            # On hardware/shot backends we'd need separate measurement circuits
            # for the penalty operators — out of scope for M1.
            logger.warning("Symmetry penalty disabled on non-statevector backend (M1 limitation)")
            return 0.0

        from qiskit.quantum_info import Statevector

        # Build bound circuit (mirrors _compute_energy_statevector)
        if self.ansatz.circuit is None:
            self.ansatz.build_circuit()
        circ = self.ansatz.circuit
        if hasattr(circ, 'bind_parameters'):
            try:
                circ.bind_parameters(parameters)
            except TypeError:
                pass
        if hasattr(circ, 'to_qiskit'):
            qiskit_circuit = circ.to_qiskit()
        else:
            qiskit_circuit = circ
        if qiskit_circuit.num_parameters > 0:
            param_dict = {qiskit_circuit.parameters[i]: parameters[i] for i in range(len(parameters))}
            bound_circuit = qiskit_circuit.assign_parameters(param_dict)
        else:
            bound_circuit = qiskit_circuit

        n_qubits = bound_circuit.num_qubits
        self._build_penalty_operators(n_qubits)

        statevector = Statevector.from_instruction(bound_circuit)
        penalty = 0.0
        if self.lambda_N > 0 and self._n_penalty_op is not None:
            penalty += self.lambda_N * float(statevector.expectation_value(self._n_penalty_op).real)
        if self.lambda_Sz > 0 and self._sz_penalty_op is not None:
            penalty += self.lambda_Sz * float(statevector.expectation_value(self._sz_penalty_op).real)
        if self.lambda_S2 > 0 and self._s2_penalty_op is not None:
            penalty += self.lambda_S2 * float(statevector.expectation_value(self._s2_penalty_op).real)
        return penalty

    # ===== M2 PR-3: Parameter cache helpers =============================

    def _save_to_cache(self, scipy_result, solve_start_time):
        """Persist the converged θ* to the parameter cache."""
        import time
        from datetime import datetime
        try:
            import kanad
            from kanad.core.cache import (
                get_default_cache, CachedRun,
            )

            key, atom_syms, atom_coords = self._build_cache_key_if_possible()
            if key is None:
                return  # can't build a stable key, skip

            cache = get_default_cache()
            walltime = time.time() - solve_start_time

            telemetry = self.results.get('telemetry', {}) if hasattr(self, 'results') else {}
            run = CachedRun(
                parameters=np.asarray(scipy_result.x, dtype=float),
                final_energy=float(scipy_result.fun),
                n_iterations=int(self.iteration_count),
                walltime_seconds=float(walltime),
                init_strategy=self._init_strategy,
                framework_version=getattr(kanad, '__version__', 'unknown'),
                final_n_variance=telemetry.get('final_n_variance'),
                final_s2_variance=telemetry.get('final_s2_variance'),
                final_gradient_norm=telemetry.get('final_gradient_norm'),
                atom_symbols=tuple(atom_syms),
                atom_coords=np.asarray(atom_coords, dtype=float),
                created_at=datetime.utcnow(),
            )
            cache.put(key, run)
            logger.info(
                f"💾 Cached converged run (E={run.final_energy:.6f} Ha, "
                f"{run.n_iterations} evals, walltime={walltime:.2f}s)"
            )
        except Exception as exc:
            logger.debug(f"Cache save failed (non-fatal): {exc}")

    def _build_cache_key_if_possible(self):
        """Return (CacheKey, atom_symbols, atom_coords) or (None, None, None).

        Returns None if we can't extract a stable cache key — e.g. when the
        solver was initialized in components mode without a molecule, or when
        atom positions aren't available.
        """
        atom_syms = None
        atom_coords = None
        # Path 1: molecule has .atoms with .symbol and .position
        mol = getattr(self, 'molecule', None)
        if mol is not None and hasattr(mol, 'atoms'):
            try:
                atoms = mol.atoms
                if all(hasattr(a, 'symbol') and hasattr(a, 'position') for a in atoms):
                    atom_syms = [a.symbol for a in atoms]
                    atom_coords = np.array([a.position for a in atoms], dtype=float)
            except Exception:
                pass
        # Path 2: bond has .atom_1, .atom_2
        bond = getattr(self, 'bond', None)
        if atom_syms is None and bond is not None:
            try:
                if hasattr(bond, 'atom_1') and hasattr(bond, 'atom_2'):
                    a1, a2 = bond.atom_1, bond.atom_2
                    atom_syms = [a1.symbol, a2.symbol]
                    atom_coords = np.array([a1.position, a2.position], dtype=float)
            except Exception:
                pass
        if atom_syms is None or atom_coords is None:
            return None, None, None

        basis = getattr(self.hamiltonian, 'basis_name', None) or getattr(mol, 'basis', None) or 'sto-3g'

        try:
            from kanad.core.cache import build_cache_key
            key = build_cache_key(
                atom_syms, atom_coords, self.ansatz,
                str(self.mapper_type), str(basis),
            )
            return key, atom_syms, atom_coords
        except Exception as exc:
            logger.debug(f"Cache-key construction failed (non-fatal): {exc}")
            return None, None, None

    # ===== M2 PR-1: Parameter-shift gradient =============================

    def _compute_gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Analytical gradient of the augmented loss via parameter-shift rule.

        For any parameterized circuit whose gates are of the form e^{-iθP/2}
        (Pauli rotations like Ry, Rz, Rx — all that HEA uses), the
        parameter-shift rule gives the exact gradient with two energy
        evaluations per parameter:

            ∂L/∂θ_k = (1/2) [L(θ + π/2·e_k) − L(θ - π/2·e_k)]

        where L(θ) = ⟨H⟩(θ) + λ_N⟨(N̂-N)²⟩(θ) + λ_Sz⟨(Ŝ_z-S_z)²⟩(θ) is the
        augmented loss the optimizer sees.

        Cost: 2·P energy evaluations + (if penalty enabled) 2·P penalty
        evaluations per gradient call.

        Returns:
            np.ndarray of shape (n_parameters,) — ∂L/∂θ_k for each parameter.
        """
        n_params = len(parameters)
        grad = np.zeros(n_params)
        shift = np.pi / 2

        for k in range(n_params):
            p_plus = parameters.copy()
            p_plus[k] += shift
            p_minus = parameters.copy()
            p_minus[k] -= shift

            # Evaluate the FULL augmented loss at each shifted point.
            # The parameter-shift rule is linear in the observable, so it
            # applies independently to ⟨H⟩ and the penalty terms.
            l_plus = self._compute_energy(p_plus) + self._compute_penalty(p_plus)
            l_minus = self._compute_energy(p_minus) + self._compute_penalty(p_minus)

            grad[k] = 0.5 * (l_plus - l_minus)

        # Track gradient call count separately from function-eval count
        # so users can compare optimizers fairly.
        self._gradient_call_count = getattr(self, '_gradient_call_count', 0) + 1
        return grad

    # ===== M2.7: Adjoint-state gradient ==================================

    def _build_adjoint_calculator(self):
        """Build (and cache) the AdjointGradientCalculator for the current
        circuit + augmented Hamiltonian (energy + penalty terms).

        Constructs once per VQESolver instance — the calculator caches
        per-gate generator info and the sparse Hamiltonian matrix.
        """
        from kanad.core.vqe_gradients import AdjointGradientCalculator
        from qiskit.quantum_info import SparsePauliOp

        # Build the combined observable: H + λ_N·(N̂−N)² + λ_Sz·(Ŝ_z−S_z)² + λ_S²·(Ŝ²−S(S+1))²
        # All terms are Hermitian Pauli sums; adjoint gradient applies
        # to any Hermitian observable so we combine them once and use a single
        # calculator. This avoids 2-3× cost from running adjoint per term.
        if self._sparse_pauli_op is None:
            # Trigger Pauli-Hamiltonian construction via a dummy energy eval.
            # The Hamiltonian is then cached on self._sparse_pauli_op.
            from qiskit.quantum_info import Statevector
            dummy = np.zeros(self.n_parameters)
            _ = self._compute_energy_statevector(dummy)
        H = self._sparse_pauli_op
        n_qubits = H.num_qubits

        # Ensure penalty operators are built for the right qubit count.
        if self.lambda_N > 0 or self.lambda_Sz > 0 or self.lambda_S2 > 0:
            self._build_penalty_operators(n_qubits)

        combined_terms = [H]
        if self.lambda_N > 0 and self._n_penalty_op is not None:
            combined_terms.append(self.lambda_N * self._n_penalty_op)
        if self.lambda_Sz > 0 and self._sz_penalty_op is not None:
            combined_terms.append(self.lambda_Sz * self._sz_penalty_op)
        if self.lambda_S2 > 0 and self._s2_penalty_op is not None:
            combined_terms.append(self.lambda_S2 * self._s2_penalty_op)
        combined = combined_terms[0]
        for op in combined_terms[1:]:
            combined = combined + op
        combined = combined.simplify()

        # The circuit may have been re-bound from a prior call. Force a fresh
        # build so the calculator sees the symbolic parameters.
        if self.ansatz.circuit is None:
            self.ansatz.build_circuit()
        circ = self.ansatz.circuit
        if hasattr(circ, 'to_qiskit'):
            qc = circ.to_qiskit()
        else:
            qc = circ

        if getattr(self, 'backend_name', '') == 'planck':
            from kanad.backends.planck_adapter import PlanckAdjointGradient
            return PlanckAdjointGradient(qc, combined)
        return AdjointGradientCalculator(qc, combined)

    def _compute_gradient_adjoint(self, parameters: np.ndarray) -> np.ndarray:
        """Adjoint-state gradient of the augmented loss (energy + penalty).

        Single forward + backward pass through the circuit. O(N_gates)
        statevector ops total — orders of magnitude faster than the
        parameter-shift or finite-diff alternatives on large ansatze.
        """
        if not hasattr(self, '_adjoint_calculator') or self._adjoint_calculator is None:
            self._adjoint_calculator = self._build_adjoint_calculator()
        # Map scipy's numpy parameter array → Qiskit Parameter dict
        # (the calculator's parameter ordering matches what we collected in init).
        qc_params = self._adjoint_calculator.parameters
        param_dict = {p: float(v) for p, v in zip(qc_params, parameters)}
        grad_dict = self._adjoint_calculator.gradient(param_dict)
        grad = np.array([grad_dict[p] for p in qc_params], dtype=float)
        self._gradient_call_count = getattr(self, '_gradient_call_count', 0) + 1
        return grad

    def _compute_energy_statevector(self, parameters: np.ndarray) -> float:
        """
        Compute energy using classical statevector simulation.

        PERFORMANCE: Uses sparse Pauli operators for efficiency (100-1000x faster).
        MEMORY SAFE: No dense matrix construction for large molecules.
        """
        from qiskit.quantum_info import Statevector

        # Build circuit if not already built
        if self.ansatz.circuit is None:
            self.ansatz.build_circuit()

        # Bind parameters / convert to Qiskit. Different ansatze return either a
        # Kanad circuit (has .bind_parameters + .to_qiskit) or a raw Qiskit
        # QuantumCircuit (assign_parameters only; .bind_parameters was removed
        # in Qiskit 2.x).
        circ = self.ansatz.circuit
        if hasattr(circ, 'bind_parameters'):
            try:
                circ.bind_parameters(parameters)
            except TypeError:
                pass
        if hasattr(circ, 'to_qiskit'):
            qiskit_circuit = circ.to_qiskit()
        else:
            qiskit_circuit = circ

        # If circuit still has parameters, bind them
        if qiskit_circuit.num_parameters > 0:
            param_dict = {qiskit_circuit.parameters[i]: parameters[i] for i in range(len(parameters))}
            bound_circuit = qiskit_circuit.assign_parameters(param_dict)
        else:
            bound_circuit = qiskit_circuit

        # Get statevector from circuit. The planck backend builds |psi> on-GPU
        # (rocm-planck); the result is wrapped in a Qiskit Statevector so every
        # downstream consumer (penalties, RDMs) is byte-identical.
        if getattr(self, 'backend_name', '') == 'planck':
            # Build |psi> on-GPU and wrap in a Statevector so downstream consumers
            # (penalties, RDMs) are byte-identical. NOTE: template caching was attempted
            # but kanad binds the ansatz circuit in place and the symbolic circuit is not
            # cleanly available at this hook — the per-call path is correct; revisit
            # caching via a dedicated symbolic-template accessor (see planck docs).
            from kanad.backends.planck_adapter import planck_statevector
            self._planck_sv = planck_statevector(bound_circuit)
            statevector = Statevector(self._planck_sv.to_numpy())
        else:
            statevector = Statevector.from_instruction(bound_circuit)

        # DEBUG: Check statevector first time
        if not hasattr(self, '_statevector_checked'):
            print(f"🔍 Statevector: {len(statevector.data)} components")
            print(f"🔍 Bound circuit: {bound_circuit.num_qubits} qubits, depth {bound_circuit.depth()}")
            self._statevector_checked = True

        # Get n_qubits from ansatz
        n_qubits = self.ansatz.n_qubits if hasattr(self.ansatz, 'n_qubits') else 2 * self.hamiltonian.n_orbitals

        # CRITICAL PERFORMANCE IMPROVEMENT: Use sparse Pauli operators instead of dense matrices
        # Check if Hamiltonian has sparse method (covalent, ionic, molecular hamiltonians)
        # CRITICAL BUG FIX: Must check if cached Hamiltonian matches current mapper!

        # Determine mapper type to pass to Hamiltonian
        mapper_name = getattr(self, 'mapper_type', 'jordan_wigner')
        if mapper_name.lower() in ['bravyikitaevmapper', 'bravyi_kitaev']:
            mapper_arg = 'bravyi_kitaev'
        else:
            mapper_arg = 'jordan_wigner'

        # Check if we need to rebuild Hamiltonian (cache miss or mapper changed)
        cache_invalid = (self._sparse_pauli_op is None or
                        self._cached_mapper != mapper_arg)

        # Check if Hamiltonian is already a SparsePauliOp
        from qiskit.quantum_info import SparsePauliOp
        if isinstance(self.hamiltonian, SparsePauliOp):
            # Already sparse - just use it directly
            if self._sparse_pauli_op is None:
                self._sparse_pauli_op = self.hamiltonian
                self._cached_mapper = mapper_arg
                self._use_sparse = True
                # A directly-supplied SparsePauliOp has num_qubits matching the
                # ansatz, so no statevector padding is required. Set this flag
                # explicitly to avoid an AttributeError on the sparse fast path.
                self._needs_padding = False
                print(f"📊 Using provided SparsePauliOp: {len(self._sparse_pauli_op)} Pauli terms")
        elif hasattr(self.hamiltonian, 'to_sparse_hamiltonian') and cache_invalid:
            # Build sparse Pauli operator (FAST, memory-efficient)
            logger.info(f"Building sparse Pauli Hamiltonian with {mapper_arg} mapper")
            print(f"🔧 Building sparse Pauli Hamiltonian with {mapper_arg} mapper")

            # Build Hamiltonian with correct mapper
            self._sparse_pauli_op = self.hamiltonian.to_sparse_hamiltonian(mapper=mapper_arg)
            self._cached_mapper = mapper_arg  # Update cache tracker
            self._use_sparse = True

            print(f"📊 Sparse Hamiltonian built: {len(self._sparse_pauli_op)} Pauli terms")

            # DEBUG: Check identity coefficient
            identity_coeff = 0.0
            for i, pauli_str in enumerate(self._sparse_pauli_op.paulis):
                if all(c == 'I' for c in str(pauli_str)):
                    identity_coeff += self._sparse_pauli_op.coeffs[i]
            print(f"🔍 VQE Hamiltonian identity coefficient: {identity_coeff:.8f} Ha")
            print(f"🔍 Nuclear repulsion from molecule: {self.hamiltonian.nuclear_repulsion:.8f} Ha")

            # Check qubit count consistency
            if self._sparse_pauli_op.num_qubits != n_qubits:
                # Fail fast. Silently mutating self.ansatz.n_qubits here leaves
                # the ansatz inconsistent with its already-built circuit and
                # changes n_parameters mid-optimization (breaking the in-flight
                # parameter vector). The ansatz and Hamiltonian must be built for
                # the same number of qubits before constructing the solver.
                raise ValueError(
                    f"Qubit-count mismatch: Hamiltonian has "
                    f"{self._sparse_pauli_op.num_qubits} qubits but ansatz has "
                    f"{n_qubits}. The ansatz and Hamiltonian must be built for the "
                    f"same number of qubits. Rebuild the ansatz with "
                    f"n_qubits={self._sparse_pauli_op.num_qubits} before "
                    f"constructing VQESolver (the high-level bond=/molecule= API "
                    f"does this automatically)."
                )
            else:
                print(f"✅ Qubit counts match: {n_qubits} qubits")
                self._needs_padding = False

        # Compute energy using sparse or dense method
        if hasattr(self, '_use_sparse') and self._use_sparse:
            # FAST PATH: Sparse Pauli operator
            # Pad statevector if needed
            if getattr(self, '_needs_padding', False):
                psi = statevector.data
                psi_padded = np.zeros(2 ** self._full_qubits, dtype=complex)
                psi_padded[:len(psi)] = psi
                statevector_padded = Statevector(psi_padded)
                energy = statevector_padded.expectation_value(self._sparse_pauli_op).real
            else:
                # Direct expectation value computation (FAST!)
                if getattr(self, 'backend_name', '') == 'planck':
                    # contract <psi|H|psi> on-GPU against the state built above
                    from kanad.backends.planck_adapter import expectation as _planck_expect
                    energy = _planck_expect(self._planck_sv, self._sparse_pauli_op)
                else:
                    energy = statevector.expectation_value(self._sparse_pauli_op).real

                # DEBUG: Print first few energy evaluations
                if not hasattr(self, '_debug_counter'):
                    self._debug_counter = 0
                if self._debug_counter < 3:
                    print(f"🔍 Energy evaluation {self._debug_counter + 1}: {energy:.8f} Ha")
                    self._debug_counter += 1
        else:
            # FALLBACK: Dense matrix path (only for small test systems)
            logger.warning("Using SLOW dense matrix Hamiltonian - consider using sparse method")

            # Get Hamiltonian matrix if not cached
            if self._hamiltonian_matrix is None:
                # MEMORY SAFETY CHECK
                # Handle different Hamiltonian types
                from qiskit.quantum_info import SparsePauliOp
                if isinstance(self.hamiltonian, SparsePauliOp):
                    # SparsePauliOp passed directly (from tests)
                    # Get n_qubits from ansatz or Hamiltonian
                    if hasattr(self, 'circuit') and self.circuit:
                        full_n_qubits = self.circuit.num_qubits
                    elif hasattr(self.ansatz, 'n_qubits'):
                        full_n_qubits = self.ansatz.n_qubits
                    else:
                        # Get from Hamiltonian
                        full_n_qubits = self.hamiltonian.num_qubits
                elif hasattr(self.hamiltonian, 'n_orbitals'):
                    # MolecularHamiltonian
                    full_n_qubits = 2 * self.hamiltonian.n_orbitals
                else:
                    raise TypeError(
                        f"Unsupported Hamiltonian type: {type(self.hamiltonian)}. "
                        f"Expected MolecularHamiltonian or SparsePauliOp"
                    )

                required_memory_gb = (2 ** full_n_qubits) ** 2 * 16 / 1e9

                if required_memory_gb > 16:  # 16 GB limit
                    raise MemoryError(
                        f"Dense Hamiltonian matrix requires {required_memory_gb:.1f} GB RAM!\n"
                        f"System: {self.hamiltonian.n_orbitals} orbitals → {full_n_qubits} qubits\n"
                        f"Matrix size: {2**full_n_qubits} × {2**full_n_qubits} = {(2**full_n_qubits)**2:,} elements\n"
                        f"SOLUTION: Your Hamiltonian should implement to_sparse_hamiltonian() method"
                    )

                logger.info(f"Building dense Hamiltonian matrix ({required_memory_gb:.2f} GB)")

                # Handle different Hamiltonian types for matrix conversion
                from qiskit.quantum_info import SparsePauliOp
                if isinstance(self.hamiltonian, SparsePauliOp):
                    # SparsePauliOp can be converted to matrix directly
                    self._hamiltonian_matrix = self.hamiltonian.to_matrix()
                    self._needs_padding = False
                else:
                    # Check if to_matrix supports n_qubits parameter
                    import inspect
                    to_matrix_sig = inspect.signature(self.hamiltonian.to_matrix)
                    has_n_qubits_param = 'n_qubits' in to_matrix_sig.parameters

                    if has_n_qubits_param:
                        self._hamiltonian_matrix = self.hamiltonian.to_matrix(n_qubits=full_n_qubits, use_mo_basis=True)
                        self._needs_padding = (n_qubits < full_n_qubits)
                        self._ansatz_qubits = n_qubits
                        self._full_qubits = full_n_qubits
                    else:
                        # Simple test Hamiltonian
                        H_core = self.hamiltonian.to_matrix()
                        dim = 2 ** n_qubits
                        self._hamiltonian_matrix = np.kron(H_core, np.eye(dim // H_core.shape[0]))
                        self._needs_padding = False

            # Pad statevector if needed
            psi = statevector.data
            if hasattr(self, '_needs_padding') and self._needs_padding:
                psi_padded = np.zeros(2 ** self._full_qubits, dtype=complex)
                psi_padded[:len(psi)] = psi
                psi = psi_padded

            # Compute expectation value: E = <psi|H|psi>
            energy = np.real(np.conj(psi) @ self._hamiltonian_matrix @ psi)

        return float(energy)

    def _compute_energy_from_counts(self, counts: dict, pauli_hamiltonian) -> float:
        """Estimate ⟨ψ|H|ψ⟩ from Z-basis measurement counts.

        Delegates to the single-source core estimator
        ``core.error_mitigation.expectation_from_counts`` (reorg B5), replacing the
        former inline implementation. The inline version was numerically correct
        (it reversed BOTH the bitstring and the Pauli string — a self-consistent
        double reversal) but (a) did not strip IBM register-space bitstrings and
        (b) duplicated the estimator. The canonical version strips register spaces
        and raises NotImplementedError on ANY X/Y Pauli term (X/Y cannot be read
        from Z-basis counts). The sole caller (the BlueQubit-counts branch in
        ``_compute_energy_quantum``) wraps this in a try/except that falls back to
        the statevector energy, so an X/Y-bearing molecular Hamiltonian routes to
        the exact statevector path rather than a silently-truncated counts energy.

        Args:
            counts: Measurement counts dict (bitstring -> count).
            pauli_hamiltonian: SparsePauliOp representing the Hamiltonian.

        Returns:
            Estimated energy expectation value.
        """
        from kanad.core.error_mitigation import expectation_from_counts
        return expectation_from_counts(pauli_hamiltonian, counts)

    def _compute_energy_quantum(self, parameters: np.ndarray) -> float:
        """
        Compute energy using quantum backend (sampling-based).

        Supports IBM Quantum and BlueQubit backends.
        """
        from qiskit.quantum_info import SparsePauliOp

        # Build circuit if not already built
        if self.ansatz.circuit is None:
            self.ansatz.build_circuit()

        # Resolve circuit (Kanad or Qiskit-native) and bind parameters.
        circ = self.ansatz.circuit
        if hasattr(circ, 'bind_parameters'):
            try:
                circ.bind_parameters(parameters)
            except TypeError:
                pass
        if hasattr(circ, 'to_qiskit'):
            qiskit_circuit = circ.to_qiskit()
        else:
            qiskit_circuit = circ

        # Bind parameters if circuit has them
        if qiskit_circuit.num_parameters > 0:
            param_dict = {qiskit_circuit.parameters[i]: parameters[i]
                         for i in range(len(parameters))}
            bound_circuit = qiskit_circuit.assign_parameters(param_dict)
        else:
            bound_circuit = qiskit_circuit

        # Get Pauli representation of Hamiltonian
        try:
            from kanad.core.hamiltonians.pauli_converter import PauliConverter
            pauli_hamiltonian = PauliConverter.to_sparse_pauli_op(
                self.hamiltonian,
                self.mapper,
                use_qiskit_nature=True
            )
        except Exception as e:
            logger.error(f"Failed to convert Hamiltonian to Pauli operators: {e}")
            logger.warning("Falling back to statevector simulation")
            import traceback
            traceback.print_exc()
            return self._compute_energy_statevector(parameters)

        # Use IBM backend if available
        if hasattr(self, '_ibm_backend') and self._ibm_backend is not None:
            logger.info(f"Submitting to IBM Quantum (iteration {self.iteration_count})")

            # Broadcast to frontend if experiment_id is available
            if self.experiment_id:
                try:
                    from api.utils import broadcast_log_sync
                    broadcast_log_sync(self.experiment_id, f"🚀 Submitting job to IBM Quantum (function eval {self.iteration_count})")
                except Exception:
                    print(f"🚀 Submitting job to IBM Quantum (function eval {self.iteration_count})")
            else:
                print(f"🚀 Submitting job to IBM Quantum (function eval {self.iteration_count})")

            try:
                # Submit job to IBM
                result = self._ibm_backend.run_batch(
                    circuits=[bound_circuit],
                    observables=[pauli_hamiltonian],
                    shots=self.shots
                )

                job_id = result['job_id']
                logger.info(f"IBM job submitted: {job_id}")

                # Track cloud job information
                self.cloud_job_ids.append(job_id)
                self.cloud_provider = 'ibm'
                self.execution_mode = 'batch'  # IBM uses batch mode

                # Broadcast job ID to frontend
                if self.experiment_id:
                    try:
                        from api.utils import broadcast_log_sync
                        broadcast_log_sync(self.experiment_id, f"✅ IBM job submitted: {job_id}")
                        broadcast_log_sync(self.experiment_id, f"🔗 Track job at: https://quantum.ibm.com/jobs/{job_id}")
                    except Exception:
                        print(f"✅ IBM job submitted: {job_id}")
                        print(f"🔗 Track job at: https://quantum.ibm.com/jobs/{job_id}")
                else:
                    print(f"✅ IBM job submitted: {job_id}")
                    print(f"🔗 Track job at: https://quantum.ibm.com/jobs/{job_id}")

                # Wait for job to complete with cancellation checking
                job = self._ibm_backend.service.job(job_id)
                logger.info(f"Waiting for IBM job {job_id}...")

                if self.experiment_id:
                    try:
                        from api.utils import broadcast_log_sync
                        broadcast_log_sync(self.experiment_id, f"⏳ Waiting for IBM job to complete...")
                    except Exception:
                        print(f"⏳ Waiting for IBM job to complete...")
                else:
                    print(f"⏳ Waiting for IBM job to complete...")

                # Poll job status with cancellation checks (instead of blocking job.result())
                import time
                poll_interval = 5  # Check every 5 seconds
                job_result = None

                while True:
                    # Check for cancellation FIRST before checking job status
                    if self.experiment_id and self.job_id:
                        try:
                            from api.services.experiment_service import check_cancellation
                            check_cancellation(self.experiment_id, self.job_id)
                        except Exception as cancel_error:
                            # Cancellation detected - cancel IBM job and raise
                            logger.warning(f"Cancellation detected, cancelling IBM job {job_id}...")
                            try:
                                from api.utils import broadcast_log_sync
                                broadcast_log_sync(self.experiment_id, f"🚫 Cancelling IBM job {job_id}...")
                            except:
                                pass

                            try:
                                self._ibm_backend.cancel_job(job_id)
                                logger.info(f"IBM job {job_id} cancelled successfully")
                            except Exception as e:
                                logger.error(f"Failed to cancel IBM job {job_id}: {e}")

                            # Re-raise the cancellation exception
                            raise cancel_error

                    # Check job status
                    status = self._ibm_backend.get_job_status(job_id)
                    logger.debug(f"IBM job {job_id} status: {status}")

                    # Check if job is done (handles different status formats)
                    status_upper = str(status).upper()
                    if 'DONE' in status_upper or 'COMPLETED' in status_upper:
                        logger.info(f"IBM job {job_id} completed, retrieving result...")
                        job_result = job.result()
                        break
                    elif 'ERROR' in status_upper or 'CANCELLED' in status_upper or 'FAILED' in status_upper:
                        logger.error(f"IBM job {job_id} failed with status: {status}")
                        raise RuntimeError(f"IBM job failed: {status}")

                    # Job still running, wait before next poll
                    time.sleep(poll_interval)

                # Extract energy from Estimator result
                # Handle both V1 and V2 primitive formats
                if hasattr(job_result, 'values'):
                    # V1 Estimator result format
                    energy = float(job_result.values[0])
                elif hasattr(job_result, '__getitem__'):
                    # V2 Estimator result format (EstimatorV2)
                    # Result is a PrimitiveResult object with PubResult items
                    # Each PubResult has .data.evs (expectation values)
                    energy = float(job_result[0].data.evs[0])
                else:
                    raise AttributeError(f"Unknown Estimator result format: {type(job_result)}")

                logger.info(f"IBM job {job_id} completed: E = {energy:.8f} Ha")
                return energy

            except Exception as e:
                # Check if this is a cancellation exception - if so, re-raise it
                if 'ExperimentCancelledException' in type(e).__name__ or 'cancelled' in str(e).lower():
                    logger.warning(f"🚫 Cancellation detected during IBM backend execution - stopping")
                    raise  # Re-raise cancellation exception to stop optimizer

                # For other errors, fall back to statevector
                logger.error(f"IBM backend execution failed: {e}")
                logger.warning("Falling back to statevector simulation")
                return self._compute_energy_statevector(parameters)

        # Use BlueQubit backend if available
        elif hasattr(self, '_bluequbit_backend') and self._bluequbit_backend is not None:
            logger.info(f"Submitting to BlueQubit (iteration {self.iteration_count})")

            # Broadcast to frontend if experiment_id is available
            if self.experiment_id:
                try:
                    from api.utils import broadcast_log_sync
                    broadcast_log_sync(self.experiment_id, f"🚀 Submitting job to BlueQubit (function eval {self.iteration_count})")
                except Exception:
                    print(f"🚀 Submitting job to BlueQubit (function eval {self.iteration_count})")
            else:
                print(f"🚀 Submitting job to BlueQubit (function eval {self.iteration_count})")

            try:
                # For BlueQubit CPU/MPS devices, use statevector mode for accuracy
                # (counts-based measurement requires basis rotation which is complex)
                use_shots = self.shots
                if hasattr(self._bluequbit_backend, 'device'):
                    device = self._bluequbit_backend.device
                    if device in ['cpu', 'mps.cpu', 'mps.gpu']:
                        use_shots = None  # Force statevector mode
                        logger.info(f"Using statevector mode for {device} (more accurate than sampling)")

                # Submit to BlueQubit (asynchronous to allow cancellation)
                job_info = self._bluequbit_backend.run_circuit(
                    circuit=bound_circuit,
                    shots=use_shots,  # None = statevector mode
                    asynchronous=True  # Changed to async for cancellation support
                )

                job_id = job_info['job_id']
                logger.info(f"BlueQubit job submitted: {job_id}")

                # Track cloud job information
                self.cloud_job_ids.append(job_id)
                self.cloud_provider = 'bluequbit'

                # Broadcast job submission
                if self.experiment_id:
                    try:
                        from api.utils import broadcast_log_sync
                        broadcast_log_sync(self.experiment_id, f"✅ BlueQubit job submitted: {job_id}")
                        broadcast_log_sync(self.experiment_id, f"🔗 Track job at: https://app.bluequbit.io/jobs/{job_id}")
                        broadcast_log_sync(self.experiment_id, f"⏳ Waiting for BlueQubit job to complete...")
                    except Exception:
                        print(f"✅ BlueQubit job submitted: {job_id}")
                        print(f"🔗 Track job at: https://app.bluequbit.io/jobs/{job_id}")
                else:
                    print(f"✅ BlueQubit job submitted: {job_id}")
                    print(f"🔗 Track job at: https://app.bluequbit.io/jobs/{job_id}")

                # Check for cancellation before waiting for job
                # (Note: BlueQubit's wait_for_job is blocking, so we check before calling it)
                if self.experiment_id and self.job_id:
                    try:
                        from api.services.experiment_service import check_cancellation
                        check_cancellation(self.experiment_id, self.job_id)
                    except Exception as cancel_error:
                        # Cancellation detected - cancel BlueQubit job and raise
                        logger.warning(f"Cancellation detected, cancelling BlueQubit job {job_id}...")
                        try:
                            from api.utils import broadcast_log_sync
                            broadcast_log_sync(self.experiment_id, f"🚫 Cancelling BlueQubit job {job_id}...")
                        except:
                            pass

                        try:
                            self._bluequbit_backend.cancel_job(job_id)
                            logger.info(f"BlueQubit job {job_id} cancelled successfully")
                        except Exception as e:
                            logger.error(f"Failed to cancel BlueQubit job {job_id}: {e}")

                        # Re-raise the cancellation exception
                        raise cancel_error

                # Wait for job to complete (blocking call)
                result = self._bluequbit_backend.wait_for_job(job_id)
                logger.info(f"BlueQubit job {job_id} completed successfully")

                # Extract statevector or counts
                if 'statevector' in result:
                    statevector = np.array(result['statevector'])

                    # Convert Pauli operator to matrix
                    pauli_matrix = pauli_hamiltonian.to_matrix()

                    # Compute expectation value: <ψ|H|ψ>
                    energy = float(np.real(np.conj(statevector) @ pauli_matrix @ statevector))

                elif 'counts' in result:
                    # Sampling-based energy estimation
                    counts = result['counts']
                    logger.info(f"Computing energy from counts: {sum(counts.values())} total shots")

                    # Compute expectation value from Pauli measurements
                    energy = self._compute_energy_from_counts(
                        counts=counts,
                        pauli_hamiltonian=pauli_hamiltonian
                    )
                    logger.info(f"Counts-based energy: {energy:.8f} Ha")
                else:
                    raise ValueError("BlueQubit result missing statevector or counts")

                logger.info(f"BlueQubit energy: {energy:.8f} Ha")
                return energy

            except Exception as e:
                # Check if this is a cancellation exception - if so, re-raise it
                if 'ExperimentCancelledException' in type(e).__name__ or 'cancelled' in str(e).lower():
                    logger.warning(f"🚫 Cancellation detected during BlueQubit backend execution - stopping")
                    raise  # Re-raise cancellation exception to stop optimizer

                # For other errors, fall back to statevector
                logger.error(f"BlueQubit backend execution failed: {e}")
                logger.warning("Falling back to statevector simulation")
                return self._compute_energy_statevector(parameters)

        elif hasattr(self, '_ionq_backend') and self._ionq_backend is not None:
            # IonQ cloud backend — uses IonQ simulator or QPU via Qiskit IonQ
            try:
                # Reuse the local pauli_hamiltonian and bound_circuit already
                # computed above (mirrors the IBM/BlueQubit branches).
                qc_qiskit = bound_circuit.to_qiskit() if hasattr(bound_circuit, 'to_qiskit') else bound_circuit
                shots = getattr(self, 'shots', None) or 8192

                energy = self._ionq_backend.get_expectation_value(
                    circuit=qc_qiskit,
                    observable=pauli_hamiltonian,
                    shots=shots,
                )
                logger.info(f"IonQ energy: {energy:.8f} Ha")
                return energy
            except Exception as e:
                logger.error(f"IonQ backend execution failed: {e}")
                logger.warning("Falling back to statevector simulation")
                return self._compute_energy_statevector(parameters)

        else:
            logger.warning("No quantum backend available, using statevector")
            return self._compute_energy_statevector(parameters)

    def compute_energy(self, parameters: np.ndarray) -> float:
        """
        Public API: Compute energy expectation value for given parameters.

        Args:
            parameters: Circuit parameters

        Returns:
            Energy expectation value (Hartree)
        """
        return self._compute_energy(parameters)

    def get_energy_variance(self, parameters: np.ndarray) -> float:
        """
        Compute energy variance for given parameters.

        Variance = <H²> - <H>²

        Args:
            parameters: Circuit parameters

        Returns:
            Energy variance
        """
        from qiskit.quantum_info import Statevector

        # Build circuit if not already built
        if self.ansatz.circuit is None:
            self.ansatz.build_circuit()

        # Bind parameters / convert to Qiskit. Different ansatze return either a
        # Kanad circuit (has .bind_parameters + .to_qiskit) or a raw Qiskit
        # QuantumCircuit (assign_parameters only; .bind_parameters was removed
        # in Qiskit 2.x).
        circ = self.ansatz.circuit
        if hasattr(circ, 'bind_parameters'):
            try:
                circ.bind_parameters(parameters)
            except TypeError:
                pass
        if hasattr(circ, 'to_qiskit'):
            qiskit_circuit = circ.to_qiskit()
        else:
            qiskit_circuit = circ

        # If circuit still has parameters, bind them
        if qiskit_circuit.num_parameters > 0:
            param_dict = {qiskit_circuit.parameters[i]: parameters[i] for i in range(len(parameters))}
            bound_circuit = qiskit_circuit.assign_parameters(param_dict)
        else:
            bound_circuit = qiskit_circuit

        # Get statevector from circuit
        statevector = Statevector.from_instruction(bound_circuit)
        psi = statevector.data

        # Get Hamiltonian matrix if not cached
        if self._hamiltonian_matrix is None:
            # Get n_qubits from ansatz (more reliable than 2*n_orbitals)
            n_qubits = self.ansatz.n_qubits if hasattr(self.ansatz, 'n_qubits') else 2 * self.hamiltonian.n_orbitals

            # Check if to_matrix supports n_qubits parameter (for real hamiltonians)
            import inspect
            to_matrix_sig = inspect.signature(self.hamiltonian.to_matrix)
            has_n_qubits_param = 'n_qubits' in to_matrix_sig.parameters

            if has_n_qubits_param:
                # WORKAROUND: For hardware-efficient ansätze with fewer qubits than full space,
                # pad the statevector to match full Hamiltonian dimension
                full_n_qubits = 2 * self.hamiltonian.n_orbitals
                self._hamiltonian_matrix = self.hamiltonian.to_matrix(n_qubits=full_n_qubits, use_mo_basis=True)
                self._needs_padding = (n_qubits < full_n_qubits)
                self._ansatz_qubits = n_qubits
                self._full_qubits = full_n_qubits
            else:
                # Simple test Hamiltonian - just call to_matrix() without parameters
                H_core = self.hamiltonian.to_matrix()
                # Expand to full qubit space (2^n_qubits x 2^n_qubits)
                dim = 2 ** n_qubits
                self._hamiltonian_matrix = np.kron(H_core, np.eye(dim // H_core.shape[0]))
                self._needs_padding = False

        # Pad statevector if needed (for hardware-efficient ansätze with fewer qubits)
        if hasattr(self, '_needs_padding') and self._needs_padding:
            psi_padded = np.zeros(2 ** self._full_qubits, dtype=complex)
            psi_padded[:len(psi)] = psi
            psi = psi_padded

        # Compute <H>
        H_expectation = np.real(np.conj(psi) @ self._hamiltonian_matrix @ psi)

        # Compute <H²>
        H_squared = self._hamiltonian_matrix @ self._hamiltonian_matrix
        H2_expectation = np.real(np.conj(psi) @ H_squared @ psi)

        # Variance = <H²> - <H>²
        variance = H2_expectation - H_expectation**2

        return float(variance)

    def _objective_function(self, parameters: np.ndarray) -> float:
        """
        Objective function for classical optimization.

        Returns the augmented loss (energy + symmetry penalty if enabled).
        `energy_history` keeps the *raw* ⟨H⟩ trace; `loss_history` and
        `penalty_history` keep the augmented quantities for diagnostics.

        Args:
            parameters: Current parameters

        Returns:
            Loss value to minimize (energy + λ_N·var_N + λ_Sz·var_Sz)
        """
        energy = self._compute_energy(parameters)
        penalty = self._compute_penalty(parameters)
        loss = energy + penalty

        # Track history (energy = raw ⟨H⟩; penalty + loss = augmented)
        self.energy_history.append(energy)
        self.penalty_history.append(penalty)
        self.loss_history.append(loss)
        self.parameter_history.append(parameters.copy())
        self.iteration_count += 1

        # Detect optimizer iterations: energy changes significantly or it's the first eval
        # This heuristic detects when optimizer completes an iteration and starts a new one
        energy_changed = (self._last_energy is None or
                         abs(energy - self._last_energy) > 1e-12)

        if energy_changed and self.iteration_count > 1:
            self.optimizer_iteration += 1
            self._last_energy = energy
        elif self.iteration_count == 1:
            self.optimizer_iteration = 1
            self._last_energy = energy

        # CRITICAL FIX: Enforce max_iterations limit via function evaluation count
        # Use iteration_count (function evals) not optimizer_iteration (broken heuristic)
        # This is the actual number of calls to the objective function
        if self.iteration_count >= self.max_iterations:
            logger.info(f"🛑 Max iterations ({self.max_iterations}) reached at {self.iteration_count} function evals, stopping optimizer")
            # Raise exception to force optimizer to stop
            raise StopIteration(f"Maximum iterations ({self.max_iterations}) reached")

        # Call user callback if provided (used for progress broadcasting from API layer)
        # Pass: (optimizer_iteration, energy, parameters, function_eval_count)
        if hasattr(self, '_callback') and self._callback is not None:
            try:
                # Check if callback accepts 4 arguments (new signature)
                import inspect
                sig = inspect.signature(self._callback)
                if len(sig.parameters) >= 4:
                    # New signature: (iteration, energy, parameters, function_evals)
                    self._callback(self.optimizer_iteration, energy, parameters, self.iteration_count)
                else:
                    # Old signature: (iteration, energy, parameters) - pass optimizer iteration
                    self._callback(self.optimizer_iteration, energy, parameters)
            except Exception as e:
                # Check if this is a cancellation exception - if so, re-raise it to stop optimizer
                if 'ExperimentCancelledException' in type(e).__name__ or 'cancelled' in str(e).lower():
                    logger.warning(f"🚫 Experiment cancelled - stopping optimizer")
                    raise  # Re-raise to stop optimizer
                else:
                    # For other callback errors, log but don't stop the optimizer
                    print(f"⚠️ Callback failed: {e}")
                    import traceback
                    traceback.print_exc()
        elif self.iteration_count == 1:
            # Debug: print on first iteration if callback is missing
            print(f"🔍 DEBUG: _callback exists={hasattr(self, '_callback')}, is None={getattr(self, '_callback', 'MISSING') is None}")

        # Log progress (every 10 function evals)
        if self.iteration_count % 10 == 0:
            # Estimate optimizer iteration (rough approximation)
            est_iter = self.iteration_count // 40 if self.optimizer_method in ['SLSQP', 'L-BFGS-B'] else self.iteration_count
            if penalty != 0.0:
                logger.info(
                    f"Function eval {self.iteration_count} (~iter {est_iter}): "
                    f"E = {energy:.8f} Ha, penalty = {penalty:.3e}"
                )
                print(
                    f"📊 Progress: {self.iteration_count} function evals (~{est_iter} iterations), "
                    f"E = {energy:.8f} Ha, penalty = {penalty:.3e}"
                )
            else:
                logger.info(f"Function eval {self.iteration_count} (~iter {est_iter}): E = {energy:.8f} Ha")
                print(f"📊 Progress: {self.iteration_count} function evals (~{est_iter} iterations), E = {energy:.8f} Ha")

        # Optimizer receives the augmented loss when penalty is enabled.
        return loss

    def _spsa_minimize(self, initial_parameters: np.ndarray) -> tuple:
        """
        SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

        Key advantage: Only 2 function evaluations per iteration regardless of parameter count!
        Perfect for cloud quantum backends where function evaluations are expensive.

        Args:
            initial_parameters: Initial parameter values

        Returns:
            (optimized_parameters, final_energy)
        """
        params = initial_parameters.copy()
        best_energy = float('inf')
        best_params = params.copy()

        # SPSA hyperparameters (standard values from Spall 1998)
        a = 0.16  # Step size coefficient
        c = 0.1   # Perturbation size coefficient
        A = 0.1 * self.max_iterations  # Stability constant
        alpha = 0.602  # Step size decay
        gamma = 0.101  # Perturbation decay

        prev_energy = None

        for k in range(self.max_iterations):
            # Compute decay schedules
            a_k = a / (k + 1 + A)**alpha
            c_k = c / (k + 1)**gamma

            # Random perturbation direction (Bernoulli ±1)
            delta = 2 * np.random.randint(0, 2, size=len(params)) - 1

            # Evaluate at perturbed points (only 2 evaluations!)
            params_plus = params + c_k * delta
            params_minus = params - c_k * delta

            try:
                energy_plus = self._objective_function(params_plus)
                energy_minus = self._objective_function(params_minus)
            except StopIteration as e:
                # Max-iterations limiter (enforced in _objective_function) was
                # hit. Stop and return the best state tracked so far.
                logger.info(f"SPSA stopped at max iterations: {e}")
                break

            # Gradient approximation
            gradient = (energy_plus - energy_minus) / (2 * c_k) * delta

            # Update parameters
            params = params - a_k * gradient

            # Track best result
            current_energy = min(energy_plus, energy_minus)
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = params_plus if energy_plus < energy_minus else params_minus

            # Check convergence
            if prev_energy is not None:
                energy_change = abs(current_energy - prev_energy)
                if energy_change < self.conv_threshold:
                    print(f"✅ SPSA converged at iteration {k+1}")
                    break

            prev_energy = current_energy

            if (k + 1) % 5 == 0:
                print(f"  SPSA iter {k+1}/{self.max_iterations}: E = {best_energy:.8f} Ha")

        return best_params, best_energy

    def solve(self, initial_parameters: Optional[np.ndarray] = None, callback: Optional[callable] = None) -> 'SolverResult':
        """Solve for the ground state and return a unified :class:`SolverResult`.

        The full legacy result dict is preserved on ``self.results`` and mapped
        into the result's stable core + ``extra`` (parameters, telemetry, RDMs,
        ...). ``mode='hivqe'`` is tagged ``solver='ci'``.
        """
        from kanad.core.solver_result import SolverResult
        raw = self._solve_dispatch(initial_parameters, callback)
        if isinstance(raw, SolverResult):
            return raw
        tag = 'ci' if raw.get('mode') == 'hivqe' else 'vqe'
        return SolverResult.from_mapping(raw, solver=tag, backend=self.backend_name)

    def _solve_dispatch(self, initial_parameters: Optional[np.ndarray] = None, callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Solve for ground state energy using VQE or Hi-VQE.

        Args:
            initial_parameters: Initial parameter guess (random if None)
            callback: Optional callback function(iteration, energy, params)

        Returns:
            Dictionary with comprehensive results:
                - energy: Ground state energy (Hartree)
                - parameters: Optimized parameters (None for Hi-VQE)
                - converged: Convergence status
                - iterations: Number of iterations
                - hf_energy: Hartree-Fock reference energy
                - correlation_energy: E_VQE - E_HF
                - energy_history: Energy at each iteration
                - analysis: Detailed analysis (if enabled)
                - optimization_stats: Circuit optimization stats (if enabled)
                - mode: 'standard' or 'hivqe'
                - hivqe_stats: Hi-VQE specific stats (if mode='hivqe')
        """
        # Store callback (only if explicitly provided, don't overwrite __init__ callback)
        if callback is not None:
            self._callback = callback

        # Route to appropriate solver based on mode
        if self.mode == 'hivqe':
            # Check if hamiltonian supports Hi-VQE (needs num_qubits attribute)
            can_use_hivqe = False
            if hasattr(self, 'bond') and self.bond is not None:
                can_use_hivqe = True
            elif hasattr(self, 'hamiltonian'):
                ham = self.hamiltonian
                if hasattr(ham, 'to_sparse_hamiltonian') or hasattr(ham, 'num_qubits'):
                    can_use_hivqe = True

            if can_use_hivqe:
                logger.info("🔥 Starting Hi-VQE optimization...")
                return self._solve_hivqe()
            else:
                logger.warning(
                    "⚠️  Hi-VQE not supported for this hamiltonian type. "
                    "Falling back to standard VQE."
                )
                return self._solve_standard_vqe(initial_parameters)
        else:
            logger.info("🔬 Standard quantum VQE mode - real quantum computation")
            logger.info("Starting VQE optimization...")
            return self._solve_standard_vqe(initial_parameters)

    def _solve_standard_vqe(self, initial_parameters: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Standard VQE solve with variational optimization."""
        import time
        _solve_start_time = time.time()

        # Get HF reference energy
        hf_energy = self.get_reference_energy()
        if hf_energy is not None:
            logger.info(f"HF reference energy: {hf_energy:.8f} Hartree")

        # Initial parameters
        if initial_parameters is not None:
            self._init_strategy = 'user'
        else:
            # M2 PR-3 — try the parameter cache first
            cache_key, atom_syms, atom_coords = self._build_cache_key_if_possible()
            cached_run = None
            if self.use_cache and cache_key is not None:
                from kanad.core.cache import get_default_cache
                try:
                    cache = get_default_cache()
                    cached_run = cache.get(cache_key)
                    if cached_run is None and atom_coords is not None:
                        cached_run = cache.find_similar(
                            cache_key, atom_syms, atom_coords, max_rmsd_angstrom=0.1,
                        )
                        if cached_run is not None:
                            self._init_strategy = 'cached_similar'
                            logger.info(
                                f"🎯 Warm-start from cached nearby geometry "
                                f"(E={cached_run.final_energy:.6f} Ha)"
                            )
                    elif cached_run is not None:
                        self._init_strategy = 'cached'
                        logger.info(
                            f"🎯 Warm-start from exact cache hit "
                            f"(E={cached_run.final_energy:.6f} Ha)"
                        )
                except Exception as exc:
                    logger.debug(f"Cache lookup failed (non-fatal): {exc}")
                    cached_run = None

            if cached_run is not None and cached_run.parameters.shape[0] == self.n_parameters:
                # Add a tiny jitter so the optimizer escapes any saddle the
                # previous run happened to converge to.
                initial_parameters = cached_run.parameters.copy()
                initial_parameters += np.random.uniform(-1e-4, 1e-4, size=self.n_parameters)
                self._cache_hit = True
            elif hasattr(self.ansatz, 'get_smart_initial_params'):
                # MP2-based smart init (only for ansatze that support it)
                try:
                    logger.info(f"🎯 Using smart MP2-based initialization...")
                    initial_parameters = self.ansatz.get_smart_initial_params(hamiltonian=self.hamiltonian)
                    logger.info(f"✅ MP2 initialization successful (range: [{np.min(initial_parameters):.4f}, {np.max(initial_parameters):.4f}])")
                    self._init_strategy = 'mp2'
                except Exception as e:
                    logger.warning(f"MP2 initialization failed: {e}, falling back to random")
                    initial_parameters = np.random.uniform(-0.1, 0.1, size=self.n_parameters)
                    self._init_strategy = 'random'
            else:
                # Use uniform random initialization - provides good exploration
                # Range [-0.1, 0.1] is standard VQE practice
                initial_parameters = np.random.uniform(-0.1, 0.1, size=self.n_parameters)
                logger.info(f"Generated random initial parameters in range [-0.1, 0.1]")
                self._init_strategy = 'random'

        # Auto-select SPSA for cloud backends (huge cost savings)
        original_optimizer = self.optimizer_method
        if self.backend_name in ['ibm', 'bluequbit'] and self.optimizer_method not in ['SPSA', 'COBYLA', 'POWELL']:
            switch_msg = (
                f"☁️  CLOUD BACKEND OPTIMIZATION ☁️\n"
                f"   Backend: {self.backend_name}\n"
                f"   Original optimizer: {self.optimizer_method} (gradient-based)\n"
                f"   Auto-switching to: SPSA\n"
                f"   \n"
                f"   📉 Efficiency gain:\n"
                f"      {self.optimizer_method}: ~40 function evaluations/iteration\n"
                f"      SPSA: 2 function evaluations/iteration\n"
                f"      Expected speedup: 20x fewer quantum jobs\n"
                f"   \n"
                f"   💡 SPSA (Simultaneous Perturbation Stochastic Approximation) is\n"
                f"      specifically designed for noisy quantum hardware and cloud execution.\n"
                f"      It uses finite differences with random perturbations instead of gradients."
            )
            logger.warning(switch_msg)
            print(switch_msg)
            self.optimizer_method = 'SPSA'

            # Adjust max_iterations for SPSA (it typically needs fewer iterations)
            if self.max_iterations > 100:
                old_max_iter = self.max_iterations
                self.max_iterations = min(100, self.max_iterations)
                logger.info(f"📊 Adjusted max_iterations: {old_max_iter} → {self.max_iterations} (SPSA converges faster)")
                print(f"📊 Adjusted max_iterations: {old_max_iter} → {self.max_iterations}")

        logger.info(f"Optimizing {self.n_parameters} parameters using {self.optimizer_method}")

        # Reset tracking
        self.energy_history = []
        self.parameter_history = []
        self.iteration_count = 0  # Function evaluations
        self.optimizer_iteration = 0  # Real optimizer iterations
        self._last_energy = None  # For iteration detection

        # Classical optimization - Simple user-controlled approach
        # User sets max_iterations, optimizer uses natural convergence behavior
        # This is more predictable than trying to control function evaluations

        # CRITICAL FIX: Iteration limit enforced via callback in _objective_function
        # This works for ALL optimizers including COBYLA (which ignores maxfun in options)
        opt_options = {
            'maxiter': 10000,  # Set high, we enforce via callback
            'disp': False  # Suppress optimizer output
        }

        logger.info(f"Optimizer: {self.optimizer_method}, max_iterations: {self.max_iterations} (enforced via callback)")
        print(f"🔧 Optimizer: {self.optimizer_method} with {self.max_iterations} iterations")

        # Use SPSA for cloud backends or if explicitly requested
        if self.optimizer_method == 'SPSA':
            print(f"📊 Using SPSA: 2 function evaluations per iteration (efficient for cloud)")
            final_params, final_energy = self._spsa_minimize(initial_parameters)

            # Create scipy-like result object for consistency
            class SPSAResult:
                def __init__(self, x, fun, nit, success=True, message="SPSA converged"):
                    self.x = x
                    self.fun = fun
                    self.nit = nit
                    self.success = success
                    self.message = message

            result = SPSAResult(
                x=final_params,
                fun=final_energy,
                nit=self.optimizer_iteration,
                success=True,
                message=f"SPSA completed {self.optimizer_iteration} iterations"
            )
        else:
            # Standard scipy optimizers
            # M2 PR-1: detect gradient-capable scipy method and pass jac=
            # for parameter-shift Jacobian. The methods below all accept a
            # `jac` callable; others (COBYLA, Powell, Nelder-Mead) are
            # gradient-free.
            _GRADIENT_METHODS = (
                'L-BFGS-B', 'BFGS', 'CG', 'Newton-CG', 'SLSQP', 'TNC',
                'trust-ncg', 'trust-krylov', 'trust-exact',
            )
            # Gradient routing (in priority order):
            #   1. Adjoint-state gradient (M2.7) — one forward+backward pass per
            #      gradient call, O(N) total cost. Used when the ansatz
            #      declares `_supports_adjoint_gradient = True` (currently
            #      `GivensSDAnsatz` post-decompose).
            #   2. Parameter-shift gradient (M2 PR-1) — 2N energy evals per
            #      gradient call. Used when `_supports_parameter_shift = True`.
            #   3. scipy finite-difference — (N+1) evals per gradient call.
            #      Used as a fallback if neither flag is set.
            ansatz_supports_pshift = getattr(self.ansatz, '_supports_parameter_shift', True)
            ansatz_supports_adjoint = getattr(self.ansatz, '_supports_adjoint_gradient', False)
            optimizer_takes_jac = self.optimizer_method in _GRADIENT_METHODS
            self._gradient_mode = 'none'
            if optimizer_takes_jac and self._use_statevector and self.n_parameters > 0:
                if ansatz_supports_adjoint:
                    self._gradient_mode = 'adjoint'
                elif ansatz_supports_pshift:
                    self._gradient_mode = 'parameter_shift'
                else:
                    self._gradient_mode = 'scipy_finite_diff'

            if self._gradient_mode == 'adjoint':
                logger.info(
                    f"Using adjoint-state gradient for {self.optimizer_method} "
                    f"({self.n_parameters} parameters; one forward+backward pass per gradient call)"
                )
                print(f"⚡ Adjoint-state gradient enabled for {self.optimizer_method}")
            elif self._gradient_mode == 'parameter_shift':
                logger.info(
                    f"Using parameter-shift gradient for {self.optimizer_method} "
                    f"({self.n_parameters} parameters → {2*self.n_parameters} evals/grad call)"
                )
                print(f"⚡ Parameter-shift gradient enabled for {self.optimizer_method}")
            elif self._gradient_mode == 'scipy_finite_diff':
                logger.info(
                    f"Ansatz {type(self.ansatz).__name__} supports neither adjoint nor "
                    f"parameter-shift gradient; scipy finite-diff fallback."
                )

            minimize_kwargs = dict(
                fun=self._objective_function,
                x0=initial_parameters,
                method=self.optimizer_method,
                options=opt_options,
                tol=self.conv_threshold,
            )
            if self._gradient_mode == 'adjoint':
                minimize_kwargs['jac'] = self._compute_gradient_adjoint
            elif self._gradient_mode == 'parameter_shift':
                minimize_kwargs['jac'] = self._compute_gradient

            try:
                result = minimize(**minimize_kwargs)
            except StopIteration as e:
                # Max iterations reached - create result from current state
                logger.info(f"✅ Optimization stopped: {e}")
                class MaxIterResult:
                    def __init__(self, x, fun, nit, success=True, message="Max iterations reached"):
                        self.x = x
                        self.fun = fun
                        self.nit = nit
                        self.success = success
                        self.message = message

                # Select the BEST-by-loss point seen, not the last evaluated one.
                # L-BFGS-B's last evaluation may be a worse line-search trial than
                # the best iterate; loss_history is the augmented objective scipy
                # actually minimizes, so argmin over it gives the true optimum.
                if self.loss_history:
                    best_i = int(np.argmin(self.loss_history))
                    result = MaxIterResult(
                        x=self.parameter_history[best_i],
                        fun=self.loss_history[best_i],
                        nit=self.optimizer_iteration,
                        success=True,
                        message=f"Stopped at max iterations ({self.max_iterations})"
                    )
                else:
                    result = MaxIterResult(
                        x=initial_parameters,
                        fun=float('inf'),
                        nit=self.optimizer_iteration,
                        success=True,
                        message=f"Stopped at max iterations ({self.max_iterations})"
                    )

        # M2 PR-1: re-bind ansatz to the optimal θ* before returning so downstream
        # code (e.g. tests inspecting `solver.ansatz.circuit.to_qiskit()`) sees
        # the converged state. Without this, the ansatz holds whatever the LAST
        # parameter-shift evaluation bound (NOT θ*), because the kanad-side
        # `bind_parameters` mutates the ansatz's parameter values in-place.
        # We do one extra _compute_energy call at result.x to re-establish
        # the correct binding.
        #
        # _compute_energy returns the RAW variational ⟨H⟩ (the penalty is only
        # added inside _objective_function, never here). result.fun, on the
        # normal scipy path, is the AUGMENTED loss (energy + penalty), whereas
        # on the StopIteration path it is the raw energy. Capture the raw ⟨H⟩
        # here and report it as results['energy'] on BOTH paths so the two
        # return paths agree on the physical quantity, and keep the augmented
        # loss separately under 'loss'.
        variational_energy = result.fun  # fallback if rebind eval fails
        try:
            variational_energy = self._compute_energy(np.asarray(result.x))
        except Exception as exc:  # pragma: no cover — robustness only
            logger.debug(f"Optimal-θ rebind failed (non-fatal): {exc}")

        # Store results
        self.results = {
            'energy': variational_energy,
            'loss': result.fun,
            'parameters': result.x,
            'converged': result.success,
            'iterations': result.nit if hasattr(result, 'nit') else self.iteration_count,  # Use optimizer iterations, not function evals
            'function_evaluations': self.iteration_count,  # Track function evaluations separately
            'energy_history': np.array(self.energy_history),
            'parameter_history': np.array(self.parameter_history),
            'optimizer_message': result.message,
            'mode': 'standard'  # Indicate this was standard VQE
        }

        # M2 PR-3: telemetry — convergence metadata for the cache + future ML
        self.results['telemetry'] = {
            'n_function_evals': int(self.iteration_count),
            'n_gradient_calls': int(getattr(self, '_gradient_call_count', 0)),
            'init_strategy': self._init_strategy,
            'cache_hit': self._cache_hit,
            'final_n_variance': None,
            'final_s2_variance': None,
            'final_gradient_norm': None,
            'optimizer': self.optimizer_method,
        }
        # Compute residual symmetry variances + gradient norm at the optimal
        # point (these matter for both diagnostics and the cache record).
        try:
            from qiskit.quantum_info import SparsePauliOp
            if self._use_statevector and (
                self._n_penalty_op is not None
                or self._sz_penalty_op is not None
                or self._s2_penalty_op is not None
            ):
                # _compute_penalty already builds these — we just need the raw
                # residuals at result.x. The rebind above ensures the ansatz
                # holds result.x, so a single _compute_penalty call gives both.
                penalty_at_opt = self._compute_penalty(np.asarray(result.x))
                # Decompose into N and Sz components if both penalty ops exist
                # (compute the raw operator expectation values, not λ·var).
                self.results['telemetry']['penalty_at_optimum'] = float(penalty_at_opt)
        except Exception as exc:
            logger.debug(f"Telemetry collection failed (non-fatal): {exc}")

        # M2 PR-3: save the converged run to the cache.
        if self.use_cache and self._init_strategy != 'user':
            self._save_to_cache(result, _solve_start_time)

        # Add HF reference and correlation energy
        if hf_energy is not None:
            self.results['hf_energy'] = hf_energy
            self.results['correlation_energy'] = variational_energy - hf_energy

            logger.info(f"VQE energy: {variational_energy:.8f} Hartree")
            logger.info(f"Correlation energy: {variational_energy - hf_energy:.8f} Hartree")

            # Convergence guard (planck audit #3): a correlated method must land at
            # or below its HF reference. If VQE finished meaningfully ABOVE HF, the
            # optimizer diverged / under-converged (seen on larger HEA systems with a
            # weak optimizer budget — e.g. H₂O landing tens of Ha above HF, and the
            # planck/statevector trajectories diverging). Flag it so consumers don't
            # trust a non-physical energy. We DON'T silently "fix" the energy — we
            # surface that it's unreliable (raise the iteration budget or use
            # optimizer='L-BFGS-B', the gradient default).
            if variational_energy > hf_energy + 1e-4:
                self.results['converged'] = False
                warn_msg = (
                    f"VQE energy {variational_energy:.6f} Ha is ABOVE the HF reference "
                    f"{hf_energy:.6f} Ha (Δ=+{(variational_energy - hf_energy)*1000:.2f} mHa): "
                    f"the optimizer did not reach a correlated state. Result is unreliable — "
                    f"increase max_iterations or use optimizer='L-BFGS-B'."
                )
                self.results['convergence_warning'] = warn_msg
                logger.warning(warn_msg)

        # M3 PR-2: Quantum 1-RDM extraction. Run unconditionally in statevector
        # mode (independent of `enable_analysis`) so PropertyCalculator can
        # consume the wavefunction-derived density. The pre-M3 implementation
        # gated this on `enable_analysis=True` AND silently fell back to HF
        # via a `hasattr(hamiltonian, 'set_quantum_density_matrix')` check —
        # which passed `False` for the polyatomic `MolecularHamiltonian` path,
        # so every multi-atom dipole/NMR was HF in disguise.
        if self._use_statevector:
            try:
                from qiskit.quantum_info import Statevector
                from kanad.core.density import QuantumRDMExtractor

                if self.ansatz.circuit is None:
                    self.ansatz.build_circuit()
                circ = self.ansatz.circuit
                qiskit_circuit = circ.to_qiskit() if hasattr(circ, 'to_qiskit') else circ
                if qiskit_circuit.num_parameters > 0:
                    param_dict = {
                        qiskit_circuit.parameters[i]: float(result.x[i])
                        for i in range(len(result.x))
                    }
                    bound_circuit = qiskit_circuit.assign_parameters(param_dict)
                else:
                    bound_circuit = qiskit_circuit
                statevector = Statevector.from_instruction(bound_circuit)

                extractor = QuantumRDMExtractor(
                    n_orbitals=self.hamiltonian.n_orbitals,
                    n_electrons=self.hamiltonian.n_electrons,
                    mapper='jordan_wigner',
                )
                # Active-MO (or full-MO for non-active-space Hamiltonians) 1-RDM.
                # Trace-validated inside the extractor.
                quantum_density_mo = extractor.extract_1rdm(statevector)
                self.results['quantum_1rdm'] = quantum_density_mo
                # Legacy key — keep until downstream callers are migrated.
                self.results['quantum_rdm1'] = quantum_density_mo

                # Store in Hamiltonian. The Hamiltonian's set_quantum_density_matrix
                # handles active-MO → full-MO embedding (ActiveHamiltonian) and the
                # MO → AO transform. No silent hasattr fallback: if the Hamiltonian
                # type lacks the method, raise so the bug is loud.
                if not hasattr(self.hamiltonian, 'set_quantum_density_matrix'):
                    raise RuntimeError(
                        f"Hamiltonian {type(self.hamiltonian).__name__} does not implement "
                        "set_quantum_density_matrix(). Add it so wavefunction-derived "
                        "observables (dipole, NMR, polarizability) work."
                    )
                self.hamiltonian.set_quantum_density_matrix(quantum_density_mo)
                logger.info(
                    f"Stored quantum 1-RDM on hamiltonian: trace = "
                    f"{float(np.trace(quantum_density_mo)):.6f} "
                    f"(n_e_active = {self.hamiltonian.n_electrons})"
                )

                if self.enable_analysis:
                    self._add_analysis_to_results(result.fun, quantum_density_mo)
                    logger.info("✅ Analysis data added (using wavefunction-derived 1-RDM)")
            except Exception as e:
                logger.error(f"Quantum 1-RDM extraction failed: {e}")
                import traceback
                traceback.print_exc()
                if self.enable_analysis:
                    # Fall back so analysis pipeline doesn't crash, but log loudly.
                    try:
                        density_matrix, _ = self.hamiltonian.solve_scf(max_iterations=50, conv_tol=1e-6)
                        self._add_analysis_to_results(result.fun, density_matrix)
                    except Exception:
                        pass
        else:
            # Non-statevector backend: 1-RDM extraction via sampling is M4 work.
            # Analysis path uses HF density with a loud warning.
            if self.enable_analysis:
                logger.warning("Non-statevector backend: analysis using HF density (not VQE-correlated)")
                try:
                    density_matrix, _ = self.hamiltonian.solve_scf(max_iterations=50, conv_tol=1e-6)
                    self._add_analysis_to_results(result.fun, density_matrix)
                except Exception as e:
                    logger.error(f"HF analysis fallback failed: {e}")

        # Add optimization stats if enabled
        if self.enable_optimization:
            self._add_optimization_stats()

        # Validate results
        validation = self.validate_results()
        self.results['validation'] = validation

        if not validation['passed']:
            logger.warning("VQE results failed validation checks!")

        # ADD ENHANCED DATA FOR ANALYSIS SERVICE
        try:
            # Store molecule geometry for ADME and other analyses
            if self.molecule is not None:
                self.results['geometry'] = [
                    (atom.symbol, tuple(atom.position))
                    for atom in self.molecule.atoms
                ]
                self.results['atoms'] = [atom.symbol for atom in self.molecule.atoms]
                self.results['n_atoms'] = self.molecule.n_atoms
                self.results['n_electrons'] = self.molecule.n_electrons
                self.results['charge'] = getattr(self.molecule, 'charge', 0)
                self.results['multiplicity'] = getattr(self.molecule, 'multiplicity', 1)
                logger.info(f"✅ Stored molecule geometry for analysis")

            # Store nuclear repulsion energy
            if hasattr(self.hamiltonian, 'nuclear_repulsion'):
                self.results['nuclear_repulsion'] = float(self.hamiltonian.nuclear_repulsion)

            # Stash the 1-RDM (MO basis) for serialization. The AO-basis copy
            # lives on the Hamiltonian and is consumed by PropertyCalculator.
            try:
                if 'quantum_1rdm' in self.results:
                    self.results['rdm1'] = self.results['quantum_1rdm'].tolist()
                    logger.debug("Stored MO-basis quantum 1-RDM in results['rdm1']")
                elif hasattr(self.hamiltonian, 'mf') and hasattr(self.hamiltonian.mf, 'make_rdm1'):
                    self.results['rdm1'] = self.hamiltonian.mf.make_rdm1().tolist()
                    logger.warning("Stored HF RDM1 (no statevector-mode quantum density was extracted)")
            except Exception as e:
                logger.warning(f"Could not serialize RDM1: {e}")

            # Try to get orbital energies
            try:
                logger.debug(f"🔍 Checking hamiltonian for orbital energies: hasattr(mf)={hasattr(self.hamiltonian, 'mf')}")
                if hasattr(self.hamiltonian, 'mf'):
                    logger.debug(f"🔍 Hamiltonian has mf attribute, checking mo_energy: hasattr(mo_energy)={hasattr(self.hamiltonian.mf, 'mo_energy')}")
                    if hasattr(self.hamiltonian.mf, 'mo_energy'):
                        orb_energies = self.hamiltonian.mf.mo_energy
                        logger.debug(f"🔍 Found orbital energies: shape={orb_energies.shape}, dtype={orb_energies.dtype}")
                        self.results['orbital_energies'] = orb_energies.tolist()
                        logger.info(f"✅ Stored orbital energies for DOS analysis")
                    else:
                        logger.debug(f"Hamiltonian.mf does not have mo_energy attribute")
                else:
                    logger.debug(f"Hamiltonian does not have mf attribute (type: {type(self.hamiltonian).__name__}) - orbital energies not available")
            except Exception as e:
                logger.debug(f"Could not extract orbital energies: {e}")

            # Dipole moment via PySCF, consuming the AO-basis quantum density
            # if available (HF density otherwise). The pre-M3 code at this site
            # passed MO-basis 1-RDM to `scf.hf.dip_moment(mol, dm)` which expects
            # AO, producing nonsense values silently labelled "quantum dipole".
            try:
                if hasattr(self.hamiltonian, 'mf') and hasattr(self.hamiltonian, 'get_density_matrix'):
                    from pyscf import scf as _scf
                    dm_ao = self.hamiltonian.get_density_matrix(basis='ao')
                    dipole = _scf.hf.dip_moment(
                        self.hamiltonian.mf.mol, dm_ao, unit='Debye', verbose=0,
                    )
                    self.results['dipole'] = dipole.tolist()
                    src = 'quantum' if 'quantum_1rdm' in self.results else 'HF'
                    logger.info(f"Stored dipole (Debye) from {src} density: {dipole}")
            except Exception as e:
                logger.warning(f"Could not calculate dipole: {e}")

        except Exception as e:
            logger.error(f"Error storing enhanced data: {e}")

        # Add cloud provider job information to results
        if self.cloud_provider:
            self.results['cloud_provider'] = self.cloud_provider
            if self.cloud_job_ids:
                self.results['cloud_job_ids'] = self.cloud_job_ids
                # Generate job URLs
                if self.cloud_provider == 'bluequbit':
                    # BlueQubit job URLs
                    self.results['cloud_job_urls'] = [
                        f"https://app.bluequbit.io/jobs/{jid}" for jid in self.cloud_job_ids
                    ]
                elif self.cloud_provider == 'ibm':
                    # IBM Quantum job URLs
                    self.results['cloud_job_urls'] = [
                        f"https://quantum.ibm.com/jobs/{jid}" for jid in self.cloud_job_ids
                    ]
            if self.execution_mode:
                self.results['execution_mode'] = self.execution_mode
            logger.info(f"✅ Added cloud job info: provider={self.cloud_provider}, jobs={len(self.cloud_job_ids)}")

        logger.info(f"VQE optimization complete: {result.success}, {self.iteration_count} iterations")

        return self.results

    def solve_with_restarts(self, n_restarts=3, callback=None):
        """
        Run VQE multiple times with different random initializations.
        Returns the best result among all attempts.

        This significantly improves reliability for stochastic optimization.

        Args:
            n_restarts: Number of VQE attempts (default: 3)
            callback: Optional callback function for progress tracking

        Returns:
            dict: Best VQE result with lowest energy
        """
        logger.info(f"🔄 Starting multi-start VQE with {n_restarts} restarts")

        best_energy = float('inf')
        best_result = None
        all_energies = []

        # Disable the parameter cache for the duration of the restart loop so
        # each attempt draws a fresh random/MP2 init. Otherwise attempts 2..n
        # warm-start from the cached θ* written by attempt 1, defeating the
        # whole point of random-restart exploration.
        saved_use_cache = self.use_cache
        self.use_cache = False
        try:
            for attempt in range(1, n_restarts + 1):
                logger.info(f"🎯 VQE attempt {attempt}/{n_restarts}")

                # Run VQE with new random initialization
                result = self.solve(callback=callback)

                energy = result.energy
                all_energies.append(energy)

                # Track best result
                if energy < best_energy:
                    best_energy = energy
                    best_result = result
                    logger.info(f"   ✅ New best: {best_energy:.8f} Ha")
                else:
                    logger.info(f"   Energy: {energy:.8f} Ha (not better than {best_energy:.8f})")
        finally:
            self.use_cache = saved_use_cache

        # Add multi-start metadata (SolverResult is frozen; extra is a mutable dict)
        best_result.extra['multi_start'] = {
            'n_restarts': n_restarts,
            'all_energies': all_energies,
            'best_attempt': all_energies.index(best_energy) + 1,
            'energy_std': float(np.std(all_energies)),
            'energy_range': float(max(all_energies) - min(all_energies))
        }

        logger.info(f"🏆 Multi-start VQE complete:")
        logger.info(f"   Best energy: {best_energy:.8f} Ha (attempt {all_energies.index(best_energy) + 1})")
        logger.info(f"   Energy range: {max(all_energies) - min(all_energies):.8f} Ha")
        logger.info(f"   Energy std: {np.std(all_energies):.8f} Ha")

        return best_result

    def print_summary(self):
        """Print comprehensive VQE results summary."""
        print("=" * 80)
        print("VQE SOLVER RESULTS")
        print("=" * 80)

        # System information
        atoms = getattr(self, 'atoms', []) or []
        print(f"\nMolecule: {'-'.join([a.symbol for a in atoms]) or '?'}")
        bt = self.bond.bond_type if getattr(self, 'bond', None) is not None else getattr(self, '_bond_type', 'molecular')
        print(f"Bond Type: {bt}")
        print(f"Electrons: {self.molecule.n_electrons}")
        print(f"Orbitals: {self.hamiltonian.n_orbitals}")
        print(f"Qubits: {2 * self.hamiltonian.n_orbitals}")

        # Method details
        print(f"\nMethod: VQE")
        print(f"Ansatz: {self.ansatz_type.upper()}")
        print(f"Parameters: {self.n_parameters}")
        print(f"Mapper: {self.mapper_type}")
        print(f"Optimizer: {self.optimizer_method}")
        print(f"Backend: {self.backend_name}")

        # Energy results
        print("\n" + "-" * 80)
        print("ENERGY RESULTS")
        print("-" * 80)

        if 'energy' in self.results:
            print(f"\nVQE Energy:     {self.results['energy']:14.8f} Hartree")

        if 'hf_energy' in self.results:
            print(f"HF Reference:   {self.results['hf_energy']:14.8f} Hartree")

        if 'correlation_energy' in self.results:
            corr = self.results['correlation_energy']
            print(f"Correlation:    {corr:14.8f} Hartree ({corr * 627.509:.2f} kcal/mol)")

        # Convergence
        print("\n" + "-" * 80)
        print("CONVERGENCE")
        print("-" * 80)

        if 'converged' in self.results:
            status = "✓ Converged" if self.results['converged'] else "✗ Not Converged"
            print(f"\nStatus: {status}")

        if 'iterations' in self.results:
            print(f"Iterations: {self.results['iterations']}")

        if 'optimizer_message' in self.results:
            print(f"Message: {self.results['optimizer_message']}")

        # Energy history (convergence plot in text)
        if 'energy_history' in self.results and len(self.results['energy_history']) > 0:
            print("\nEnergy Convergence:")
            history = self.results['energy_history']
            # Show first, middle, last few points
            if len(history) > 10:
                indices = [0, 1, 2, len(history)//2, -3, -2, -1]
                for idx in indices:
                    if idx < 0:
                        idx = len(history) + idx
                    if 0 <= idx < len(history):
                        print(f"  Iter {idx:4d}: {history[idx]:14.8f} Ha")
            else:
                for i, E in enumerate(history):
                    print(f"  Iter {i:4d}: {E:14.8f} Ha")

        # Analysis results
        if 'analysis' in self.results and self.results['analysis']:
            print("\n" + "-" * 80)
            print("ANALYSIS")
            print("-" * 80)

            analysis = self.results['analysis']

            # Bonding analysis
            if analysis.get('bonding'):
                print("\nBonding Analysis:")
                bonding = analysis['bonding']
                for key, value in bonding.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key:25s}: {value:.6f}")

            # Properties
            if analysis.get('properties'):
                print("\nMolecular Properties:")
                props = analysis['properties']
                for key, value in props.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key:25s}: {value:.6f}")

        # Optimization stats
        if 'optimization_stats' in self.results:
            opt = self.results['optimization_stats']
            if any(opt.get('circuit', {}).values()):
                print("\n" + "-" * 80)
                print("CIRCUIT OPTIMIZATION")
                print("-" * 80)
                if opt['circuit'].get('gates_before') and opt['circuit'].get('gates_after'):
                    print(f"\nGates: {opt['circuit']['gates_before']} → {opt['circuit']['gates_after']}")
                    reduction = 100 * (1 - opt['circuit']['gates_after'] / opt['circuit']['gates_before'])
                    print(f"Reduction: {reduction:.1f}%")

        # Validation
        if 'validation' in self.results:
            validation = self.results['validation']
            if not validation['passed']:
                print("\n" + "-" * 80)
                print("⚠ VALIDATION WARNINGS")
                print("-" * 80)
                for check in validation['checks']:
                    if not check['passed']:
                        print(f"✗ {check['name']}: {check['message']}")
            else:
                print("\n✓ All validation checks passed")

        # Final summary
        print("\n" + "=" * 80)
        if self.results.get('converged'):
            print("✓ VQE OPTIMIZATION SUCCESSFUL")
        else:
            print("⚠ VQE OPTIMIZATION INCOMPLETE")
        print("=" * 80)
