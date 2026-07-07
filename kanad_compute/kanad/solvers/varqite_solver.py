"""
Variational Quantum Imaginary Time Evolution (VarQITE) Solver

VarQITE provides an alternative to VQE with several advantages:
1. No classical optimization loop - parameters evolve via differential equations
2. Monotonic energy decrease guaranteed
3. No barren plateau problem (natural landscape)
4. Natural extension to real-time dynamics (VarQRTE)

Theory:
------
Imaginary time evolution: |ψ(τ)⟩ = e^{-Hτ}|ψ(0)⟩ / ||...||

For parametrized state |ψ(θ)⟩, McLachlan variational principle gives:
    A · dθ/dτ = -C

Where:
    A_kl = Re[⟨∂_k ψ|∂_l ψ⟩]  (quantum Fisher information / metric tensor)
    C_k = Re[⟨∂_k ψ|(H-E)|ψ⟩]  (gradient vector)

Energy decreases monotonically:
    dE/dτ = -2 Var(H) ≤ 0

References:
----------
1. Yuan et al. (2019) Quantum 3, 191 - VarQITE theory
2. McArdle et al. (2019) npj Quantum Inf 5, 75 - VarQITE implementation
3. Zoufal et al. (2021) PRX Quantum 2, 010309 - Generalized VQE via VarQITE
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple, Any
from scipy.linalg import lstsq, pinv
from scipy.integrate import solve_ivp

from kanad.solvers.base_solver import BaseSolver
from kanad.core.solver_result import SolverResult

logger = logging.getLogger(__name__)

# The legacy ``@dataclass VarQITEResult`` / ``@dataclass VarQRTEResult`` were deleted in
# the unified-solver-protocol migration: ``solve()`` / ``evolve_real_time()`` now return a
# :class:`SolverResult`. The names are kept as back-compat aliases so
# ``from kanad.solvers import VarQITEResult, VarQRTEResult`` and the package ``__all__``
# export keep importing cleanly. Use ``SolverResult`` directly in new code.
VarQITEResult = SolverResult
VarQRTEResult = SolverResult


class VarQITESolver(BaseSolver):
    """
    Variational Quantum Imaginary Time Evolution Solver.

    VarQITE finds the ground state by evolving in imaginary time.
    No classical optimizer needed - parameters evolve via ODE.

    Advantages over VQE:
    - Guaranteed monotonic energy decrease
    - No barren plateaus
    - Natural convergence criterion (variance → 0)
    - Can extend to real-time dynamics

    Usage:
    ------
    >>> from kanad import BondFactory
    >>> from kanad.solvers import VarQITESolver

    >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
    >>> solver = VarQITESolver(bond, ansatz_type='hardware_efficient')
    >>> result = solver.solve()
    >>> print(f"Energy: {result.energy:.6f} Ha")
    """

    def __init__(
        self,
        system=None,
        *,
        bond_or_molecule=None,
        ansatz_type: str = 'hardware_efficient',
        backend: str = 'statevector',
        regularization: float = 1e-2,  # 1e-4 left the QFI metric near-singular -> pinv overshoot diverges; 1e-2 is the verified-convergent value
        convergence_threshold: float = 1e-6,
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        **backend_kwargs,
    ):
        """
        Initialize VarQITE solver.

        Args:
            system: Unified-protocol positional input (Bond / Molecule / bare
                Hamiltonian / builder QuantumSystem). Mapped onto the legacy
                ``bond_or_molecule`` slot unless ``bond_or_molecule=`` is given.
            bond_or_molecule: Legacy keyword for the system input (Bond/Molecule).
            ansatz_type: Type of ansatz ('hardware_efficient', 'governance')
            backend: Backend name resolved via ``make_backend`` (statevector by default).
            regularization: Regularization for metric tensor (avoids singularity)
            convergence_threshold: Energy variance threshold for convergence
            enable_analysis: Enable BaseSolver analysis tooling.
            enable_optimization: Enable BaseSolver optimization tooling.
        """
        # Unified solver protocol: the positional `system` is the high-level input.
        # Map it onto the legacy `bond_or_molecule` slot unless an explicit kwarg
        # was given.
        if system is not None and bond_or_molecule is None:
            bond_or_molecule = system
        if bond_or_molecule is None:
            raise ValueError("Must provide a system (Bond / Molecule / Hamiltonian)")

        self.ansatz_type = ansatz_type
        self.regularization = regularization
        self.convergence_threshold = convergence_threshold

        # Resolve the system through BaseSolver: sets self.hamiltonian / self.molecule /
        # self.bond and builds the BaseBackend object on self.backend.
        super().__init__(
            bond_or_molecule,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **backend_kwargs,
        )

        # self.backend is now a BaseBackend object; self.backend_name is the string.
        # The internal VQESolver (built lazily in _initialize) takes a backend *name*.
        from kanad.backends.statevector_backend import StatevectorBackend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        # Will be initialized on first solve
        self._solver = None  # VQE solver for energy/gradient evaluation
        self._circuit = None
        self._hamiltonian_pauli = None
        self._n_params = None

        logger.info(f"VarQITESolver initialized")
        logger.info(f"  Regularization: {regularization}")
        logger.info(f"  Convergence threshold: {convergence_threshold}")

    def _initialize(self):
        """Initialize internal VQE solver for energy evaluation."""
        if self._solver is not None:
            return

        from kanad.solvers import VQESolver

        # VQESolver takes a backend *name* string; self.backend is now a
        # BaseBackend object, so pass self.backend_name. The internal solver is
        # only used for energy/gradient evaluation, so the bond/hamiltonian is
        # routed through whichever the system resolved to.
        _system = self.bond if self.bond is not None else self.hamiltonian
        self._solver = VQESolver(
            _system,
            ansatz_type=self.ansatz_type,
            backend=self.backend_name,
            optimizer='COBYLA',
            max_iterations=1  # Just for initialization
        )

        # Get circuit info
        if hasattr(self._solver, 'ansatz'):
            self._n_params = self._solver.ansatz.n_parameters if hasattr(self._solver.ansatz, 'n_parameters') else 12
        else:
            self._n_params = 12  # Default for hardware-efficient

        logger.debug(f"Initialized with {self._n_params} parameters")

    def _state(self, p: np.ndarray) -> np.ndarray:
        """
        Build the statevector |ψ(θ)⟩ for parameters p from the ansatz circuit.

        Args:
            p: Circuit parameters

        Returns:
            Complex statevector amplitudes (numpy array)
        """
        from qiskit.quantum_info import Statevector

        circ = self._solver.ansatz.circuit or self._solver.ansatz.build_circuit()
        qc = circ.to_qiskit() if hasattr(circ, 'to_qiskit') else circ
        if qc.num_parameters > 0:
            qc = qc.assign_parameters({qc.parameters[i]: p[i] for i in range(len(p))})
        return Statevector.from_instruction(qc).data

    def _hamiltonian_op(self, params: np.ndarray):
        """
        Return the SparsePauliOp Hamiltonian operator matching the ansatz.

        VQESolver builds/caches its SparsePauliOp lazily on the first energy
        evaluation, so we trigger one compute_energy call to populate it.
        """
        if getattr(self._solver, '_sparse_pauli_op', None) is None:
            self._solver.compute_energy(params)
        return self._solver._sparse_pauli_op

    def compute_metric_tensor(self, params: np.ndarray) -> np.ndarray:
        """
        Compute quantum Fisher information matrix (metric tensor).

        A_kl = Re[⟨∂_k ψ|∂_l ψ⟩] - Re[⟨∂_k ψ|ψ⟩]Re[⟨ψ|∂_l ψ⟩]

        Computed from state derivatives via central finite differences on the
        ansatz statevector (the true McLachlan/QFI overlap metric), not from
        energy second-differences.

        Args:
            params: Current circuit parameters

        Returns:
            A: (n_params, n_params) metric tensor
        """
        n = len(params)
        eps = 1e-6
        psi = self._state(params)

        # State derivatives via central differences
        dpsi = []
        for k in range(n):
            pp = params.copy()
            pp[k] += eps
            pm = params.copy()
            pm[k] -= eps
            dpsi.append((self._state(pp) - self._state(pm)) / (2 * eps))

        A = np.zeros((n, n))
        for k in range(n):
            for l in range(k, n):
                val = (
                    np.real(np.vdot(dpsi[k], dpsi[l]))
                    - np.real(np.vdot(dpsi[k], psi)) * np.real(np.vdot(psi, dpsi[l]))
                )
                A[k, l] = val
                A[l, k] = val

        # Add regularization to ensure positive definiteness
        A += self.regularization * np.eye(n)

        return A

    def compute_gradient_vector(self, params: np.ndarray, energy: float) -> np.ndarray:
        """
        Compute gradient vector C.

        C_k = Re[⟨∂_k ψ|(H-E)|ψ⟩] = (1/2)[E(θ_k+π/2) - E(θ_k-π/2)]

        This is the same as the VQE parameter gradient.

        Args:
            params: Current circuit parameters
            energy: Current energy expectation value

        Returns:
            C: (n_params,) gradient vector
        """
        n = len(params)
        C = np.zeros(n)
        shift = np.pi / 2

        for k in range(n):
            params_plus = params.copy()
            params_plus[k] += shift
            e_plus = self._solver.compute_energy(params_plus)

            params_minus = params.copy()
            params_minus[k] -= shift
            e_minus = self._solver.compute_energy(params_minus)

            C[k] = 0.5 * (e_plus - e_minus)

        return C

    def compute_energy_variance(self, params: np.ndarray, energy: float) -> float:
        """
        Compute energy variance Var(H) = ⟨H²⟩ - ⟨H⟩².

        The variance determines the rate of convergence:
        dE/dτ = -2 Var(H)

        When Var(H) → 0, we've found an eigenstate.

        Args:
            params: Circuit parameters
            energy: Current energy

        Returns:
            variance: Energy variance
        """
        # True energy variance Var(H) = ⟨H²⟩ - ⟨H⟩² on the bound statevector,
        # not the ||∇E||²/4 gradient-norm proxy (which is not Var(H)).
        from qiskit.quantum_info import Statevector

        psi = Statevector(self._state(params))
        H_op = self._hamiltonian_op(params)
        h_exp = psi.expectation_value(H_op)
        h2_exp = psi.expectation_value(H_op @ H_op)
        variance = np.real(h2_exp - h_exp ** 2)
        return float(variance)

    def imaginary_time_derivative(self, tau: float, params: np.ndarray) -> np.ndarray:
        """
        Compute dθ/dτ from McLachlan principle.

        A · dθ/dτ = -C

        Solve: dθ/dτ = -A^{-1} · C

        Args:
            tau: Current imaginary time (unused but required by ODE solver)
            params: Current parameters

        Returns:
            dtheta_dtau: Parameter derivatives
        """
        energy = self._solver.compute_energy(params)
        A = self.compute_metric_tensor(params)
        C = self.compute_gradient_vector(params, energy)

        # Solve linear system A · x = -C
        # Use pseudo-inverse for stability
        A_pinv = pinv(A)
        dtheta_dtau = -A_pinv @ C

        return dtheta_dtau

    def solve(
        self,
        max_tau: float = 10.0,
        dtau: float = 0.1,
        initial_params: Optional[np.ndarray] = None,
        use_adaptive: bool = True,
        verbose: bool = False,
        callback: Optional[callable] = None
    ) -> SolverResult:
        """
        Solve for ground state using imaginary time evolution.

        Args:
            max_tau: Maximum imaginary time
            dtau: Time step (for fixed step) or initial step (for adaptive)
            initial_params: Starting parameters (random if None)
            use_adaptive: Use adaptive step size ODE solver
            verbose: Print progress
            callback: Optional progress callback invoked once per recorded
                      imaginary-time step as callback(iteration, energy, params).
                      Lets the API layer stream the convergence curve. (For the
                      adaptive ODE path, points are emitted over the solver's
                      output trajectory, not every internal RK substep.)

        Returns:
            SolverResult with the final energy and converged parameters. The
            VarQITE-specific fields (optimal_parameters, tau_final,
            parameter_history, energy_variance, method) live under ``.extra``.
        """
        self._initialize()

        # Initialize parameters
        if initial_params is None:
            initial_params = np.random.random(self._n_params) * 2 * np.pi

        params = initial_params.copy()
        energy_history = []
        parameter_history = []

        logger.info(f"Starting VarQITE evolution")
        logger.info(f"  max_tau: {max_tau}")
        logger.info(f"  Initial params: {len(params)}")

        def _emit_progress(iteration, energy, params_vec):
            """Invoke the user callback (if any) with VQESolver-style args.
            Re-raises cancellation; swallows other callback errors so a broken
            callback never breaks the evolution."""
            if callback is None:
                return
            try:
                import inspect
                sig = inspect.signature(callback)
                if len(sig.parameters) >= 3:
                    callback(iteration, energy, params_vec)
                else:
                    callback(iteration, energy)
            except Exception as cb_exc:
                if 'Cancelled' in type(cb_exc).__name__ or 'cancelled' in str(cb_exc).lower():
                    raise
                logger.warning(f"VarQITE progress callback failed: {cb_exc}")

        if use_adaptive:
            # Use scipy ODE solver with adaptive stepping
            def deriv(tau, p):
                return self.imaginary_time_derivative(tau, p)

            # Solve ODE
            sol = solve_ivp(
                deriv,
                [0, max_tau],
                params,
                method='RK45',
                max_step=dtau,
                rtol=1e-4,
                atol=1e-6
            )

            # Extract trajectory
            for i, t in enumerate(sol.t):
                p = sol.y[:, i]
                e = self._solver.compute_energy(p)
                energy_history.append(e)
                parameter_history.append(p.copy())
                _emit_progress(i + 1, e, p)

                if verbose and i % 10 == 0:
                    logger.info(f"  τ={t:.3f}: E = {e:.6f} Ha")

            final_params = sol.y[:, -1]
            tau_final = sol.t[-1]
        else:
            # Fixed step Euler integration
            tau = 0.0
            n_steps = int(max_tau / dtau)
            _plateau = 0

            for step in range(n_steps):
                energy = self._solver.compute_energy(params)
                energy_history.append(energy)
                parameter_history.append(params.copy())
                _emit_progress(step + 1, energy, params)

                # Energy-plateau early-stop: imaginary-time evolution converges the
                # ENERGY well before the variance reaches the (strict) 1e-6 threshold, so
                # waiting for the variance ran the full max_tau needlessly (e.g. H2 took
                # ~100 steps for a result reached by ~20). Stop once the energy stops
                # moving — keeps the variance check below as the physical backstop.
                if len(energy_history) >= 2 and abs(energy_history[-1] - energy_history[-2]) < 1e-7:
                    _plateau += 1
                    if _plateau >= 3:
                        logger.info(f"Energy converged (plateau) at τ={tau:.3f}, E={energy:.6f} Ha")
                        break
                else:
                    _plateau = 0

                # Check convergence (variance — the physical convergence measure)
                variance = self.compute_energy_variance(params, energy)
                if variance < self.convergence_threshold:
                    logger.info(f"Converged at τ={tau:.3f} with variance {variance:.2e}")
                    break

                if verbose and step % 10 == 0:
                    logger.info(f"  τ={tau:.3f}: E = {energy:.6f} Ha, Var = {variance:.2e}")

                # Euler step
                dtheta = self.imaginary_time_derivative(tau, params)
                params = params + dtau * dtheta
                tau += dtau

            final_params = params
            tau_final = tau

        # Final evaluation
        final_energy = self._solver.compute_energy(final_params)
        final_variance = self.compute_energy_variance(final_params, final_energy)

        converged = final_variance < self.convergence_threshold

        logger.info(f"VarQITE complete")
        logger.info(f"  Final energy: {final_energy:.6f} Ha")
        logger.info(f"  Final variance: {final_variance:.2e}")
        logger.info(f"  Converged: {converged}")

        raw = {
            'energy': float(final_energy),
            'converged': bool(converged),
            'iterations': len(energy_history),
            'energy_history': [float(e) for e in energy_history],
            'parameters': np.asarray(final_params),
            'optimal_parameters': np.asarray(final_params),
            'tau_final': float(tau_final),
            'energy_variance': float(final_variance),
            'parameter_history': parameter_history,
            'method': 'varqite',
        }
        return SolverResult.from_mapping(raw, solver='varqite', backend=self.backend_name)

    def evolve_real_time(
        self,
        initial_params: np.ndarray,
        total_time: float,
        dt: float,
        observable_ops: Optional[Dict[str, Any]] = None
    ) -> SolverResult:
        """
        Real-time evolution via VarQRTE.

        Instead of imaginary time (τ), evolves in real time (t):
        A · dθ/dt = -i·C  →  A · dθ/dt = C' (different gradient)

        Args:
            initial_params: Starting parameters
            total_time: Total evolution time
            dt: Time step
            observable_ops: Observables to track (name → operator)

        Returns:
            VarQRTEResult with time evolution
        """
        self._initialize()
        from qiskit.quantum_info import Statevector

        params = initial_params.copy()
        n_steps = int(total_time / dt)
        H_op = self._hamiltonian_op(params)
        eps = 1e-6  # finite-difference step for state derivatives

        time_history = []
        param_history = []
        obs_history = {name: [] for name in (observable_ops or {})}

        for step in range(n_steps + 1):
            t = step * dt
            time_history.append(t)
            param_history.append(params.copy())

            # Record each requested observable's expectation on the bound
            # statevector (real-time dynamics tracks observables, not energy).
            if observable_ops:
                psi = Statevector(self._state(params))
                for name, op in observable_ops.items():
                    obs_history[name].append(
                        np.real(psi.expectation_value(op))
                    )

            if step < n_steps:
                # VarQRTE McLachlan equation: A · dθ/dt = C', with the
                # imaginary-part RHS C'_k = Im[⟨∂_k ψ|H|ψ⟩] (no minus sign),
                # distinct from the imaginary-time real-part gradient.
                A = self.compute_metric_tensor(params)
                psi = self._state(params)
                Hpsi = H_op.to_matrix(sparse=True).dot(psi)
                C_prime = np.zeros(len(params))
                for k in range(len(params)):
                    pp = params.copy(); pp[k] += eps
                    pm = params.copy(); pm[k] -= eps
                    dpsi_k = (self._state(pp) - self._state(pm)) / (2 * eps)
                    C_prime[k] = np.imag(np.vdot(dpsi_k, Hpsi))

                A_pinv = pinv(A)
                dtheta = A_pinv @ C_prime  # A · dθ/dt = C' (no minus sign)

                params = params + dt * dtheta

        # Real-time evolution tracks observables rather than minimizing energy;
        # the canonical SolverResult.energy is ⟨H⟩ on the final evolved state.
        final_energy = float(self._solver.compute_energy(params))
        raw = {
            'energy': final_energy,
            'converged': True,
            'iterations': len(time_history),
            'final_state': np.asarray(params),
            'parameters': np.asarray(params),
            'time_history': np.array(time_history),
            'parameter_history': np.array(param_history),
            'observable_history': {k: np.array(v) for k, v in obs_history.items()},
            'method': 'varqrte',
        }
        return SolverResult.from_mapping(raw, solver='varqrte', backend=self.backend_name)


def create_varqite_solver(bond_or_molecule, **kwargs) -> VarQITESolver:
    """
    Convenience function to create a VarQITE solver.

    Args:
        bond_or_molecule: Bond or Molecule object
        **kwargs: Additional arguments for VarQITESolver

    Returns:
        Configured VarQITESolver instance
    """
    return VarQITESolver(bond_or_molecule, **kwargs)
