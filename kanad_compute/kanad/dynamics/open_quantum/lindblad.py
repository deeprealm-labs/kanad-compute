"""
Lindblad Master Equation Evolver

Implements the Lindblad (GKSL) master equation for open quantum systems.

Theory:
------
The Lindblad equation describes non-unitary evolution of a density matrix:

    dρ/dt = -i[H,ρ] + Σ_k γ_k D[L_k](ρ)

Where the dissipator is:
    D[L](ρ) = L ρ L† - ½{L†L, ρ}

This preserves:
1. Trace: Tr(ρ) = 1
2. Positivity: ρ ≥ 0
3. Hermiticity: ρ† = ρ

Common Lindblad Operators:
-------------------------
- Dephasing: L = σ_z (rate 1/T2*)
- Amplitude damping: L = σ⁻ (rate 1/T1)
- Thermal excitation: L = σ⁺ (rate determined by temperature)
- Vibrational relaxation: L = a (phonon emission)

References:
----------
1. Lindblad (1976) Commun. Math. Phys. 48, 119
2. Gorini, Kossakowski, Sudarshan (1976) J. Math. Phys. 17, 821
3. Breuer & Petruccione "Theory of Open Quantum Systems" (2002)
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.linalg import expm

logger = logging.getLogger(__name__)


@dataclass
class LindbladResult:
    """Result from Lindblad evolution."""
    final_rho: np.ndarray           # Final density matrix
    time_history: np.ndarray        # Time points
    rho_history: List[np.ndarray]   # Density matrices at each time
    purity_history: np.ndarray      # Tr(ρ²) over time
    trace_history: np.ndarray       # Tr(ρ) over time (should be 1)
    energy_history: np.ndarray      # ⟨H⟩ over time
    converged_to_steady: bool       # Whether reached steady state


class LindbladEvolver:
    """
    Lindblad master equation evolver for open quantum systems.

    Evolves density matrix under both Hamiltonian dynamics and
    Lindblad dissipation.

    Example:
    -------
    >>> from kanad.dynamics.open_quantum import LindbladEvolver
    >>> from kanad.dynamics.open_quantum import create_dephasing_operator

    >>> # Create 2-level system
    >>> H = np.array([[0, 1], [1, 0]])  # σ_x
    >>> L_ops = [create_dephasing_operator(n_qubits=1)]
    >>> rates = [0.1]  # Dephasing rate

    >>> evolver = LindbladEvolver(H, L_ops, rates)
    >>> rho_0 = np.array([[1, 0], [0, 0]])  # |0⟩ state
    >>> result = evolver.evolve(rho_0, total_time=10.0, dt=0.1)
    >>> print(f"Final purity: {result.purity_history[-1]:.4f}")
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        lindblad_ops: List[np.ndarray],
        rates: List[float],
        use_sparse: bool = False
    ):
        """
        Initialize Lindblad evolver.

        Args:
            hamiltonian: System Hamiltonian matrix (Hermitian)
            lindblad_ops: List of Lindblad operators [L_1, L_2, ...]
            rates: Corresponding decay rates [γ_1, γ_2, ...]
            use_sparse: Use sparse matrix operations (for large systems)
        """
        self.H = np.asarray(hamiltonian, dtype=complex)
        self.L_ops = [np.asarray(L, dtype=complex) for L in lindblad_ops]
        self.rates = rates
        self.use_sparse = use_sparse

        # Validate inputs
        self._validate_inputs()

        self.dim = self.H.shape[0]
        logger.info(f"LindbladEvolver initialized")
        logger.info(f"  Dimension: {self.dim}")
        logger.info(f"  Lindblad operators: {len(self.L_ops)}")

    def _validate_inputs(self):
        """Validate Hamiltonian and Lindblad operators."""
        # Check Hamiltonian is Hermitian
        if not np.allclose(self.H, self.H.conj().T):
            logger.warning("Hamiltonian is not Hermitian!")

        # Check dimensions match
        for i, L in enumerate(self.L_ops):
            if L.shape != self.H.shape:
                raise ValueError(f"Lindblad operator {i} has wrong shape: "
                               f"{L.shape} vs {self.H.shape}")

        if len(self.rates) != len(self.L_ops):
            raise ValueError("Number of rates must match number of Lindblad operators")

    def lindblad_superoperator(self) -> np.ndarray:
        """
        Construct the Lindbladian superoperator L.

        The master equation dρ/dt = L(ρ) can be vectorized as:
        d|ρ⟩⟩/dt = L |ρ⟩⟩

        Where |ρ⟩⟩ is the vectorized density matrix.

        Returns:
            L: (dim², dim²) superoperator matrix
        """
        dim = self.dim
        dim2 = dim * dim

        # Identity
        I = np.eye(dim, dtype=complex)

        # Hamiltonian part: -i[H, ρ] = -i(H⊗I - I⊗H^T) |ρ⟩⟩
        L_H = -1j * (np.kron(self.H, I) - np.kron(I, self.H.T))

        # Dissipator part
        L_D = np.zeros((dim2, dim2), dtype=complex)
        for L, gamma in zip(self.L_ops, self.rates):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L

            # D[L](ρ) = L ρ L† - ½{L†L, ρ}
            # Vectorized: (L⊗L*)|ρ⟩⟩ - ½(L†L⊗I + I⊗(L†L)^T)|ρ⟩⟩
            L_D += gamma * (
                np.kron(L, L.conj())
                - 0.5 * np.kron(L_dag_L, I)
                - 0.5 * np.kron(I, L_dag_L.T)
            )

        return L_H + L_D

    def dissipator(self, rho: np.ndarray) -> np.ndarray:
        """
        Compute dissipator D(ρ) = Σ_k γ_k D[L_k](ρ).

        Args:
            rho: Density matrix

        Returns:
            D(ρ): Dissipator contribution
        """
        result = np.zeros_like(rho, dtype=complex)

        for L, gamma in zip(self.L_ops, self.rates):
            L_dag = L.conj().T
            L_dag_L = L_dag @ L

            # D[L](ρ) = L ρ L† - ½{L†L, ρ}
            result += gamma * (
                L @ rho @ L_dag
                - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
            )

        return result

    def drho_dt(self, t: float, rho_vec: np.ndarray) -> np.ndarray:
        """
        Compute dρ/dt for ODE solver.

        Args:
            t: Time (unused but required by solve_ivp)
            rho_vec: Vectorized density matrix

        Returns:
            Vectorized dρ/dt
        """
        # Reshape to matrix
        rho = rho_vec.reshape((self.dim, self.dim))

        # Hamiltonian evolution: -i[H, ρ]
        hamiltonian_term = -1j * (self.H @ rho - rho @ self.H)

        # Dissipator
        dissipator_term = self.dissipator(rho)

        # Total
        drho = hamiltonian_term + dissipator_term

        return drho.flatten()

    def evolve(
        self,
        rho_0: np.ndarray,
        total_time: float,
        dt: float = 0.1,
        method: str = 'RK45',
        observables: Dict[str, np.ndarray] = None
    ) -> LindbladResult:
        """
        Evolve density matrix under Lindblad dynamics.

        Args:
            rho_0: Initial density matrix
            total_time: Total evolution time
            dt: Time step for output
            method: Integration method ('RK45', 'euler', 'expm')
            observables: Dict of observables to track {name: operator}

        Returns:
            LindbladResult with evolution history
        """
        rho_0 = np.asarray(rho_0, dtype=complex)
        logger.info(f"Starting Lindblad evolution")
        logger.info(f"  Total time: {total_time}")
        logger.info(f"  Method: {method}")

        if method == 'expm':
            # Matrix exponential method (exact for time-independent)
            result = self._evolve_expm(rho_0, total_time, dt)
        elif method == 'euler':
            # Simple Euler (for testing)
            result = self._evolve_euler(rho_0, total_time, dt)
        else:
            # scipy ODE solver
            result = self._evolve_ode(rho_0, total_time, dt, method)

        return result

    @staticmethod
    def check_cptp(result, tol: float = 1e-6) -> dict:
        """Validate that an evolution stayed physical (CPTP) — inspection D9.

        The Lindblad master equation is completely-positive and trace-preserving
        by construction, but a finite-step integrator (especially plain Euler) can
        drive ρ(t) slightly negative or off-trace. This scans the trajectory and
        reports the worst-case positivity (smallest eigenvalue of the Hermitized
        ρ) and trace deviation, so no observable derived from ρ is trusted before
        the density matrix is verified physical.

        Returns a dict: ``cptp_ok``, ``positive_semidefinite``, ``min_eigenvalue``,
        ``trace_preserving``, ``max_trace_deviation``.
        """
        min_eig = np.inf
        max_trace_dev = 0.0
        for rho in result.rho_history:
            rho = np.asarray(rho)
            w = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
            min_eig = min(min_eig, float(w[0]))
            max_trace_dev = max(max_trace_dev,
                                abs(float(np.real(np.trace(rho))) - 1.0))
        positive = min_eig >= -tol
        trace_preserving = max_trace_dev <= tol
        return {
            'cptp_ok': bool(positive and trace_preserving),
            'positive_semidefinite': bool(positive),
            'min_eigenvalue': float(min_eig),
            'trace_preserving': bool(trace_preserving),
            'max_trace_deviation': float(max_trace_dev),
        }

    def _evolve_ode(
        self,
        rho_0: np.ndarray,
        total_time: float,
        dt: float,
        method: str
    ) -> LindbladResult:
        """Evolve using scipy ODE solver."""
        t_eval = np.arange(0, total_time + dt, dt)

        sol = solve_ivp(
            self.drho_dt,
            [0, total_time],
            rho_0.flatten(),
            method=method,
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-8
        )

        # Extract results
        rho_history = []
        purity_history = []
        trace_history = []
        energy_history = []

        for i, t in enumerate(sol.t):
            rho = sol.y[:, i].reshape((self.dim, self.dim))
            rho_history.append(rho)

            # Purity: Tr(ρ²)
            purity = np.real(np.trace(rho @ rho))
            purity_history.append(purity)

            # Trace (should be 1)
            trace = np.real(np.trace(rho))
            trace_history.append(trace)

            # Energy
            energy = np.real(np.trace(self.H @ rho))
            energy_history.append(energy)

        # Check if reached steady state
        if len(purity_history) > 10:
            purity_change = abs(purity_history[-1] - purity_history[-10])
            converged = purity_change < 1e-6
        else:
            converged = False

        return LindbladResult(
            final_rho=rho_history[-1],
            time_history=sol.t,
            rho_history=rho_history,
            purity_history=np.array(purity_history),
            trace_history=np.array(trace_history),
            energy_history=np.array(energy_history),
            converged_to_steady=converged
        )

    def _evolve_euler(
        self,
        rho_0: np.ndarray,
        total_time: float,
        dt: float
    ) -> LindbladResult:
        """Simple Euler evolution (for testing)."""
        n_steps = int(total_time / dt)
        rho = rho_0.copy()

        time_history = [0.0]
        rho_history = [rho.copy()]
        purity_history = [np.real(np.trace(rho @ rho))]
        trace_history = [np.real(np.trace(rho))]
        energy_history = [np.real(np.trace(self.H @ rho))]

        for step in range(n_steps):
            drho = self.drho_dt(step * dt, rho.flatten()).reshape((self.dim, self.dim))
            rho = rho + dt * drho

            time_history.append((step + 1) * dt)
            rho_history.append(rho.copy())
            purity_history.append(np.real(np.trace(rho @ rho)))
            trace_history.append(np.real(np.trace(rho)))
            energy_history.append(np.real(np.trace(self.H @ rho)))

        converged = abs(purity_history[-1] - purity_history[-10]) < 1e-6 if len(purity_history) > 10 else False

        return LindbladResult(
            final_rho=rho_history[-1],
            time_history=np.array(time_history),
            rho_history=rho_history,
            purity_history=np.array(purity_history),
            trace_history=np.array(trace_history),
            energy_history=np.array(energy_history),
            converged_to_steady=converged
        )

    def _evolve_expm(
        self,
        rho_0: np.ndarray,
        total_time: float,
        dt: float
    ) -> LindbladResult:
        """Matrix exponential evolution (exact for time-independent)."""
        # Get Lindbladian superoperator
        L = self.lindblad_superoperator()

        # Propagator for time step dt
        propagator = expm(L * dt)

        n_steps = int(total_time / dt)
        rho_vec = rho_0.flatten()

        time_history = [0.0]
        rho_history = [rho_0.copy()]
        purity_history = [np.real(np.trace(rho_0 @ rho_0))]
        trace_history = [np.real(np.trace(rho_0))]
        energy_history = [np.real(np.trace(self.H @ rho_0))]

        for step in range(n_steps):
            rho_vec = propagator @ rho_vec
            rho = rho_vec.reshape((self.dim, self.dim))

            time_history.append((step + 1) * dt)
            rho_history.append(rho.copy())
            purity_history.append(np.real(np.trace(rho @ rho)))
            trace_history.append(np.real(np.trace(rho)))
            energy_history.append(np.real(np.trace(self.H @ rho)))

        converged = abs(purity_history[-1] - purity_history[-10]) < 1e-6 if len(purity_history) > 10 else False

        return LindbladResult(
            final_rho=rho_history[-1],
            time_history=np.array(time_history),
            rho_history=rho_history,
            purity_history=np.array(purity_history),
            trace_history=np.array(trace_history),
            energy_history=np.array(energy_history),
            converged_to_steady=converged
        )

    def compute_steady_state(self) -> np.ndarray:
        """
        Compute the steady state ρ_ss where dρ/dt = 0.

        Solves L(ρ_ss) = 0 subject to Tr(ρ_ss) = 1.

        Returns:
            ρ_ss: Steady state density matrix
        """
        from scipy.linalg import null_space

        L = self.lindblad_superoperator()

        # Find null space of L
        ns = null_space(L)

        if ns.shape[1] == 0:
            logger.warning("No steady state found (Lindbladian has no null space)")
            return None

        # The steady state is in the null space
        # Reshape and normalize trace
        rho_ss = ns[:, 0].reshape((self.dim, self.dim))

        # Ensure Hermiticity
        rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)

        # Normalize trace
        trace = np.trace(rho_ss)
        if np.abs(trace) > 1e-10:
            rho_ss = rho_ss / trace

        logger.info(f"Steady state found")
        logger.info(f"  Purity: {np.real(np.trace(rho_ss @ rho_ss)):.4f}")

        return rho_ss


def create_dephasing_operator(n_qubits: int = 1, qubit_idx: int = 0) -> np.ndarray:
    """
    Create dephasing (Z) Lindblad operator.

    Scaled so the supplied rate γ equals 1/T₂* exactly. The GKSL dissipator drives
    the coherence as dρ₀₁/dt = -‖ΔL‖²·γ·ρ₀₁ where ΔL = L₀₀-L₁₁. For the BARE Pauli
    L = σ_z = diag(1,-1), ΔL = 2 → coherence decays as exp(-2γt), i.e. 1/T₂* = 2γ, so
    a user passing rate=1/T₂* gets a T₂* short by exactly 2×. Returning L = σ_z/√2 gives
    ΔL² = 2·(1/2) = 1 → exp(-γt), making the documented "rate = 1/T₂*" hold literally.
    (σ_z/2 would over-correct to 1/T₂* = γ/2.)

    Args:
        n_qubits: Total number of qubits
        qubit_idx: Which qubit to apply dephasing to

    Returns:
        Lindblad operator as matrix
    """
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex) / np.sqrt(2.0)

    if n_qubits == 1:
        return sigma_z

    # Build tensor product
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit_idx] = sigma_z

    result = ops[0]
    for i in range(1, n_qubits):
        result = np.kron(result, ops[i])

    return result


def create_amplitude_damping_operator(n_qubits: int = 1, qubit_idx: int = 0) -> np.ndarray:
    """
    Create amplitude damping (σ⁻) Lindblad operator.

    Amplitude damping: L = σ⁻ = |0⟩⟨1| causes T1 decay.

    Args:
        n_qubits: Total number of qubits
        qubit_idx: Which qubit to apply damping to

    Returns:
        Lindblad operator as matrix
    """
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

    if n_qubits == 1:
        return sigma_minus

    # Build tensor product
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit_idx] = sigma_minus

    result = ops[0]
    for i in range(1, n_qubits):
        result = np.kron(result, ops[i])

    return result


def create_thermal_operators(
    n_qubits: int = 1,
    qubit_idx: int = 0,
    temperature: float = 300.0,
    omega: float = 0.01
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Create thermal bath Lindblad operators.

    At finite temperature, both emission and absorption occur:
    - Emission: L = σ⁻, rate = γ(1 + n_th)
    - Absorption: L = σ⁺, rate = γ·n_th

    Where n_th = 1/(exp(ℏω/kT) - 1) is the thermal occupation.

    Args:
        n_qubits: Total number of qubits
        qubit_idx: Which qubit
        temperature: Temperature in Kelvin
        omega: Energy gap in Hartree

    Returns:
        (operators, rates): Lindblad operators and rates
    """
    # Boltzmann factor
    kT = 3.166811563e-6 * temperature  # kB*T in Hartree (kB = 3.1668e-6 Ha/K)

    if kT > 0:
        n_th = 1.0 / (np.exp(omega / kT) - 1) if omega > 0 else 0
    else:
        n_th = 0

    # Base decay rate (arbitrary, would come from bath spectral density)
    gamma_base = 0.01  # Ha⁻¹

    # Emission operator
    sigma_minus = create_amplitude_damping_operator(n_qubits, qubit_idx)

    # Absorption operator (σ⁺)
    sigma_plus = sigma_minus.conj().T

    operators = [sigma_minus, sigma_plus]
    rates = [gamma_base * (1 + n_th), gamma_base * n_th]

    return operators, rates
