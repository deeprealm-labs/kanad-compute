"""
Smart Solver - Automatically chooses the most efficient method.

This solver addresses a critical issue: VQE is WASTEFUL for small molecules!

Reality Check:
- H2 (4 qubits): Classical FCI takes 4ms, VQE takes 500+ iterations
- LiH (12 qubits): Classical FCI takes <1s, VQE takes thousands of evaluations
- H2O (14 qubits): Classical FCI takes ~1s, still faster than VQE

Quantum advantage only starts at ~50+ qubits!

SmartSolver automatically:
1. Uses classical FCI for small systems (< 24 qubits) - EXACT & FAST
2. Uses VQE only when classical methods become intractable
3. Provides honest cost estimates before running quantum circuits

This is scientifically honest and practically useful.

Example:
    >>> from kanad.solvers import SmartSolver
    >>> solver = SmartSolver(bond)
    >>> result = solver.solve()  # -> SolverResult
    >>> print(result.extra['method'])  # 'classical_fci' or 'vqe'
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import time

from kanad.solvers.base_solver import BaseSolver

logger = logging.getLogger(__name__)


class SmartSolver(BaseSolver):
    """
    Intelligent solver that automatically chooses the most efficient method.

    For small systems (< 24 qubits): Uses classical exact diagonalization
    For large systems (>= 24 qubits): Uses VQE with optimized settings

    This is scientifically honest - we don't waste quantum resources
    on problems that classical computers solve instantly.
    """

    # Thresholds for method selection
    CLASSICAL_FCI_THRESHOLD = 24  # qubits (2^24 = 16M, still fast)
    CLASSICAL_APPROX_THRESHOLD = 40  # qubits (use DMRG/selected CI)

    def __init__(
        self,
        system=None,
        *,
        bond=None,
        hamiltonian=None,
        force_method: Optional[str] = None,
        verbose: bool = True,
        backend: str = 'statevector',
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        **backend_kwargs,
    ):
        """
        Initialize SmartSolver.

        Args:
            system: Bond / Molecule / Hamiltonian / QuantumSystem (unified solver
                protocol). Mapped onto the legacy ``bond`` slot.
            bond: Bond object (legacy alias for ``system``).
            hamiltonian: Hamiltonian object (alternative to bond/system).
            force_method: Force specific method ('classical', 'vqe').
            verbose: Print method selection rationale.
            backend: Backend name resolved via the backend factory.
            enable_analysis: Enable automatic analysis (default: True).
            enable_optimization: Enable automatic optimization (default: True).
            **backend_kwargs: Backend construction params forwarded to make_backend.
        """
        # Unified solver protocol: collapse the legacy bond=/hamiltonian= kwargs
        # onto the positional `system` that BaseSolver._resolve_system handles.
        if system is None:
            if bond is not None:
                system = bond
            elif hamiltonian is not None:
                system = hamiltonian
            else:
                raise ValueError("Must provide either system, bond, or hamiltonian")

        super().__init__(
            system,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **backend_kwargs,
        )

        from kanad.backends.statevector_backend import StatevectorBackend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        self.force_method = force_method
        self.verbose = verbose

        # Determine system size (n_qubits = 2 * n_orbitals, set after super().__init__
        # so self.hamiltonian is resolved).
        self.n_qubits = 2 * self.hamiltonian.n_orbitals
        self.hilbert_dim = 2 ** self.n_qubits

        if verbose:
            print(f"\n{'='*60}")
            print(f"SmartSolver: {self.n_qubits} qubits, {self.hilbert_dim:,} Hilbert dim")
            print(f"{'='*60}")

    def _select_method(self) -> str:
        """Select optimal method based on system size."""
        if self.force_method:
            # Normalize documented tokens to internal dispatch keys; the public
            # 'classical' value must map to 'classical_fci' or solve() falls
            # through to the VQE branch. Raise on anything unrecognized.
            _alias = {'classical': 'classical_fci', 'vqe': 'vqe'}
            normalized = _alias.get(self.force_method, self.force_method)
            if normalized not in ('classical_fci', 'classical_approx', 'vqe'):
                raise ValueError(
                    f"Unknown force_method '{self.force_method}'. "
                    f"Expected one of: 'classical', 'vqe'."
                )
            return normalized

        if self.n_qubits < self.CLASSICAL_FCI_THRESHOLD:
            return 'classical_fci'
        elif self.n_qubits < self.CLASSICAL_APPROX_THRESHOLD:
            return 'classical_approx'
        else:
            return 'vqe'

    def _estimate_vqe_cost(self) -> Dict[str, Any]:
        """Estimate VQE computational cost."""
        # Rough estimates based on empirical data
        n_params = 4 * self.n_qubits  # Approximate for HW-efficient
        n_iterations = 50 + 10 * self.n_qubits
        evals_per_iter = 2 * n_params + 1  # For gradient estimation
        total_evals = n_iterations * evals_per_iter

        # Cost on various platforms (rough estimates)
        cost_per_eval_ibm = 0.01  # $ per circuit on IBM
        cost_per_eval_ionq = 0.10  # $ per circuit on IonQ

        return {
            'n_parameters': n_params,
            'estimated_iterations': n_iterations,
            'estimated_evaluations': total_evals,
            'estimated_cost_ibm': total_evals * cost_per_eval_ibm,
            'estimated_cost_ionq': total_evals * cost_per_eval_ionq
        }

    def solve(self) -> 'SolverResult':
        """
        Solve for ground state using optimal method.

        Returns:
            A :class:`SolverResult` (solver tag ``"smart"``). The chosen method
            (``classical_fci`` / ``classical_approx`` / ``vqe``) and cost-saving
            metadata live in ``result.extra``; the canonical energy is
            ``result.energy``.
        """
        from kanad.core.solver_result import SolverResult

        method = self._select_method()

        if self.verbose:
            print(f"\nSelected method: {method.upper()}")

        start_time = time.time()

        if method == 'classical_fci':
            result = self._solve_classical_fci()
        elif method == 'classical_approx':
            result = self._solve_classical_approx()
        else:
            result = self._solve_vqe()

        elapsed = time.time() - start_time
        result['time'] = elapsed
        result['method'] = method

        # Estimate savings
        if method == 'classical_fci':
            vqe_cost = self._estimate_vqe_cost()
            result['vqe_would_cost'] = vqe_cost
            result['quantum_circuits_saved'] = vqe_cost['estimated_evaluations']

            if self.verbose:
                print(f"\n✓ Classical FCI completed in {elapsed*1000:.1f} ms")
                print(f"✓ Result is EXACT (machine precision)")
                print(f"✓ Saved ~{vqe_cost['estimated_evaluations']:,} quantum circuit evaluations")
                print(f"✓ Saved ~${vqe_cost['estimated_cost_ibm']:.2f} on IBM Quantum")

        # Keep the legacy dict on self.results for inspection, then wrap in the
        # unified SolverResult envelope. The math is unchanged — only the return
        # type changes.
        self.results = dict(result)
        return SolverResult.from_mapping(result, solver="smart", backend=self.backend_name)

    def _solve_classical_fci(self) -> Dict[str, Any]:
        """Solve using PySCF FCI (FAST and EXACT)."""
        from pyscf import scf, fci

        if self.verbose:
            print(f"  Running PySCF FCI (fast, exact)...")

        # Get PySCF mol object from Hamiltonian
        mol = self.hamiltonian.mol

        # Run HF (fast)
        mf = scf.RHF(mol)
        mf.verbose = 0
        mf.kernel()

        # Run FCI (exact)
        cisolver = fci.FCI(mf)
        e_fci, fcivec = cisolver.kernel()

        ground_state_energy = float(e_fci)

        return {
            'energy': ground_state_energy,
            'hf_energy': float(mf.e_tot),
            'correlation_energy': ground_state_energy - float(mf.e_tot),
            'fci_vector': fcivec,
            'accuracy': 'exact (FCI)',
            'error_estimate': 1e-10  # PySCF convergence
        }

    def _solve_classical_approx(self) -> Dict[str, Any]:
        """Solve using classical approximations (DMRG, selected CI)."""
        # For now, fall back to sparse diagonalization
        # In future, could use DMRG from pyscf or other methods

        from scipy.sparse.linalg import eigsh

        if self.verbose:
            print(f"  Using sparse diagonalization (Lanczos)...")

        sparse_ham = self.hamiltonian.to_sparse_hamiltonian()
        # Keep the operator sparse: SparsePauliOp.to_matrix() defaults to a dense
        # 2^n x 2^n array (MemoryError for 24-40 qubits). sparse=True hands eigsh
        # the sparse matrix it expects.
        H_sparse = sparse_ham.to_matrix(sparse=True)

        # Find lowest eigenvalue
        eigenvalues, eigenvectors = eigsh(H_sparse, k=1, which='SA')

        return {
            'energy': float(eigenvalues[0]),
            'ground_state': eigenvectors[:, 0],
            'accuracy': 'near-exact (Lanczos)',
            'error_estimate': 1e-10
        }

    def _solve_vqe(self) -> Dict[str, Any]:
        """Solve using VQE (only for large systems)."""
        from kanad.solvers import VQESolver
        from kanad.core.ansatze import HardwareEfficientAnsatz

        if self.verbose:
            cost = self._estimate_vqe_cost()
            print(f"\n⚠️  VQE required for {self.n_qubits}-qubit system")
            print(f"   Estimated cost: ~{cost['estimated_evaluations']:,} circuit evaluations")
            print(f"   Estimated time: significant (depends on backend)")

        # Use optimized settings discovered earlier
        ansatz = HardwareEfficientAnsatz(
            n_qubits=self.n_qubits,
            n_electrons=self.hamiltonian.n_electrons,
            n_layers=2
        )

        solver = VQESolver(
            hamiltonian=self.hamiltonian,
            ansatz=ansatz,
            # Use the gradient-based default (L-BFGS-B). The old COBYLA + all-zero
            # start sat on the reference-state plateau (∂E/∂θ at θ=0 is structurally
            # zero for HEA) and the gradient-free optimizer never escaped, so 'VQE'
            # silently returned <HF-circuit|H|HF-circuit> instead of the variational min.
            max_iterations=200,
            backend=self.backend_name,
        )

        # Let VQESolver pick its own non-trivial init (random θ∈[-0.1,0.1]); do NOT
        # pass all-zeros, which would re-create the plateau the optimizer cannot leave.
        # VQESolver.solve() now returns a SolverResult — read .energy / .extra, not
        # subscripts.
        inner_result = solver.solve()

        return {
            'energy': inner_result.energy,
            # VQESolver surfaces the converged params under 'parameters' in extra;
            # SolverResult.to_dict() flattens extra to the top level.
            'optimal_parameters': inner_result.to_dict().get('parameters', []),
            'accuracy': 'approximate (VQE)',
            'error_estimate': 0.002  # ~2 mHa typical
        }


def solve_smart(bond=None, hamiltonian=None, verbose=True) -> 'SolverResult':
    """
    Convenience function to solve using SmartSolver.

    Args:
        bond: Bond object
        hamiltonian: Hamiltonian object
        verbose: Print status

    Returns:
        A :class:`SolverResult` (solver tag ``"smart"``).
    """
    solver = SmartSolver(bond=bond, hamiltonian=hamiltonian, verbose=verbose)
    return solver.solve()
