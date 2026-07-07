"""Solver auto-dispatch for the Molecular Builder.

One tunable place that maps ``(n_qubits, backend)`` to a solver route. The
thresholds encode the framework owner's rule: classical CI is exact and
instant for small systems, VQE is the quantum method in the ~20–30 qubit band,
and SQD (sample-based diagonalization) carries the large / hardware regime.

This is intentionally separate from `SmartSolver` (solvers/smart_solver.py),
which has its own {classical_fci, classical_approx, vqe} vocabulary and
external consumers. `SolverRouter` is the single source of truth for the
builder's ``solver='auto'`` path.
"""

from __future__ import annotations


class SolverRouter:
    """Route ``solver='auto'`` to a concrete method by system size / backend."""

    # Below this, exact classical CI (CASCI/FCI on the active space) is both
    # exact and effectively instant — no reason to run an approximate quantum
    # method. 2^20 ≈ 1M-dim Hilbert space.
    CI_MAX_QUBITS = 20
    # Owner's rule: VQE is the quantum method for ~20–30 qubit simulations.
    VQE_MAX_QUBITS = 30
    # Sampling-native backends always route to SQD (VQE/CI assume a statevector
    # or exact integrals; a shot/hardware backend is what SQD is built for).
    SAMPLING_BACKENDS = ('ibm', 'bluequbit', 'ionq', 'qasm')

    @classmethod
    def select(cls, n_qubits: int, backend: str = 'statevector') -> str:
        """Return one of ``'ci'`` | ``'vqe'`` | ``'sqd'``.

        Args:
            n_qubits: ``2 * n_active_orbitals`` for the materialized system.
            backend: execution backend; any sampling/hardware backend forces SQD.
        """
        if backend in cls.SAMPLING_BACKENDS:
            return 'sqd'
        if n_qubits <= cls.CI_MAX_QUBITS:
            return 'ci'
        if n_qubits <= cls.VQE_MAX_QUBITS:
            return 'vqe'
        return 'sqd'
