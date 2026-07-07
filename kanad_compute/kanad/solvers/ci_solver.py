"""
Classical Configuration Interaction in a sampled subspace.

This solver runs what the framework historically called "Hi-VQE" — but Hi-VQE is
not a variational quantum eigensolver: it samples Z-basis configurations from a
parameterized circuit, builds a configuration subspace, and diagonalizes the
Hamiltonian classically in that subspace. On systems where the subspace equals
the full Hilbert space (e.g. ≤14-qubit STO-3G molecules) it returns FCI by
construction. The wavefunction is built classically; there is no variational
quantum step.

`CISolver` exposes this algorithm under its true name. For real VQE, use
`VQESolver` (default mode is now 'standard').
"""

from kanad.solvers.vqe_solver import VQESolver


class CISolver(VQESolver):
    """Classical CI in a sampled configuration subspace.

    Wraps `VQESolver(mode='hivqe')` so the algorithm is callable under a name
    that matches what it actually does. The implementation lives on
    `HiVQESolverMixin._solve_hivqe` (used by VQESolver when mode='hivqe').
    """

    def __init__(self, system=None, **kwargs):
        # Unified solver protocol: a single positional `system` (Bond /
        # QuantumSystem / Molecule / bare Hamiltonian) forwarded to VQESolver.
        # Force mode='hivqe' — drop any caller-supplied mode kwarg.
        kwargs.pop('mode', None)
        # Suppress the deprecation warning VQESolver emits for mode='hivqe' —
        # CISolver is the supported entry point for this algorithm.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            super().__init__(system, mode='hivqe', **kwargs)
