"""Bridge from kanad's VQE hot path onto the public rocm-planck core.

Depends on ``planck`` (the public, open-source ROCm/HIP statevector core). The
dependency is strictly one-directional: ``planck`` never imports ``kanad``. This
keeps DeepRealm chemistry IP out of the public package while letting kanad
offload its three hot-path operations (gate apply, expectation, adjoint gradient)
to the GPU.

Activated by ``VQESolver(..., backend='planck')``. With any other backend none of
this is imported, so the default path is unaffected.
"""
from __future__ import annotations

import numpy as np

# planck is the public package. Import lazily-safe at module load: this module is
# only imported when backend='planck' is explicitly selected.
from planck.circuit import Circuit
from planck.gradient import adjoint_gradient
from planck.pauli import PauliSum


from kanad.backends.base_backend import BaseBackend


class PlanckBackend(BaseBackend):
    """Exact statevector backend backed by the rocm-planck GPU core. Drop-in peer of
    StatevectorBackend: `estimate_expectation` builds |psi> on-GPU and contracts the
    observable (Pauli sums fully on-GPU; non-Pauli observables on the planck-built
    state via Qiskit). `sample` draws bitstrings from the GPU statevector's
    probabilities. Falls back to nothing — requires the `planck` package.
    """

    name = "planck"

    def __init__(self, seed: int | None = None, dtype: str = "complex128", **_ignored):
        self._rng = np.random.default_rng(seed)
        self._dtype = dtype

    def _state(self, circuit):
        from planck.statevector import StateVector
        sv = StateVector(circuit.num_qubits, dtype=self._dtype)
        return Circuit.from_qiskit(circuit).run(sv)

    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        from qiskit.quantum_info import SparsePauliOp, Statevector
        sv = self._state(circuit)
        if isinstance(observable, SparsePauliOp):
            return float(PauliSum.from_qiskit(observable).expectation(sv))   # fully on-GPU
        # non-Pauli observable: contract on the GPU-built state via Qiskit
        return float(np.real_if_close(Statevector(sv.to_numpy()).expectation_value(observable)))

    def sample(self, circuit, shots: int) -> dict:
        sv = self._state(circuit)
        probs = np.abs(sv.to_numpy()) ** 2
        probs = probs / probs.sum()
        n = circuit.num_qubits
        idx = self._rng.choice(len(probs), size=shots, p=probs)
        counts: dict = {}
        for i in idx:
            key = format(int(i), f"0{n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts


def planck_statevector(bound_circuit):
    """Build |psi> for an already-bound Qiskit circuit on-GPU; return planck.StateVector."""
    return Circuit.from_qiskit(bound_circuit).run()


def statevector_data(bound_circuit) -> np.ndarray:
    """|psi> as a dense complex128 array (for wrapping in qiskit Statevector)."""
    return planck_statevector(bound_circuit).to_numpy()


def expectation(planck_sv, sparse_pauli_op) -> float:
    """<psi|H|psi> for a planck.StateVector and a Qiskit SparsePauliOp."""
    return PauliSum.from_qiskit(sparse_pauli_op).expectation(planck_sv)


def energy_from_bound(bound_circuit, sparse_pauli_op) -> float:
    """One call: build |psi> and contract against H, both on-GPU.

    Note: transpiles + parses H every call. For an iterative optimizer, prefer
    PlanckVQEEvaluator (caches both, binds per call) — see vqe_solver integration.
    """
    return expectation(planck_statevector(bound_circuit), sparse_pauli_op)


class PlanckVQEEvaluator:
    """Caches the transpiled circuit template AND the parsed Hamiltonian once, then
    binds parameters per call — eliminating the per-iteration `transpile` +
    `SparsePauliOp` parse that dominated the VQE inner loop.

    `theta` is a parameter array in `qiskit_circuit.parameters` order (the order
    kanad's VQESolver binds in), which matches Circuit.from_qiskit's param indexing.
    """

    def __init__(self, qiskit_circuit, sparse_pauli_op):
        self.parameters = list(qiskit_circuit.parameters)
        self._template = Circuit.from_qiskit(qiskit_circuit)   # transpile ONCE
        self._h = PauliSum.from_qiskit(sparse_pauli_op)        # parse ONCE
        self._last_state = None

    def state(self, theta):
        """|psi(theta)> as a planck.StateVector (cached for a same-theta energy reuse)."""
        self._last_state = self._template.bind(np.asarray(theta, dtype=float)).run()
        return self._last_state

    def energy(self, theta) -> float:
        return self._h.expectation(self.state(theta))


class PlanckAdjointGradient:
    """Duck-typed drop-in for ``AdjointGradientCalculator``.

    Exposes the same surface (``.circuit``, ``.hamiltonian``, ``.parameters``,
    ``.gradient(param_dict)``) so ``VQESolver._build_adjoint_calculator`` can
    return it unchanged. Translation (transpile to the planck basis) happens once
    at construction; each ``gradient`` call binds and runs one forward + one
    backward sweep on the GPU, with H never materialized as a matrix.
    """

    def __init__(self, circuit, sparse_pauli_op):
        self.circuit = circuit
        self.hamiltonian = sparse_pauli_op
        self.parameters = list(circuit.parameters)
        self._template = Circuit.from_qiskit(circuit)
        self._h = PauliSum.from_qiskit(sparse_pauli_op)

    def gradient(self, param_dict):
        theta = np.array([float(param_dict[p]) for p in self.parameters])
        g = adjoint_gradient(self._template.bind(theta), self._h)
        return {p: float(g[i]) for i, p in enumerate(self.parameters)}
