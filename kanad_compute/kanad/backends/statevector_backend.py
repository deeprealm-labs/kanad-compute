"""Exact statevector backend (replaces the legacy ``_use_statevector`` flag)."""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Statevector

from kanad.backends.base_backend import BaseBackend


class StatevectorBackend(BaseBackend):
    """Classical exact backend: full statevector simulation.

    ``estimate_expectation`` is exact (no shot noise). ``sample`` draws
    bitstrings from the exact probability distribution using a seedable RNG.
    Accepts and ignores arbitrary backend kwargs so solver-specific extras
    forwarded through ``make_backend`` never crash the statevector path.
    """

    name = "statevector"

    def __init__(self, seed: int | None = None, **_ignored):
        self._rng = np.random.default_rng(seed)

    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        sv = Statevector(circuit)
        return float(np.real_if_close(sv.expectation_value(observable)))

    def sample(self, circuit, shots: int) -> dict[str, int]:
        sv = Statevector(circuit)
        probs = np.asarray(sv.probabilities())
        n = circuit.num_qubits
        idx = self._rng.choice(len(probs), size=shots, p=probs)
        counts: dict[str, int] = {}
        for i in idx:
            key = format(int(i), f"0{n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts
