"""Backend protocol: the two operations every Kanad solver needs.

Solvers consume backends through exactly two operations:
  - ``estimate_expectation`` — expectation value of an observable (VQE family),
  - ``sample`` — bitstring counts from measuring a circuit (SQD family).

``statevector`` is a real :class:`~kanad.backends.statevector_backend.StatevectorBackend`
implementing both exactly; cloud backends implement them on top of their native
``run_*`` execution paths. This replaces the legacy ``_use_statevector`` boolean
flag + string-dispatch in ``BaseSolver._init_backend``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBackend(ABC):
    name: str = "base"

    @abstractmethod
    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        """Return ``<circuit| observable |circuit>``.

        ``shots=None`` requests an exact value where the backend supports it
        (statevector); shot-based backends use a default shot count.
        """

    @abstractmethod
    def sample(self, circuit, shots: int) -> dict[str, int]:
        """Return ``{bitstring: count}`` from measuring ``circuit``.

        Bitstrings use little-endian Qiskit qubit order (qubit 0 is the
        rightmost character), matching ``qiskit.result.Counts``.
        """


def expectation_from_counts(counts: dict[str, int], observable) -> float:
    """Diagonal (Z-basis) expectation value from measurement counts.

    Shared by the cloud backends. Only valid for observables whose Pauli terms
    contain only I and Z; a non-diagonal observable must be measured in a
    rotated basis by the caller, so this raises for X/Y terms rather than
    returning a silently-wrong number.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for pauli, coeff in zip(observable.paulis, observable.coeffs):
        label = pauli.to_label()
        if set(label) - {"I", "Z"}:
            raise NotImplementedError(
                "expectation_from_counts requires a Z-basis (diagonal) observable; "
                f"got non-diagonal term {label!r}"
            )
        val = 0.0
        for bits, c in counts.items():
            # label is big-endian (qubit n-1 .. 0); bits string is also big-endian
            # in Qiskit Counts. Parity over the Z positions.
            parity = sum(int(b) for b, p in zip(bits, label) if p == "Z")
            val += c * ((-1) ** parity)
        exp += float(np.real(coeff)) * val / total
    return exp
