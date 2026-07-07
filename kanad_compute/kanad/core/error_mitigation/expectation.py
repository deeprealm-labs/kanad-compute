"""Count -> expectation-value conversion (core.error_mitigation.expectation).

Indigenous home for estimating <psi|H|psi> from computational-basis (Z) shot
counts. Previously reimplemented in three solvers (physics_vqe, vqe_solver,
deterministic_ci). Reorg Phase B5, 2026-05-31.

Convention: CORRECT Qiskit big-endian. ``observable.paulis.to_labels()[k][i]``
is the Pauli on qubit ``n-1-i``; a Qiskit ``get_counts`` bitstring has
``bitstring[i]`` = qubit ``n-1-i`` — so label position ``i`` pairs with bitstring
position ``i`` directly (no reversal). This matches physics_vqe._energy_from_counts.

LOAD-BEARING HONESTY GUARD: X/Y Pauli terms CANNOT be read from Z-basis counts
(they need a rotated measurement basis). Raising is mandatory — silently treating
them as identity/zero returns a plausible-but-wrong energy. Pure numpy + a Qiskit
SparsePauliOp argument; NO kanad.backends / kanad.solvers imports.
"""

from __future__ import annotations


def expectation_from_counts(observable, counts) -> float:
    """Estimate ``<psi|H|psi>`` from Z-basis measurement counts.

    Args:
        observable: Hamiltonian as a Qiskit ``SparsePauliOp``.
        counts: ``{bitstring: shots}`` from ``get_counts`` (IBM register spaces ok).

    Returns:
        Real expectation value.

    Raises:
        NotImplementedError: if any term contains an X or Y Pauli (not measurable
            from Z-basis counts without basis rotation).
    """
    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("expectation_from_counts: counts sum to zero shots.")

    energy = 0.0
    for label, coeff in zip(observable.paulis.to_labels(), observable.coeffs):
        c = complex(coeff).real
        if abs(c) < 1e-12:
            continue
        if all(ch == 'I' for ch in label):
            energy += c
            continue
        term = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')  # strip IBM register spaces
            eigenvalue = 1.0
            for i, ch in enumerate(label):
                if ch == 'I':
                    continue
                if ch == 'Z':
                    eigenvalue *= (1 - 2 * int(bits[i]))  # big-endian: label[i] <-> bits[i]
                elif ch in ('X', 'Y'):
                    raise NotImplementedError(
                        f"expectation_from_counts cannot estimate '{ch}' Pauli terms "
                        f"from Z-basis counts; measure in the rotated basis or use the "
                        f"statevector path."
                    )
            term += eigenvalue * count
        energy += c * (term / total_shots)
    return energy
