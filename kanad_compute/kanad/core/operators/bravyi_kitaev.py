"""
Bravyi-Kitaev fermion-to-qubit transformation.

Thin wrapper around OpenFermion's verified BK implementation. The previous
in-house transform was structurally broken (anticommutator off by O(1), spectra
off from JW by ~0.37 Ha) and was replaced by this module in M1 (2026-05-26).

OpenFermion (`openfermion==1.7.x`) is already a Kanad dependency.

Conventions
-----------
- The output dict keys use the same Pauli-string convention as
  `kanad.core.operators.jordan_wigner`: a string of length ``n_qubits``,
  read **right-to-left** as qubit 0, 1, …, n−1 (Qiskit little-endian).
- Coefficients are complex; identity is the all-``I`` string.
- All transforms here are bit-correct against OpenFermion (regression tests in
  `tests/unit/test_mapper_algebra.py` and
  `tests/validation/test_mapper_jw_bk_equivalence.py`).

References
----------
- Bravyi & Kitaev, Ann. Phys. 298, 210 (2002).
- Seeley, Richard, Love, J. Chem. Phys. 137, 224109 (2012).
"""

import numpy as np
from typing import Dict, Tuple
from qiskit.quantum_info import SparsePauliOp

from openfermion.transforms import bravyi_kitaev as _of_bravyi_kitaev
from openfermion.ops import FermionOperator as _OFFermionOperator

from kanad.core.operators.fermion_operator import FermionOperator


def _kanad_to_openfermion(fermion_op: FermionOperator) -> _OFFermionOperator:
    """Convert Kanad `FermionOperator` → OpenFermion `FermionOperator`.

    Both store terms as ``{((orbital, action), …): coeff}`` with action=1 for
    creation and 0 for annihilation, so the conversion is a direct rebuild.
    """
    of = _OFFermionOperator()
    for term, coeff in fermion_op.terms.items():
        of += _OFFermionOperator(term, complex(coeff))
    return of


def _openfermion_qubit_op_to_pauli_dict(qubit_op, n_qubits: int) -> Dict[str, complex]:
    """Convert OpenFermion QubitOperator → Kanad pauli-string dict.

    OpenFermion stores terms as tuples of ``(qubit_index, pauli_letter)`` —
    left-to-right qubit ordering. Kanad uses Qiskit little-endian where the
    rightmost string position is qubit 0.
    """
    result: Dict[str, complex] = {}
    for term, coeff in qubit_op.terms.items():
        pauli = ['I'] * n_qubits
        for q, p in term:
            # OpenFermion qubit q → Qiskit position n_qubits - 1 - q
            pauli[n_qubits - 1 - q] = p
        s = ''.join(pauli)
        if s in result:
            result[s] += complex(coeff)
        else:
            result[s] = complex(coeff)
    return {k: v for k, v in result.items() if abs(v) > 1e-15}


def _get_max_orbital(fermion_op: FermionOperator) -> int:
    """Maximum orbital index in the operator (or -1 for identity-only)."""
    max_orb = -1
    for term in fermion_op.terms.keys():
        for orb, _ in term:
            max_orb = max(max_orb, orb)
    return max_orb


def bravyi_kitaev(fermion_op: FermionOperator, n_qubits: int = None) -> Dict[str, complex]:
    """Apply BK transform → Kanad pauli-dict.

    Args:
        fermion_op: input fermionic operator.
        n_qubits: total number of qubits. If ``None``, inferred from the
            operator's highest orbital index. **Strongly recommended to pass
            explicitly** — OpenFermion's BK encoding depends on n_qubits, and
            partial operators (e.g. an isolated ``a_j``) must be transformed in
            the context of the full system size.
    """
    if n_qubits is None:
        n_qubits = _get_max_orbital(fermion_op) + 1
        n_qubits = max(n_qubits, 1)

    of_op = _kanad_to_openfermion(fermion_op)
    of_qubit = _of_bravyi_kitaev(of_op, n_qubits=n_qubits)
    return _openfermion_qubit_op_to_pauli_dict(of_qubit, n_qubits)


def bravyi_kitaev_sparse_pauli_op(fermion_op: FermionOperator, n_qubits: int = None) -> SparsePauliOp:
    """Apply BK transform → Qiskit ``SparsePauliOp``."""
    pauli_dict = bravyi_kitaev(fermion_op, n_qubits=n_qubits)

    if len(pauli_dict) == 0:
        n = n_qubits if n_qubits is not None else 1
        return SparsePauliOp(['I' * n], [0.0])

    pauli_strings = list(pauli_dict.keys())
    coefficients = list(pauli_dict.values())
    return SparsePauliOp(pauli_strings, coefficients)


def build_molecular_hamiltonian_bk(
    h_mo: np.ndarray,
    eri_mo: np.ndarray,
    nuclear_repulsion: float,
    n_electrons: int = None,
) -> SparsePauliOp:
    """Build molecular Hamiltonian in BK encoding.

    Same fermionic Hamiltonian construction as
    `kanad.core.operators.jordan_wigner.build_molecular_hamiltonian_jw`; only
    the qubit transform differs.

    Args:
        h_mo: one-electron integrals in MO basis ``(n_orb, n_orb)``.
        eri_mo: two-electron integrals in MO basis, chemist notation
            ``⟨pq|rs⟩`` indexed ``[p, q, r, s]``.
        nuclear_repulsion: scalar nuclear repulsion energy.
        n_electrons: accepted for API parity with the JW builder; not used.

    Returns:
        Qiskit ``SparsePauliOp`` on ``2 * n_orb`` qubits (spin orbitals).
    """
    n_orbitals = h_mo.shape[0]
    n_qubits = 2 * n_orbitals

    fermion_ham = FermionOperator((), nuclear_repulsion)

    # One-body
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            if abs(h_mo[p, q]) < 1e-12:
                continue
            fermion_ham += FermionOperator(((2*p,     1), (2*q,     0)), h_mo[p, q])
            fermion_ham += FermionOperator(((2*p + 1, 1), (2*q + 1, 0)), h_mo[p, q])

    # Two-body
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    g = eri_mo[p, q, r, s]
                    if abs(g) < 1e-12:
                        continue
                    coeff = 0.5 * g
                    fermion_ham += FermionOperator(((2*p,     1), (2*r,     1), (2*s,     0), (2*q,     0)), coeff)
                    fermion_ham += FermionOperator(((2*p,     1), (2*r + 1, 1), (2*s + 1, 0), (2*q,     0)), coeff)
                    fermion_ham += FermionOperator(((2*p + 1, 1), (2*r,     1), (2*s,     0), (2*q + 1, 0)), coeff)
                    fermion_ham += FermionOperator(((2*p + 1, 1), (2*r + 1, 1), (2*s + 1, 0), (2*q + 1, 0)), coeff)

    return bravyi_kitaev_sparse_pauli_op(fermion_ham, n_qubits=n_qubits)
