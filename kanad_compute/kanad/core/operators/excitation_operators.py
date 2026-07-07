"""Fermionic excitation generators (core.operators.excitation_operators).

Indigenous home for the Hermitian single/double excitation generator
``G = T - T†`` as a ``SparsePauliOp`` — previously built five different ways
(physics_vqe, hardware_vqe, givens_sd, physics_driven, efficient_excitation),
all producing the SAME matrix for the canonical convention. Reorg Phase B4.

CANON: the physics_vqe construction (the only validated true-VQE loop, <=1 mHa on
H2/HeH+/LiH/HF/HCl/F2/H2O STO-3G). Single (i->a): T = a+_a a_i - a+_i a_a.
Double ((i,j)->(a,b)): T = a+_a a+_b a_j a_i - a+_i a+_j a_b a_a. Then
``jordan_wigner(T)`` and a hand Hermitian extraction (imag-dominant terms *-1j,
take real) — NO .simplify(), bit-for-bit identical to physics_vqe. Pair with
``PauliEvolutionGate(H, time=-theta)`` => exp(+i*theta*H) (the canon gate sign).
Verified A==B (this generator == givens_sd's (-1j)*coeffs form) for single &
double at machine precision.
"""

from __future__ import annotations


def build_excitation_generator(occ, virt, n_qubits: int, mapper='jordan_wigner'):
    """Hermitian excitation generator ``G = T - T†`` as a ``SparsePauliOp``.

    Args:
        occ: occupied spin-orbital index/indices — ``(i,)`` single, ``(i, j)`` double.
        virt: virtual spin-orbital index/indices — ``(a,)`` single, ``(a, b)`` double.
        n_qubits: total spin-orbitals (qubits).
        mapper: ``'jordan_wigner'`` (canon) / ``'bravyi_kitaev'`` or a mapper instance.

    Returns:
        ``SparsePauliOp`` Hermitian generator (NOT simplified — matches the
        physics_vqe reference bit-for-bit). Use with ``PauliEvolutionGate(H, time=-theta)``.
    """
    from qiskit.quantum_info import SparsePauliOp
    from kanad.core.operators.fermion_operator import FermionOperator

    occ = tuple(occ); virt = tuple(virt)
    if len(occ) == 1 and len(virt) == 1:
        i, a = occ[0], virt[0]
        T = FermionOperator(((a, 1), (i, 0)), 1.0)
        T += FermionOperator(((i, 1), (a, 0)), -1.0)
    elif len(occ) == 2 and len(virt) == 2:
        i, j = occ[0], occ[1]
        a, b = virt[0], virt[1]
        T = FermionOperator(((a, 1), (b, 1), (j, 0), (i, 0)), 1.0)
        T += FermionOperator(((i, 1), (j, 1), (b, 0), (a, 0)), -1.0)
    else:
        raise ValueError(
            f"build_excitation_generator expects single (1 occ, 1 virt) or double "
            f"(2 occ, 2 virt); got occ={occ}, virt={virt}."
        )

    is_jw = (mapper == 'jordan_wigner' or 'JordanWigner' in type(mapper).__name__)
    is_bk = (mapper == 'bravyi_kitaev' or 'BravyiKitaev' in type(mapper).__name__)
    if is_jw:
        from kanad.core.operators.jordan_wigner import jordan_wigner
        pauli_dict = jordan_wigner(T, n_qubits)
    elif is_bk:
        from kanad.core.operators.bravyi_kitaev import bravyi_kitaev
        pauli_dict = bravyi_kitaev(T, n_qubits)
    else:
        raise NotImplementedError(
            f"build_excitation_generator supports 'jordan_wigner'/'bravyi_kitaev'; got {mapper!r}."
        )

    # Hermitian extraction (verbatim from physics_vqe canon): imag-dominant
    # terms -> (-1j*c).real, else c.real. NO simplify (bit-identical to reference).
    pauli_strings, coeffs = [], []
    for ps, c in pauli_dict.items():
        pauli_strings.append(ps)
        if abs(c.imag) > abs(c.real):
            coeffs.append((-1j * c).real)
        else:
            coeffs.append(c.real)
    return SparsePauliOp(pauli_strings, coeffs)
