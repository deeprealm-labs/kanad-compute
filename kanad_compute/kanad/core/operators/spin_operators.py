"""Spin / number operators in the qubit basis (core.operators.spin_operators).

Indigenous home for the number (N), spin-projection (Sz), and total-spin (S2)
operators as ``SparsePauliOp`` — previously built two separate ways: the JW
``FermionOperator`` route in core/density/quantum_rdm.py (Sz, S2) and raw Pauli-Z
literals in solvers/vqe_solver.py (N, Sz penalties). Reorg Phase B4, 2026-05-31.

CANON: INTERLEAVED Jordan-Wigner (alpha = even qubit 2p, beta = odd 2p+1) on
``2*n_orbitals`` qubits — the convention already used by build_molecular_hamiltonian_jw,
the vqe_solver penalties, and spin_squared_from_statevector. The JW path here is
bit-identical (after .simplify()) to all of those:
  N  = sum_q n_q                       == (n_qubits/2) I - 1/2 sum_q Z_q   (vqe N_op)
  Sz = 1/2 sum_p (n_2p - n_2p+1)       == sum_p (-0.25 Z_2p + 0.25 Z_2p+1) (vqe Sz_op)
  S2 = Sz^2 + 1/2 (S+ S- + S- S+),  S+ = sum_p a+_2p a_2p+1,  S- = h.c.    (quantum_rdm S2)
"""

from __future__ import annotations


def build_spin_operators(n_orbitals: int, mapper='jordan_wigner'):
    """Return ``(N, Sz, S2)`` as ``SparsePauliOp`` on ``2*n_orbitals`` qubits.

    Args:
        n_orbitals: number of spatial orbitals (qubits = 2*n_orbitals).
        mapper: ``'jordan_wigner'`` (or a ``JordanWignerMapper`` instance). The
            interleaved-JW path is the validated canon. Other mappers raise
            ``NotImplementedError`` until their spin-operator route is validated.

    Returns:
        ``(N, Sz, S2)`` each a simplified ``SparsePauliOp``.
    """
    from qiskit.quantum_info import SparsePauliOp
    from kanad.core.operators.fermion_operator import FermionOperator

    is_jw = (mapper == 'jordan_wigner'
             or 'JordanWigner' in type(mapper).__name__)
    if not is_jw:
        raise NotImplementedError(
            f"build_spin_operators currently supports interleaved Jordan-Wigner only; "
            f"got mapper={mapper!r}. Add a validated route before using other mappers."
        )

    from kanad.core.operators.jordan_wigner import jordan_wigner
    n_qubits = 2 * n_orbitals

    def f2p(ferm):
        d = jordan_wigner(ferm, n_qubits=n_qubits)
        if not d:
            return SparsePauliOp(['I' * n_qubits], [0.0])
        return SparsePauliOp(list(d.keys()), list(d.values()))

    # N = sum over spin-orbitals of the number operator
    N = SparsePauliOp(['I' * n_qubits], [0.0])
    for q in range(n_qubits):
        N = N + f2p(FermionOperator(((q, 1), (q, 0))))
    N = N.simplify()

    # Sz = 1/2 sum_p (n_pα - n_pβ), interleaved (alpha=2p, beta=2p+1)
    Sz = SparsePauliOp(['I' * n_qubits], [0.0])
    for p in range(n_orbitals):
        Sz = Sz + 0.5 * f2p(FermionOperator(((2 * p, 1), (2 * p, 0))))
        Sz = Sz + (-0.5) * f2p(FermionOperator(((2 * p + 1, 1), (2 * p + 1, 0))))
    Sz = Sz.simplify()

    # S+ = sum_p a+_pα a_pβ ;  S- = sum_p a+_pβ a_pα
    s_plus = SparsePauliOp(['I' * n_qubits], [0.0])
    s_minus = SparsePauliOp(['I' * n_qubits], [0.0])
    for p in range(n_orbitals):
        s_plus = s_plus + f2p(FermionOperator(((2 * p, 1), (2 * p + 1, 0))))
        s_minus = s_minus + f2p(FermionOperator(((2 * p + 1, 1), (2 * p, 0))))
    s_plus = s_plus.simplify()
    s_minus = s_minus.simplify()

    # S2 = Sz^2 + 1/2 (S+ S- + S- S+)
    S2 = ((Sz @ Sz) + 0.5 * ((s_plus @ s_minus) + (s_minus @ s_plus))).simplify()

    return N, Sz, S2
