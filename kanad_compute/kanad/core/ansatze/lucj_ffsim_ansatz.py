"""CCSD-initialized LUCJ ansatz via ffsim — a CORRELATED, physically-parametrized SQD seed
state with NO VQE optimization (the IBM-SQD standard).

kanad's native ``LUCJAnsatz`` leaves parameters free (the builder seeds them with a hand-tuned
heuristic); with unset params it collapses to Hartree-Fock. This ansatz instead initializes the
LUCJ generators from **classical CCSD t2 amplitudes** (run once on the active space), giving a
correlated trial state whose computational-basis samples populate the dominant determinants —
exactly what SQD needs. Uses ``ffsim`` (the validated qiskit-addon-sqd companion) for the
CCSD->LUCJ map.

Convention: ffsim orders qubits spin-blocked (alpha orbitals then beta orbitals); kanad/JW is
interleaved (alpha at 2p, beta at 2p+1). We append the ffsim gates on a remapped qubit list so
the produced circuit — and the bitstrings it samples — are in kanad's interleaved convention,
ready to feed kanad's selected-CI / the hybrid det-CI engine directly.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class LUCJFfsimAnsatz:
    """LUCJ initialized from active-space CCSD t2 amplitudes (correlated, no VQE).

    Build from a kanad ActiveHamiltonian (`from_hamiltonian`) or directly from a pyscf
    mean-field + active-orbital list. `build_circuit()` returns a Qiskit circuit in kanad's
    interleaved-JW convention; `.t2`, `.ccsd_energy` are exposed for diagnostics.
    """

    def __init__(self, mf, active_orbitals: List[int], n_electrons_active: int,
                 n_reps: int = 2):
        self.mf = mf
        self.active_orbitals = sorted(int(i) for i in active_orbitals)
        self.norb = len(self.active_orbitals)
        self.n_electrons = int(n_electrons_active)
        self.na = self.nb = self.n_electrons // 2
        if self.n_electrons % 2:
            raise NotImplementedError("LUCJFfsimAnsatz: closed-shell (Sz=0) active spaces only")
        self.n_reps = int(n_reps)
        self.n_qubits = 2 * self.norb
        self.t2 = None
        self.ccsd_energy = None
        self.circuit = None

    @classmethod
    def from_hamiltonian(cls, ham, n_reps: int = 2):
        """Build from a kanad ActiveHamiltonian (uses ham.mf + ham.active_orbitals)."""
        active = getattr(ham, "active_orbitals", None)
        if active is None:
            raise ValueError("from_hamiltonian needs an ActiveHamiltonian with .active_orbitals")
        return cls(ham.mf, list(active), int(ham.n_electrons), n_reps=n_reps)

    def build_circuit(self):
        from pyscf import cc
        import ffsim
        from ffsim.qiskit import PrepareHartreeFockJW, UCJOpSpinBalancedJW
        from qiskit import QuantumCircuit

        n_all = self.mf.mo_coeff.shape[1]
        # active-space CCSD: freeze EVERYTHING outside the active window (core + out-of-CAS
        # virtuals), so t2 has shape (n_occ_act, n_occ_act, n_virt_act, n_virt_act).
        frozen = [i for i in range(n_all) if i not in self.active_orbitals]
        mycc = cc.CCSD(self.mf, frozen=frozen)
        mycc.verbose = 0
        self.ccsd_energy = float(mycc.kernel()[0]) + float(self.mf.e_tot)
        self.t2 = mycc.t2
        nocc = self.t2.shape[0]
        if nocc != self.na:
            raise RuntimeError(f"active CCSD nocc={nocc} != expected {self.na}; check active space")

        op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(self.t2, n_reps=self.n_reps)

        # ffsim spin-blocked qubit i -> kanad interleaved: alpha orb p (i=p) -> 2p ;
        # beta orb p (i=norb+p) -> 2p+1.
        qmap = [2 * p for p in range(self.norb)] + [2 * p + 1 for p in range(self.norb)]
        qc = QuantumCircuit(self.n_qubits)
        qc.append(PrepareHartreeFockJW(self.norb, (self.na, self.nb)), qmap)
        qc.append(UCJOpSpinBalancedJW(op), qmap)
        self.circuit = qc.decompose(reps=4)        # lower ffsim gates to a runnable circuit
        return self.circuit

    def to_qiskit(self):
        return self.circuit if self.circuit is not None else self.build_circuit()
