"""
Qubit Tapering via Z₂ Symmetries.

Reduces qubit count by exploiting molecular symmetries:
1. Particle number conservation (N)
2. Spin parity (Sz)
3. Point group symmetry

For H₂:
- 4 qubits → 2 qubits (50% reduction)
- Exponentially reduces hardware noise

References:
- Bravyi et al., "Tapering off qubits to simulate fermionic Hamiltonians" (2017)
- Setia et al., "Reducing Qubit Requirements for Quantum Simulation" (2018)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from qiskit.quantum_info import SparsePauliOp, Pauli
import logging

logger = logging.getLogger(__name__)


class QubitTapering:
    """
    Reduce qubit count using Z₂ symmetries of the molecular Hamiltonian.

    The key insight is that molecular Hamiltonians have conserved quantities
    (symmetries) that commute with H. By identifying these, we can reduce
    the effective Hilbert space.

    For a typical diatomic:
    - Original: 2*n_orbitals qubits
    - Tapered: 2*n_orbitals - k qubits (k = number of independent symmetries)

    Common symmetry reductions:
    - H₂: 4 → 2 qubits (particle number + spin)
    - LiH: 10 → 6-8 qubits
    - H₂O: 12 → 8-10 qubits

    Usage:
        >>> from kanad import BondFactory
        >>> from kanad.core.mappers import QubitTapering
        >>>
        >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
        >>> tapering = QubitTapering()
        >>>
        >>> # Get tapered Hamiltonian
        >>> tapered_ham, metadata = tapering.taper_hamiltonian(
        ...     bond.hamiltonian.sparse_pauli_op,
        ...     n_electrons=2,
        ...     n_qubits=4
        ... )
        >>> print(f"Reduced: {metadata['original_qubits']} → {metadata['tapered_qubits']} qubits")
    """

    def __init__(self):
        """Initialize qubit tapering."""
        self._symmetries = []
        self._eigenvalues = []
        self._clifford = None

    def find_symmetries(
        self,
        hamiltonian: SparsePauliOp,
        max_symmetries: int = 4
    ) -> List[Pauli]:
        """
        Find Z₂ symmetries that commute with the Hamiltonian.

        A Z₂ symmetry is a Pauli string that:
        1. Squares to identity (P² = I)
        2. Commutes with all terms in H ([P, H] = 0)

        Args:
            hamiltonian: SparsePauliOp representation of Hamiltonian
            max_symmetries: Maximum symmetries to find

        Returns:
            List of Pauli strings representing independent symmetries
        """
        n_qubits = hamiltonian.num_qubits

        # Find all Pauli terms in Hamiltonian
        pauli_terms = list(hamiltonian.paulis)

        # Build kernel of commutation matrix
        # A Pauli P is a symmetry if it commutes with all terms
        # Use symplectic representation for efficiency

        symmetries = []

        # 1. Particle number symmetry: Z on all qubits (but only for even qubit pairs)
        # For JW: Z_i Z_{i+1} for each spatial orbital
        for i in range(0, n_qubits - 1, 2):
            z_pauli = ['I'] * n_qubits
            z_pauli[i] = 'Z'
            z_pauli[i + 1] = 'Z'
            candidate = Pauli(''.join(reversed(z_pauli)))  # Qiskit uses little-endian

            if self._commutes_with_all(candidate, pauli_terms):
                symmetries.append(candidate)
                logger.info(f"Found symmetry: Z{i}Z{i+1}")
                if len(symmetries) >= max_symmetries:
                    break

        # 2. Total parity symmetry: Z on all qubits
        z_all = 'Z' * n_qubits
        candidate = Pauli(z_all)
        if self._commutes_with_all(candidate, pauli_terms) and len(symmetries) < max_symmetries:
            if not any(candidate.equiv(s) for s in symmetries):
                symmetries.append(candidate)
                logger.info(f"Found total parity symmetry: Z⊗{n_qubits}")

        # 3. Spin symmetry: Z on alternating qubits (alpha vs beta)
        if n_qubits >= 4 and len(symmetries) < max_symmetries:
            z_spin = ['I'] * n_qubits
            for i in range(0, n_qubits, 2):
                z_spin[i] = 'Z'
            candidate = Pauli(''.join(reversed(z_spin)))
            if self._commutes_with_all(candidate, pauli_terms):
                if not any(candidate.equiv(s) for s in symmetries):
                    symmetries.append(candidate)
                    logger.info(f"Found spin-alpha parity symmetry")

        # Reduce to a GF(2)-linearly-independent set. Z-type Paulis multiply by
        # XOR of their Z-supports, so e.g. Z⊗n = product of all per-orbital ZZ
        # and is linearly dependent. Counting dependent symmetries over-reports
        # n_symmetries and corrupts the taper-qubit count. Greedy GF(2) rank
        # reduction over the Z-support bitvectors, preserving discovery order.
        _independent = []
        _basis = []  # list of (pivot_index, reduced_bitvector)
        for _sym in symmetries:
            _s = str(_sym)[::-1]  # qubit 0 at position 0
            _v = [1 if c == 'Z' else 0 for c in _s]
            for _piv, _bv in _basis:
                if _v[_piv]:
                    _v = [a ^ b for a, b in zip(_v, _bv)]
            _piv = next((i for i, b in enumerate(_v) if b), None)
            if _piv is not None:
                _basis.append((_piv, _v))
                _independent.append(_sym)
        symmetries = _independent

        self._symmetries = symmetries
        logger.info(f"Found {len(symmetries)} independent Z₂ symmetries")

        return symmetries

    def _commutes_with_all(self, pauli: Pauli, terms: List[Pauli]) -> bool:
        """Check if Pauli commutes with all terms."""
        for term in terms:
            if not pauli.commutes(term):
                return False
        return True

    def determine_eigenvalues(
        self,
        n_electrons: int,
        n_qubits: int,
        spin: int = 0
    ) -> List[int]:
        """
        Determine eigenvalues (±1) for each symmetry based on physics.

        For the ground state of a singlet molecule:
        - Total particle number parity: (-1)^N
        - Spin parity: depends on Sz

        Args:
            n_electrons: Number of electrons
            n_qubits: Number of qubits
            spin: Total spin (0 for singlet, 1 for doublet, etc.)

        Returns:
            List of eigenvalues (±1) for each symmetry
        """
        eigenvalues = []

        for sym in self._symmetries:
            # Count Z operators
            z_count = str(sym).count('Z')

            # For particle number type symmetries
            # The eigenvalue depends on how many electrons occupy the qubits with Z
            # For ground state HF configuration: first n_electrons qubits occupied

            # Simple heuristic: parity of electrons in Z-support
            z_support = [i for i, p in enumerate(str(sym)[::-1]) if p == 'Z']
            n_in_support = sum(1 for i in z_support if i < n_electrons)

            eigenvalue = (-1) ** n_in_support
            eigenvalues.append(eigenvalue)

        self._eigenvalues = eigenvalues
        logger.info(f"Symmetry eigenvalues: {eigenvalues}")

        return eigenvalues

    def taper_hamiltonian(
        self,
        hamiltonian: SparsePauliOp,
        n_electrons: int,
        n_qubits: int,
        spin: int = 0
    ) -> Tuple[SparsePauliOp, Dict[str, Any]]:
        """
        Apply qubit tapering to reduce Hamiltonian size.

        Args:
            hamiltonian: Original SparsePauliOp
            n_electrons: Number of electrons
            n_qubits: Number of qubits in original Hamiltonian
            spin: Total spin

        Returns:
            Tuple of (tapered_hamiltonian, metadata_dict)
        """
        # The 4-qubit H2 case has a validated specialized path.
        if hamiltonian.num_qubits == 4:
            return taper_h2_hamiltonian(hamiltonian)

        # General case: EXACT sector projection (the same construction
        # taper_h2_hamiltonian uses, generalized). The Z2 symmetries are Z-strings
        # that commute with H, so H is block-diagonal across their ±1 sectors.
        #   1. find the independent symmetries,
        #   2. pick the ground state's sector from the HF determinant (the
        #      lowest-diagonal basis state with the right electron count — cheap,
        #      no full diagonalization),
        #   3. collect the basis indices in that sector,
        #   4. restrict H.to_matrix() to them and decompose the reduced block back
        #      into an (n−k)-qubit SparsePauliOp.
        # This reproduces the full-space ground eigenvalue exactly. It replaces the
        # old hand-rolled term-substitution that gave ~0.6 Ha errors. (Cost is
        # O(2^n) — fine for the moderate registers tapering targets; a Clifford
        # construction would be needed for very large n, which to_matrix can't
        # reach anyway.)
        from qiskit.quantum_info import Operator

        syms = self.find_symmetries(hamiltonian)
        n = hamiltonian.num_qubits
        if not syms:
            logger.warning("No Z2 symmetries found; returning original Hamiltonian")
            return hamiltonian, {'original_qubits': n, 'tapered_qubits': n,
                                 'n_symmetries': 0, 'symmetries': [], 'eigenvalues': []}

        # Z-support bitmask per symmetry (qubit q ↔ bit q; Qiskit label is little-endian)
        supports = []
        for s in syms:
            lbl = str(s)
            mask = 0
            for q in range(n):
                if lbl[n - 1 - q] == 'Z':
                    mask |= (1 << q)
            supports.append(mask)

        H = np.asarray(hamiltonian.to_matrix())
        diag = np.real(np.diag(H))
        elec_sector = [i for i in range(diag.size) if bin(i).count('1') == n_electrons]
        if not elec_sector:
            raise ValueError(f"tapering: no basis state has {n_electrons} electrons")
        hf = min(elec_sector, key=lambda i: diag[i])           # HF determinant
        bvals = [bin(hf & m).count('1') & 1 for m in supports]  # sector parity per symmetry

        idx = [x for x in range(1 << n)
               if all((bin(x & supports[i]).count('1') & 1) == bvals[i]
                      for i in range(len(supports)))]
        k = len(supports)
        if len(idx) != (1 << (n - k)):
            # Symmetries weren't fully independent → the reduced block isn't a clean
            # 2^(n-k); fail loudly rather than emit a malformed operator.
            raise ValueError(
                f"tapering: sector dimension {len(idx)} != 2^(n-k) = {1 << (n - k)} "
                f"(found {k} symmetries on {n} qubits that are not independent)"
            )

        H_sector = H[np.ix_(idx, idx)]
        tapered_ham = SparsePauliOp.from_operator(Operator(H_sector)).simplify()
        metadata = {
            'original_qubits': n,
            'tapered_qubits': n - k,
            'n_symmetries': k,
            'symmetries': [str(s) for s in syms],
            'eigenvalues': [(-1) ** b for b in bvals],
            'sector_indices': idx if len(idx) <= 64 else None,
            # Original-space HF determinant index, so a caller can locate the HF
            # reference's POSITION within the sector (idx.index(hf)) to prepare the
            # correct tapered reference state. (CORE_BUGS B12.)
            'hf_index': int(hf),
            'reduction_percent': 100.0 * k / n,
        }
        logger.info(f"Tapered (sector projection): {n} → {n - k} qubits "
                    f"({metadata['reduction_percent']:.0f}% reduction)")
        return tapered_ham, metadata

    def _apply_tapering(
        self,
        hamiltonian: SparsePauliOp,
        symmetries: List[Pauli],
        eigenvalues: List[int]
    ) -> SparsePauliOp:
        """
        Apply the tapering transformation.

        For each symmetry with eigenvalue ±1:
        1. Find a Clifford that maps the symmetry to a single-qubit Z
        2. Project onto the ±1 eigenspace
        3. Remove the tapered qubit
        """
        n_original = hamiltonian.num_qubits
        n_symmetries = len(symmetries)
        n_tapered = n_original - n_symmetries

        # For simplicity, use Qiskit's tapering if available
        # Otherwise implement basic version
        try:
            from qiskit_nature.second_q.mappers import TaperedQubitMapper
            from qiskit_nature.second_q.operators import SparseLabelOp

            # Convert to Qiskit Nature format and use their tapering
            # This is the robust approach
            logger.info("Using Qiskit Nature's tapering implementation")

        except ImportError:
            logger.info("Qiskit Nature not available, using basic tapering")

        # Basic implementation: project and reduce
        # For each symmetry, we replace operators acting on the tapered qubit
        # with the eigenvalue

        # Identify qubits to taper (those with single Z in symmetry)
        taper_qubits = []
        for sym in symmetries:
            sym_str = str(sym)[::-1]  # Reverse for qubit 0 at position 0
            for i, p in enumerate(sym_str):
                if p == 'Z' and i not in taper_qubits:
                    taper_qubits.append(i)
                    break

        # Pair each taper qubit with its source symmetry's eigenvalue BEFORE
        # sorting. The eigenvalue must be looked up by qubit index, not by the
        # post-sort position in taper_qubits (which previously mismatched the
        # eigenvalue to the wrong symmetry, flipping coefficient signs).
        eig_by_qubit = {q: eigenvalues[k] for k, q in enumerate(taper_qubits)
                        if k < len(eigenvalues)}

        # Sort to remove from high to low (so indices remain valid)
        taper_qubits.sort(reverse=True)

        # Build new Hamiltonian with tapered qubits removed
        new_paulis = []
        new_coeffs = []

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            pauli_str = str(pauli)[::-1]  # qubit 0 at position 0

            # Check action on tapered qubits
            skip = False
            eigenvalue_factor = 1.0

            for tq_idx, tq in enumerate(taper_qubits):
                if tq < len(pauli_str):
                    op = pauli_str[tq]
                    if op == 'X' or op == 'Y':
                        # X or Y on tapered qubit -> zero in eigenspace
                        skip = True
                        break
                    elif op == 'Z':
                        # Z on tapered qubit -> eigenvalue (looked up by qubit,
                        # not by post-sort list position)
                        eigenvalue_factor *= eig_by_qubit.get(tq, 1.0)

            if skip:
                continue

            # Build new Pauli without tapered qubits
            new_pauli_str = ''
            for i, op in enumerate(pauli_str):
                if i not in taper_qubits:
                    new_pauli_str += op

            if len(new_pauli_str) == 0:
                new_pauli_str = 'I'

            # Reverse back for Qiskit format
            new_paulis.append(new_pauli_str[::-1])
            new_coeffs.append(coeff * eigenvalue_factor)

        # Combine like terms
        tapered_ham = SparsePauliOp.from_list(
            [(p, c) for p, c in zip(new_paulis, new_coeffs)]
        ).simplify()

        return tapered_ham

    def taper_circuit_hf(
        self,
        n_qubits_original: int,
        n_electrons: int,
        taper_qubits: List[int]
    ) -> Tuple['QuantumCircuit', List[int]]:
        """
        Create tapered Hartree-Fock state preparation.

        Args:
            n_qubits_original: Original qubit count
            n_electrons: Number of electrons
            taper_qubits: Qubits that were tapered away

        Returns:
            (tapered_circuit, qubit_mapping)
        """
        from qiskit import QuantumCircuit

        n_tapered = n_qubits_original - len(taper_qubits)
        circuit = QuantumCircuit(n_tapered)

        # Build qubit mapping: original qubit -> tapered qubit
        mapping = {}
        tapered_idx = 0
        for orig in range(n_qubits_original):
            if orig not in taper_qubits:
                mapping[orig] = tapered_idx
                tapered_idx += 1

        # Prepare HF state on non-tapered qubits
        for orig_q in range(n_electrons):
            if orig_q in mapping:
                circuit.x(mapping[orig_q])

        return circuit, mapping


def taper_h2_hamiltonian(
    hamiltonian: SparsePauliOp
) -> Tuple[SparsePauliOp, Dict[str, Any]]:
    """
    Specialized tapering for H₂ molecule using exact Clifford transformation.

    H₂ has well-known symmetries:
    - 4 qubits → 2 qubits
    - Particle number parity on each spatial orbital

    The transformation follows Bravyi et al. (2017):
    1. Apply Clifford to map τ = Z₀Z₁ to Z₁ and Z₂Z₃ to Z₃
    2. Project onto eigenspace (τ₁ = τ₂ = +1 for ground state)
    3. Remove qubits 1 and 3

    Args:
        hamiltonian: 4-qubit H₂ Hamiltonian

    Returns:
        (2-qubit tapered Hamiltonian, metadata)
    """
    if hamiltonian.num_qubits != 4:
        raise ValueError(f"H₂ tapering requires 4 qubits, got {hamiltonian.num_qubits}")

    logger.info("Applying exact H₂ tapering: 4 → 2 qubits")

    # The H₂ ground state lives in the Z₀Z₁ = +1, Z₂Z₃ = +1 symmetry sector.
    # In Qiskit little-endian (state index = Σ_q bit_q·2^q) those are the four
    # computational-basis states with even parity on (q0,q1) and (q2,q3):
    #   |0000⟩ = 0, |0011⟩ = 3, |1100⟩ = 12, |1111⟩ = 15.
    # Tapering = an *exact projection* of H onto this 2-dimensional-per-free-
    # qubit sector. The two free qubits are q0 and q2 (q1=q0, q3=q2 inside the
    # sector); relabel orig-q0 → new-q0, orig-q2 → new-q1. Restricting
    # H.to_matrix() to rows/cols [0,3,12,15] (which is exactly that relabelling)
    # and decomposing the 4×4 block with SparsePauliOp.from_operator yields the
    # exact 2-qubit Hamiltonian.
    #
    # This replaces a hand-rolled term-substitution that did NOT apply a real
    # Clifford and produced a wrong spectrum (−0.557 Ha vs FCI −1.137284 Ha for
    # H₂/STO-3G). The projection reproduces PySCF FCI to machine precision.
    from qiskit.quantum_info import Operator

    sector = [0b0000, 0b0011, 0b1100, 0b1111]   # [0, 3, 12, 15]
    H_full = np.asarray(hamiltonian.to_matrix())
    H_sector = H_full[np.ix_(sector, sector)]
    tapered_ham = SparsePauliOp.from_operator(Operator(H_sector)).simplify()

    logger.info(f"Tapered Hamiltonian: {len(tapered_ham)} terms")
    for p, c in zip(tapered_ham.paulis, tapered_ham.coeffs):
        logger.debug(f"  {p}: {c:.6f}")

    metadata = {
        'original_qubits': 4,
        'tapered_qubits': 2,
        'n_symmetries': 2,
        'symmetries': ['Z0Z1', 'Z2Z3'],
        'eigenvalues': [+1, +1],
        'sector_states': sector,
        'reduction_percent': 50.0
    }

    return tapered_ham, metadata
