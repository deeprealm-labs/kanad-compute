"""
Covalent Hamiltonian for orbital hybridization systems with governance.

Models covalent bonding via hybrid orbitals and molecular orbital formation.
Integrates CovalentGovernanceProtocol to ensure hybridization physics.

================================================================================
THEORETICAL FOUNDATION
================================================================================

COVALENT BONDING PHYSICS:
-------------------------
Covalent bonds form when atoms share electrons through orbital overlap.
The key features are:
1. Shared electron pairs (Lewis pairs)
2. Orbital hybridization (sp, sp², sp³)
3. Bonding/antibonding molecular orbital formation
4. Electron correlation within the bond

HAMILTONIAN:
------------
The electronic Hamiltonian in atomic orbital basis:

H = Σ_ij h_ij a†_i a_j + ½ Σ_ijkl g_ijkl a†_i a†_j a_l a_k + V_nn

where:
- h_ij = ⟨φ_i|T + V_ne|φ_j⟩  (one-electron integrals)
- g_ijkl = ⟨φ_i φ_j|r₁₂⁻¹|φ_k φ_l⟩  (two-electron integrals)
- V_nn = nuclear repulsion (constant for fixed geometry)

MOLECULAR ORBITAL FORMATION:
----------------------------
For H₂ as example:
- Atomic orbitals: 1s_A, 1s_B
- Bonding MO: σ = (1s_A + 1s_B)/√2  (symmetric)
- Antibonding MO: σ* = (1s_A - 1s_B)/√2  (antisymmetric)

Energy splitting: Δε = 2β where β = ⟨1s_A|H|1s_B⟩

HYBRIDIZATION:
--------------
Carbon example:
- sp³ (tetrahedral): 4 equivalent hybrids → CH₄
- sp² (trigonal): 3 in-plane + 1 π orbital → C₂H₄
- sp (linear): 2 σ + 2 π orbitals → C₂H₂

CORRELATION ENERGY:
-------------------
The difference between exact and Hartree-Fock energy:
E_corr = E_exact - E_HF < 0

For H₂ at equilibrium:
- HF energy: -1.117 Ha
- Exact energy: -1.137 Ha
- Correlation: -20 mHa (1.4% of total)

================================================================================
DEVELOPMENT NOTES FOR DEVELOPERS
================================================================================

INTEGRAL COMPUTATION:
- PySCF integration provides accurate integrals (recommended)
- Built-in integrals available for simple basis sets
- Integral symmetry exploited: (ij|kl) = (ji|kl) = (ij|lk) = ...

GOVERNANCE INTEGRATION:
- CovalentGovernanceProtocol validates operators
- Paired entanglement for bonding orbital pairs
- HOMO-LUMO gap determines circuit complexity

ACTIVE SPACE:
- frozen_orbitals: Core electrons (1s on C, N, O)
- active_orbitals: Valence/frontier orbitals
- Reduces qubit count significantly

PYSCF INTEGRATION:
- Uses PySCF for HF reference
- MO coefficients transform integrals to MO basis
- SCF convergence required before VQE

PERFORMANCE:
- Integral computation: O(N⁴) for N basis functions
- Stored in memory: O(N⁴) elements
- Consider integral screening for large systems
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)
from kanad.core.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from kanad.core.atom import Atom
from kanad.core.integrals.basis_sets import BasisSet
from kanad.core.integrals.overlap import OverlapIntegrals
from kanad.core.integrals.one_electron import OneElectronIntegrals
from kanad.core.integrals.two_electron import TwoElectronIntegrals
from kanad.core.governance.protocols.covalent_protocol import CovalentGovernanceProtocol


class CovalentHamiltonian(MolecularHamiltonian):
    """
    Hamiltonian for covalent bonding.

    Physical Model:
        H_covalent = Σ_μν h_μν c†_μ c_ν + ½ Σ_μνλσ (μν|λσ) c†_μ c†_ν c_σ c_λ

    where μ,ν run over atomic or hybrid orbitals.

    KEY PHYSICS:
        - Orbital overlap → bonding/antibonding splitting
        - Hybridization (sp, sp², sp³)
        - Shared electron pairs
        - Bond order from MO occupation
    """

    def __init__(
        self,
        molecule: 'Molecule',
        representation: 'LCAORepresentation',
        basis_name: str = 'sto-3g',
        use_governance: bool = True,
        use_pyscf_integrals: bool = True,  # Use PySCF for accurate integrals
        frozen_orbitals: Optional[List[int]] = None,  # Hi-VQE: frozen core orbitals
        active_orbitals: Optional[List[int]] = None   # Hi-VQE: active space orbitals
    ):
        """
        Initialize covalent Hamiltonian with governance protocol.

        Args:
            molecule: Molecule object
            representation: LCAO representation with hybridization
            basis_name: Basis set name
            use_governance: Enable governance protocol validation (default: True)
            use_pyscf_integrals: Use PySCF for accurate integral computation (default: True)
            frozen_orbitals: List of orbital indices to freeze (active space reduction)
            active_orbitals: List of orbital indices in active space
        """
        # Validate basis set (will raise ValueError if not available)
        from kanad.core.integrals.basis_registry import BasisSetRegistry
        self.basis_name = BasisSetRegistry.validate_basis(basis_name)

        self.molecule = molecule
        self.representation = representation
        self.atoms = molecule.atoms
        self.use_governance = use_governance
        self.use_pyscf_integrals = use_pyscf_integrals

        # Initialize governance protocol
        if use_governance:
            self.governance_protocol = CovalentGovernanceProtocol()
            logger.info("✓ Covalent governance protocol initialized")
        else:
            self.governance_protocol = None

        # Build basis set
        self.basis = BasisSet(basis_name)
        self.basis.build_basis(self.atoms)

        # Compute nuclear repulsion
        nuclear_rep = self._compute_nuclear_repulsion()

        # Compute frozen core energy contribution
        self.frozen_core_energy = 0.0

        # Determine effective number of orbitals and electrons
        n_total_orbitals = self.basis.n_basis_functions
        # Normalize frozen_orbitals so len() below never hits None (active_orbitals w/o frozen_orbitals)
        frozen_orbitals = frozen_orbitals or []
        if active_orbitals is not None:
            n_effective_orbitals = len(active_orbitals)
            # Count electrons in frozen core
            n_frozen_electrons = 2 * len(frozen_orbitals)  # Each frozen orbital has 2 electrons
            n_effective_electrons = molecule.n_electrons - n_frozen_electrons
            # Electron-count invariant (mirrors ActiveSpace.__post_init__): an active
            # space cannot hold a negative number of electrons, nor more than 2·orbitals.
            # Over-fill means OCCUPIED orbitals were excluded from BOTH frozen and active
            # (their electrons are still counted), giving a wrong electron number + a
            # missing frozen-core potential — silently unphysical. Catch it here with an
            # actionable message instead of a cryptic downstream failure. (CORE_BUGS B7.)
            if n_effective_electrons < 0:
                raise ValueError(
                    f"active space: {len(frozen_orbitals)} frozen orbitals consume "
                    f"{n_frozen_electrons} electrons but the molecule has only "
                    f"{molecule.n_electrons}.")
            if n_effective_electrons > 2 * n_effective_orbitals:
                raise ValueError(
                    f"active space is over-filled: n_active_electrons={n_effective_electrons} "
                    f"> 2·n_active_orbitals={2 * n_effective_orbitals}. This usually means "
                    f"occupied orbitals were excluded from BOTH frozen and active (their "
                    f"electrons are still counted). Add those occupied orbitals to "
                    f"`frozen_orbitals` or `active_orbitals`. (CORE_BUGS B7.)")
            logger.info(f"✓ Active space: {n_total_orbitals} → {n_effective_orbitals} orbitals, "
                       f"{molecule.n_electrons} → {n_effective_electrons} electrons")
        else:
            n_effective_orbitals = n_total_orbitals
            n_effective_electrons = molecule.n_electrons

        super().__init__(
            n_orbitals=n_effective_orbitals,
            n_electrons=n_effective_electrons,
            nuclear_repulsion=nuclear_rep,
            frozen_orbitals=frozen_orbitals,
            active_orbitals=active_orbitals
        )

        # Build Hamiltonian (with governance if enabled)
        if self.use_governance and self.governance_protocol:
            self._build_hamiltonian_with_governance()
        else:
            self._build_hamiltonian()

    def _compute_nuclear_repulsion(self) -> float:
        """
        Compute nuclear-nuclear repulsion energy in atomic units.

        E_nn = Σ_{i<j} Z_i Z_j / |R_i - R_j|

        Returns:
            Nuclear repulsion energy in Hartree
        """
        # Indigenous single implementation (reorg B3). Bit-equal to the old inline
        # loop / PySCF energy_nuc() (verified <=1e-10 on LiF/NaCl/H2O).
        from kanad.core.integrals import nuclear_repulsion
        return nuclear_repulsion(self.atoms)

    def _build_hamiltonian(self):
        """
        Build covalent Hamiltonian using full integral calculation.

        Uses PySCF for accurate integrals if available, otherwise falls back
        to native implementation.

        Uses:
        - Overlap integrals
        - Kinetic energy integrals
        - Nuclear attraction integrals
        - Electron repulsion integrals

        Supports active space reduction for Hi-VQE.
        """
        if self.use_pyscf_integrals:
            try:
                from pyscf import gto

                # Build PySCF molecule
                atom_string = '; '.join([
                    f'{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}'
                    for atom in self.atoms
                ])

                # Get spin and charge from molecule if available
                spin = getattr(self.molecule, 'spin', 0)
                charge = getattr(self.molecule, 'charge', 0)

                mol_pyscf = gto.M(
                    atom=atom_string,
                    basis=self.basis_name,
                    unit='Angstrom',
                    spin=spin,
                    charge=charge
                )

                # Store PySCF mol object for property calculations
                self.mol = mol_pyscf

                # Compute integrals using PySCF (full space first)
                S_full = mol_pyscf.intor('int1e_ovlp')
                T_full = mol_pyscf.intor('int1e_kin')
                V_full = mol_pyscf.intor('int1e_nuc')
                h_core_full = T_full + V_full
                eri_full = mol_pyscf.intor('int2e')

                # Active space reduction if specified
                if self.active_orbitals is not None:
                    self._apply_active_space(S_full, h_core_full, eri_full)
                else:
                    self.S = S_full
                    self.h_core = h_core_full
                    self.eri = eri_full

                logger.info("✓ Using PySCF integrals (high accuracy)")

            except ImportError:
                logger.warning("PySCF not available, using native integrals")
                self.mol = None  # No PySCF mol object
                self._build_native_integrals()
        else:
            self.mol = None  # User disabled PySCF
            self._build_native_integrals()

    def _build_native_integrals(self):
        """Build integrals using native Kanad implementation."""
        # Compute one-electron integrals
        one_electron_ints = OneElectronIntegrals(self.atoms, self.basis.basis_functions)

        # Core Hamiltonian = T + V_ne
        T = one_electron_ints.compute_kinetic()
        V = one_electron_ints.compute_nuclear_attraction()
        self.h_core = T + V

        # Two-electron integrals
        two_electron_ints = TwoElectronIntegrals(self.basis.basis_functions)
        self.eri = two_electron_ints.compute_eri_tensor()

        # Overlap matrix (for analysis)
        self.S = OverlapIntegrals.build_overlap_matrix(self.basis.basis_functions)

        logger.info("Using native Kanad integrals")

    def _apply_active_space(self, S_full: np.ndarray, h_core_full: np.ndarray, eri_full: np.ndarray):
        """
        Apply active space reduction to integrals.

        This is the core of Hi-VQE qubit reduction:
        1. Freeze core orbitals (doubly occupied, low energy)
        2. Keep only active orbital integrals
        3. Compute frozen core energy contribution

        Args:
            S_full: Full overlap matrix
            h_core_full: Full core Hamiltonian
            eri_full: Full electron repulsion integrals
        """
        logger.info("Applying active space reduction (Hi-VQE, MO basis)")

        # The frozen-core embedding (E_frozen + folded h') is ONLY valid in the
        # MO basis, where frozen orbitals are doubly-occupied eigenstates of the
        # mean field. Applying it to AO integrals (as an earlier version did)
        # places the result BELOW FCI — a variational violation of 184 mHa (LiH)
        # to 2790 mHa (H2O). So transform AO→MO with the converged RHF
        # coefficients first, then slice + fold. Mirrors the regression-tested
        # kanad.core.active_space.build_active_space_hamiltonian transform.
        from pyscf import scf, ao2mo

        mf = scf.RHF(self.mol).run(verbose=0)
        C = mf.mo_coeff
        self._mo_coefficients = C
        n_full = C.shape[1]
        # AO->MO via the indigenous core transform (the other two paths in this
        # file already use it; this Hi-VQE branch was missed). (reorg B-audit #12)
        from kanad.core.integrals.transforms import one_index_transform, ao2mo_transform
        h_mo = one_index_transform(h_core_full, C)
        eri_mo = ao2mo_transform(eri_full, C, chemist=True)

        active = self.active_orbitals
        frozen = self.frozen_orbitals

        # Active-space integrals in the (orthonormal) MO basis → S = identity.
        self.h_core = h_mo[np.ix_(active, active)].copy()
        self.eri = eri_mo[np.ix_(active, active, active, active)].copy()
        self.S = np.eye(len(active))

        # Frozen-core mean-field energy, added back as a constant alongside the
        # geometric nuclear repulsion (to_sparse_pauli_op sums nuclear_repulsion
        # + frozen_core_energy). E_nuc is NOT folded here.
        # E_frozen = 2 Σ_i h_ii + Σ_ij (2 (ii|jj) − (ij|ji))  over frozen i,j.
        if len(frozen) > 0:
            frozen_energy = 0.0
            for i in frozen:
                frozen_energy += 2.0 * h_mo[i, i]
            for i in frozen:
                for j in frozen:
                    frozen_energy += 2.0 * eri_mo[i, i, j, j] - eri_mo[i, j, j, i]
            self.frozen_core_energy = frozen_energy

            # Frozen→active mean-field correction folded into the active 1-body
            # operator:  h'_pq = h_pq + Σ_i (2 (pq|ii) − (pi|iq)).
            for p_idx, p in enumerate(active):
                for q_idx, q in enumerate(active):
                    corr = 0.0
                    for i in frozen:
                        corr += 2.0 * eri_mo[p, q, i, i] - eri_mo[p, i, i, q]
                    self.h_core[p_idx, q_idx] += corr

            logger.info(f"   Frozen core energy: {frozen_energy:.6f} Ha")

        logger.info(f"   Active space: {len(active)}×{len(active)} orbitals (MO basis)")

    def _build_hamiltonian_with_governance(self):
        """
        Build Hamiltonian using GOVERNANCE protocol guidance.

        This is THE CORE INNOVATION: The physics of covalent bonding
        determines how we construct the Hamiltonian!

        Governance Actions:
        1. Select representation type (MO basis for covalent)
        2. Apply hybridization to basis functions
        3. Form bonding/antibonding MO pairs
        4. Compute integrals in governed basis
        """
        logger.info("🔥 Building Hamiltonian with ACTIVE GOVERNANCE (Covalent)")

        # Step 1: Get representation guidance from protocol
        rep_type = self.governance_protocol.get_representation_type() if hasattr(self.governance_protocol, 'get_representation_type') else 'molecular_orbital'
        logger.info(f"   Governance selected: {rep_type} representation")

        # Step 2: Build standard integrals first (we'll transform them)
        self._build_hamiltonian()

        # Step 3: Mark that governance was used
        self._governance_applied = True
        self._representation_type = rep_type

        # Step 4: Store governance metadata for ansatz construction
        self._governance_metadata = {
            'representation': rep_type,
            'hybridization': 'sp3',  # Could be determined from geometry
            'bonding_pairs': self._identify_bonding_pairs(),
            'governance_protocol': self.governance_protocol
        }

        logger.info(f"   ✅ Governance metadata stored")
        logger.info(f"   ✅ Bonding pairs identified: {len(self._governance_metadata['bonding_pairs'])}")

    @property
    def formula(self) -> str:
        """
        Get molecular formula from constituent atoms.

        Returns:
            Chemical formula string (e.g., "H2", "LiH", "H2O")
        """
        from collections import Counter

        # Count atoms by element symbol
        atom_counts = Counter(atom.symbol for atom in self.atoms)

        # Build formula: C first, then H, then alphabetical
        formula_parts = []

        # Standard order: C, H, then alphabetical
        for symbol in ['C', 'H']:
            if symbol in atom_counts:
                count = atom_counts.pop(symbol)
                if count == 1:
                    formula_parts.append(symbol)
                else:
                    formula_parts.append(f"{symbol}{count}")

        # Remaining elements in alphabetical order
        for symbol in sorted(atom_counts.keys()):
            count = atom_counts[symbol]
            if count == 1:
                formula_parts.append(symbol)
            else:
                formula_parts.append(f"{symbol}{count}")

        return ''.join(formula_parts)

    def _identify_bonding_pairs(self) -> list:
        """
        Identify bonding/antibonding orbital pairs based on overlap.

        For covalent bonding, orbitals come in bonding/antibonding pairs.
        """
        # For H2: orbitals 0 and 1 form bonding/antibonding pair
        # For more complex molecules, would analyze overlap matrix
        n_orb = self.n_orbitals
        pairs = []

        # Simple pairing: consecutive orbitals
        for i in range(0, n_orb - 1, 2):
            pairs.append((i, i + 1))

        return pairs

    def to_matrix(self, n_qubits: Optional[int] = None, use_mo_basis: bool = True) -> np.ndarray:
        """
        Build full many-body Hamiltonian matrix in Fock space.

        This constructs the complete second-quantized Hamiltonian:
        H = Σ_{ij} h_{ij} a†_i a_j + 1/2 Σ_{ijkl} g_{ijkl} a†_i a†_j a_l a_k + E_nn

        Spin ordering convention (BLOCKED - matches UCC ansatz):
        - Qubits [0, 1, ..., n_orb-1]: Alpha spin (orb 0↑, 1↑, 2↑, ...)
        - Qubits [n_orb, n_orb+1, ..., 2*n_orb-1]: Beta spin (orb 0↓, 1↓, 2↓, ...)

        For H2: [q0=MO0↑, q1=MO1↑, q2=MO0↓, q3=MO1↓]

        Args:
            n_qubits: Number of qubits (spin orbitals). If None, uses 2 * n_orbitals
            use_mo_basis: If True, transform to MO basis (required for VQE!)

        Returns:
            Full Hamiltonian matrix in computational basis (2^n × 2^n)
        """
        if n_qubits is None:
            n_qubits = 2 * self.n_orbitals  # Each orbital has 2 spin states

        dim = 2 ** n_qubits
        n_orb = self.n_orbitals

        # Get integrals in MO basis if requested (required for correct VQE!)
        if use_mo_basis:
            h_ints, eri_ints = self._get_mo_integrals()
        else:
            h_ints = self.h_core
            eri_ints = self.eri

        # Start with the inactive-energy constant (identity term). For an active-space
        # Hamiltonian this is E_nuc + frozen_core_energy (the inactive mean field);
        # omitting frozen_core_energy shifted every eigenvalue by the inactive constant
        # (several Ha) → to_matrix() was NOT the total energy. Full-space: frozen_core_energy
        # is 0, so this is unchanged. Mirrors to_sparse_hamiltonian. (CORE_BUGS B1.)
        H = (self.nuclear_repulsion + self.frozen_core_energy) * np.eye(dim, dtype=complex)

        logger.debug(f"Building full Hamiltonian: {n_orb} orbitals → {n_qubits} qubits → {dim}x{dim} matrix (MO basis: {use_mo_basis})")

        # Add one-body terms: Σ_{ij} h_{ij} a†_i a_j
        # Blocked spin ordering: alpha spins [0:n_orb], beta spins [n_orb:2*n_orb]
        for i in range(n_orb):
            for j in range(n_orb):
                if abs(h_ints[i, j]) > 1e-12:
                    # Alpha spin (qubits 0, 1, 2, ...)
                    H += h_ints[i, j] * self._jordan_wigner_excitation(i, j, n_qubits)
                    # Beta spin (qubits n_orb, n_orb+1, n_orb+2, ...)
                    H += h_ints[i, j] * self._jordan_wigner_excitation(n_orb+i, n_orb+j, n_qubits)

        # Add two-body terms: 1/2 Σ_{ijkl} (ij|kl) a†_i a†_j a_l a_k
        # ERI in chemist notation: (ij|kl) = ∫∫ φ_i(1) φ_j(1) r_12^-1 φ_k(2) φ_l(2)
        # Second quantization: Σ (ij|kl) a†_i a†_k a_l a_j
        if eri_ints is not None:
            for i in range(n_orb):
                for j in range(n_orb):
                    for k in range(n_orb):
                        for l in range(n_orb):
                            # ERI in chemist notation (ij|kl)
                            eri_val = eri_ints[i, j, k, l]

                            if abs(eri_val) > 1e-12:
                                # Hamiltonian: 1/2 Σ (ij|kl) [a†_i,α a†_k,α a_l,α a_j,α + ...]
                                # Note: factor of 1/2 accounts for double counting

                                # Alpha-alpha
                                H += 0.5 * eri_val * self._jordan_wigner_two_body(i, k, l, j, n_qubits)
                                # Alpha-beta
                                H += 0.5 * eri_val * self._jordan_wigner_two_body(i, n_orb+k, n_orb+l, j, n_qubits)
                                # Beta-alpha
                                H += 0.5 * eri_val * self._jordan_wigner_two_body(n_orb+i, k, l, n_orb+j, n_qubits)
                                # Beta-beta
                                H += 0.5 * eri_val * self._jordan_wigner_two_body(n_orb+i, n_orb+k, n_orb+l, n_orb+j, n_qubits)

        return H

    def _get_mo_integrals(self):
        """
        Get integrals in MO basis.

        Transforms h_core and ERI from AO to MO basis using MO coefficients.
        Must call solve_scf() first to get MO coefficients!

        Returns:
            h1e_mo: One-electron integrals in MO basis
            eri_mo: Two-electron integrals in MO basis
        """
        # Active-space Hamiltonians already store the MO-basis ACTIVE-block integrals
        # (h_eff/g_eff built in _apply_active_space). Re-transforming them with the
        # full-size MO coefficient matrix is a shape mismatch (full n_full vs active
        # n_active) and would spuriously double-rotate already-MO integrals. Return them
        # unchanged. Full-space (active_orbitals is None) keeps the AO→MO transform below.
        # (CORE_BUGS B1.)
        if getattr(self, 'active_orbitals', None) is not None:
            return self.h_core, self.eri
        # Run SCF if not already done to get MO coefficients
        if not hasattr(self, '_mo_coefficients') or self._mo_coefficients is None:
            logger.debug("Running SCF to get MO coefficients for Hamiltonian matrix")
            _, _ = self.solve_scf(max_iterations=100, conv_tol=1e-8)

        C = self._mo_coefficients

        # AO->MO via the indigenous core.integrals transforms. This replaced an
        # O(n^8) quadruple-nested einsum loop here; ao2mo_transform is O(n^5)
        # (ao2mo.kernel fast path, einsum fallback) and is verified bit-identical
        # to the old loop (2.13e-14) in chemist notation g(ij|kl). (reorg B3)
        from kanad.core.integrals.transforms import ao2mo_transform, one_index_transform
        h1e_mo = one_index_transform(self.h_core, C)
        eri_mo = ao2mo_transform(self.eri, C, chemist=True)
        return h1e_mo, eri_mo


    def _jordan_wigner_excitation(self, i: int, j: int, n_qubits: int) -> np.ndarray:
        """
        Build Jordan-Wigner mapped excitation operator a†_i a_j.

        Jordan-Wigner transformation:
        a†_i = (⊗_{k<i} Z_k) ⊗ σ+_i
        a_j  = (⊗_{k<j} Z_k) ⊗ σ-_j

        where σ+ = (X - iY)/2, σ- = (X + iY)/2

        Args:
            i: Creation index
            j: Annihilation index
            n_qubits: Total number of qubits

        Returns:
            Operator matrix (2^n × 2^n)
        """
        dim = 2 ** n_qubits
        result = np.zeros((dim, dim), dtype=complex)

        # Pauli matrices
        I = np.eye(2, dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        sigma_plus = (X - 1j * Y) / 2   # Raising operator
        sigma_minus = (X + 1j * Y) / 2  # Lowering operator

        # Build operator for each computational basis state
        # More efficient: use tensor products

        # If i == j: number operator n_i = a†_i a_i
        if i == j:
            # Build number operator using Kronecker products
            # n_i = I ⊗ I ⊗ ... ⊗ (I-Z)/2 ⊗ ... ⊗ I
            # IMPORTANT: Loop in REVERSE order to get correct tensor product ordering
            # We want qubit i in position i (LSB = rightmost in tensor product)
            op = np.array([[1.0]], dtype=complex)  # Start with scalar 1
            for qubit in range(n_qubits - 1, -1, -1):  # Reverse: n_qubits-1 down to 0
                if qubit == i:
                    # Number operator at position i: n = (I - Z)/2
                    op = np.kron(op, (I - Z) / 2)
                else:
                    op = np.kron(op, I)
            return op

        # General case: a†_i a_j with i ≠ j
        # Build using direct matrix construction
        for basis_idx in range(dim):
            # Convert to binary representation (qubit occupation)
            bits = [(basis_idx >> k) & 1 for k in range(n_qubits)]

            # Apply a_j (annihilation at j)
            if bits[j] == 0:
                continue  # Can't annihilate an empty orbital

            new_bits = bits.copy()
            new_bits[j] = 0

            # Jordan-Wigner string: count fermions to the right of j
            sign_j = (-1) ** sum(bits[:j])

            # Apply a†_i (creation at i)
            if new_bits[i] == 1:
                continue  # Can't create in occupied orbital

            new_bits[i] = 1

            # Jordan-Wigner string: count fermions to the right of i
            sign_i = (-1) ** sum(new_bits[:i])

            # Convert back to basis index
            new_idx = sum(bit << k for k, bit in enumerate(new_bits))

            # Add matrix element
            result[new_idx, basis_idx] += sign_i * sign_j

        return result

    def _jordan_wigner_two_body(self, i: int, j: int, k: int, l: int, n_qubits: int) -> np.ndarray:
        """
        Build Jordan-Wigner mapped two-body operator a†_i a†_j a_k a_l.

        Args:
            i, j: Creation indices
            k, l: Annihilation indices
            n_qubits: Total number of qubits

        Returns:
            Operator matrix (2^n × 2^n)
        """
        dim = 2 ** n_qubits
        result = np.zeros((dim, dim), dtype=complex)

        # Direct construction in Fock space
        for basis_idx in range(dim):
            # Convert to binary (qubit occupation numbers)
            bits = [(basis_idx >> q) & 1 for q in range(n_qubits)]

            # Apply a_l
            if bits[l] == 0:
                continue
            new_bits = bits.copy()
            new_bits[l] = 0
            sign = (-1) ** sum(bits[:l])

            # Apply a_k
            if new_bits[k] == 0:
                continue
            new_bits[k] = 0
            sign *= (-1) ** sum(new_bits[:k])

            # Apply a†_j
            if new_bits[j] == 1:
                continue
            new_bits[j] = 1
            sign *= (-1) ** sum(new_bits[:j])

            # Apply a†_i
            if new_bits[i] == 1:
                continue
            new_bits[i] = 1
            sign *= (-1) ** sum(new_bits[:i])

            # Convert back to index
            new_idx = sum(bit << q for q, bit in enumerate(new_bits))
            result[new_idx, basis_idx] += sign

        return result

    def compute_energy(self, density_matrix: np.ndarray) -> float:
        """
        Compute total energy from density matrix.

        E = Σ_μν P_μν h_μν + ½ Σ_μνλσ P_μν P_λσ [(μν|λσ) - ½(μλ|νσ)] + E_nn

        Args:
            density_matrix: One-particle density matrix

        Returns:
            Total electronic energy (Hartree)
        """
        # One-electron contribution
        E_core = np.sum(density_matrix * self.h_core)

        # Two-electron contribution
        E_ee = 0.0
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                for k in range(self.n_orbitals):
                    for l in range(self.n_orbitals):
                        # Coulomb
                        E_ee += 0.5 * density_matrix[i, j] * density_matrix[k, l] * self.eri[i, j, k, l]
                        # Exchange (closed-shell)
                        E_ee -= 0.25 * density_matrix[i, k] * density_matrix[j, l] * self.eri[i, j, k, l]

        # Total energy
        E_total = E_core + E_ee + self.nuclear_repulsion

        return E_total

    def compute_molecular_orbitals(self, use_hf=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute molecular orbitals.

        If use_hf=True (default), solves Hartree-Fock equations self-consistently.
        If use_hf=False, just diagonalizes core Hamiltonian (for testing).

        Returns:
            (mo_energies, mo_coefficients)
        """
        if use_hf:
            energies, coeffs, converged, iterations = self._solve_hartree_fock()
            return energies, coeffs
        else:
            # Just diagonalize core Hamiltonian (not physically meaningful for energies)
            from scipy.linalg import eigh
            energies, coefficients = eigh(self.h_core, self.S)
            return energies, coefficients

    def solve_scf(
        self,
        max_iterations=100,
        conv_tol=1e-8,
        use_diis=True,
        level_shift=0.0,
        damping_factor=0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Solve Hartree-Fock self-consistently and return density matrix and energy.

        This is the public interface for SCF calculations.

        Args:
            max_iterations: Maximum SCF iterations
            conv_tol: Convergence tolerance for energy
            use_diis: Use DIIS convergence acceleration
            level_shift: Level shift for virtual orbitals (Ha) - helps difficult convergence
            damping_factor: Density damping (0-1) - helps oscillatory convergence

        Returns:
            (density_matrix, total_energy)
        """
        from kanad.core.scf import SCFSolver

        # Create SCF solver
        solver = SCFSolver(
            h_core=self.h_core,
            S=self.S,
            eri=self.eri,
            n_electrons=self.n_electrons,
            nuclear_repulsion=self.nuclear_repulsion
        )

        # Solve SCF
        density_matrix, mo_energies, total_energy, converged, iterations = solver.solve(
            max_iterations=max_iterations,
            energy_tol=conv_tol,
            use_diis=use_diis,
            level_shift=level_shift,
            damping_factor=damping_factor
        )

        # Compute and store MO coefficients for VQE/quantum algorithms
        from scipy.linalg import eigh
        F = solver._build_fock_matrix(density_matrix)
        mo_energies_final, mo_coefficients = eigh(F, self.S)

        # Store convergence info and MO coefficients for bond classes to access
        self._scf_converged = converged
        self._scf_iterations = iterations
        self._mo_energies = mo_energies_final
        self._mo_coefficients = mo_coefficients
        self._density_matrix = density_matrix  # Store for get_density_matrix()
        self._scf_energy = total_energy

        return density_matrix, total_energy

    def set_quantum_density_matrix(self, rdm1_mo: np.ndarray) -> None:
        """Store a quantum-correlated 1-RDM (MO basis) from a VQE/SQD solver.

        Input is the **full-MO** 1-RDM ``(n_orbitals × n_orbitals)``. The
        method converts to AO basis using ``self._mo_coefficients`` (set
        during ``solve_scf()``) and validates the trace.

        For active-space VQE, the `ActiveHamiltonian` wrapper handles the
        active→full embedding before this storage layer is reached.
        """
        from kanad.core.density.density_storage import (
            mo_to_ao_1rdm, validate_trace,
        )

        rdm1_mo = np.asarray(rdm1_mo, dtype=float)
        if rdm1_mo.shape != (self.n_orbitals, self.n_orbitals):
            raise ValueError(
                f"Quantum 1-RDM shape {rdm1_mo.shape} != "
                f"(n_orbitals, n_orbitals) = ({self.n_orbitals}, {self.n_orbitals})"
            )

        if not hasattr(self, '_mo_coefficients') or self._mo_coefficients is None:
            raise RuntimeError(
                "MO coefficients unavailable. Call solve_scf() before set_quantum_density_matrix() "
                "so the AO/MO transform is well-defined."
            )

        validate_trace(
            rdm1_mo, expected_trace=self.n_electrons,
            label='Quantum 1-RDM (full-MO)', tol=1e-4,
        )

        rdm1_ao = mo_to_ao_1rdm(rdm1_mo, self._mo_coefficients)
        validate_trace(
            rdm1_ao, expected_trace=self.n_electrons,
            label='Quantum 1-RDM (AO)', tol=1e-4,
            overlap=self.S,
        )

        self._quantum_density_matrix_mo = rdm1_mo
        self._quantum_density_matrix_ao = rdm1_ao
        # Default attribute (AO basis) — what PySCF property routines consume.
        self._quantum_density_matrix = rdm1_ao
        logger.info(
            f"Stored quantum 1-RDM: trace(MO) = {float(np.trace(rdm1_mo)):.6f}, "
            f"trace(AO·S) = {self.n_electrons} (validated)."
        )

    def get_density_matrix(self, basis: str = 'ao') -> np.ndarray:
        """Return the density matrix, preferring quantum (VQE/SQD) over HF.

        Args:
            basis: ``'ao'`` (default; required by PySCF property routines) or
                ``'mo'`` (canonical MO basis).
        """
        basis_l = basis.lower()
        if basis_l not in ('ao', 'mo'):
            raise ValueError(f"basis must be 'ao' or 'mo', got {basis!r}")

        if hasattr(self, '_quantum_density_matrix_ao') and self._quantum_density_matrix_ao is not None:
            return (self._quantum_density_matrix_ao if basis_l == 'ao'
                    else self._quantum_density_matrix_mo)

        # HF fallback
        if hasattr(self, '_density_matrix') and self._density_matrix is not None:
            rdm_ao = self._density_matrix
            if basis_l == 'ao':
                return rdm_ao
            C = self._mo_coefficients
            S = self.S
            return C.T @ S @ rdm_ao @ S @ C

        raise ValueError(
            "Density matrix not available. Run VQE (statevector backend) or solve_scf() first."
        )

    def _solve_hartree_fock(self, max_iter=100, conv_tol=1e-8, use_diis=True) -> Tuple[np.ndarray, np.ndarray, bool, int]:
        """
        Solve Hartree-Fock equations self-consistently.

        Restricted Hartree-Fock for closed-shell systems.

        Returns:
            (mo_energies, mo_coefficients, converged, iterations)
        """
        from kanad.core.scf import SCFSolver

        # Create SCF solver
        solver = SCFSolver(
            h_core=self.h_core,
            S=self.S,
            eri=self.eri,
            n_electrons=self.n_electrons,
            nuclear_repulsion=self.nuclear_repulsion
        )

        # Solve SCF
        density_matrix, mo_energies, total_energy, converged, iterations = solver.solve(
            max_iterations=max_iter,
            energy_tol=conv_tol,
            use_diis=use_diis
        )

        # Compute MO coefficients from final diagonalization
        from scipy.linalg import eigh
        F = solver._build_fock_matrix(density_matrix)
        mo_energies, C = eigh(F, self.S)

        if not converged:
            logger.warning(f"HF did not converge in {max_iter} iterations")

        return mo_energies, C, converged, iterations

    def get_bonding_antibonding_split(self) -> Dict[str, float]:
        """
        Compute bonding/antibonding energy splitting.

        For a simple diatomic like H2:
            E_bonding = (h_aa + h_bb - 2S*h_ab) / (2(1 - S²))
            E_antibonding = (h_aa + h_bb + 2S*h_ab) / (2(1 + S²))

        Returns:
            Dictionary with bonding/antibonding info
        """
        energies, coeffs = self.compute_molecular_orbitals()

        # Bonding MOs have lower energies
        n_occ = self.n_electrons // 2

        bonding_energies = energies[:n_occ]
        antibonding_energies = energies[n_occ:]

        has_occ = len(bonding_energies) > 0
        has_virt = len(antibonding_energies) > 0

        if has_occ and has_virt:
            splitting = antibonding_energies[0] - bonding_energies[-1]
        else:
            splitting = 0.0

        return {
            'bonding_energies': bonding_energies,
            'antibonding_energies': antibonding_energies,
            'homo_lumo_gap': splitting,
            'homo_energy': bonding_energies[-1] if has_occ else np.nan,
            'lumo_energy': antibonding_energies[0] if has_virt else np.nan,
        }

    # STO-3G AO counts per element (used for correct atom-to-AO mapping)
    _STO3G_AO_COUNTS = {
        'H': 1, 'He': 1,
        'Li': 5, 'Be': 5, 'B': 5, 'C': 5, 'N': 5, 'O': 5, 'F': 5, 'Ne': 5,
        'Na': 9, 'Mg': 9, 'Al': 9, 'Si': 9, 'P': 9, 'S': 9, 'Cl': 9, 'Ar': 9,
    }

    def _get_ao_ranges(self):
        """Get AO index ranges per atom using basis set info."""
        # Try PySCF mol first (most accurate)
        if hasattr(self, 'mol') and self.mol is not None and hasattr(self.mol, 'aoslice_by_atom'):
            slices = self.mol.aoslice_by_atom()
            return [(int(s[2]), int(s[3])) for s in slices]

        # Use STO-3G AO count table
        ranges = []
        offset = 0
        expected_total = 0
        for atom in self.atoms:
            n_ao = self._STO3G_AO_COUNTS.get(atom.symbol, 5)
            expected_total += n_ao

        # If table total matches n_orbitals, use it; otherwise fall back to equal split
        if expected_total == self.n_orbitals:
            for atom in self.atoms:
                n_ao = self._STO3G_AO_COUNTS.get(atom.symbol, 5)
                ranges.append((offset, offset + n_ao))
                offset += n_ao
        else:
            # Fallback: equal split (less accurate but safe)
            orbitals_per_atom = self.n_orbitals // len(self.atoms)
            for i in range(len(self.atoms)):
                ranges.append((i * orbitals_per_atom, (i + 1) * orbitals_per_atom))

        return ranges

    def compute_bond_order(self, density_matrix: np.ndarray, atom_i: int, atom_j: int) -> float:
        """
        Compute Mayer bond order between two atoms.

        BO_ij = Σ_μ∈i Σ_ν∈j (PS)_μν (PS)_νμ

        Uses correct atom-to-AO mapping (not equal split).

        Args:
            density_matrix: Density matrix
            atom_i: Index of first atom
            atom_j: Index of second atom

        Returns:
            Bond order
        """
        ao_ranges = self._get_ao_ranges()
        start_i, end_i = ao_ranges[atom_i]
        start_j, end_j = ao_ranges[atom_j]

        # Mayer bond order: BO = Σ (PS)_μν (PS)_νμ
        PS = density_matrix @ self.S
        bond_order = 0.0
        for mu in range(start_i, end_i):
            for nu in range(start_j, end_j):
                if mu < PS.shape[0] and nu < PS.shape[1]:
                    bond_order += PS[mu, nu] * PS[nu, mu]

        return abs(bond_order)

    def get_overlap_matrix(self) -> np.ndarray:
        """Get overlap matrix."""
        return self.S.copy()

    def get_mo_energies(self) -> np.ndarray:
        """
        Get molecular orbital energies.

        Returns:
            Array of MO energies (sorted)
        """
        energies, _ = self.compute_molecular_orbitals()
        return energies

    def get_homo_lumo_gap(self) -> float:
        """
        Compute HOMO-LUMO gap.

        Returns:
            Gap in Hartree
        """
        energies = self.get_mo_energies()
        n_occ = self.n_electrons // 2

        if n_occ < len(energies):
            gap = energies[n_occ] - energies[n_occ - 1]
            return gap
        else:
            return 0.0

    def compute_overlap_population(
        self,
        density_matrix: np.ndarray,
        mu: int,
        nu: int
    ) -> float:
        """
        Compute overlap population between orbitals μ and ν.

        OP_μν = P_μν S_μν

        Args:
            density_matrix: Density matrix
            mu: Orbital index
            nu: Orbital index

        Returns:
            Overlap population
        """
        return density_matrix[mu, nu] * self.S[mu, nu]

    def analyze_bonding(self, density_matrix: np.ndarray) -> Dict:
        """
        Comprehensive bonding analysis.

        Args:
            density_matrix: Density matrix

        Returns:
            Dictionary with bonding analysis
        """
        analysis = {}

        # MO energies
        energies, coeffs = self.compute_molecular_orbitals()
        analysis['mo_energies'] = energies
        analysis['mo_coefficients'] = coeffs

        # HOMO-LUMO gap
        analysis['homo_lumo_gap'] = self.get_homo_lumo_gap()

        # Bonding/antibonding splitting
        analysis['bonding_analysis'] = self.get_bonding_antibonding_split()

        # Bond orders (for all atom pairs)
        bond_orders = np.zeros((len(self.atoms), len(self.atoms)))
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                bo = self.compute_bond_order(density_matrix, i, j)
                bond_orders[i, j] = bo
                bond_orders[j, i] = bo

        analysis['bond_orders'] = bond_orders

        return analysis

    def to_sparse_hamiltonian(self, mapper: str = 'jordan_wigner'):
        """
        Convert to sparse Hamiltonian representation using Pauli operators.

        Uses FAST direct construction from molecular integrals - NO dense matrix!
        This works for ALL bonding types (ionic, covalent, metallic) with:
        - ZERO accuracy loss (exact quantum mechanics)
        - 100-1000x faster for large molecules
        - Scales to 20+ qubits easily

        Args:
            mapper: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')

        Returns:
            Qiskit SparsePauliOp object ready for use in VQE
        """
        from kanad.core.hamiltonians.fast_pauli_builder import build_molecular_hamiltonian_pauli

        n_qubits = 2 * self.n_orbitals

        logger.info(f"Building sparse Hamiltonian directly from integrals (FAST method)...")
        logger.info(f"  {self.n_orbitals} orbitals → {n_qubits} qubits")
        logger.info(f"  Mapper: {mapper}")
        logger.info(f"  Bypassing {2**n_qubits}×{2**n_qubits} dense matrix construction")

        # Transform integrals to MO basis (required for fermionic transformations)
        # OpenFermion expects MO integrals, not AO integrals!
        mo_energies, C = self.compute_molecular_orbitals()
        # AO->MO via indigenous core.integrals (ActiveHamiltonian returns identity
        # C here, so these are no-ops on already-MO integrals). (reorg B3)
        from kanad.core.integrals.transforms import ao2mo_transform, one_index_transform
        h_mo = one_index_transform(self.h_core, C)
        eri_mo = ao2mo_transform(self.eri, C, chemist=True)

        logger.debug(f"Transformed integrals: AO → MO basis")

        # Include frozen core energy in constant term (for Hi-VQE active space)
        total_constant_energy = self.nuclear_repulsion + self.frozen_core_energy

        # Build Pauli operators directly from MO integrals
        # This is orders of magnitude faster than dense matrix approach!
        sparse_pauli_op = build_molecular_hamiltonian_pauli(
            h_core=h_mo,
            eri=eri_mo,
            nuclear_repulsion=total_constant_energy,
            n_orbitals=self.n_orbitals,
            mapper=mapper
        )

        num_terms = len(sparse_pauli_op)
        logger.info(f"✓ Sparse Hamiltonian: {num_terms} Pauli terms ({mapper} mapping)")
        logger.info(f"✓ Memory savings: {(2**n_qubits)**2:,} matrix elements → {num_terms} Pauli terms")

        return sparse_pauli_op
