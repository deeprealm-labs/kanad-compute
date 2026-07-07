"""
Metallic Hamiltonian for delocalized electron systems.

Models:
- Tight-binding Hamiltonian in second quantization
- Periodic boundary conditions
- Hubbard U term for electron-electron repulsion
- Band structure and Fermi surface
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from kanad.core.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from kanad.core.atom import Atom
from kanad.core.representations.base_representation import Molecule


class MetallicHamiltonian(MolecularHamiltonian):
    """
    Hamiltonian for metallic bonding systems.

    Physical Model:
        H = H_hopping + H_onsite + H_coulomb

        H_hopping = Σ_<ij>,σ t_ij (a†_iσ a_jσ + h.c.)  [electron hopping]
        H_onsite = Σ_i,σ ε_i a†_iσ a_iσ               [on-site energy]
        H_coulomb = U Σ_i n_i↑ n_i↓                    [Hubbard repulsion]

    where:
        t_ij: hopping parameter between sites i,j
        ε_i: on-site energy at site i
        U: Coulomb repulsion (Hubbard U)
        a†_iσ: creation operator for electron at site i with spin σ

    KEY PHYSICS:
        - Delocalized electrons (GHZ-like entanglement)
        - Band structure with Fermi surface
        - Metallic conductivity (DOS ≠ 0 at E_F)
        - Periodic boundary conditions

    UNITS:
        All energies in this empirical model are in electron-volts (eV):
        hopping_parameter t, onsite_energy ε, and hubbard_u U. The onsite energy
        ε is the energy reference; there is no nuclear-repulsion term (it is set
        to 0.0), so do not mix Hartree-scale quantities into this Hamiltonian.
    """

    def __init__(
        self,
        molecule: Molecule,
        lattice_type: str = '1d_chain',
        hopping_parameter: float = -1.0,
        onsite_energy: float = 0.0,
        hubbard_u: float = 0.0,
        periodic: bool = True,
        temperature: Optional[float] = None,
        use_governance: bool = True,
        basis_name: str = 'sto-3g',
        frozen_orbitals: Optional[List[int]] = None,  # Hi-VQE: frozen core orbitals
        active_orbitals: Optional[List[int]] = None   # Hi-VQE: active space orbitals
    ):
        """
        Initialize metallic Hamiltonian.

        Args:
            molecule: Molecule with atoms
            lattice_type: Lattice structure ('1d_chain', '2d_square', etc.)
            hopping_parameter: Electron hopping strength t (eV)
            onsite_energy: On-site energy ε (eV)
            hubbard_u: Coulomb repulsion U (eV), 0 for non-interacting
            periodic: Use periodic boundary conditions
            temperature: Temperature in Kelvin (for thermal effects)
            use_governance: Enable governance protocol (default: True)
            basis_name: Basis set name (default: 'sto-3g')
            frozen_orbitals: List of orbital indices to freeze (active space reduction)
            active_orbitals: List of orbital indices in active space
        """
        # Validate basis set (will raise ValueError if not available)
        from kanad.core.integrals.basis_registry import BasisSetRegistry
        self.basis_name = BasisSetRegistry.validate_basis(basis_name)

        self.frozen_core_energy = 0.0

        # Active space support (MetallicHamiltonian doesn't call super().__init__)
        self.frozen_orbitals = frozen_orbitals if frozen_orbitals is not None else []
        self.active_orbitals = active_orbitals

        # Store parameters before calling super().__init__
        self.lattice_type = lattice_type
        self.hopping_parameter = hopping_parameter
        self.onsite_energy = onsite_energy
        self.hubbard_u = hubbard_u
        self.periodic = periodic
        self.temperature = temperature
        self.use_governance = use_governance

        # Initialize governance protocol
        if use_governance:
            from kanad.core.governance.protocols.metallic_protocol import MetallicGovernanceProtocol
            self.governance_protocol = MetallicGovernanceProtocol()
        else:
            self.governance_protocol = None

        self.molecule = molecule
        self.n_sites = len(molecule.atoms)

        # For metallic systems, n_orbitals = n_sites (one orbital per site in tight-binding)
        # With active space, use effective number
        n_total_sites = self.n_sites
        if self.active_orbitals is not None:
            self.n_orbitals = len(self.active_orbitals)
        else:
            self.n_orbitals = n_total_sites

        # Each atom contributes its valence electrons
        # For alkali metals (Li, Na, K): 1 valence electron per atom
        n_total_electrons = sum(atom.n_valence for atom in molecule.atoms)
        if self.active_orbitals is not None:
            n_frozen_electrons = 2 * len(self.frozen_orbitals)
            self.n_electrons = n_total_electrons - n_frozen_electrons
        else:
            self.n_electrons = n_total_electrons

        # One orbital per site holds at most 2 electrons (spin ↑/↓). For NON-monovalent
        # metals (Cu/Fe/Au/Ag/Al: n_valence > 2) this single s-band model is massively
        # over-filled, and get_fermi_energy / the density build silently return
        # unphysical results (wrong electron count, top band returned as E_F). Reject it
        # at construction rather than fail silently. (CORE_BUGS B2.)
        if self.n_electrons > 2 * self.n_orbitals:
            raise ValueError(
                f"MetallicHamiltonian: {self.n_electrons} valence electrons exceed the "
                f"capacity 2·n_orbitals={2 * self.n_orbitals} of the one-orbital-per-site "
                f"tight-binding model (n_sites={self.n_orbitals}). This single s-band model "
                f"is valid only for monovalent metals (≤2 e⁻/site, e.g. Li/Na/K). For d-band "
                f"metals (Cu/Fe/Au/Ag/Al…) supply an explicit multi-orbital basis/active space.")

        # Nuclear repulsion: this is an empirical tight-binding/Hubbard model whose
        # energies are in eV, with the onsite energy eps as the energy reference.
        # The Hubbard/tight-binding model has no nuclear-repulsion term, so adding a
        # Hartree-scale Coulomb sum here would inject a spurious constant offset
        # (unit mismatch: Hartree vs eV) that pollutes reported totals. Set to 0.0.
        self.nuclear_repulsion = 0.0

        # Build tight-binding Hamiltonian matrix
        self.h_tight_binding = self._build_tight_binding_hamiltonian()

        # For compatibility with MolecularHamiltonian interface
        # In tight-binding, h_core represents the single-particle part
        self.h_core = self.h_tight_binding.copy()

        # ERI tensor for Hubbard U (on-site repulsion only)
        # eri[i,j,k,l] = U if i=j=k=l, else 0
        # Built unconditionally: when hubbard_u == 0.0 this is an all-zeros tensor,
        # so the JW/BK two-body loops short-circuit on every term (all |eri| < 1e-12)
        # and produce exactly the intended non-interacting tight-binding Hamiltonian.
        # to_sparse_hamiltonian() requires a concrete tensor (None would crash).
        self.eri = self._build_hubbard_eri()

        # Overlap matrix (identity for orthogonal tight-binding)
        self.S = np.eye(self.n_orbitals)

    def _compute_nuclear_repulsion(self) -> float:
        """
        Compute nuclear repulsion energy.

        For metals with large lattice spacing, this is often negligible
        compared to electronic energies.
        """
        atoms = self.molecule.atoms
        e_nuc = 0.0

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                Z_i = atoms[i].atomic_number
                Z_j = atoms[j].atomic_number
                r_ij = np.linalg.norm(atoms[i].position - atoms[j].position)

                if r_ij > 1e-10:  # Avoid division by zero
                    # Convert from Angstroms to Bohr for consistency
                    from kanad.core.constants.conversion_factors import ConversionFactors
                    r_ij_bohr = r_ij * ConversionFactors.ANGSTROM_TO_BOHR
                    e_nuc += Z_i * Z_j / r_ij_bohr

        return e_nuc

    def _build_tight_binding_hamiltonian(self) -> np.ndarray:
        """
        Build tight-binding Hamiltonian matrix.

        H[i,j] = ε_i         if i == j (on-site energy)
                 t_ij        if i,j are neighbors (hopping)
                 0           otherwise

        Returns:
            Tight-binding Hamiltonian matrix (n_sites × n_sites)
        """
        n = self.n_sites
        H = np.zeros((n, n))

        # On-site energies (diagonal)
        for i in range(n):
            H[i, i] = self.onsite_energy

        # Hopping terms (off-diagonal)
        if self.lattice_type == '1d_chain':
            # Nearest-neighbor hopping
            for i in range(n - 1):
                H[i, i+1] = self.hopping_parameter
                H[i+1, i] = self.hopping_parameter

            # Periodic boundary conditions
            if self.periodic and n > 2:
                H[0, n-1] = self.hopping_parameter
                H[n-1, 0] = self.hopping_parameter

        elif self.lattice_type in ('2d_square', '3d_cubic'):
            # 2D/3D real-space adjacency requires lattice dimensions (Lx, Ly[, Lz])
            # which are not stored on this object. Previously this branch built a
            # plain 1D chain matrix while get_band_structure() reported the true
            # 2D/3D dispersion E(k) = eps + 2t[cos(kx)+cos(ky)(+cos(kz))], so the
            # real-space Hamiltonian and the band structure disagreed. Rather than
            # emit contradictory observables, refuse construction until proper
            # multi-dimensional neighbor finding is implemented.
            raise NotImplementedError(
                f"Real-space tight-binding for '{self.lattice_type}' is not implemented "
                f"(needs lattice dimensions for nearest-neighbor adjacency). "
                f"Only '1d_chain' is currently supported."
            )

        else:
            raise ValueError(f"Lattice type '{self.lattice_type}' not implemented. Supported: '1d_chain', '2d_square', '3d_cubic'")

        return H

    def _build_hubbard_eri(self) -> np.ndarray:
        """
        Build ERI tensor for Hubbard U term.

        In Hubbard model, only on-site repulsion:
            eri[i,i,i,i] = U
            eri[i,j,k,l] = 0  otherwise

        Returns:
            ERI tensor (n_orbitals × n_orbitals × n_orbitals × n_orbitals)
        """
        n = self.n_orbitals
        eri = np.zeros((n, n, n, n))

        # On-site Coulomb repulsion
        for i in range(n):
            eri[i, i, i, i] = self.hubbard_u

        return eri

    def get_band_structure(self, n_k: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute band structure E_n(k).

        Dispersion relations:
        - 1D chain: E(k) = ε + 2t cos(k)
        - 2D square: E(k) = ε + 2t[cos(kx) + cos(ky)]
        - 3D cubic: E(k) = ε + 2t[cos(kx) + cos(ky) + cos(kz)]

        Args:
            n_k: Number of k-points per dimension

        Returns:
            Dictionary with k-points and energies
        """
        if self.lattice_type == '1d_chain':
            k_points = np.linspace(-np.pi, np.pi, n_k)

            # 1D dispersion: E(k) = ε + 2t cos(k)
            energies = np.zeros((n_k, 1))

            for ik, k in enumerate(k_points):
                energies[ik, 0] = self.onsite_energy + 2 * self.hopping_parameter * np.cos(k)

            return {
                'k_points': k_points,
                'energies': energies
            }

        elif self.lattice_type in ('2d_square', '3d_cubic'):
            # The real-space tight-binding matrix for these lattices is not
            # implemented (see _build_tight_binding_hamiltonian), so returning a
            # true 2D/3D analytic dispersion here would contradict the real-space
            # Hamiltonian. Refuse until proper multi-dimensional support exists.
            raise NotImplementedError(
                f"Band structure for '{self.lattice_type}' is not implemented; "
                f"only '1d_chain' is currently supported."
            )

        else:
            raise NotImplementedError(f"Band structure for '{self.lattice_type}' not implemented")

    def get_fermi_energy(self, eigenvalues: Optional[np.ndarray] = None) -> float:
        """
        Compute Fermi energy E_F.

        For non-interacting system at T=0:
            E_F = energy of HOMO (highest occupied molecular orbital)

        Args:
            eigenvalues: Band energies (if None, compute from h_tight_binding)

        Returns:
            Fermi energy in eV
        """
        if eigenvalues is None:
            eigenvalues = np.linalg.eigvalsh(self.h_tight_binding)

        # Number of filled bands (each band holds 2 electrons: spin up/down)
        n_bands_occupied = int(np.ceil(self.n_electrons / 2.0))

        if n_bands_occupied <= 0:
            return float(eigenvalues[0]) if len(eigenvalues) > 0 else 0.0
        if n_bands_occupied > len(eigenvalues):
            # Over-filled lattice (non-monovalent metal in a single-band model). The old
            # code silently returned eigenvalues[-1] (the TOP band) as E_F — physically
            # meaningless. Fail loudly instead. (CORE_BUGS B2.)
            raise ValueError(
                f"get_fermi_energy: {n_bands_occupied} occupied bands but only "
                f"{len(eigenvalues)} bands available — the lattice is over-filled. The "
                f"one-orbital-per-site model supports only ≤2 e⁻/site (monovalent metals).")
        # Valid: n_bands_occupied in [1, len(eigenvalues)].
        if self.n_electrons % 2 == 0 and n_bands_occupied < len(eigenvalues):
            E_F = (eigenvalues[n_bands_occupied - 1] + eigenvalues[n_bands_occupied]) / 2.0
        else:
            E_F = eigenvalues[n_bands_occupied - 1]
        return E_F

    def compute_dos_at_fermi(self, eigenvalues: np.ndarray, delta_E: float = 0.1) -> int:
        """
        Compute density of states at Fermi level.

        DOS(E_F) ≠ 0 indicates metallic character.

        Args:
            eigenvalues: Band energies
            delta_E: Energy window (eV)

        Returns:
            Number of states within ±delta_E of E_F
        """
        E_F = self.get_fermi_energy(eigenvalues)
        
        # For metallic character, we need to check if there are states
        # at the Fermi level, but we need to be careful about numerical zeros
        # For even electron systems, if the Fermi level is exactly between
        # HOMO and LUMO, there should be no states at E_F
        
        # Check if Fermi level is exactly between two states (insulating)
        n_bands_occupied = int(np.ceil(self.n_electrons / 2.0))
        
        if self.n_electrons % 2 == 0 and n_bands_occupied < len(eigenvalues):
            # Even electrons: Fermi level between HOMO and LUMO
            homo_energy = eigenvalues[n_bands_occupied - 1]
            lumo_energy = eigenvalues[n_bands_occupied]
            
            # If Fermi level is exactly between HOMO and LUMO, system is insulating
            if abs(E_F - (homo_energy + lumo_energy) / 2.0) < 1e-10:
                # Check if there's actually a gap
                gap = lumo_energy - homo_energy
                if gap > 1e-6:  # Significant gap
                    return 0  # Insulating
                else:
                    # No gap - check if there are states at Fermi level
                    # For numerical zeros, we need to be more careful
                    states_at_fermi = np.abs(eigenvalues - E_F) < 1e-12
                    if np.any(states_at_fermi):
                        # There are states exactly at Fermi level
                        return np.sum(states_at_fermi)
                    else:
                        return 0  # No states at Fermi level
        
        # Otherwise, count states near Fermi level
        dos = np.sum(np.abs(eigenvalues - E_F) < delta_E)
        return dos

    def is_metallic(self) -> bool:
        """
        Check if system is metallic.

        Metallic if Fermi level crosses bands (DOS(E_F) > 0).

        Returns:
            True if metallic, False otherwise
        """
        eigenvalues = np.linalg.eigvalsh(self.h_tight_binding)
        dos_fermi = self.compute_dos_at_fermi(eigenvalues)
        return dos_fermi > 0

    def to_fermionic_operator(self):
        """
        Convert to native Kanad FermionOperator.

        This allows integration with VQE and quantum algorithms using
        Kanad's native operators (no OpenFermion dependency).

        Returns:
            kanad.core.operators.FermionOperator
        """
        from kanad.core.operators.fermion_operator import FermionOperator

        # Build fermionic operator: H = Σ h_ij a†_i a_j
        result = FermionOperator()

        # Single-particle terms (hopping + on-site)
        for i in range(self.n_orbitals):
            for j in range(self.n_orbitals):
                if abs(self.h_tight_binding[i, j]) > 1e-10:
                    coeff = self.h_tight_binding[i, j]

                    # Spin-up: a†_{2i} a_{2j}
                    result += FermionOperator(((2*i, 1), (2*j, 0)), coeff)

                    # Spin-down: a†_{2i+1} a_{2j+1}
                    result += FermionOperator(((2*i+1, 1), (2*j+1, 0)), coeff)

        # Hubbard U term: U Σ_i n_i↑ n_i↓
        if self.hubbard_u != 0.0:
            for i in range(self.n_orbitals):
                # n_i↑ n_i↓ = a†_i↑ a_i↑ a†_i↓ a_i↓
                result += FermionOperator(
                    ((2*i, 1), (2*i, 0), (2*i+1, 1), (2*i+1, 0)),
                    self.hubbard_u
                )

        return result

    def to_matrix(self) -> np.ndarray:
        """Single-particle tight-binding matrix (``h_tight_binding``, n_sites × n_sites).

        This is the ONE-BODY hopping matrix and does NOT contain the on-site Hubbard
        ``U`` interaction (a two-body term that lives in many-body Fock space, not in the
        n_sites × n_sites single-particle space). Previously this silently returned the
        non-interacting matrix even for U≠0, so any diagonalization of it gave
        non-interacting results regardless of U (the C1 finding). For the INTERACTING
        many-body Hamiltonian (U≠0) use ``to_sparse_hamiltonian()`` / its ``.to_matrix()``,
        which builds the full Fock-space operator including the on-site U.

        Raises:
            NotImplementedError: if ``hubbard_u != 0`` (the single-particle matrix cannot
                represent the interaction — fail loudly instead of dropping U silently).
        """
        if abs(self.hubbard_u) > 1e-12:
            raise NotImplementedError(
                f"MetallicHamiltonian.to_matrix() returns only the single-particle "
                f"tight-binding matrix and CANNOT represent the Hubbard U={self.hubbard_u} "
                f"interaction (a two-body term). Use to_sparse_hamiltonian().to_matrix() for "
                f"the interacting many-body Hamiltonian, or set hubbard_u=0 for tight-binding.")
        return self.h_tight_binding.copy()

    def to_sparse_hamiltonian(self, mapper: str = 'jordan_wigner'):
        """
        Convert to sparse Hamiltonian representation using Pauli operators.

        Uses fast direct construction from tight-binding/Hubbard parameters.

        Args:
            mapper: Fermion-to-qubit mapping ('jordan_wigner' or 'bravyi_kitaev')

        Returns:
            Qiskit SparsePauliOp object ready for use in VQE
        """
        from kanad.core.hamiltonians.fast_pauli_builder import build_molecular_hamiltonian_pauli

        # Include frozen core energy in constant term (for Hi-VQE active space)
        total_constant_energy = self.nuclear_repulsion + self.frozen_core_energy

        # Build Pauli operators directly from single-particle + Hubbard U terms
        sparse_pauli_op = build_molecular_hamiltonian_pauli(
            h_core=self.h_core,
            eri=self.eri,
            nuclear_repulsion=total_constant_energy,
            n_orbitals=self.n_orbitals,
            mapper=mapper
        )

        return sparse_pauli_op

    def compute_energy(self, density_matrix: np.ndarray) -> float:
        """
        Compute total energy from density matrix.

        E = Tr[P * H] + E_Hubbard + E_nuc

        For Hubbard model:
            E_Hubbard = (U/2) * Σ_i n_i↑ n_i↓
                      = (U/2) * Σ_i P_ii (1 - P_ii)  (mean-field approximation)

        Args:
            density_matrix: One-particle density matrix (spin-summed)

        Returns:
            Total electronic energy
        """
        # One-electron energy: Tr[P * H_tight_binding]
        E_one_electron = np.trace(density_matrix @ self.h_tight_binding)

        # Hubbard U correlation energy (on-site repulsion)
        # Mean-field: E_U = (U/2) * Σ_i n_i (2 - n_i) where n_i = occupation of site i
        # For spin-unpolarized: n_i = 2 * P_ii (factor of 2 for spin)
        # NOTE: the mean-field term (U/4)·Σn_i² is well-defined and correctly
        # signed for ANY nonzero U — including the ATTRACTIVE Hubbard model (U<0,
        # bipolaron/pairing physics). The old `> 0` guard silently returned bare
        # tight-binding energy for U<0, disagreeing with the qubit Hamiltonian
        # (to_fermionic_operator keeps U≠0). (CORE_BUGS B8.)
        if self.hubbard_u != 0.0:
            n_orbitals = density_matrix.shape[0]
            occupations = np.diag(density_matrix)  # Site occupations (per spin)

            # Hubbard energy: U * Σ_i n_i↑ * n_i↓
            # For mean-field with equal spin: n_i↑ = n_i↓ = n_i/2
            # E_U = U * Σ_i (n_i/2)^2 = (U/4) * Σ_i n_i^2
            E_hubbard = (self.hubbard_u / 4.0) * np.sum(occupations ** 2)
        else:
            E_hubbard = 0.0

        # Add nuclear repulsion
        E_total = E_one_electron + E_hubbard + self.nuclear_repulsion

        return E_total

    def __repr__(self) -> str:
        """String representation."""
        return (f"MetallicHamiltonian({self.lattice_type}, "
                f"n_sites={self.n_sites}, "
                f"t={self.hopping_parameter:.2f} eV, "
                f"U={self.hubbard_u:.2f} eV)")
