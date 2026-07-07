"""
Second quantization representation for ionic bonding.

In ionic bonding, electrons are localized on specific atoms.
This representation uses localized atomic orbitals with minimal
entanglement between sites.

Physical picture:
- Electron transfer: a†_i a_j (electron hops from j to i)
- Localized states: minimal entanglement
- Hubbard-like model with on-site repulsion
"""

import numpy as np
from typing import Dict, List, Optional
from kanad.core.representations.base_representation import BaseRepresentation, Molecule


class SecondQuantizationRepresentation(BaseRepresentation):
    """
    Second quantization (Fock space) representation for ionic systems.

    Uses atomic orbital basis with creation/annihilation operators.
    Optimized for systems with electron transfer between localized sites.

    Example: Na+ Cl-
    - Na: loses electron (donor site)
    - Cl: gains electron (acceptor site)
    - Hamiltonian: H = ε_Na n_Na + ε_Cl n_Cl + t(a†_Na a_Cl + h.c.)
    """

    def __init__(self, molecule: Molecule, include_spin: bool = True, basis_name: str = 'sto-3g'):
        """
        Initialize second quantization representation.

        Args:
            molecule: Molecule object
            include_spin: Whether to include spin degrees of freedom
        """
        super().__init__(molecule)
        self.include_spin = include_spin

        # AUTHORITATIVE orbital count = number of basis functions for the molecule's
        # basis — must match the (PySCF-backed) IonicHamiltonian and the ansatz/solver,
        # which read representation.n_qubits (bonds/ionic_bond.py:144,235). The previous
        # n_orbitals = molecule.n_atoms (one orbital per atom) disagreed with the real
        # Hamiltonian dimension for any non-minimal/multi-function-per-atom basis.
        try:
            from kanad.core.integrals.basis_sets import BasisSet
            _bs = BasisSet(basis_name)
            _bs.build_basis(self.molecule.atoms)
            self.n_orbitals = int(_bs.n_basis_functions)
        except Exception:
            self.n_orbitals = molecule.n_atoms

        # Number of spin orbitals
        self.n_spin_orbitals = 2 * self.n_orbitals if include_spin else self.n_orbitals

        # Number of qubits (Jordan-Wigner mapping: 1 qubit per spin-orbital)
        self.n_qubits = self.n_spin_orbitals

        # Calculate total number of electrons (charge-aware: BondMolecule.n_electrons
        # subtracts the molecule-level charge, which the atoms alone do not carry)
        self.n_electrons = self.molecule.n_electrons

    def build_hamiltonian(self) -> 'IonicHamiltonian':
        """
        Build ionic Hamiltonian in second quantized form.

        H = Σ_i ε_i n_i + Σ_ij t_ij a†_i a_j + Σ_i U_i n_i↑ n_i↓

        where:
            ε_i: on-site energy (electronegativity)
            t_ij: hopping/transfer integral
            U_i: on-site Coulomb repulsion (Hubbard U)

        Returns:
            IonicHamiltonian object
        """
        from kanad.core.hamiltonians.ionic_hamiltonian import IonicHamiltonian

        # Build ionic Hamiltonian with charge transfer and site energies
        return IonicHamiltonian(
            molecule=self.molecule,
            representation=self
        )

    def get_reference_state(self) -> np.ndarray:
        """
        Get reference state (Hartree-Fock or simple product state).

        For ionic systems, this is typically the charge-separated state.
        E.g., for NaCl: |Na+ Cl-⟩ = |0⟩_Na ⊗ |↑↓⟩_Cl

        Returns:
            Reference state vector
        """
        # For simplified one-orbital-per-atom model, we fill spin orbitals
        # based on the minimal representation (not full electron count)
        state_dim = 2 ** self.n_qubits
        ref_state = np.zeros(state_dim)

        # Construct reference determinant
        # For ionic bonds, typically occupy the more electronegative atom's orbitals
        # For now, fill lowest energy spin orbitals up to n_qubits
        # This gives the proper charge-separated state
        hf_occupation = 0
        n_occ = min(self.n_electrons, self.n_qubits)  # Can't exceed number of qubits
        for i in range(n_occ):
            hf_occupation |= (1 << i)

        ref_state[hf_occupation] = 1.0

        return ref_state

    def compute_observables(self, state: np.ndarray) -> Dict[str, float]:
        """
        Compute observables for ionic system.

        Args:
            state: Quantum state vector

        Returns:
            Dictionary of observables:
                - 'charge_transfer': Amount of charge transferred
                - 'site_occupations': Occupation numbers per site
                - 'energy': Expectation value of energy
        """
        observables = {}

        # Compute site occupations from quantum state
        site_occupations = np.zeros(self.n_orbitals)

        # Parse state vector to get occupations for each site
        for i in range(self.n_orbitals):
            # Count occupation of spin-up and spin-down on site i
            site_occupations[i] = self._compute_site_occupation(state, i)

        observables['site_occupations'] = site_occupations

        # Charge transfer vs neutral atoms. n_orbitals is the basis-function count
        # (not n_atoms), so aggregate the per-orbital occupations into per-ATOM
        # occupations (basis functions -> atoms) before comparing to neutral
        # valence. The old code subtracted a length-n_atoms neutral vector from a
        # length-n_orbitals occupation vector -> shape crash for any atom with >1
        # basis function (LiH, H2O, C/N/O, ...).
        n_atoms = len(self.molecule.atoms)
        atom_occ = np.zeros(n_atoms)
        try:
            from pyscf import gto
            mol_p = gto.M(
                atom=[[a.symbol, tuple(np.asarray(a.position, dtype=float))] for a in self.molecule.atoms],
                basis=getattr(self.molecule, 'basis', 'sto-3g') or 'sto-3g', unit='Angstrom',
                spin=getattr(self.molecule, 'spin', 0) or 0,
                charge=getattr(self.molecule, 'charge', 0) or 0, verbose=0,
            )
            for A, sl in enumerate(mol_p.aoslice_by_atom()):
                atom_occ[A] = float(np.sum(site_occupations[int(sl[2]):int(sl[3])]))
        except Exception:
            per = max(1, len(site_occupations) // max(1, n_atoms))
            for A in range(n_atoms):
                atom_occ[A] = float(np.sum(site_occupations[A * per:(A + 1) * per]))
        neutral_occupations = np.array([atom.n_valence for atom in self.molecule.atoms], dtype=float)
        charge_transfer = atom_occ - neutral_occupations
        observables['charge_transfer'] = charge_transfer
        observables['total_charge_transfer'] = float(np.sum(np.abs(charge_transfer)) / 2)

        return observables

    def _compute_site_occupation(self, state: np.ndarray, site: int) -> float:
        """
        Compute occupation number for a site.

        n_i = ⟨a†_i↑ a_i↑ + a†_i↓ a_i↓⟩

        Args:
            state: Quantum state
            site: Site index

        Returns:
            Occupation number (0 to 2)
        """
        # Compute expectation value ⟨ψ|n_i|ψ⟩ where n_i = n_{i↑} + n_{i↓}
        # by summing |⟨basis|ψ⟩|² weighted by occupation in each basis state
        occupation = 0.0
        state_dim = len(state)

        for basis_state in range(state_dim):
            amplitude = state[basis_state]
            if abs(amplitude) > 1e-10:
                # Count occupation in this basis state using bit operations
                # Spin-up orbital for site i
                if basis_state & (1 << (2 * site)):
                    occupation += abs(amplitude) ** 2

                # Spin-down orbital for site i
                if basis_state & (1 << (2 * site + 1)):
                    occupation += abs(amplitude) ** 2

        return occupation

    def to_qubit_operator(self) -> Dict[str, complex]:
        """
        Map the ionic Hamiltonian to qubit operators.

        Delegates to the PySCF-backed IonicHamiltonian.to_sparse_hamiltonian()
        and converts the resulting SparsePauliOp into a {Pauli string: coeff} dict.
        (The previous implementation assembled a Hubbard-toy Hamiltonian from
        hardcoded hopping t_0/d_0 and electronegativity-derived on-site energies,
        not real molecular integrals; it has been removed.)

        Returns:
            Dictionary mapping Pauli strings to complex coefficients
        """
        # Build the PySCF-backed Hamiltonian if not already cached
        if not hasattr(self, 'hamiltonian') or self.hamiltonian is None:
            self.hamiltonian = self.build_hamiltonian()

        sparse_pauli = self.hamiltonian.to_sparse_hamiltonian()

        pauli_hamiltonian = {
            str(pauli): complex(coeff)
            for pauli, coeff in zip(sparse_pauli.paulis, sparse_pauli.coeffs)
        }

        # Clean up near-zero terms
        pauli_hamiltonian = {k: v for k, v in pauli_hamiltonian.items() if abs(v) > 1e-12}

        return pauli_hamiltonian

    def get_num_qubits(self) -> int:
        """Get number of qubits (one per spin-orbital)."""
        return self.n_qubits

    def __repr__(self) -> str:
        """String representation."""
        return f"SecondQuantization(n_qubits={self.n_qubits}, n_electrons={self.n_electrons})"
