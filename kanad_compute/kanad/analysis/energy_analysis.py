"""
Energy analysis and decomposition tools.

Provides utilities for analyzing VQE results and decomposing
molecular energies into physical components.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from kanad.core.hamiltonians.molecular_hamiltonian import MolecularHamiltonian
from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
from kanad.core.hamiltonians.ionic_hamiltonian import IonicHamiltonian


class EnergyAnalyzer:
    """
    Analyzes molecular energies and their components.

    Decomposes total energy into:
    - Nuclear repulsion
    - Kinetic energy
    - Electron-nuclear attraction
    - Electron-electron repulsion
    - Exchange energy
    - Correlation energy
    """

    def __init__(self, hamiltonian: MolecularHamiltonian):
        """
        Initialize energy analyzer.

        Args:
            hamiltonian: Molecular Hamiltonian
        """
        self.hamiltonian = hamiltonian

    def decompose_energy(self, density_matrix: np.ndarray) -> Dict[str, float]:
        """
        Decompose total energy into components.

        Args:
            density_matrix: One-particle density matrix

        Returns:
            Dictionary with energy components (Hartree)
        """
        decomposition = {}

        # Nuclear repulsion
        decomposition['nuclear_repulsion'] = self.hamiltonian.nuclear_repulsion

        # One-electron terms
        h_core = self.hamiltonian.h_core
        dm = np.asarray(density_matrix, dtype=float)
        hc_shape = np.asarray(h_core).shape
        # The density and the Hamiltonian integrals must be in the SAME orbital
        # basis. A full-space density into an active-space Hamiltonian used to
        # surface as a cryptic numpy broadcast error (F12, planck full-analysis
        # audit). When the caller passes a full-space density to an active-space
        # Hamiltonian, fall back to the active-space HF *reference* density
        # (diag of the active MO occupations, already in the active MO basis):
        # together with the active `h_core`/`eri` and the frozen-core-carrying
        # `nuclear_repulsion`, its decomposition is self-consistent and reproduces
        # the HF energy. Flagged via `density_source`.
        decomposition['density_source'] = 'provided'
        if dm.shape != hc_shape:
            # Only auto-fall-back when the caller passed the recognizable FULL
            # mean-field density (e.g. `mf.make_rdm1()`, shape = full MO count) to
            # an active-space Hamiltonian — the common convenience case. A density
            # of any other (genuinely wrong) shape still raises the clear error, so
            # a real basis mistake isn't silently masked.
            mf = getattr(self.hamiltonian, 'mf', None)
            full_dim = (int(np.asarray(mf.mo_coeff).shape[0])
                        if mf is not None and getattr(mf, 'mo_coeff', None) is not None else None)
            ref = self._active_space_reference_density()
            if (ref is not None and ref.shape == hc_shape
                    and full_dim is not None and dm.shape == (full_dim, full_dim)):
                dm = ref
                density_matrix = ref
                decomposition['density_source'] = 'active_space_hf_reference'
            else:
                raise ValueError(
                    f"density_matrix shape {dm.shape} does not match the Hamiltonian's "
                    f"one-electron integrals {hc_shape}; they must be in the same orbital "
                    f"basis. For an active-space Hamiltonian pass the active-space 1-RDM "
                    f"(or the full mean-field density `mf.make_rdm1()`, which auto-falls-back "
                    f"to the active-space HF reference)."
                )
        E_core = np.sum(dm * h_core)
        decomposition['one_electron'] = E_core

        # Two-electron terms (if available)
        if hasattr(self.hamiltonian, 'eri'):
            eri = self.hamiltonian.eri
            n = len(h_core)

            # Coulomb energy
            J = 0.0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            J += 0.5 * density_matrix[i, j] * density_matrix[k, l] * eri[i, j, k, l]

            decomposition['coulomb'] = J

            # Exchange energy
            K = 0.0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        for l in range(n):
                            K += 0.25 * density_matrix[i, k] * density_matrix[j, l] * eri[i, j, k, l]

            decomposition['exchange'] = -K  # Note: exchange lowers energy

            decomposition['two_electron'] = J - K
        else:
            decomposition['two_electron'] = 0.0

        # Total energy (do NOT include intermediate coulomb/exchange terms to avoid double-counting)
        decomposition['total'] = (
            decomposition['nuclear_repulsion'] +
            decomposition['one_electron'] +
            decomposition['two_electron']
        )

        return decomposition

    def _active_space_reference_density(self) -> Optional[np.ndarray]:
        """Active-space HF reference 1-RDM in the active MO basis: diag of the
        active-orbital occupations. Returns None if the Hamiltonian isn't an
        active-space system exposing a mean-field + active orbital indices."""
        ham = self.hamiltonian
        mf = getattr(ham, 'mf', None)
        if mf is None or not hasattr(mf, 'mo_occ'):
            return None
        active = None
        asp = getattr(ham, 'active_space', None)
        if asp is not None and getattr(asp, 'active_indices', None) is not None:
            active = list(asp.active_indices)
        elif getattr(ham, 'active_orbitals', None) is not None:
            active = list(ham.active_orbitals)
        if not active:
            return None
        occ = np.asarray(mf.mo_occ, dtype=float)[active]
        return np.diag(occ)

    def compute_binding_energy(
        self,
        molecular_energy: float,
        atomic_energies: List[float]
    ) -> float:
        """
        Compute binding energy (bond dissociation energy).

        BE = E_atoms - E_molecule

        Args:
            molecular_energy: Total molecular energy
            atomic_energies: List of atomic energies

        Returns:
            Binding energy (positive for stable molecule)
        """
        total_atomic_energy = sum(atomic_energies)
        binding_energy = total_atomic_energy - molecular_energy

        return binding_energy

    def compute_ionization_energy(
        self,
        neutral_energy: float,
        cation_energy: float
    ) -> float:
        """
        Compute ionization energy.

        IE = E_cation - E_neutral

        Args:
            neutral_energy: Neutral molecule energy
            cation_energy: Cation energy

        Returns:
            Ionization energy (positive)
        """
        return cation_energy - neutral_energy

    def compute_electron_affinity(
        self,
        neutral_energy: float,
        anion_energy: float
    ) -> float:
        """
        Compute electron affinity.

        EA = E_neutral - E_anion

        Args:
            neutral_energy: Neutral molecule energy
            anion_energy: Anion energy

        Returns:
            Electron affinity (positive for stable anion)
        """
        return neutral_energy - anion_energy

    def analyze_convergence(self, energy_history: np.ndarray) -> Dict:
        """
        Analyze VQE convergence.

        Args:
            energy_history: Energy at each iteration

        Returns:
            Convergence metrics
        """
        analysis = {
            'initial_energy': energy_history[0],
            'final_energy': energy_history[-1],
            'energy_change': energy_history[-1] - energy_history[0],
            'iterations': len(energy_history),
            'converged': self._check_convergence(energy_history),
        }

        # Convergence rate
        if len(energy_history) > 1:
            energy_diffs = np.diff(energy_history)
            analysis['mean_energy_change'] = np.mean(np.abs(energy_diffs))
            analysis['final_gradient'] = np.abs(energy_diffs[-1])

        return analysis

    def _check_convergence(
        self,
        energy_history: np.ndarray,
        threshold: float = 1e-6
    ) -> bool:
        """Check if VQE converged."""
        if len(energy_history) < 2:
            return False

        # Check last few iterations
        n_check = min(5, len(energy_history) - 1)
        recent_changes = np.abs(np.diff(energy_history[-n_check:]))

        return np.all(recent_changes < threshold)


class BondingAnalyzer:
    """
    Analyzes chemical bonding characteristics.

    Provides tools for:
    - Bond order analysis
    - Orbital population analysis
    - Hybridization analysis
    - Charge transfer analysis
    """

    def __init__(self, hamiltonian: MolecularHamiltonian):
        """
        Initialize bonding analyzer.

        Args:
            hamiltonian: Molecular Hamiltonian
        """
        self.hamiltonian = hamiltonian

    def analyze_bonding_type(self) -> Dict:
        """
        Determine bonding type (ionic, covalent, metallic).

        Returns:
            Bonding analysis
        """
        analysis = {'bonding_type': 'unknown'}

        if isinstance(self.hamiltonian, IonicHamiltonian):
            analysis['bonding_type'] = 'ionic'
            analysis['characteristics'] = [
                'Localized electrons',
                'Charge transfer',
                'Electrostatic interactions'
            ]

        elif isinstance(self.hamiltonian, CovalentHamiltonian):
            analysis['bonding_type'] = 'covalent'
            analysis['characteristics'] = [
                'Shared electrons',
                'Orbital hybridization',
                'Bonding/antibonding MOs'
            ]

            # Get HOMO-LUMO gap
            if hasattr(self.hamiltonian, 'get_homo_lumo_gap'):
                gap = self.hamiltonian.get_homo_lumo_gap()
                analysis['homo_lumo_gap'] = gap
                analysis['homo_lumo_gap_ev'] = gap * 27.211  # Convert to eV

        # Support MolecularHamiltonian (multi-atom molecules) AND active-space /
        # builder systems: the latter lack `.atoms`/`.mol` but carry a mean-field
        # (`.mf`) holding the frontier orbital energies — F8 fix so builder systems
        # report a bonding type + HOMO-LUMO gap instead of 'unknown'.
        elif (hasattr(self.hamiltonian, 'mol') and hasattr(self.hamiltonian, 'atoms')) \
                or getattr(self.hamiltonian, 'mf', None) is not None:
            # For multi-atom molecules, assume covalent bonding
            analysis['bonding_type'] = 'molecular'
            analysis['characteristics'] = [
                'Multi-atom molecule',
                'Covalent bonds between atoms',
                'Molecular orbitals'
            ]

            # Get HOMO-LUMO gap from MO energies
            if hasattr(self.hamiltonian, 'mf'):
                mo_energies = np.asarray(self.hamiltonian.mf.mo_energy)
                mo_occ = getattr(self.hamiltonian.mf, 'mo_occ', None)

                # Use actual orbital occupations (correct for open-shell/odd-electron)
                if mo_occ is not None:
                    mo_occ = np.asarray(mo_occ)
                    occ_idx = np.where(mo_occ > 0)[0]
                    virt_idx = np.where(mo_occ == 0)[0]
                    if occ_idx.size > 0 and virt_idx.size > 0:
                        homo = int(occ_idx.max())
                        lumo = int(virt_idx.min())
                        gap = float(mo_energies[lumo] - mo_energies[homo])
                        analysis['homo_lumo_gap'] = gap
                        analysis['homo_lumo_gap_ev'] = gap * 27.211  # Convert to eV
                else:
                    # Fallback to closed-shell heuristic
                    n_occ = self.hamiltonian.n_electrons // 2
                    if 0 < n_occ < len(mo_energies):
                        gap = float(mo_energies[n_occ] - mo_energies[n_occ - 1])
                        analysis['homo_lumo_gap'] = gap
                        analysis['homo_lumo_gap_ev'] = gap * 27.211  # Convert to eV

        return analysis

    def compute_mulliken_charges(
        self,
        density_matrix: np.ndarray,
        overlap_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute Mulliken atomic charges.

        Q_A = Z_A - Σ_μ∈A Σ_ν P_μν S_νμ

        Args:
            density_matrix: Density matrix
            overlap_matrix: Overlap matrix

        Returns:
            Atomic charges
        """
        # Mulliken population matrix
        PS = density_matrix @ overlap_matrix

        if isinstance(self.hamiltonian, CovalentHamiltonian):
            n_atoms = len(self.hamiltonian.atoms)
            charges = np.zeros(n_atoms)

            # Correct atom->AO mapping. Atoms carry different numbers of basis
            # functions, so equal-split (n_orbitals // n_atoms) mislabels every
            # heteronuclear molecule (e.g. assigning H and O the same orbital
            # count in H2O). Reuse the hamiltonian helper, which uses PySCF
            # aoslice_by_atom() when available, the STO-3G AO-count table
            # otherwise, and only falls back to equal-split as a last resort
            # (same mapping the Mayer bond-order sibling uses).
            ao_ranges = self.hamiltonian._get_ao_ranges()
            for atom_idx in range(n_atoms):
                start, end = ao_ranges[atom_idx]
                population = np.trace(PS[start:end, start:end])
                Z = self.hamiltonian.atoms[atom_idx].atomic_number
                charges[atom_idx] = Z - population

            return charges
        else:
            # For ionic systems, use different approach
            return np.array([])

    def analyze_bond_orders(
        self,
        density_matrix: np.ndarray
    ) -> Dict:
        """
        Analyze bond orders between atoms.

        Args:
            density_matrix: Density matrix

        Returns:
            Bond order analysis
        """
        analysis = {}

        if isinstance(self.hamiltonian, CovalentHamiltonian):
            n_atoms = len(self.hamiltonian.atoms)
            bond_orders = np.zeros((n_atoms, n_atoms))

            # Prefer Mayer bond order via PySCF mol if available
            pyscf_mol = getattr(self.hamiltonian, 'mol', None)
            if pyscf_mol is not None and hasattr(pyscf_mol, 'aoslice_by_atom'):
                S = pyscf_mol.intor('int1e_ovlp')
                atom_slices = pyscf_mol.aoslice_by_atom()
                PS = density_matrix @ S
                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        si, ei = int(atom_slices[i][2]), int(atom_slices[i][3])
                        sj, ej = int(atom_slices[j][2]), int(atom_slices[j][3])
                        bo = 0.0
                        for mu in range(si, ei):
                            for nu in range(sj, ej):
                                if mu < PS.shape[0] and nu < PS.shape[1]:
                                    bo += PS[mu, nu] * PS[nu, mu]
                        bond_orders[i, j] = abs(bo)
                        bond_orders[j, i] = abs(bo)
            else:
                # Fall back to CovalentHamiltonian's own method (now uses Mayer)
                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        bo = self.hamiltonian.compute_bond_order(density_matrix, i, j)
                        bond_orders[i, j] = bo
                        bond_orders[j, i] = bo

            analysis['bond_orders'] = bond_orders
            analysis['bond_classification'] = self._classify_bonds(bond_orders)

        # Also support MolecularHamiltonian (from kanad.core.molecule)
        elif hasattr(self.hamiltonian, 'mol') and hasattr(self.hamiltonian, 'atoms'):
            # This is a MolecularHamiltonian using PySCF backend
            n_atoms = len(self.hamiltonian.atoms)
            bond_orders = np.zeros((n_atoms, n_atoms))

            # Get overlap matrix from PySCF molecule
            S = self.hamiltonian.mol.intor('int1e_ovlp')
            n_orbitals = self.hamiltonian.n_orbitals

            # Compute Mayer bond order for each atom pair
            # BO_ij = Σ_μ∈i Σ_ν∈j (PS)_μν (PS)_νμ
            # Use aoslice_by_atom() for correct atom-to-AO mapping
            # (ao_loc_nr() returns shell offsets, NOT atom offsets!)
            atom_slices = self.hamiltonian.mol.aoslice_by_atom()
            PS = density_matrix @ S
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    start_i = atom_slices[i][2]  # ao_start
                    end_i = atom_slices[i][3]    # ao_end
                    start_j = atom_slices[j][2]
                    end_j = atom_slices[j][3]

                    bond_order = 0.0
                    for mu in range(start_i, end_i):
                        for nu in range(start_j, end_j):
                            if mu < PS.shape[0] and nu < PS.shape[1]:
                                bond_order += PS[mu, nu] * PS[nu, mu]

                    bo = abs(bond_order)
                    bond_orders[i, j] = bo
                    bond_orders[j, i] = bo

            analysis['bond_orders'] = bond_orders
            analysis['bond_classification'] = self._classify_bonds(bond_orders)

        return analysis

    def _classify_bonds(self, bond_orders: np.ndarray) -> Dict:
        """
        Classify bonds as single, double, triple, etc.

        Args:
            bond_orders: Bond order matrix

        Returns:
            Bond classification with atom symbols
        """
        classification = {}
        n_atoms = len(bond_orders)

        # Get atom symbols if available
        atoms = getattr(self.hamiltonian, 'atoms', [])

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                bo = bond_orders[i, j]

                if bo > 0.15:  # Significant bond (lowered from 0.5 for charged/polar species)
                    if bo < 1.3:
                        bond_type = 'single'
                    elif bo < 2.3:
                        bond_type = 'double'
                    elif bo < 3.3:
                        bond_type = 'triple'
                    else:
                        bond_type = 'multiple'

                    # Get atom symbols if available
                    atom_i_symbol = atoms[i].symbol if i < len(atoms) and hasattr(atoms[i], 'symbol') else f'Atom{i}'
                    atom_j_symbol = atoms[j].symbol if j < len(atoms) and hasattr(atoms[j], 'symbol') else f'Atom{j}'

                    classification[f'atom_{i}_atom_{j}'] = {
                        'type': bond_type,
                        'order': bo,
                        'atom_i': i,
                        'atom_j': j,
                        'atom_i_symbol': atom_i_symbol,
                        'atom_j_symbol': atom_j_symbol,
                        'bond_label': f'{atom_i_symbol}-{atom_j_symbol}'
                    }

        return classification

    def compute_overlap_populations(
        self,
        density_matrix: np.ndarray,
        overlap_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute overlap population matrix.

        OP_μν = P_μν S_μν

        Args:
            density_matrix: Density matrix
            overlap_matrix: Overlap matrix

        Returns:
            Overlap population matrix
        """
        return density_matrix * overlap_matrix


class CorrelationAnalyzer:
    """
    Analyzes electron correlation effects.

    Compares VQE (correlated) vs Hartree-Fock (mean-field) energies.
    """

    def __init__(self, hamiltonian: MolecularHamiltonian):
        """
        Initialize correlation analyzer.

        Args:
            hamiltonian: Molecular Hamiltonian
        """
        self.hamiltonian = hamiltonian

    def compute_correlation_energy(
        self,
        vqe_energy: float,
        hf_energy: float
    ) -> float:
        """
        Compute correlation energy.

        E_corr = E_VQE - E_HF

        Args:
            vqe_energy: VQE ground state energy
            hf_energy: Hartree-Fock energy

        Returns:
            Correlation energy (negative for stable correlation)
        """
        return vqe_energy - hf_energy

    def compute_percent_correlation(
        self,
        vqe_energy: float,
        hf_energy: float,
        exact_energy: Optional[float] = None
    ) -> float:
        """
        Compute percentage of correlation energy recovered.

        % = (E_HF - E_VQE) / (E_HF - E_exact) × 100

        Args:
            vqe_energy: VQE energy
            hf_energy: Hartree-Fock energy
            exact_energy: Exact (FCI) energy (if known)

        Returns:
            Percentage of correlation recovered
        """
        if exact_energy is None:
            return 0.0

        E_corr_total = exact_energy - hf_energy
        E_corr_vqe = vqe_energy - hf_energy

        if abs(E_corr_total) < 1e-10:
            return 100.0

        return (E_corr_vqe / E_corr_total) * 100.0

    def analyze_electron_correlation(
        self,
        vqe_result: Dict,
        hf_energy: float
    ) -> Dict:
        """
        Comprehensive correlation analysis.

        Args:
            vqe_result: VQE result dictionary
            hf_energy: Hartree-Fock energy

        Returns:
            Correlation analysis
        """
        vqe_energy = vqe_result['energy']
        correlation_energy = self.compute_correlation_energy(vqe_energy, hf_energy)

        analysis = {
            'hf_energy': hf_energy,
            'vqe_energy': vqe_energy,
            'correlation_energy': correlation_energy,
            'correlation_energy_ev': correlation_energy * 27.211,  # eV
        }

        # Correlation strength classification
        if abs(correlation_energy) < 0.001:
            analysis['correlation_strength'] = 'negligible'
        elif abs(correlation_energy) < 0.01:
            analysis['correlation_strength'] = 'weak'
        elif abs(correlation_energy) < 0.1:
            analysis['correlation_strength'] = 'moderate'
        else:
            analysis['correlation_strength'] = 'strong'

        return analysis
