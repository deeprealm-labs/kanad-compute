"""
Covalent bond with governance protocol enforcement.

Models orbital hybridization and electron sharing between atoms.
"""

from typing import Dict, Any, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

from kanad.core.bonds.base_bond import BaseBond
from kanad.core.atom import Atom
from kanad.core.representations.base_representation import Molecule
from kanad.core.representations.lcao_representation import LCAORepresentation
from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
from kanad.core.mappers.hybrid_orbital_mapper import HybridOrbitalMapper
from kanad.core.governance.protocols.covalent_protocol import CovalentGovernanceProtocol
from kanad.core.ansatze.hardware_efficient_ansatz import HardwareEfficientAnsatz


class CovalentBond(BaseBond):
    """
    Covalent bond with automatic governance.

    Models:
    - Orbital hybridization (sp, sp2, sp3)
    - Bonding/antibonding molecular orbitals
    - Electron sharing between atoms
    - Bell-pair entanglement structure

    Governance:
    - Enforces orbital pairing
    - Validates hybridization
    - Constructs bonding/antibonding ansatz
    """

    def __init__(
        self,
        atom_1: Atom,
        atom_2: Atom,
        distance: Optional[float] = None,
        hybridization: str = 'sp3',
        bond_order: int = 1,
        basis: str = 'sto-3g',
        spin: Optional[int] = None,
        charge: int = 0
    ):
        """
        Initialize covalent bond.

        Args:
            atom_1: First atom
            atom_2: Second atom
            distance: Bond distance in Angstroms (optional)
            hybridization: Hybridization type ('sp', 'sp2', 'sp3')
            bond_order: Bond order (1=single, 2=double, 3=triple)
            basis: Basis set name ('sto-3g', '6-31g', etc.)
        """
        # Validate basis set
        from kanad.core.integrals.basis_registry import BasisSetRegistry
        self.basis = BasisSetRegistry.validate_basis(basis)

        super().__init__([atom_1, atom_2], 'covalent', distance)

        self.atom_1 = atom_1
        self.atom_2 = atom_2
        self.bond_order = bond_order

        # Calculate total electrons and appropriate spin
        n_electrons = atom_1.atomic_number + atom_2.atomic_number - charge
        # Use caller-specified spin if provided; otherwise infer from electron count
        if spin is None:
            spin = n_electrons % 2

        # Store resolved spin/charge so geometry rebuilds preserve the electronic state
        self._spin = spin
        self._charge = charge

        # Create molecule
        self.molecule = Molecule([atom_1, atom_2], spin=spin, charge=charge)

        # Determine hybridization if not explicitly specified
        if hybridization == 'sp3':
            # Default was given, determine automatically
            self.hybridization = self._determine_hybridization()
        else:
            self.hybridization = hybridization

        # Set up representation (LCAO for covalent)
        self.representation = LCAORepresentation(self.molecule, basis_name=self.basis)

        # Set up Hamiltonian
        self.hamiltonian = CovalentHamiltonian(
            self.molecule,
            self.representation,
            basis_name=self.basis
        )

        # Governance protocol
        self.governance = CovalentGovernanceProtocol()

        # Mapper (Hybrid Orbital Mapper for MO pairs)
        # For diatomic, pairs are (0,1), (2,3), etc.
        n_orbitals = self.representation.n_orbitals
        mo_pairs = [(2*i, 2*i+1) for i in range(n_orbitals // 2)]
        if not mo_pairs:  # Fallback for odd number of orbitals
            mo_pairs = [(0, 1)] if n_orbitals >= 2 else [(0, 0)]
        self.mapper = HybridOrbitalMapper(mo_pairs)

    def compute_energy(
        self,
        method: str = 'VQE',
        max_iterations: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute covalent bond energy.

        Args:
            method: Computational method
                - 'VQE': Variational Quantum Eigensolver (standard)
                - 'HI-VQE' or 'HIVQE': Hi-VQE mode (recommended, 1000x fewer measurements)
                - 'SQD': Subspace Quantum Diagonalization
                - 'KRYLOV': Krylov SQD (efficient for larger systems)
                - 'HF': Hartree-Fock
                - 'EXACT': Exact diagonalization
            max_iterations: Maximum iterations
            **kwargs: Additional parameters
                - subspace_dim: SQD subspace dimension (default: 8)
                - krylov_dim: Krylov subspace dimension (default: 8)
                - hivqe_iterations: Hi-VQE max iterations (default: 10)
                - backend: Quantum backend ('statevector', 'ibm', 'bluequbit')

        Returns:
            Dictionary with results including energy, method, converged, etc.
        """
        result = {}

        if method.upper() == 'HF':
            # Hartree-Fock energy
            # Use more iterations for difficult cases (default 200 instead of 100)
            if max_iterations is None or max_iterations == 100:
                max_iterations = 200

            # Try standard SCF first
            density_matrix, hf_energy = self.hamiltonian.solve_scf(
                max_iterations=max_iterations,
                conv_tol=1e-7,  # Slightly relaxed for difficult cases
                use_diis=True
            )

            # If didn't converge, try again with level shift and damping
            converged = getattr(self.hamiltonian, '_scf_converged', False)
            if not converged:
                logger.warning("Standard SCF did not converge, retrying with level shift and damping...")
                density_matrix, hf_energy = self.hamiltonian.solve_scf(
                    max_iterations=max_iterations,
                    conv_tol=1e-7,
                    use_diis=True,
                    level_shift=0.5,  # 0.5 Ha level shift
                    damping_factor=0.2  # 20% damping
                )
                converged = getattr(self.hamiltonian, '_scf_converged', False)

            # Energy in Hartree (atomic units) - standard for quantum chemistry
            result['energy'] = hf_energy
            result['method'] = 'Hartree-Fock'

            # Get convergence info from Hamiltonian
            result['converged'] = converged
            result['iterations'] = getattr(self.hamiltonian, '_scf_iterations', 0)
            result['density_matrix'] = density_matrix  # Legacy compatibility

            # Store HF 1-RDM in hamiltonian for standardized access.
            # solve_scf() returns the AO-basis density and already stores it as
            # self._density_matrix (consumed by get_density_matrix()); the HF
            # reference is NOT a quantum-correlated 1-RDM, so we must NOT route
            # it through set_quantum_density_matrix(), which expects a full-MO
            # 1-RDM and would crash on the AO-vs-MO trace mismatch (e.g. H-H
            # trace 1.20 vs expected 2). The AO density is already persisted by
            # solve_scf(), so no extra storage call is needed here.

            # Get molecular orbitals
            mo_energies, mo_coeffs = self.hamiltonian.compute_molecular_orbitals()
            result['mo_energies'] = mo_energies  # Also in Hartree
            result['mo_coefficients'] = mo_coeffs

        elif method.upper() == 'VQE':
            # VQE with covalent governance ansatz
            from kanad.solvers.vqe_solver import VQESolver

            # Get system size
            n_qubits = self.representation.n_qubits
            n_electrons = self.molecule.n_electrons

            # Create covalent governance ansatz
            ansatz = HardwareEfficientAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=2,
            )

            # Create VQE solver
            vqe = VQESolver(
                hamiltonian=self.hamiltonian,
                ansatz=ansatz,
                mapper=self.mapper,
                max_iterations=max_iterations
            )

            # Solve
            # AUDIT H5: solve() returns a frozen SolverResult (not subscriptable);
            # flatten via to_dict() like the KRYLOV branch. Solver-specific fields
            # (parameters, energy_history) are merged in from extra by to_dict().
            vqe_result = vqe.solve().to_dict()

            # Energy in Hartree (atomic units)
            result['energy'] = vqe_result['energy']
            result['method'] = 'VQE (Covalent Governance)'
            result['converged'] = vqe_result['converged']
            result['iterations'] = vqe_result['iterations']
            result['parameters'] = vqe_result['parameters']
            result['energy_history'] = vqe_result['energy_history']

            # Compute HF for comparison
            _, hf_energy = self.hamiltonian.solve_scf(max_iterations=50)
            result['hf_energy'] = hf_energy
            result['correlation_energy'] = vqe_result['energy'] - hf_energy

        elif method.upper() == 'EXACT':
            # Exact diagonalization
            H_matrix = self.hamiltonian.to_matrix()
            eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

            # Energy in Hartree (atomic units)
            result['energy'] = eigenvalues[0]
            result['method'] = 'Exact'
            result['converged'] = True
            result['eigenvalues'] = eigenvalues
            result['ground_state'] = eigenvectors[:, 0]

        elif method.upper() == 'SQD':
            # Subspace Quantum Diagonalization
            from kanad.solvers.deterministic_ci import DeterministicCI

            solver = DeterministicCI(
                self,
                subspace_dim=kwargs.get('subspace_dim', 8),
                backend=kwargs.get('backend', 'statevector')
            )
            # AUDIT H5: solve() returns a frozen SolverResult (not subscriptable);
            # flatten via to_dict() like the KRYLOV branch before reading fields.
            sqd_result = solver.solve().to_dict()

            result['energy'] = sqd_result['energy']
            result['method'] = 'SQD (Subspace Quantum Diagonalization)'
            result['converged'] = sqd_result.get('converged', True)
            result['iterations'] = sqd_result.get('iterations', 0)
            result['subspace_size'] = sqd_result.get('subspace_dim', 0)

            # Get HF reference
            _, hf_energy = self.hamiltonian.solve_scf(max_iterations=50)
            result['hf_energy'] = hf_energy
            result['correlation_energy'] = sqd_result['energy'] - hf_energy

        elif method.upper() == 'KRYLOV':
            # Krylov SQD for larger systems
            from kanad.solvers.lanczos_solver import LanczosSolver

            solver = LanczosSolver(
                self,
                krylov_dim=kwargs.get('krylov_dim', 8),
                backend=kwargs.get('backend', 'statevector')
            )
            krylov_result = solver.solve().to_dict()

            result['energy'] = krylov_result['energy']
            result['method'] = 'Krylov SQD'
            result['converged'] = krylov_result.get('converged', True)
            result['iterations'] = krylov_result.get('iterations', 0)

            # Get HF reference
            _, hf_energy = self.hamiltonian.solve_scf(max_iterations=50)
            result['hf_energy'] = hf_energy
            result['correlation_energy'] = krylov_result['energy'] - hf_energy

        elif method.upper() in ('HI-VQE', 'HIVQE'):
            # Hi-VQE mode (subspace expansion VQE)
            from kanad.solvers.vqe_solver import VQESolver

            n_qubits = self.representation.n_qubits
            n_electrons = self.molecule.n_electrons

            ansatz = HardwareEfficientAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=2,
            )

            vqe = VQESolver(
                hamiltonian=self.hamiltonian,
                ansatz=ansatz,
                mapper=self.mapper,
                max_iterations=max_iterations,
                mode='hivqe',
                hivqe_max_iterations=kwargs.get('hivqe_iterations', 10)
            )

            hivqe_result = vqe.solve()

            result['energy'] = hivqe_result['energy']
            result['method'] = 'Hi-VQE (Covalent Governance)'
            result['converged'] = hivqe_result['converged']
            result['iterations'] = hivqe_result['iterations']
            result['hivqe_stats'] = hivqe_result.get('hivqe_stats', {})

            # Get HF reference
            _, hf_energy = self.hamiltonian.solve_scf(max_iterations=50)
            result['hf_energy'] = hf_energy
            result['correlation_energy'] = hivqe_result['energy'] - hf_energy

        else:
            raise ValueError(f"Unknown method: {method}. Supported: HF, VQE, EXACT, SQD, KRYLOV, HI-VQE")

        # Add bond analysis
        result['bond_analysis'] = self.analyze()
        result['bond_length'] = self.get_bond_length()

        return result

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze covalent bond properties.

        Returns:
            Dictionary with:
                - bond_type: 'covalent'
                - bond_order: Bond order
                - hybridization: Hybridization type
                - homo_lumo_gap: HOMO-LUMO gap
                - overlap: Orbital overlap
                - ionic_character: % ionic character
                - covalent_character: % covalent character
        """
        # Get HOMO-LUMO gap
        if hasattr(self.hamiltonian, 'get_homo_lumo_gap'):
            homo_lumo_gap = self.hamiltonian.get_homo_lumo_gap()
        else:
            homo_lumo_gap = None

        # Estimate ionic character from EN difference
        delta_en = abs(
            self.atom_1.electronegativity - self.atom_2.electronegativity
        )
        ionic_character = 1.0 - np.exp(-0.25 * delta_en**2)
        covalent_character = 1.0 - ionic_character

        # Get overlap from Hamiltonian if available
        if hasattr(self.hamiltonian, 'S'):
            # Overlap matrix exists
            overlap_matrix = self.hamiltonian.S
            # Get off-diagonal elements (atom 1 - atom 2 overlap)
            n_aos_per_atom = len(overlap_matrix) // 2
            if n_aos_per_atom > 0:
                overlap = np.mean(np.abs(
                    overlap_matrix[:n_aos_per_atom, n_aos_per_atom:]
                ))
            else:
                overlap = None
        else:
            overlap = None

        analysis = {
            'bond_type': 'covalent',
            'bond_order': self.bond_order,
            'hybridization': self.hybridization,
            'bond_length': self.get_bond_length(),
            'ionic_character': ionic_character,
            'covalent_character': covalent_character,
            'electronegativity_difference': delta_en,
            'entanglement_type': 'Bell-pair (bonding/antibonding)',
            'governance_protocol': 'CovalentGovernanceProtocol'
        }

        if homo_lumo_gap is not None:
            analysis['homo_lumo_gap'] = homo_lumo_gap
            analysis['homo_lumo_gap_ev'] = homo_lumo_gap * 27.211

        if overlap is not None:
            analysis['overlap'] = overlap

        # Add MO analysis if available
        if hasattr(self.hamiltonian, 'compute_molecular_orbitals'):
            mo_energies, _ = self.hamiltonian.compute_molecular_orbitals()
            analysis['mo_energies'] = mo_energies
            if len(mo_energies) >= 2:
                analysis['bonding_energy'] = mo_energies[0]
                analysis['antibonding_energy'] = mo_energies[1]
                analysis['mo_splitting'] = mo_energies[1] - mo_energies[0]

        return analysis

    def get_bond_order(self, density_matrix: Optional[np.ndarray] = None) -> float:
        """
        Compute bond order from density matrix.

        Args:
            density_matrix: Density matrix (optional, will compute if not provided)

        Returns:
            Bond order (1=single, 2=double, 3=triple, etc.)
        """
        if density_matrix is None:
            # Compute HF density matrix
            density_matrix, _ = self.hamiltonian.solve_scf()

        # Use Hamiltonian's bond order method if available
        if hasattr(self.hamiltonian, 'compute_bond_order'):
            return self.hamiltonian.compute_bond_order(density_matrix, 0, 1)
        else:
            # Return preset bond order
            return float(self.bond_order)

    def get_molecular_orbitals(self) -> tuple:
        """
        Get molecular orbital energies and coefficients.

        Returns:
            Tuple of (energies, coefficients)
        """
        return self.hamiltonian.compute_molecular_orbitals()

    def optimize_bond_length(
        self,
        r_min: float = 0.5,
        r_max: float = 3.0,
        n_points: int = 20,
        method: str = 'HF'
    ) -> Dict[str, Any]:
        """
        Optimize bond length by scanning potential energy surface.

        Args:
            r_min: Minimum bond length to scan (Angstroms)
            r_max: Maximum bond length to scan (Angstroms)
            n_points: Number of points to scan
            method: Energy calculation method ('HF' or 'VQE')

        Returns:
            Dictionary with optimized distance, energy, and scan data
        """
        import numpy as np

        distances = np.linspace(r_min, r_max, n_points)
        energies = []

        # Save original positions
        orig_pos_1 = self.atom_1.position.copy()
        orig_pos_2 = self.atom_2.position.copy()

        print(f"Optimizing bond length for {self.atom_1.symbol}-{self.atom_2.symbol}...")
        print(f"Scanning from {r_min:.2f} to {r_max:.2f} Å ({n_points} points)")

        for i, r in enumerate(distances):
            # Set new positions (along x-axis)
            self.atom_1.position = np.array([0.0, 0.0, 0.0])
            self.atom_2.position = np.array([r, 0.0, 0.0])

            # Rebuild molecule, representation, and Hamiltonian
            self.molecule = Molecule([self.atom_1, self.atom_2], spin=self._spin, charge=self._charge)
            self.representation = LCAORepresentation(self.molecule, basis_name=self.basis)
            self.hamiltonian = CovalentHamiltonian(
                self.molecule,
                self.representation,
                basis_name=self.basis
            )

            # Compute energy
            try:
                result = self.compute_energy(method=method, max_iterations=100)
                energies.append(result['energy'])
                converged = "✓" if result.get('converged', False) else "✗"
                print(f"  Point {i+1}/{n_points}: r={r:.3f} Å, E={result['energy']:.6f} Ha {converged}")
            except Exception as e:
                print(f"  Point {i+1}/{n_points}: r={r:.3f} Å FAILED - {e}")
                energies.append(np.nan)

        energies = np.array(energies)

        # Find minimum (excluding NaN values)
        valid_mask = ~np.isnan(energies)
        if not np.any(valid_mask):
            # All failed
            print("ERROR: All energy calculations failed!")
            # Restore original positions
            self.atom_1.position = orig_pos_1
            self.atom_2.position = orig_pos_2
            self._rebuild_system()
            return {'success': False}

        valid_distances = distances[valid_mask]
        valid_energies = energies[valid_mask]

        min_idx = np.argmin(valid_energies)
        opt_distance = valid_distances[min_idx]
        opt_energy = valid_energies[min_idx]

        print(f"\nOptimized bond length: {opt_distance:.4f} Å")
        print(f"Minimum energy: {opt_energy:.6f} Ha")

        # Set optimized geometry
        self.atom_1.position = np.array([0.0, 0.0, 0.0])
        self.atom_2.position = np.array([opt_distance, 0.0, 0.0])
        self._rebuild_system()

        return {
            'success': True,
            'optimized_distance': opt_distance,
            'optimized_energy': opt_energy,
            'distances': distances,
            'energies': energies,
            'original_distance': np.linalg.norm(orig_pos_2 - orig_pos_1)
        }

    def _rebuild_system(self):
        """Rebuild molecule, representation, and Hamiltonian after geometry change."""
        self.molecule = Molecule([self.atom_1, self.atom_2], spin=self._spin, charge=self._charge)
        self.representation = LCAORepresentation(self.molecule, basis_name=self.basis)
        self.hamiltonian = CovalentHamiltonian(
            self.molecule,
            self.representation,
            basis_name=self.basis
        )

    def _determine_hybridization(self) -> str:
        """
        Determine hybridization type based on atomic orbitals involved.

        Returns:
            Hybridization type: 's', 'sp', 'sp2', 'sp3', or 'none'
        """
        # Get atomic numbers
        Z1 = self.atom_1.atomic_number
        Z2 = self.atom_2.atomic_number

        # Hydrogen (Z=1) only has 1s orbital - no hybridization
        if Z1 == 1 and Z2 == 1:
            return 's'  # H-H bond uses s orbitals only

        # If either atom is hydrogen, the bond involves H 1s + other atom's hybrid
        if Z1 == 1 or Z2 == 1:
            # The non-H atom determines hybridization
            non_H_Z = Z2 if Z1 == 1 else Z1

            # H with C, N, O typically involves sp3, sp2, or sp hybrid from heavier atom
            # For diatomics, we can estimate based on bond order and atom
            if non_H_Z == 6:  # Carbon
                # C-H typically sp3 for alkanes, sp2 for alkenes, sp for alkynes
                # Without geometry info, assume sp3 for single bond
                return 'sp3' if self.bond_order == 1 else ('sp2' if self.bond_order == 2 else 'sp')
            elif non_H_Z == 7:  # Nitrogen
                # N-H typically sp3 (ammonia-like)
                return 'sp3'
            elif non_H_Z == 8:  # Oxygen
                # O-H typically sp3 (water-like)
                return 'sp3'
            else:
                return 's'  # Default for H with other elements

        # Both atoms are heavier than H
        # Determine based on bond order and atomic numbers
        if self.bond_order == 1:
            # Single bonds: typically sp3
            return 'sp3'
        elif self.bond_order == 2:
            # Double bonds: typically sp2
            return 'sp2'
        elif self.bond_order == 3:
            # Triple bonds: typically sp
            return 'sp'
        else:
            return 'sp3'  # Default

    def __repr__(self) -> str:
        """String representation."""
        bond_symbol = '=' if self.bond_order == 2 else ('≡' if self.bond_order == 3 else '—')
        return (f"CovalentBond({self.atom_1.symbol}{bond_symbol}{self.atom_2.symbol}, "
                f"{self.hybridization}, {self.get_bond_length():.2f} Å)")
