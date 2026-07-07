"""
Ionic bond with governance protocol enforcement.

Models electron transfer between atoms with localized orbitals.
"""

from typing import Dict, Any, Optional
import numpy as np

from kanad.core.bonds.base_bond import BaseBond
from kanad.core.atom import Atom
from kanad.core.representations.base_representation import Molecule
from kanad.core.representations.second_quantization import SecondQuantizationRepresentation
from kanad.core.hamiltonians.ionic_hamiltonian import IonicHamiltonian
from kanad.core.mappers.jordan_wigner_mapper import JordanWignerMapper
from kanad.core.governance.protocols.ionic_protocol import IonicGovernanceProtocol
from kanad.core.ansatze.hardware_efficient_ansatz import HardwareEfficientAnsatz


class IonicBond(BaseBond):
    """
    Ionic bond with automatic governance.

    Models:
    - Electron transfer from donor to acceptor
    - Localized atomic orbitals
    - Coulombic interactions
    - Minimal entanglement (charge transfer only)

    Governance:
    - Enforces localization
    - Validates charge transfer
    - Constructs minimal-entanglement ansatz
    """

    def __init__(
        self,
        atom_1: Atom,
        atom_2: Atom,
        distance: Optional[float] = None,
        basis: str = 'sto-3g',
        spin: Optional[int] = None,
        charge: int = 0
    ):
        """
        Initialize ionic bond.

        Args:
            atom_1: First atom (typically donor/cation)
            atom_2: Second atom (typically acceptor/anion)
            distance: Bond distance in Angstroms (optional)
            basis: Basis set name (default: 'sto-3g')
        """
        # Validate basis set
        from kanad.core.integrals.basis_registry import BasisSetRegistry
        self.basis = BasisSetRegistry.validate_basis(basis)

        super().__init__([atom_1, atom_2], 'ionic', distance)

        # Identify donor and acceptor based on electronegativity
        if atom_1.electronegativity < atom_2.electronegativity:
            self.donor = atom_1
            self.acceptor = atom_2
        else:
            self.donor = atom_2
            self.acceptor = atom_1

        # Calculate electrons (accounting for charge) and spin
        n_electrons = atom_1.atomic_number + atom_2.atomic_number - charge
        if spin is None:
            spin = n_electrons % 2

        # Create molecule
        self.molecule = Molecule([atom_1, atom_2], spin=spin, charge=charge)

        # Set up representation (second quantization for ionic)
        self.representation = SecondQuantizationRepresentation(self.molecule, basis_name=self.basis)

        # Set up Hamiltonian
        self.hamiltonian = IonicHamiltonian(
            self.molecule,
            self.representation,
            basis_name=self.basis
        )

        # Governance protocol
        self.governance = IonicGovernanceProtocol()

        # Mapper (Jordan-Wigner for localized states)
        self.mapper = JordanWignerMapper()

    def compute_energy(
        self,
        method: str = 'VQE',
        max_iterations: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute ionic bond energy.

        Args:
            method: Computational method
                - 'VQE': Variational Quantum Eigensolver (standard)
                - 'HI-VQE' or 'HIVQE': Hi-VQE mode (recommended, 1000x fewer measurements)
                - 'SQD': Subspace Quantum Diagonalization
                - 'KRYLOV': Krylov SQD (efficient for larger systems)
                - 'HF': Hartree-Fock (classical)
                - 'EXACT': Exact diagonalization (small systems)
            max_iterations: Maximum VQE iterations
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
            density_matrix, hf_energy = self.hamiltonian.solve_scf(
                max_iterations=max_iterations
            )
            # Energy in Hartree (atomic units) - standard for quantum chemistry
            result['energy'] = hf_energy
            result['method'] = 'Hartree-Fock'
            # Get convergence info from Hamiltonian
            result['converged'] = getattr(self.hamiltonian, '_scf_converged', False)
            result['iterations'] = getattr(self.hamiltonian, '_scf_iterations', 0)
            result['density_matrix'] = density_matrix  # Legacy compatibility
            # NOTE: do NOT call set_quantum_density_matrix() here — solve_scf() already
            # stored the HF AO density in self._density_matrix (consumed by
            # get_density_matrix()'s HF fallback). set_quantum_density_matrix expects a
            # full-MO 1-RDM and would fail MO-trace validation on the AO density.

        elif method.upper() == 'VQE':
            # VQE with governance-aware ansatz
            from kanad.solvers.vqe_solver import VQESolver

            # Get number of qubits from representation
            n_qubits = self.representation.n_qubits
            n_electrons = self.molecule.n_electrons

            # Create ionic governance ansatz
            ansatz = HardwareEfficientAnsatz(
                n_qubits=n_qubits,
                n_electrons=n_electrons,
                n_layers=2
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

            result['energy'] = vqe_result['energy']
            result['method'] = 'VQE (Ionic Governance)'
            result['converged'] = vqe_result['converged']
            result['iterations'] = vqe_result['iterations']
            result['parameters'] = vqe_result['parameters']
            result['energy_history'] = vqe_result['energy_history']

            # Compute HF for comparison
            _, hf_energy = self.hamiltonian.solve_scf(max_iterations=50)
            result['hf_energy'] = hf_energy
            result['correlation_energy'] = result['energy'] - hf_energy

        elif method.upper() == 'EXACT':
            # Exact diagonalization (for small systems)
            H_matrix = self.hamiltonian.to_matrix()
            eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

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
                n_layers=2
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
            result['method'] = 'Hi-VQE (Ionic Governance)'
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
        Analyze ionic bond properties.

        Returns:
            Dictionary with:
                - bond_type: 'ionic'
                - donor: Donor atom symbol
                - acceptor: Acceptor atom symbol
                - electronegativity_difference: ΔEN
                - charge_transfer: Estimated charge transfer
                - ionic_character: Percentage ionic character
                - coulomb_energy: Electrostatic energy
        """
        # Basic properties
        delta_en = abs(
            self.donor.electronegativity - self.acceptor.electronegativity
        )

        # Estimate charge transfer from EN difference
        # Empirical: q ≈ 1 - exp(-0.25 * ΔEN^2)
        charge_transfer = 1.0 - np.exp(-0.25 * delta_en**2)

        # Ionic character (Pauling formula)
        ionic_character = 1.0 - np.exp(-0.25 * delta_en**2)

        # Estimate Coulombic energy (simple point charge model)
        # E_coulomb = -k * q^2 / r (in atomic units, k = 1)
        bond_length_bohr = self.get_bond_length() / 0.529177  # Angstrom to Bohr
        if bond_length_bohr > 0:
            coulomb_energy = -charge_transfer**2 / bond_length_bohr
        else:
            coulomb_energy = 0.0

        analysis = {
            'bond_type': 'ionic',
            'donor': self.donor.symbol,
            'acceptor': self.acceptor.symbol,
            'electronegativity_difference': delta_en,
            'charge_transfer': charge_transfer,
            'ionic_character': ionic_character,
            'covalent_character': 1.0 - ionic_character,  # Complementary
            'coulomb_energy': coulomb_energy,
            'bond_length': self.get_bond_length(),
            'entanglement_type': 'minimal (charge transfer only)',
            'governance_protocol': 'IonicGovernanceProtocol'
        }

        # Add Hamiltonian analysis if available
        if hasattr(self.hamiltonian, 'get_charge_distribution'):
            charges = self.hamiltonian.get_charge_distribution()
            analysis['atomic_charges'] = {
                self.atoms[0].symbol: charges[0],
                self.atoms[1].symbol: charges[1]
            }

        return analysis

    def get_charge_distribution(self) -> Dict[str, float]:
        """
        Get charge distribution on each atom.

        Returns:
            Dictionary mapping atom symbols to charges
        """
        if hasattr(self.hamiltonian, 'get_charge_distribution'):
            charges = self.hamiltonian.get_charge_distribution()
            # Use construction-order indexing (Molecule([atom_1, atom_2]) at
            # __init__) to match the Hamiltonian's own atom ordering, exactly
            # as analyze() does. donor/acceptor ordering follows
            # electronegativity and can be swapped relative to construction.
            return {
                self.atoms[0].symbol: charges[0],
                self.atoms[1].symbol: charges[1]
            }
        else:
            # Estimate from electronegativity
            analysis = self.analyze()
            q = analysis['charge_transfer']
            return {
                self.donor.symbol: +q,  # Loses electron (positive)
                self.acceptor.symbol: -q  # Gains electron (negative)
            }

    def __repr__(self) -> str:
        """String representation."""
        return (f"IonicBond({self.donor.symbol}+ — {self.acceptor.symbol}-, "
                f"{self.get_bond_length():.2f} Å)")
