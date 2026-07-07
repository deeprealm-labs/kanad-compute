"""
Base representation for quantum chemical systems.

Each bonding type (ionic, covalent, metallic) requires a different
quantum representation strategy.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np



class BaseRepresentation(ABC):
    """
    Abstract base class for quantum representations of bonding.

    Different bonding types require fundamentally different representations:
    - Ionic: Localized, minimal entanglement
    - Covalent: Hybrid orbitals, paired entanglement
    - Metallic: Delocalized k-space, collective entanglement
    """

    def __init__(self, molecule: 'Molecule'):
        """
        Initialize representation.

        Args:
            molecule: Molecule object with atoms and bonds
        """
        self.molecule = molecule
        self.n_qubits: Optional[int] = None
        self.n_electrons = sum(atom.n_electrons for atom in molecule.atoms)
        self.n_orbitals: Optional[int] = None

    @abstractmethod
    def build_hamiltonian(self) -> 'Hamiltonian':
        """
        Build the Hamiltonian in this representation.

        Returns:
            Hamiltonian object specific to this representation type
        """
        pass

    def get_reference_state(self) -> np.ndarray:
        """
        Get the reference state (e.g., Hartree-Fock state).

        Not abstract: the LCAO override was removed (dead code, zero callers, and
        it built determinants in an interleaved spin convention inconsistent with
        the blocked convention CovalentHamiltonian.to_matrix uses). Keeping it
        @abstractmethod made LCAORepresentation impossible to instantiate. The
        default fails loudly if ever actually called; subclasses may override.

        Returns:
            Reference state vector
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_reference_state is not implemented "
            "(the legacy LCAO version was removed as dead/buggy)."
        )

    @abstractmethod
    def compute_observables(self, state: np.ndarray) -> Dict[str, float]:
        """
        Compute physical observables from quantum state.

        Args:
            state: Quantum state vector

        Returns:
            Dictionary of observable names to values
        """
        pass

    @abstractmethod
    def to_qubit_operator(self) -> 'QubitOperator':
        """
        Map representation to qubit operators.

        Returns:
            QubitOperator representing the Hamiltonian
        """
        pass

    @abstractmethod
    def get_num_qubits(self) -> int:
        """
        Get number of qubits required for this representation.

        Returns:
            Number of qubits
        """
        pass

    def get_num_electrons(self) -> int:
        """Get total number of electrons."""
        return self.n_electrons

    def get_num_orbitals(self) -> int:
        """Get number of spatial orbitals."""
        if self.n_orbitals is None:
            raise ValueError("Number of orbitals not set")
        return self.n_orbitals


class BondMolecule:
    """
    Lightweight atom/bond/geometry holder used by the bonds/ dispatch path.

    NOT the PySCF-backed polyatomic Hamiltonian — that is
    ``kanad.core.molecule.Molecule``. This minimal class only carries atoms,
    bonds, spin and charge for BondFactory/ionic/covalent/metallic bonds.
    The old name ``Molecule`` is kept as a deprecated alias (see bottom of file)
    to end the two-``Molecule`` confusion flagged in the 2026-05-28 cleanup.
    """

    def __init__(self, atoms: List['Atom'], bonds: Optional[List] = None, spin: int = 0, charge: int = 0):
        """
        Initialize molecule.

        Args:
            atoms: List of Atom objects
            bonds: Optional list of Bond objects
            spin: Spin multiplicity (2S, where S is total spin)
            charge: Total molecular charge
        """
        self.atoms = atoms
        self.bonds = bonds or []
        self.n_atoms = len(atoms)
        self.spin = spin
        self.charge = charge
        self.multiplicity = spin + 1  # multiplicity = 2S + 1

    @property
    def n_electrons(self) -> int:
        """Total number of electrons (accounting for charge)."""
        return sum(atom.n_electrons for atom in self.atoms) - self.charge

    @property
    def symbols(self) -> List[str]:
        """List of atomic symbols."""
        return [atom.symbol for atom in self.atoms]

    @property
    def formula(self) -> str:
        """Hill-system chemical formula (e.g. ``'H2O'``, ``'HHe+'``).

        Matches the polyatomic ``core.molecule.Molecule.formula`` so analysis
        code (ThermochemistryCalculator, etc.) can consume both Molecule
        types uniformly.
        """
        from collections import Counter
        counts = Counter(atom.symbol for atom in self.atoms)

        def sort_key(item):
            symbol = item[0]
            if symbol == 'C':
                return (0, symbol)
            elif symbol == 'H':
                return (1, symbol)
            return (2, symbol)

        parts = []
        for symbol, count in sorted(counts.items(), key=sort_key):
            parts.append(symbol if count == 1 else f"{symbol}{count}")
        formula = ''.join(parts)
        if self.charge > 0:
            formula += '+' if self.charge == 1 else f'{self.charge}+'
        elif self.charge < 0:
            formula += '-' if self.charge == -1 else f'{abs(self.charge)}-'
        return formula

    @property
    def positions(self) -> np.ndarray:
        """Atomic positions as (n_atoms, 3) array."""
        return np.array([atom.position for atom in self.atoms])

    def distance_matrix(self) -> np.ndarray:
        """
        Compute distance matrix between all atoms.

        Returns:
            (n_atoms, n_atoms) distance matrix in Angstroms
        """
        n = self.n_atoms
        D = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                d = self.atoms[i].distance_to(self.atoms[j])
                D[i, j] = d
                D[j, i] = d

        return D

    def __repr__(self) -> str:
        """String representation."""
        formula = ''.join(self.symbols)
        return f"BondMolecule({formula}, n_atoms={self.n_atoms}, n_electrons={self.n_electrons})"


# Deprecated alias — the bonds/ path historically imported this as ``Molecule``.
# Kept importable so existing code keeps working; prefer ``BondMolecule``.
Molecule = BondMolecule
