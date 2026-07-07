"""
Base class for molecular Hamiltonians.

Provides common functionality for all bonding-type-specific Hamiltonians.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np


class MolecularHamiltonian(ABC):
    """
    Base class for molecular Hamiltonians.

    General molecular electronic Hamiltonian:
        H = H_core + H_ee + H_nn

    where:
        H_core = T + V_ne (kinetic + nuclear-electron attraction)
        H_ee = electron-electron repulsion
        H_nn = nuclear-nuclear repulsion (constant energy shift)

    Each bonding type (ionic, covalent, metallic) emphasizes different terms.
    """

    def __init__(
        self,
        n_orbitals: int,
        n_electrons: int,
        nuclear_repulsion: float = 0.0,
        frozen_orbitals: Optional[List[int]] = None,
        active_orbitals: Optional[List[int]] = None
    ):
        """
        Initialize molecular Hamiltonian.

        Args:
            n_orbitals: Number of spatial orbitals
            n_electrons: Number of electrons
            nuclear_repulsion: Nuclear repulsion energy (constant)
            frozen_orbitals: List of orbital indices to freeze (Hi-VQE active space)
            active_orbitals: List of orbital indices in active space (Hi-VQE)
        """
        self.n_orbitals = n_orbitals
        self.n_electrons = n_electrons
        self.nuclear_repulsion = nuclear_repulsion

        # Hi-VQE active space support
        self.frozen_orbitals = frozen_orbitals if frozen_orbitals is not None else []
        self.active_orbitals = active_orbitals

        # Integral matrices (filled by subclasses)
        self.h_core: Optional[np.ndarray] = None  # (n_orbitals, n_orbitals)
        self.eri: Optional[np.ndarray] = None     # (n_orbitals, n_orbitals, n_orbitals, n_orbitals)

    @abstractmethod
    def to_matrix(self) -> np.ndarray:
        """
        Convert Hamiltonian to matrix representation.

        Returns:
            Hamiltonian matrix in the chosen basis
        """
        pass

    @abstractmethod
    def compute_energy(self, density_matrix: np.ndarray) -> float:
        """
        Compute total electronic energy from density matrix.

        E = Tr[P * H_core] + ½ Tr[P * G] + E_nn

        where G = 2J - K (Coulomb - Exchange)

        Args:
            density_matrix: One-particle density matrix

        Returns:
            Total electronic energy
        """
        pass

    def get_one_body_tensor(self) -> np.ndarray:
        """
        Get one-body (core) Hamiltonian tensor.

        Returns:
            h_core matrix
        """
        if self.h_core is None:
            raise ValueError("Core Hamiltonian not initialized")
        return self.h_core.copy()

    def get_two_body_tensor(self) -> np.ndarray:
        """
        Get two-body (electron repulsion) tensor.

        Returns:
            ERI tensor (n, n, n, n)
        """
        if self.eri is None:
            raise ValueError("ERI tensor not initialized")
        return self.eri.copy()

    def get_nuclear_repulsion(self) -> float:
        """Get nuclear repulsion energy."""
        return self.nuclear_repulsion

    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"n_orbitals={self.n_orbitals}, "
                f"n_electrons={self.n_electrons}, "
                f"E_nn={self.nuclear_repulsion:.6f})")
