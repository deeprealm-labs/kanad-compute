"""
One-electron integrals for molecular Hamiltonians.

Includes:
- Kinetic energy: T_ij = ⟨φ_i|-½∇²|φ_j⟩
- Nuclear attraction: V_ij = ⟨φ_i|Σ_A -Z_A/r_iA|φ_j⟩
- Core Hamiltonian: H_core = T + V
"""

import numpy as np
from typing import List
from kanad.core.integrals.basis_sets import GaussianPrimitive, ContractedGaussian
from kanad.core.integrals.overlap import OverlapIntegrals
from kanad.core.atom import Atom
from scipy.special import erf


class OneElectronIntegrals:
    """
    Compute one-electron integrals for molecular systems.
    """

    def __init__(self, atoms: List[Atom], basis_functions: List[ContractedGaussian]):
        """
        Initialize one-electron integral calculator.

        Args:
            atoms: List of Atom objects
            basis_functions: List of basis functions
        """
        self.atoms = atoms
        self.basis_functions = basis_functions
        self.n_basis = len(basis_functions)

    def compute_kinetic(self) -> np.ndarray:
        """
        Compute kinetic energy matrix T_ij = ⟨φ_i|-½∇²|φ_j⟩

        For Gaussians, this can be computed analytically using:
        ∇² exp(-αr²) = (2α)² exp(-αr²) - 6α exp(-αr²)

        Returns:
            Kinetic energy matrix (n_basis, n_basis)
        """
        T = np.zeros((self.n_basis, self.n_basis))

        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                T[i, j] = self._kinetic_contracted(
                    self.basis_functions[i],
                    self.basis_functions[j]
                )
                T[j, i] = T[i, j]  # Symmetric

        return T

    def _kinetic_contracted(
        self,
        cgf_a: ContractedGaussian,
        cgf_b: ContractedGaussian
    ) -> float:
        """
        Compute kinetic integral between contracted Gaussians.

        T = Σᵢⱼ cᵢ cⱼ Nᵢ Nⱼ ⟨gᵢ|-½∇²|gⱼ⟩
        """
        kinetic = 0.0

        for prim_a in cgf_a.primitives:
            for prim_b in cgf_b.primitives:
                kinetic += (prim_a.coefficient *
                           prim_b.coefficient *
                           prim_a._normalization_constant() *
                           prim_b._normalization_constant() *
                           self._kinetic_primitive(prim_a, prim_b))

        return kinetic

    def _kinetic_primitive(
        self,
        prim_a: GaussianPrimitive,
        prim_b: GaussianPrimitive
    ) -> float:
        """
        Compute kinetic integral between two primitives.

        Uses the relation:
        T = -1/2 ⟨φ_a|∇²|φ_b⟩
          = α(2l_a+1)S - 2α²⟨φ_a|r²|φ_b⟩ (for each direction)

        For Cartesian Gaussians, can be computed using:
        T_ij = β(2l_b+1)S_ij - 2β²S_{i,j+1} - 1/2 l_b(l_b-1)S_{i,j-1}
        where j+1 and j-1 refer to angular momentum changes.

        Reference: Molecular Electronic-Structure Theory by Helgaker, Jorgensen, Olsen
        """
        α = prim_a.exponent
        β = prim_b.exponent
        la, ma, na = prim_a.angular_momentum
        lb, mb, nb = prim_b.angular_momentum
        A = prim_a.center
        B = prim_b.center

        # Compute kinetic energy using recursion relations
        # T = T_x × S_y × S_z + S_x × T_y × S_z + S_x × S_y × T_z

        # For each Cartesian direction, kinetic integral:
        # T_{l_a,l_b} = β(2l_b+1)S_{l_a,l_b} - 2β²S_{l_a,l_b+2} - 1/2 l_b(l_b-1)S_{l_a,l_b-2}

        γ = α + β
        P = (α * A + β * B) / γ
        AB = A - B
        K = np.exp(-α * β * np.dot(AB, AB) / γ)

        # Compute 1D kinetic integrals for each direction
        T_x = self._kinetic_1d(la, lb, A[0], B[0], P[0], α, β, γ)
        T_y = self._kinetic_1d(ma, mb, A[1], B[1], P[1], α, β, γ)
        T_z = self._kinetic_1d(na, nb, A[2], B[2], P[2], α, β, γ)

        # Compute 1D overlap integrals for cross terms
        S_x = OverlapIntegrals._overlap_1d(la, lb, P[0]-A[0], P[0]-B[0], γ)
        S_y = OverlapIntegrals._overlap_1d(ma, mb, P[1]-A[1], P[1]-B[1], γ)
        S_z = OverlapIntegrals._overlap_1d(na, nb, P[2]-A[2], P[2]-B[2], γ)

        # Total kinetic energy: sum of contributions from each direction
        T_total = K * (T_x*S_y*S_z + S_x*T_y*S_z + S_x*S_y*T_z)

        return T_total

    def _kinetic_1d(self, la: int, lb: int, Ax: float, Bx: float, Px: float,
                    α: float, β: float, γ: float) -> float:
        """
        Compute 1D kinetic energy integral.

        T = β(2l_b+1)S - 2β²S(l_b+2) - 1/2·l_b(l_b-1)S(l_b-2)

        where S(l) is the overlap integral with angular momentum l.
        """
        XPA = Px - Ax
        XPB = Px - Bx

        # Main term: β(2l_b+1)S_{la,lb}
        S_main = OverlapIntegrals._overlap_1d(la, lb, XPA, XPB, γ)
        term1 = β * (2*lb + 1) * S_main

        # Second term: -2β² S_{la, lb+2}
        S_plus2 = OverlapIntegrals._overlap_1d(la, lb+2, XPA, XPB, γ)
        term2 = -2 * β**2 * S_plus2

        # Third term: -1/2 l_b(l_b-1) S_{la, lb-2}
        if lb >= 2:
            S_minus2 = OverlapIntegrals._overlap_1d(la, lb-2, XPA, XPB, γ)
            term3 = -0.5 * lb * (lb - 1) * S_minus2
        else:
            term3 = 0.0

        return term1 + term2 + term3

    def compute_nuclear_attraction(self) -> np.ndarray:
        """
        Compute nuclear-electron attraction matrix.

        V_ij = ⟨φ_i|Σ_A -Z_A/|r-R_A||φ_j⟩

        Returns:
            Nuclear attraction matrix (n_basis, n_basis)
        """
        V = np.zeros((self.n_basis, self.n_basis))

        for i in range(self.n_basis):
            for j in range(i, self.n_basis):
                V[i, j] = self._nuclear_contracted(
                    self.basis_functions[i],
                    self.basis_functions[j]
                )
                V[j, i] = V[i, j]  # Symmetric

        return V

    def _nuclear_contracted(
        self,
        cgf_a: ContractedGaussian,
        cgf_b: ContractedGaussian
    ) -> float:
        """
        Compute nuclear attraction between contracted Gaussians.

        V = Σᵢⱼ cᵢ cⱼ Nᵢ Nⱼ Σ_A ⟨gᵢ|-Z_A/r_A|gⱼ⟩
        """
        nuclear = 0.0

        for prim_a in cgf_a.primitives:
            for prim_b in cgf_b.primitives:
                for atom in self.atoms:
                    nuclear += (prim_a.coefficient *
                               prim_b.coefficient *
                               prim_a._normalization_constant() *
                               prim_b._normalization_constant() *
                               self._nuclear_primitive(prim_a, prim_b, atom))

        return nuclear

    def _nuclear_primitive(
        self,
        prim_a: GaussianPrimitive,
        prim_b: GaussianPrimitive,
        atom: Atom
    ) -> float:
        """
        Compute nuclear attraction integral for one nucleus.

        V_A = -Z_A ⟨φ_a|1/|r-R_A||φ_b⟩

        Uses Boys function for analytical evaluation.

        IMPORTANT: Atom position must be converted to Bohr to match
        basis function centers (which are already in Bohr).
        """
        from kanad.core.constants.conversion_factors import ConversionFactors

        α = prim_a.exponent
        β = prim_b.exponent
        A = prim_a.center  # Already in Bohr (from basis set building)
        B = prim_b.center  # Already in Bohr
        # Convert atom position from Angstrom to Bohr
        C = atom.position * ConversionFactors.ANGSTROM_TO_BOHR
        Z = atom.atomic_number

        # Gaussian product
        γ = α + β
        P = (α * A + β * B) / γ
        AB = A - B
        PC = P - C

        # Prefactor
        K = np.exp(-α * β * np.dot(AB, AB) / γ)
        prefactor = -Z * 2 * np.pi / γ * K

        # Angular momentum
        la, ma, na = prim_a.angular_momentum
        lb, mb, nb = prim_b.angular_momentum

        # For higher angular momentum, need to use recursion
        # The full formula involves Hermite Gaussians or McMurchie-Davidson
        # For now, implement for s and p orbitals explicitly

        total_L = la + ma + na + lb + mb + nb

        if total_L == 0:
            # Both s-orbitals
            T = γ * np.dot(PC, PC)
            F0 = self._boys_function(0, T)
            return prefactor * F0

        elif total_L == 1:
            # One p-orbital, one s-orbital
            T = γ * np.dot(PC, PC)
            F0 = self._boys_function(0, T)
            F1 = self._boys_function(1, T)

            # Determine which orbital has angular momentum
            # Use McMurchie-Davidson style expansion
            XPA = P[0] - A[0]
            YPA = P[1] - A[1]
            ZPA = P[2] - A[2]
            XPC = P[0] - C[0]
            YPC = P[1] - C[1]
            ZPC = P[2] - C[2]

            result = 0.0

            # x direction
            if la == 1:
                result += XPA * F0 - XPC * F1
            elif lb == 1:
                XPB = P[0] - B[0]
                result += XPB * F0 - XPC * F1

            # y direction
            if ma == 1:
                result += YPA * F0 - YPC * F1
            elif mb == 1:
                YPB = P[1] - B[1]
                result += YPB * F0 - YPC * F1

            # z direction
            if na == 1:
                result += ZPA * F0 - ZPC * F1
            elif nb == 1:
                ZPB = P[2] - B[2]
                result += ZPB * F0 - ZPC * F1

            return prefactor * result

        else:
            # Higher angular momentum (p-p, d-orbitals, etc.).
            # Use the exact McMurchie-Davidson Hermite-expansion recursion.
            #
            # V = prefactor * Σ_{t,u,v} E^x_t E^y_u E^z_v R_{t,u,v}(γ, P-C)
            #
            # where E^q_t are the Hermite expansion coefficients of the
            # Gaussian product in direction q (combining angular momenta of
            # a and b), and R_{t,u,v} are the Hermite Coulomb integrals built
            # from Boys functions. This naturally yields px-px != pz-pz and
            # px-py == 0 (no spurious off-diagonal p-p couplings).
            PA = P - A
            PB = P - B

            # Hermite expansion coefficients for each Cartesian direction.
            Ex = [self._hermite_E(la, lb, t, PA[0], PB[0], γ)
                  for t in range(la + lb + 1)]
            Ey = [self._hermite_E(ma, mb, u, PA[1], PB[1], γ)
                  for u in range(ma + mb + 1)]
            Ez = [self._hermite_E(na, nb, v, PA[2], PB[2], γ)
                  for v in range(na + nb + 1)]

            result = 0.0
            for t in range(la + lb + 1):
                for u in range(ma + mb + 1):
                    for v in range(na + nb + 1):
                        result += (Ex[t] * Ey[u] * Ez[v] *
                                   self._hermite_coulomb(t, u, v, 0, γ, PC))

            return prefactor * result

    @staticmethod
    def _hermite_E(i: int, j: int, t: int, PA: float, PB: float,
                   γ: float) -> float:
        """
        McMurchie-Davidson Hermite expansion coefficient E^{ij}_t.

        Expands the product of two 1D Cartesian Gaussians (angular momenta i, j)
        into Hermite Gaussians of order t about the product center.

        Recursion (Helgaker, Jorgensen, Olsen, "Molecular Electronic-Structure
        Theory", eq. 9.5.6):
            E^{i+1,j}_t = (1/2p) E^{ij}_{t-1} + PA·E^{ij}_t + (t+1) E^{ij}_{t+1}
            E^{i,j+1}_t = (1/2p) E^{ij}_{t-1} + PB·E^{ij}_t + (t+1) E^{ij}_{t+1}
        with base case E^{00}_0 = 1 and E^{ij}_t = 0 for t < 0 or t > i+j.
        """
        p = γ
        if t < 0 or t > i + j:
            return 0.0
        if i == 0 and j == 0:
            return 1.0 if t == 0 else 0.0
        if i > 0:
            return (
                (1.0 / (2.0 * p)) * OneElectronIntegrals._hermite_E(i - 1, j, t - 1, PA, PB, γ)
                + PA * OneElectronIntegrals._hermite_E(i - 1, j, t, PA, PB, γ)
                + (t + 1) * OneElectronIntegrals._hermite_E(i - 1, j, t + 1, PA, PB, γ)
            )
        else:
            return (
                (1.0 / (2.0 * p)) * OneElectronIntegrals._hermite_E(i, j - 1, t - 1, PA, PB, γ)
                + PB * OneElectronIntegrals._hermite_E(i, j - 1, t, PA, PB, γ)
                + (t + 1) * OneElectronIntegrals._hermite_E(i, j - 1, t + 1, PA, PB, γ)
            )

    @staticmethod
    def _hermite_coulomb(t: int, u: int, v: int, n: int, γ: float,
                         PC: np.ndarray) -> float:
        """
        Hermite Coulomb integral R^n_{t,u,v} for the McMurchie-Davidson scheme.

        Built from the auxiliary Boys functions F_n(γ·|PC|²) via the downward
        recursion (Helgaker, Jorgensen, Olsen, eqs. 9.9.18-9.9.20):
            R^n_{0,0,0} = (-2γ)^n F_n(T)
            R^n_{t+1,u,v} = t·R^{n+1}_{t-1,u,v} + PCx·R^{n+1}_{t,u,v}
            R^n_{t,u+1,v} = u·R^{n+1}_{t,u-1,v} + PCy·R^{n+1}_{t,u,v}
            R^n_{t,u,v+1} = v·R^{n+1}_{t,u,v-1} + PCz·R^{n+1}_{t,u,v}
        """
        if t < 0 or u < 0 or v < 0:
            return 0.0

        if t == 0 and u == 0 and v == 0:
            T = γ * np.dot(PC, PC)
            return ((-2.0 * γ) ** n) * OneElectronIntegrals._boys_function(n, T)

        if t > 0:
            return (
                (t - 1) * OneElectronIntegrals._hermite_coulomb(t - 2, u, v, n + 1, γ, PC)
                + PC[0] * OneElectronIntegrals._hermite_coulomb(t - 1, u, v, n + 1, γ, PC)
            )
        elif u > 0:
            return (
                (u - 1) * OneElectronIntegrals._hermite_coulomb(t, u - 2, v, n + 1, γ, PC)
                + PC[1] * OneElectronIntegrals._hermite_coulomb(t, u - 1, v, n + 1, γ, PC)
            )
        else:
            return (
                (v - 1) * OneElectronIntegrals._hermite_coulomb(t, u, v - 2, n + 1, γ, PC)
                + PC[2] * OneElectronIntegrals._hermite_coulomb(t, u, v - 1, n + 1, γ, PC)
            )

    @staticmethod
    def _boys_function(n: int, T: float) -> float:
        """
        Boys function F_n(T) used in nuclear attraction integrals.

        F_n(T) = ∫₀¹ t^(2n) exp(-Tt²) dt

        For n=0: F_0(T) = √(π/4T) erf(√T) for T > 0
                         = 1 for T = 0
        """
        if T < 1e-10:
            return 1.0 / (2 * n + 1)

        if n == 0:
            return 0.5 * np.sqrt(np.pi / T) * erf(np.sqrt(T))

        # Recursion for higher n
        # F_n(T) = [(2n-1)F_{n-1}(T) - exp(-T)] / (2T)
        F_prev = OneElectronIntegrals._boys_function(0, T)
        for i in range(1, n + 1):
            F_curr = ((2 * i - 1) * F_prev - np.exp(-T)) / (2 * T)
            F_prev = F_curr

        return F_prev

    def compute_core_hamiltonian(self) -> np.ndarray:
        """
        Compute core Hamiltonian H_core = T + V_ne

        Returns:
            Core Hamiltonian matrix (n_basis, n_basis)
        """
        T = self.compute_kinetic()
        V = self.compute_nuclear_attraction()
        return T + V

    def compute_nuclear_repulsion(self) -> float:
        """Nuclear-nuclear repulsion energy. Delegates to the module-level
        :func:`nuclear_repulsion` so every caller shares one implementation."""
        return nuclear_repulsion(self.atoms)


def nuclear_repulsion(atoms: List['Atom']) -> float:
    """Nuclear-nuclear repulsion energy ``E_nn = sum_{A<B} Z_A Z_B / R_AB`` (Bohr).

    Module-level free function (reorg Phase B3) so callers that only hold an atom
    list — not a constructed ``OneElectronIntegrals`` — share one implementation,
    replacing the several hand-rolled E_nn loops across the Hamiltonians. Part of
    the native floor: no PySCF dependency. Body lifted verbatim from the original
    ``OneElectronIntegrals.compute_nuclear_repulsion``.
    """
    from kanad.core.constants.conversion_factors import ConversionFactors
    E_nn = 0.0
    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            if i < j:
                R_ij = atom_i.distance_to(atom_j)
                if R_ij > 1e-10:  # Avoid division by zero
                    R_ij_bohr = ConversionFactors.length_to_bohr(R_ij, 'angstrom')
                    E_nn += atom_i.atomic_number * atom_j.atomic_number / R_ij_bohr
    return E_nn
