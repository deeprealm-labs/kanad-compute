"""Ionic Hubbard tight-binding model — pedagogical, NOT ab-initio.

Phenomenological one-orbital-per-atom Hamiltonian:

    H = Σ_i ε_i n_i + Σ_<ij> t_ij (a†_i a_j + h.c.) + Σ_i U_i n_i↑ n_i↓ + Σ_<ij> V_ij n_i n_j

where:
- ε_i = -χ_i  (Pauling electronegativity used as on-site energy proxy)
- t_ij = t_0 · exp(-r_ij / λ)  (exponentially decaying hopping)
- U_i  (Hubbard on-site repulsion, from atom-specific empirical values)
- V_ij = 1 / r_ij  (Coulomb inter-site repulsion in atomic units)

History
-------
This was the body of the old `IonicHamiltonian` (pre-M1). For real ionic
molecules it produced energies 90–105 Ha off PySCF FCI (per
`inspection/20-deep-inspection-r2.md §N1`) because real ionic chemistry
needs the kinetic + nuclear-attraction integrals, not site-energies derived
from electronegativity. In M1, `IonicHamiltonian` was rewritten to use PySCF
integrals and the Hubbard code was moved here under a clearly pedagogical name.

This model is appropriate for:
- Teaching what an ionic-Hubbard Hamiltonian looks like.
- Studying Hubbard physics (Mott transition, charge order) in toy systems.
- Building intuition for the framework's bond-classification logic.

It is NOT appropriate for:
- Computing energies of real ionic molecules (HF, LiF, NaCl, ...). Use
  `kanad.core.hamiltonians.IonicHamiltonian` (PySCF-backed) instead.

The `BondFactory` does not route to this model.
"""

from __future__ import annotations

from typing import List, Optional, Sequence
import logging

import numpy as np

from kanad.core.atom import Atom
from kanad.core.constants.conversion_factors import ConversionFactors

logger = logging.getLogger(__name__)


# Empirical Hubbard U values (eV) for selected atoms.
# Source: atomic ionization potentials minus electron affinities, rounded.
_HUBBARD_U_EV = {
    'H': 13.0,
    'Li': 5.0,
    'Na': 5.0,
    'K': 4.0,
    'F': 17.0,
    'Cl': 13.0,
    'Br': 11.0,
    'O': 15.0,
    'S': 12.0,
}


class IonicHubbardModel:
    """Phenomenological Hubbard tight-binding model for ionic systems.

    One spatial orbital per atom; one-electron h_core built from
    electronegativity + exponentially-decaying hopping; on-site U + inter-site
    Coulomb V. Useful for pedagogy, not for chemistry.

    Attributes:
        atoms: list of `Atom` objects ordered consistently with the integral arrays.
        n_orbitals: number of orbitals = `len(atoms)`.
        n_electrons: total electron count (taken from atomic_number sum minus
            optional charge).
        t0: hopping prefactor (Hartree).
        lambda_decay: hopping decay length (Bohr).
        h_core: ``(n, n)`` one-body matrix.
        eri: ``(n, n, n, n)`` two-body matrix (chemist notation).
        nuclear_repulsion: nuclear repulsion energy (Hartree).
    """

    def __init__(
        self,
        atoms: Sequence[Atom],
        charge: int = 0,
        t0: float = 0.05,
        lambda_decay: float = 1.5,
        hubbard_u: Optional[dict] = None,
    ):
        self.atoms = list(atoms)
        self.n_orbitals = len(self.atoms)
        # A one-orbital-per-atom Hubbard model holds at most 2 electrons per site, so it
        # is a VALENCE-electron model — using the full atomic number over-filled it by
        # ~(Z/valence) (e.g. 22 e⁻ for Na₂ in 4 spin-orbitals). (CORE_BUGS B21.)
        self.n_electrons = sum(a.n_valence for a in self.atoms) - charge
        self.charge = charge
        self.t0 = float(t0)
        self.lambda_decay = float(lambda_decay)
        self._hubbard_u_ev = {**_HUBBARD_U_EV, **(hubbard_u or {})}

        self.nuclear_repulsion = self._compute_nuclear_repulsion()
        self.h_core = self._build_h_core()
        self.eri = self._build_eri()

    # ----- construction helpers ----------------------------------------

    def _compute_nuclear_repulsion(self) -> float:
        # Indigenous single implementation (reorg B3).
        from kanad.core.integrals import nuclear_repulsion
        return nuclear_repulsion(self.atoms)

    def _build_h_core(self) -> np.ndarray:
        n = self.n_orbitals
        h = np.zeros((n, n))
        for i in range(n):
            # On-site energy proxy ε_i = -χ_i: raw (dimensionless) Pauling
            # electronegativity used directly as a tight-binding site energy. This is
            # an arbitrary-scale phenomenological proxy (NOT in eV, NOT converted to
            # Hartree); it is intentionally on a different scale from U/V. See module
            # docstring -- this model is pedagogical, not ab-initio.
            h[i, i] = -self.atoms[i].properties.electronegativity
        for i in range(n):
            for j in range(i + 1, n):
                t_ij = self._transfer_integral(i, j)
                h[i, j] = t_ij
                h[j, i] = t_ij
        return h

    def _transfer_integral(self, i: int, j: int) -> float:
        r_ij_angstrom = self.atoms[i].distance_to(self.atoms[j])
        # Decay length in Bohr; convert distance to Bohr too.
        r_bohr = r_ij_angstrom * ConversionFactors.ANGSTROM_TO_BOHR
        return self.t0 * np.exp(-r_bohr / self.lambda_decay)

    def _hubbard_u(self, atom: Atom) -> float:
        U_ev = self._hubbard_u_ev.get(atom.symbol, 10.0)
        return U_ev / 27.211  # eV → Hartree

    def _inter_site_coulomb(self, i: int, j: int) -> float:
        r_bohr = self.atoms[i].distance_to(self.atoms[j]) * ConversionFactors.ANGSTROM_TO_BOHR
        if r_bohr < 1e-10:
            return 0.0
        return 1.0 / r_bohr

    def _build_eri(self) -> np.ndarray:
        n = self.n_orbitals
        eri = np.zeros((n, n, n, n))
        for i in range(n):
            eri[i, i, i, i] = self._hubbard_u(self.atoms[i])
            for j in range(i + 1, n):
                v_ij = self._inter_site_coulomb(i, j)
                eri[i, i, j, j] = v_ij
                eri[j, j, i, i] = v_ij
        return eri

    # ----- diagnostics -------------------------------------------------

    @classmethod
    def from_atoms(
        cls,
        atoms: Sequence[Atom],
        charge: int = 0,
        t0: float = 0.05,
        lambda_decay: float = 1.5,
        U: Optional[dict] = None,
    ) -> 'IonicHubbardModel':
        """Construct an ionic-Hubbard model from a list of atoms."""
        return cls(atoms, charge=charge, t0=t0, lambda_decay=lambda_decay, hubbard_u=U)

    def __repr__(self) -> str:
        symbols = '-'.join(a.symbol for a in self.atoms)
        return (
            f"IonicHubbardModel({symbols}, "
            f"n_electrons={self.n_electrons}, t0={self.t0}, λ={self.lambda_decay})"
        )
