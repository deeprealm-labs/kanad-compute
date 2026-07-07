"""
Ab-initio molecular Hamiltonian for ionic-character molecules.

`IonicHamiltonian` is a thin specialization of `CovalentHamiltonian` that:
- Uses the same PySCF integral pipeline (h_core, ERI, S, E_nuc).
- Carries `IonicGovernanceProtocol` metadata so downstream bond-classification
  UI in `kanad-app` can render ionic-specific info.

NOT a Hubbard tight-binding model. The phenomenological Hubbard caricature that
used to live here is preserved at `kanad.core.models.IonicHubbardModel` for
pedagogy and never appears on the `BondFactory` dispatch path.

History (M1, 2026-05-26)
------------------------
Pre-M1 `IonicHamiltonian` built a one-orbital-per-atom Hubbard model with
diagonal h_core = −electronegativity and exponentially decaying hopping. For
real ionic molecules the resulting energies were 90–105 Ha off PySCF FCI:
HF was off by +91.4 Ha, LiF by +104.7 Ha; HeH⁺ violated the variational bound
by −513 mHa (see `inspection/20-deep-inspection-r2.md §N1`).

The fix is to route through `CovalentHamiltonian.__init__` so the integrals
come from PySCF (validated bit-correct against PySCF FCI on H₂/LiH/H₂O to
≤1 µHa). The ionic-character metadata sits on top as an attached governance
protocol, not as a separate (and wrong) Hamiltonian.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
import logging

import numpy as np

from kanad.core.hamiltonians.covalent_hamiltonian import CovalentHamiltonian
from kanad.core.governance.protocols.ionic_protocol import IonicGovernanceProtocol

if TYPE_CHECKING:
    from kanad.core.representations.base_representation import (
        Molecule,
        BaseRepresentation,
    )

logger = logging.getLogger(__name__)


class IonicHamiltonian(CovalentHamiltonian):
    """Ab-initio molecular Hamiltonian for ionic-character molecules.

    Inherits PySCF integral construction, `to_matrix`, `solve_scf`,
    `to_sparse_hamiltonian`, MO computation, and active-space machinery from
    `CovalentHamiltonian`. Only the governance protocol is replaced.

    The class exists for dispatch purposes: `BondFactory` instantiates this for
    high-ΔEN pairs (alkali halides, hydrogen halides, etc.), and downstream code
    that wants ionic-specific analyses checks `isinstance(h, IonicHamiltonian)`.
    """

    def __init__(
        self,
        molecule: 'Molecule',
        representation: 'BaseRepresentation',
        use_governance: bool = True,
        basis_name: str = 'sto-3g',
        frozen_orbitals: Optional[List[int]] = None,
        active_orbitals: Optional[List[int]] = None,
    ):
        # Delegate to CovalentHamiltonian's PySCF integral path. We pass
        # `use_governance=False` so the base class doesn't apply Covalent
        # governance during _build_hamiltonian; we attach the Ionic protocol
        # below.
        super().__init__(
            molecule=molecule,
            representation=representation,
            basis_name=basis_name,
            use_governance=False,
            use_pyscf_integrals=True,
            frozen_orbitals=frozen_orbitals,
            active_orbitals=active_orbitals,
        )

        if use_governance:
            self.governance_protocol = IonicGovernanceProtocol()
            self.use_governance = True
            logger.info("✓ Ionic governance protocol attached")

    def to_ionic_hubbard_model(self):
        """Return an `IonicHubbardModel` parameterized for this molecule's geometry.

        Useful for pedagogy or for explicit Hubbard-physics studies. The result
        is NOT what the framework uses for ab-initio calculations.
        """
        from kanad.core.models import IonicHubbardModel
        return IonicHubbardModel(self.atoms, charge=getattr(self.molecule, 'charge', 0))

    def __repr__(self) -> str:
        symbols = '-'.join(a.symbol for a in self.atoms)
        return f"IonicHamiltonian({symbols}, basis={self.basis_name})"
