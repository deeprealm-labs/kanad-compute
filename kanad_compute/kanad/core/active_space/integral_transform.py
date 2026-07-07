"""Active-space integral transform + `ActiveHamiltonian`.

The canonical MO-basis transform that the deleted `core/active_space.py`
module got wrong (it applied the frozen-core formula to AO integrals instead
of MO integrals, producing LiH energies 173 mHa BELOW FCI — a variational
violation).

Math
----
Given:
- ``mf`` — a converged PySCF mean-field with MO coefficients ``C``.
- ``ActiveSpace`` partition into ``(frozen, active, virtual)``.

Compute, in MO basis:
::

    h_mo  = C^T  h_core_AO  C
    g_mo  = (C C C C) eri_AO     (full 4-index transform)

Then:
::

    E_inactive = E_nuc + Σ_{i ∈ frozen} 2 h_mo[i,i]
                       + Σ_{i,j ∈ frozen} [ 2 g_mo[i,i,j,j] − g_mo[i,j,j,i] ]

    h_eff[p,q] = h_mo[p,q] + Σ_{i ∈ frozen} [ 2 g_mo[p,q,i,i] − g_mo[p,i,i,q] ]
                 (for p,q ∈ active)

    g_eff = g_mo[active, active, active, active]   (4-way slice in MO basis)

The total energy obeys:

::

    E_total_FCI = E_inactive + E_FCI(h_eff, g_eff, n_active_electrons)

A unit test MUST verify this against PySCF's full FCI to ~1e-10 Ha. The two
regression tests in `tests/validation/test_active_space.py` do exactly that.

`ActiveHamiltonian` exposes the standard `MolecularHamiltonian` attributes
(``h_core``, ``eri``, ``n_orbitals``, ``n_electrons``, ``nuclear_repulsion``)
and methods (``to_matrix``, ``to_sparse_hamiltonian``) so downstream solvers
consume it without modification.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING
import logging

import numpy as np

from kanad.core.active_space.selector import ActiveSpace

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

logger = logging.getLogger(__name__)


def build_active_space_hamiltonian(mf, active_space: ActiveSpace) -> 'ActiveHamiltonian':
    """Build the active-space `(h_eff, g_eff, E_inactive)` and wrap as `ActiveHamiltonian`.

    Args:
        mf: PySCF mean-field (RHF) already converged. ``mf.mo_coeff`` provides
            the orbital basis the active-space indices refer to.
        active_space: `ActiveSpace` describing the frozen / active / virtual
            partition (must satisfy the invariants in `ActiveSpace.__post_init__`).

    Returns:
        `ActiveHamiltonian` — duck-types as `MolecularHamiltonian`.

    Guaranteed invariant (enforced by `tests/validation/test_active_space.py`):
        ``E_total_FCI = E_inactive + FCI(h_eff, g_eff, n_active_electrons)``
    """
    from pyscf import ao2mo

    C = active_space.mo_coeff
    if C is None:
        C = mf.mo_coeff

    mol = mf.mol
    h_ao = mf.get_hcore()
    eri_ao = mol.intor('int2e')

    # AO → MO transform via indigenous core.integrals (was inline ao2mo.kernel;
    # identical chemist-notation transform, now the single shared implementation). (reorg B3)
    from kanad.core.integrals.transforms import ao2mo_transform, one_index_transform
    h_mo = one_index_transform(h_ao, C)
    n_orb = C.shape[1]
    eri_mo = ao2mo_transform(eri_ao, C, chemist=True)

    frozen = active_space.frozen_indices
    active = active_space.active_indices
    n_active = active_space.n_active_orbitals
    n_active_electrons = active_space.n_active_electrons

    # E_inactive: nuclear repulsion + frozen-core mean-field energy. Added ONLY
    # at the end; never folded into h_eff.
    E_inactive = float(mol.energy_nuc())
    for i in frozen:
        E_inactive += 2.0 * h_mo[i, i]
    for i in frozen:
        for j in frozen:
            E_inactive += 2.0 * eri_mo[i, i, j, j] - eri_mo[i, j, j, i]

    # h_eff[p, q] = h_mo[p, q] + frozen-active mean field correction.
    h_eff = np.empty((n_active, n_active))
    for p_idx, p in enumerate(active):
        for q_idx, q in enumerate(active):
            h_eff[p_idx, q_idx] = h_mo[p, q]
            for i in frozen:
                h_eff[p_idx, q_idx] += 2.0 * eri_mo[p, q, i, i] - eri_mo[p, i, i, q]

    # g_eff: pure slice of MO-basis ERIs over the active block.
    act = np.array(active, dtype=int)
    g_eff = eri_mo[np.ix_(act, act, act, act)].copy()

    logger.debug(
        f"build_active_space_hamiltonian: frozen={frozen}, active={active}, "
        f"n_active_electrons={n_active_electrons}, E_inactive={E_inactive:.10f}"
    )

    return ActiveHamiltonian(
        h_core=h_eff,
        eri=g_eff,
        nuclear_repulsion=E_inactive,
        n_orbitals=n_active,
        n_electrons=n_active_electrons,
        active_space=active_space,
        mf=mf,
        h_mo_full=h_mo,
        eri_mo_full=eri_mo,
    )


class ActiveHamiltonian:
    """Active-space Hamiltonian duck-typed as `MolecularHamiltonian`.

    Exposes the same interface as `CovalentHamiltonian` so downstream solvers
    (`VQESolver`, `PhysicsVQE`, `SQDSolver`) consume it without modification.
    The energy reported by a solver on this Hamiltonian is the **active-space
    eigenvalue** that already includes ``nuclear_repulsion`` (which equals
    ``E_inactive`` for an active-space Hamiltonian) as an identity term, so
    the value is the total-system energy after adding `E_nuc` from the inner
    structure — there is no separate frozen-core addition step for the user.
    """

    def __init__(
        self,
        h_core: np.ndarray,
        eri: np.ndarray,
        nuclear_repulsion: float,
        n_orbitals: int,
        n_electrons: int,
        active_space: ActiveSpace,
        mf,
        h_mo_full: Optional[np.ndarray] = None,
        eri_mo_full: Optional[np.ndarray] = None,
    ):
        # `nuclear_repulsion` here is the active-space "E_inactive" (E_nuc +
        # frozen-core mean field). Naming it `nuclear_repulsion` keeps the
        # duck-typing contract with `MolecularHamiltonian` so solvers don't
        # need special handling.
        self.h_core = h_core
        self.eri = eri
        self.nuclear_repulsion = float(nuclear_repulsion)
        self.n_orbitals = int(n_orbitals)
        self.n_electrons = int(n_electrons)
        self.active_space = active_space
        self.mf = mf
        self.mol = mf.mol       # CovalentHamiltonian's PySCF attribute

        # Expose `atoms` as kanad Atom objects (symbol + position in Å) derived from
        # the PySCF Mole. Analysis consumers (NMRCalculator, RamanIRCalculator, ...)
        # reach for `hamiltonian.atoms` and iterate `atom.symbol`; ActiveHamiltonian
        # previously had no `.atoms`, so they raised "Hamiltonian has no atoms".
        from kanad.core.atom import Atom
        _BOHR_TO_ANG = 0.52917721092
        try:
            self.atoms = [
                Atom(mf.mol.atom_symbol(i),
                     np.asarray(mf.mol.atom_coord(i)) * _BOHR_TO_ANG)
                for i in range(mf.mol.natm)
            ]
        except Exception:                       # pragma: no cover — defensive
            self.atoms = []

        # `molecule` is consumed by analysis tools, the VQE symmetry penalty,
        # and various solver paths that expect the framework's `Molecule`-like
        # interface (n_electrons / atoms / spin) rather than PySCF's `Mole`
        # (nelectron / atom / spin). Build a minimal shim. (`atoms` here is now the
        # real Atom list, not the previous list of nuclear charges.)
        from types import SimpleNamespace
        self.molecule = SimpleNamespace(
            n_electrons=int(n_electrons),
            spin=int(getattr(mf.mol, 'spin', 0)),
            charge=int(getattr(mf.mol, 'charge', 0)),
            atoms=self.atoms,
            mol=mf.mol,
            mf=mf,
            n_orbitals=int(n_orbitals),
        )
        self._h_mo_full = h_mo_full
        self._eri_mo_full = eri_mo_full

        # Compatibility attributes some callers reach for.
        self.frozen_orbitals = list(active_space.frozen_indices)
        self.active_orbitals = list(active_space.active_indices)
        # No overlap matrix is meaningful in the active-MO basis (already orthonormal).
        self.S = np.eye(self.n_orbitals)

    # ---- solver-side API duck-typed against MolecularHamiltonian -------

    def to_sparse_hamiltonian(self, mapper: str = 'jordan_wigner'):
        """Build a Qiskit `SparsePauliOp` for the active-space Hamiltonian."""
        mapper_l = mapper.lower()
        if mapper_l in ('jordan_wigner', 'jw'):
            from kanad.core.operators.jordan_wigner import build_molecular_hamiltonian_jw
            return build_molecular_hamiltonian_jw(
                self.h_core, self.eri, self.nuclear_repulsion
            )
        elif mapper_l in ('bravyi_kitaev', 'bk'):
            from kanad.core.operators.bravyi_kitaev import build_molecular_hamiltonian_bk
            return build_molecular_hamiltonian_bk(
                self.h_core, self.eri, self.nuclear_repulsion
            )
        raise ValueError(f"Unknown mapper: {mapper!r}; supported: 'jordan_wigner', 'bravyi_kitaev'")

    def to_matrix(self, n_qubits: Optional[int] = None, use_mo_basis: bool = True):
        """Dense Hamiltonian matrix via the sparse Pauli path."""
        sparse = self.to_sparse_hamiltonian()
        return sparse.to_matrix()

    def compute_molecular_orbitals(self):
        """The active-space orbitals are already MOs — return identity coefficients."""
        return (
            np.zeros(self.n_orbitals),
            np.eye(self.n_orbitals),
        )

    # ---- M3 quantum-density storage -------------------------------------

    def set_quantum_density_matrix(self, rdm1_active_mo: np.ndarray) -> None:
        """Store a quantum 1-RDM produced by VQE on the active space.

        The input is the **active-MO** 1-RDM (`n_active × n_active`) from
        ``QuantumRDMExtractor.extract_1rdm(statevector)``. This method:

        1. Embeds it into the full-MO 1-RDM using the active-space partition
           (frozen orbitals doubly occupied; virtuals empty).
        2. Transforms to AO basis using ``self.mf.mo_coeff``.
        3. Validates that ``tr(D_AO · S) = n_electrons_total`` (the only
           physically meaningful trace check; the full-MO basis is
           orthonormal so its plain trace must also equal n_electrons).

        After this call, ``get_density_matrix(basis='ao'|'mo')`` returns the
        quantum-correlated density for the *full* system (not the active
        subspace) — that's what PropertyCalculator's dipole/polarizability/
        NMR routines consume.
        """
        from kanad.core.density.density_storage import (
            embed_active_to_full_mo, mo_to_ao_1rdm, validate_trace,
        )

        rdm_active = np.asarray(rdm1_active_mo, dtype=float)
        if rdm_active.shape != (self.n_orbitals, self.n_orbitals):
            raise ValueError(
                f"Active 1-RDM shape {rdm_active.shape} != "
                f"(n_active, n_active) = ({self.n_orbitals}, {self.n_orbitals})"
            )

        # Use the orbital basis the active/frozen INDICES actually refer to.
        # For 'mp2no'/'avas' the ActiveSpace carries ROTATED coefficients
        # (natural / AVAS orbitals, = mf.mo_coeff @ rot), and frozen/active_indices
        # index into THOSE rotated columns. Using canonical mf.mo_coeff here both
        # misaligns the index positions and applies the wrong AO rotation, giving a
        # physically wrong AO density — silently, because the rotated basis is also
        # S-orthonormal so the trace check still passes. For 'manual'/'frontier'/
        # 'frozen_core' active_space.mo_coeff == mf.mo_coeff, so this is a no-op.
        # (CORE_BUGS B4 / B6.)
        C = np.asarray(self.active_space.mo_coeff)
        n_mo_full = int(C.shape[1])
        rdm_full_mo = embed_active_to_full_mo(
            rdm_active,
            frozen_indices=list(self.active_space.frozen_indices),
            active_indices=list(self.active_space.active_indices),
            n_mo_full=n_mo_full,
        )

        n_electrons_total = int(self.mf.mol.nelectron)
        validate_trace(
            rdm_full_mo, expected_trace=n_electrons_total,
            label='Quantum 1-RDM (full-MO after active-space embedding)',
            tol=1e-4,
        )

        rdm_ao = mo_to_ao_1rdm(rdm_full_mo, C)
        validate_trace(
            rdm_ao, expected_trace=n_electrons_total,
            label='Quantum 1-RDM (AO)', tol=1e-4,
            overlap=self.mf.mol.intor('int1e_ovlp'),
        )

        self._quantum_density_matrix_mo = rdm_full_mo
        self._quantum_density_matrix_ao = rdm_ao
        self._quantum_density_matrix_active_mo = rdm_active
        # CovalentHamiltonian-compatible attribute consumed by older code paths.
        self._quantum_density_matrix = rdm_ao

    def get_density_matrix(self, basis: str = 'ao') -> np.ndarray:
        """Return the density matrix, preferring quantum (VQE) over HF.

        For an `ActiveHamiltonian`, the returned density spans the *full*
        molecular orbital set (not just the active block) — frozen orbitals
        contribute `2·I` and virtuals contribute zero per the closed-shell
        embedding convention.

        Args:
            basis: ``'ao'`` (default; consumed by PySCF property routines) or
                ``'mo'`` (canonical full-MO basis).
        """
        basis_l = basis.lower()
        if basis_l not in ('ao', 'mo'):
            raise ValueError(f"basis must be 'ao' or 'mo', got {basis!r}")

        if hasattr(self, '_quantum_density_matrix_ao') and self._quantum_density_matrix_ao is not None:
            return (self._quantum_density_matrix_ao if basis_l == 'ao'
                    else self._quantum_density_matrix_mo)

        # HF fallback (uses the full PySCF mean-field, NOT the active block).
        rdm_ao = self.mf.make_rdm1()
        if basis_l == 'ao':
            return rdm_ao
        S = self.mf.mol.intor('int1e_ovlp')
        C = self.mf.mo_coeff
        return C.T @ S @ rdm_ao @ S @ C

    def __repr__(self) -> str:
        return (
            f"ActiveHamiltonian(n_orbitals={self.n_orbitals}, "
            f"n_electrons={self.n_electrons}, E_inactive={self.nuclear_repulsion:.6f}, "
            f"method={self.active_space.method!r})"
        )
