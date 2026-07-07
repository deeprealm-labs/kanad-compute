"""`SystemSpec` — the geometry-agnostic specification + `materialize_at`.

A `SystemSpec` captures everything needed to build a quantum-chemistry system
*except* the geometry it is evaluated at: composition, charge/spin, basis,
active-space strategy, ansatz, solver, backend. `materialize_at(atoms_bohr)`
replays the canonical pipeline (``gto.M → RHF → active space →
ActiveHamiltonian``) at any geometry.

This single method is shared by `MolecularBuilder.build()` (reference geometry)
and `QuantumSystem.energy_fn()` (arbitrary displaced geometries for dynamics /
reactions), so the proven wiring lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

# CODATA; matches dynamics/quantum_forces.py to round-trip cleanly.
BOHR_TO_ANGSTROM = 0.52917721092
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

AtomTemplate = Tuple[str, Tuple[float, float, float]]


@dataclass
class SystemSpec:
    """Geometry-agnostic spec for a quantum-chemistry system.

    Every field has a sensible default; the builder overrides what it needs.
    The active-space and solver strategies accept ``'auto'`` (smart default) or
    an explicit choice — the "freedom at every stage" the design calls for.
    """

    atoms: Tuple[AtomTemplate, ...]              # reference geometry (Angstrom)
    charge: int = 0
    spin: int = 0                                # 2S (PySCF convention)
    basis: str = 'sto-3g'

    active_space_strategy: str = 'auto'          # auto|full|frozen_core|frontier|mp2no|manual
    active_space_kwargs: Dict[str, Any] = field(default_factory=dict)
    active_space_policy: str = 'freeze'          # freeze|reselect (continuity along scans)

    ansatz_type: str = 'auto'                    # auto|lucj|hardware_efficient|givens_sd|...
    ansatz_kwargs: Dict[str, Any] = field(default_factory=dict)
    mapper: str = 'jordan_wigner'

    solver: str = 'auto'                         # auto|ci|vqe|sqd
    solver_kwargs: Dict[str, Any] = field(default_factory=dict)
    backend: str = 'statevector'

    # M9 conditions: solvent (real PCM), pH (Henderson-Hasselbalch), and
    # thermal=True turns on real RRHO free-energy corrections (ZPE + H(T) − T·S
    # at the given T, P). 'vacuum' = gas phase; pressure in atm.
    conditions: Dict[str, Any] = field(default_factory=lambda: {
        'solvent': 'vacuum', 'pH': None, 'temperature': 298.15, 'pressure': 1.0,
        'thermal': False, 'ph_sites': None,
    })

    # Freeze-policy cache: the reference geometry's (frozen, active) indices.
    # Reused at every subsequent geometry so index-based active spaces stay
    # continuous across a scan. Not part of the user-facing spec.
    _frozen_selection: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = field(
        default=None, repr=False
    )

    # ----- geometry → mean field ----------------------------------------

    def _atom_string(self, coords_ang: np.ndarray) -> str:
        return '; '.join(
            f'{sym} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}'
            for (sym, _ref), c in zip(self.atoms, coords_ang)
        )

    def _build_mol(self, coords_ang: np.ndarray):
        from pyscf import gto
        return gto.M(
            atom=self._atom_string(coords_ang),
            basis=self.basis,
            charge=self.charge,
            spin=self.spin,
            verbose=0,
        )

    def _run_scf(self, mol):
        from pyscf import scf
        mf = scf.ROHF(mol) if self.spin != 0 else scf.RHF(mol)
        mf.verbose = 0
        mf.run()
        return mf

    def materialize_at(self, atoms_bohr: Optional[np.ndarray] = None):
        """Replay the pipeline at a geometry → ``(mf, ActiveHamiltonian)``.

        Args:
            atoms_bohr: ``(n_atoms, 3)`` coordinates in Bohr. ``None`` uses the
                spec's reference geometry (Angstrom). The Bohr convention
                matches the dynamics force engine's energy_fn contract.
        """
        if atoms_bohr is None:
            coords_ang = np.array([list(c) for (_s, c) in self.atoms], dtype=float)
            displaced = False
        else:
            coords_ang = np.asarray(atoms_bohr, dtype=float) * BOHR_TO_ANGSTROM
            displaced = True

        mol = self._build_mol(coords_ang)
        mf = self._run_scf(mol)
        active_space = self._resolve_active_space(mf, displaced)

        from kanad.core.active_space import build_active_space_hamiltonian
        ham = build_active_space_hamiltonian(mf, active_space)
        return mf, ham

    # ----- active-space resolution --------------------------------------

    def _resolve_active_space(self, mf, displaced: bool):
        from kanad.core.active_space import ActiveSpaceSelector
        from kanad.solvers.solver_router import SolverRouter

        selector = ActiveSpaceSelector(mf)
        n_orb_total = selector.n_orbitals_total

        # Freeze policy: reuse the reference geometry's index partition so the
        # active space is identical at every step of a scan (continuity).
        if self.active_space_policy == 'freeze' and self._frozen_selection is not None:
            frozen, active = self._frozen_selection
            return selector.manual(frozen=list(frozen), active=list(active))

        strat = self.active_space_strategy
        if strat == 'auto':
            # Don't reduce a system already small enough for exact CI; otherwise
            # let MP2 natural orbitals choose the correlated active space.
            strat = 'full' if (2 * n_orb_total) <= SolverRouter.CI_MAX_QUBITS else 'mp2no'

        if strat == 'mp2no':
            if displaced:
                raise NotImplementedError(
                    "active_space='mp2no' is not supported along a geometry scan "
                    "(dynamics/reactions): MP2 natural orbitals can reorder between "
                    "geometries, breaking energy continuity. Use a geometry-stable "
                    "active space for scans — active_space('manual', frozen=..., "
                    "active=...) or active_space('frontier', n_occ=..., n_virt=...)."
                )
            active_space = selector.mp2_natural_orbitals(**self.active_space_kwargs)
        elif strat == 'avas':
            if displaced:
                raise NotImplementedError(
                    "active_space='avas' is not supported along a geometry scan "
                    "(orbital selection is geometry-dependent → energy "
                    "discontinuity). Use 'manual' or 'frontier' for scans.")
            active_space = selector.avas(**self.active_space_kwargs)
        elif strat == 'frontier':
            # frontier(n_occ, n_virt) requires both; supply HOMO-2..LUMO+2 defaults so
            # active_space='frontier' with no kwargs (the API/UI path) doesn't crash.
            fkw = {'n_occ': 3, 'n_virt': 3, **self.active_space_kwargs}
            active_space = selector.frontier(**fkw)
        elif strat == 'frozen_core':
            active_space = selector.frozen_core()
        elif strat == 'manual':
            active_space = selector.manual(**self.active_space_kwargs)
        elif strat == 'full':
            active_space = selector.manual(frozen=[], active=list(range(n_orb_total)))
        else:
            raise ValueError(
                f"Unknown active_space strategy {strat!r}; expected one of "
                "'auto', 'full', 'frozen_core', 'frontier', 'mp2no', 'manual'."
            )

        # Cache index-based selections for freeze continuity. mp2no is NOT cached
        # (its NO basis is geometry-specific; scans raise above), so a frozen
        # mp2no reference never silently reuses a stale rotation.
        if self.active_space_policy == 'freeze' and active_space.method not in ('mp2no', 'avas'):
            self._frozen_selection = (active_space.frozen_indices, active_space.active_indices)
        return active_space
