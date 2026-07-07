"""Periodic solver — the materials-domain reference (Stage 2).

Wraps a :class:`PeriodicHamiltonian` (PySCF PBC KRHF/KROHF) behind the
``MaterialsProvider`` capability so the Materials lab consumes ``band_structure()``
and ``density_of_states()`` by capability rather than importing the Hamiltonian.

Periodic systems have no single molecular ground state in the ``SolverResult`` sense;
``solve()`` returns the KRHF total energy per unit cell (a real, finite number) to keep
the universal ``"energy"`` capability honest, but the materials contract is the band
structure / DOS, not ``solve()``.

Units: SCF-mesh ``band_energies`` and ``fermi_energy`` are in Hartree (PySCF mo_energy);
``compute_band_structure`` returns eV, so the high-symmetry-path branch converts back to
Hartree for the ``BandStructureResult.band_energies`` contract. The band-gap / DOS dicts
stay in eV (the de-facto materials unit, matching ``api/routes/materials.py``).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from kanad.solvers.base_solver import BaseSolver
from kanad.solvers.capabilities import BandStructureResult
from kanad.core.solver_result import SolverResult

_HA_TO_EV = 27.2114


class PeriodicSolver(BaseSolver):
    """Periodic HF band structure + DOS via PySCF PBC (materials domain).

    Accepts a ``PeriodicHamiltonian`` directly, or a periodic ``Molecule`` (whose
    ``.hamiltonian`` property lazily builds one). Does NOT route through
    ``BaseSolver._resolve_system`` (which would reject a bare PeriodicHamiltonian and
    build a meaningless statevector backend).
    """

    def __init__(self, system, *, default_kpath: Optional[str] = None,
                 enable_analysis: bool = False, enable_optimization: bool = False):
        from kanad.core.hamiltonians.periodic_hamiltonian import PeriodicHamiltonian

        if isinstance(system, PeriodicHamiltonian) or (
            hasattr(system, "band_energies")
            and hasattr(system, "compute_band_structure")
            and hasattr(system, "k_weights")
        ):
            ham = system
            self.molecule = None
        elif getattr(system, "is_periodic", False):
            self.molecule = system
            ham = system.hamiltonian  # Molecule.hamiltonian lazily builds a PeriodicHamiltonian
        else:
            raise TypeError(
                "PeriodicSolver needs a periodic Molecule or a PeriodicHamiltonian, "
                f"got {type(system).__name__}"
            )

        self.hamiltonian = ham
        self.bond = None
        self.atoms = getattr(ham, "atoms", [])
        self._bond_type = "periodic"
        self.backend_name = "pyscf_pbc_hf"
        self.results = {}
        self.enable_analysis = enable_analysis
        self.enable_optimization = enable_optimization
        self.default_kpath = default_kpath
        # If the Hamiltonian already carries band energies (pre-solved or a synthetic
        # injection), no SCF is needed.
        self._scf_done = getattr(ham, "band_energies", None) is not None

    def _ensure_scf(self, *, max_iterations: int = 50, conv_tol: float = 1e-6,
                    verbose: int = 0) -> None:
        """Run the periodic SCF once (idempotent). No-op if band energies already exist."""
        if self._scf_done or getattr(self.hamiltonian, "band_energies", None) is not None:
            self._scf_done = True
            return
        self.hamiltonian.solve_scf(max_iterations=max_iterations, conv_tol=conv_tol,
                                   verbose=verbose)
        self._scf_done = True

    def solve(self, *, warm_state: Optional[Any] = None, **kwargs) -> SolverResult:
        """KRHF total energy per unit cell (Ha). Materials domain consumes band_structure()."""
        self._ensure_scf()
        hf = getattr(self.hamiltonian, "hf_energy", None)
        if hf is None:
            raise RuntimeError(
                "PeriodicSolver.solve: no periodic SCF energy available (call solve_scf "
                "or provide a solved/real PeriodicHamiltonian)."
            )
        return SolverResult(
            energy=float(hf),
            converged=bool(getattr(self.hamiltonian, "_scf_converged", True)),
            solver="periodic_hf", backend="pyscf_pbc_hf",
            hf_energy=float(hf), correlation_energy=0.0,
            extra={
                "fermi_energy_ha": float(getattr(self.hamiltonian, "fermi_energy", 0.0) or 0.0),
                "n_kpoints": int(np.asarray(self.hamiltonian.k_points).shape[0]),
                "note": "KRHF total energy per unit cell (Ha); the materials domain "
                        "consumes band_structure(), not solve().",
            },
        )

    def band_structure(self, k_path=None) -> BandStructureResult:
        """Band structure (capability ``"band_structure"``).

        ``k_path=None`` returns the SCF-mesh bands (already in Hartree — no ``mf``
        needed). A lattice-name string or explicit (N,3) k-points computes bands along
        that path via ``compute_band_structure`` (eV) and converts to Hartree.
        """
        self._ensure_scf()
        ham = self.hamiltonian
        if k_path is None:
            band_e_ha = np.asarray(ham.band_energies, dtype=float)
            kpts = np.asarray(ham.k_points, dtype=float)
        else:
            if isinstance(k_path, str):
                from kanad.core.io import get_kpath
                kpts, _labels, _pos = get_kpath(k_path, n_points=30)
            else:
                kpts = np.asarray(k_path, dtype=float)
            bs = ham.compute_band_structure(kpts)
            band_e_ha = np.asarray(bs["band_energies"], dtype=float) / _HA_TO_EV  # eV -> Ha
            kpts = np.asarray(bs.get("k_points", kpts), dtype=float)
        gap = ham.get_band_gap()  # eV dict, preserves method/caveat honesty keys
        fermi = float(getattr(ham, "fermi_energy", 0.0) or 0.0)
        return BandStructureResult(band_energies=band_e_ha, k_points=kpts,
                                   fermi_energy=fermi, band_gap=gap)

    def density_of_states(self, energy_grid=None) -> dict:
        """Total density of states (eV). Delegates to ``DOSCalculator``."""
        self._ensure_scf()
        from kanad.analysis.dos_calculator import DOSCalculator
        dos = DOSCalculator(self.hamiltonian)
        be_ev = np.asarray(self.hamiltonian.band_energies, dtype=float) * _HA_TO_EV
        if energy_grid is None:
            emin, emax, npts = float(be_ev.min()) - 3.0, float(be_ev.max()) + 3.0, 400
        else:
            grid = np.asarray(energy_grid, dtype=float)
            emin, emax, npts = float(grid.min()), float(grid.max()), len(grid)
        res = dos.compute_dos(energy_range=(emin, emax), n_points=npts, sigma=0.15,
                              units="eV")
        return {
            "energies": np.asarray(res["energies"]),
            "dos": np.asarray(res["dos"]),
            "fermi_energy": float(res["fermi_energy"]),
            "idos": np.asarray(res.get("idos", [])),
            "units": "eV",
            "n_electrons": int(res.get("n_electrons_actual", self.hamiltonian.n_electrons)),
        }
