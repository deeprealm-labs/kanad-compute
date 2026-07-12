"""Capability interfaces for the solver protocol (Stage 1, additive).

A solver declares *what it can do* in ``META.capabilities`` (see ``meta.py``) and
implements the matching accessor. This module defines:

- ``HamiltonianLike`` — the explicit, versioned surface external solvers may rely on
  (formalizes the duck-type ``BaseSolver._resolve_system`` already accepts).
- Typed result dataclasses (``GradientResult``, ``HessianResult``, ``ExcitedStateData``,
  ``BandStructureResult``).
- Capability ``Protocol``s (``EnergyProvider``, ``PropertyProvider``, ``ForceProvider``,
  ``HessianProvider``, ``ExcitedStatesProvider``, ``CouplingProvider``,
  ``FieldResponseProvider``, ``MaterialsProvider``) — the contracts consumers (labs,
  dynamics, reactions, analysis, workshop) program against.
- Default mixins (``FiniteDifferenceForceMixin``, ``FiniteDifferenceHessianMixin``)
  that synthesize forces/Hessians from an ``energy_fn`` closure so a solver gets them
  for free once it serves the md/reaction domains. (Wiring into concrete solvers lands
  in Stage 2; Stage 1 ships the contract + a generic capability-gated ``get_one_rdm``
  on ``BaseSolver``.)

Units contract (fixed once, see ``docs/design/SOLVER_PROTOCOL_PLAN.md``): positions
**Bohr**, energy **Hartree**, gradient/forces **(n_atoms, 3) Ha/Bohr** with
``forces = -gradient``, hessian **(3N, 3N) Ha/Bohr²**, frequencies **cm⁻¹** (negative =
imaginary), masses **amu**, 1-RDM **MO basis, trace == n_electrons within tol**, field
strengths **a.u.**
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np


# ── The Hamiltonian surface an external solver may depend on ────────────────
@runtime_checkable
class HamiltonianLike(Protocol):
    """Explicit version of the duck-type ``BaseSolver._resolve_system`` accepts.

    External solver authors type-annotate ``self.hamiltonian`` against this. The
    object exposes electron/orbital counts and a fermion->qubit Hamiltonian. It
    must NOT itself carry a ``.hamiltonian`` attribute (the resolver uses that to
    distinguish a bare Hamiltonian from a Bond/QuantumSystem wrapper).
    """
    n_electrons: int
    n_orbitals: int

    def to_sparse_hamiltonian(self, mapper: str = "jordan_wigner") -> Any: ...


# ── Typed capability results ────────────────────────────────────────────────
@dataclass
class GradientResult:
    """Nuclear gradient / forces at a geometry."""
    gradient: np.ndarray          # (n_atoms, 3) Ha/Bohr  (dE/dR)
    forces: np.ndarray            # (n_atoms, 3) Ha/Bohr  (= -gradient)
    energy: float                 # Ha at this geometry
    warm_state: Any = None
    method: str = ""              # 'analytic' | 'finite_difference' | 'hf_analytic'
    valid_off_equilibrium: bool = True  # frozen-theta HF forces => False (honesty)


@dataclass
class HessianResult:
    """Hessian + harmonic analysis at a stationary point."""
    hessian: np.ndarray           # (3N, 3N) Ha/Bohr^2
    frequencies_cm: np.ndarray    # (n_modes,) cm^-1 (negative = imaginary)
    normal_modes: np.ndarray      # (3N, n_modes)
    reduced_masses: np.ndarray    # (n_modes,) amu
    n_imaginary: int
    zpe_ha: float


@dataclass
class ExcitedStateData:
    """One canonical shape for excited-state output (replaces the 3 today)."""
    state_energies_ha: np.ndarray        # (n,) ABSOLUTE, ascending, [0] = ground
    excitation_energies_ev: np.ndarray   # (n-1,)
    oscillator_strengths: Optional[np.ndarray] = None   # (n-1,)  [transition_properties]
    transition_dipoles: Optional[np.ndarray] = None     # (n-1, 3) a.u. [transition_properties]
    eigenvectors: Optional[list] = None                 # per-state CI vectors (NAC / tracking)
    spin_multiplicities: Optional[np.ndarray] = None


@dataclass
class BandStructureResult:
    """Periodic band structure (materials domain)."""
    band_energies: np.ndarray     # (n_k, n_bands) Ha
    k_points: np.ndarray          # (n_k, 3)
    fermi_energy: float
    band_gap: dict                # {'gap','vbm','cbm','type':'direct'|'indirect'}


# ── Capability Protocols (what consumers program against) ───────────────────
@runtime_checkable
class EnergyProvider(Protocol):
    """Capability ``"energy"`` — the universal baseline."""
    def solve(self, *, warm_state: Optional[Any] = None, **kwargs) -> Any: ...


@runtime_checkable
class PropertyProvider(Protocol):
    """Capabilities ``"one_rdm"`` / ``"dipole"`` / ``"orbital_energies"``.

    All wavefunction observables flow through the real 1-RDM. ``get_one_rdm`` MUST
    raise on a trace mismatch rather than silently substitute HF (honesty rule).
    """
    def get_one_rdm(self, *, basis: str = "mo") -> np.ndarray: ...
    def get_dipole(self) -> np.ndarray: ...                # capability "dipole"
    def get_orbital_energies(self) -> dict: ...            # capability "orbital_energies"


@runtime_checkable
class ForceProvider(Protocol):
    """Capability ``"nuclear_gradient"`` (analytic). The FD floor over ``energy_fn``
    is always available via ``FiniteDifferenceForceMixin`` so consumers call
    ``nuclear_gradient`` unconditionally and never branch on capability presence."""
    def energy_fn(self) -> Callable[[np.ndarray, Optional[Any]], tuple]: ...
    def nuclear_gradient(self, atoms_bohr: np.ndarray, *,
                         warm_state: Optional[Any] = None) -> GradientResult: ...


@runtime_checkable
class HessianProvider(Protocol):
    """Capability ``"hessian"``."""
    def hessian(self, atoms_bohr: np.ndarray, *,
                warm_state: Optional[Any] = None) -> HessianResult: ...


@runtime_checkable
class ExcitedStatesProvider(Protocol):
    """Capabilities ``"excited_states"`` (+ ``"transition_properties"``)."""
    def solve_excited_states(self, n_states: int, *, spin: Optional[float] = None,
                             warm_state: Optional[Any] = None) -> Any: ...
    def get_excited_state_data(self) -> ExcitedStateData: ...


@runtime_checkable
class CouplingProvider(Protocol):
    """Capability ``"nonadiabatic_couplings"`` (photochemistry / FSSH)."""
    def nonadiabatic_coupling(self, atoms_bohr: np.ndarray, state_i: int, state_j: int,
                              *, warm_state: Optional[Any] = None) -> np.ndarray: ...
    def excited_state_gradient(self, atoms_bohr: np.ndarray, state: int,
                               *, warm_state: Optional[Any] = None) -> GradientResult: ...
    def state_overlap(self, atoms_a: np.ndarray, atoms_b: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class FieldResponseProvider(Protocol):
    """Capability ``"field_response"`` — energy under a static applied field; the
    honest superset for polarizability / Raman / (future) NMR. RAISES if the solver
    cannot apply a field (so the app fences rather than emitting fake values)."""
    def energy_under_field(self, atoms_bohr: np.ndarray,
                           e_field: np.ndarray, b_field: np.ndarray,
                           *, warm_state: Optional[Any] = None) -> tuple: ...


@runtime_checkable
class MaterialsProvider(Protocol):
    """Capability ``"band_structure"`` (materials domain)."""
    def band_structure(self, k_path: Optional[Any] = None) -> BandStructureResult: ...
    def density_of_states(self, energy_grid: Optional[Any] = None) -> dict: ...


# ── Default mixins (synthesize forces/Hessian from an energy closure) ───────
# Wiring into concrete solvers is Stage 2; defined here so the contract is whole.
class FiniteDifferenceForceMixin:
    """Default ``nuclear_gradient`` via central FD over ``energy_fn`` (Pulay-complete,
    matches ``dynamics/quantum_forces.compute_numerical_forces``: delta = 0.01 Bohr)."""
    _FD_DELTA_BOHR = 0.01

    def nuclear_gradient(self, atoms_bohr: np.ndarray, *,
                         warm_state: Optional[Any] = None) -> GradientResult:
        efn = self.energy_fn()  # type: ignore[attr-defined]
        atoms = np.asarray(atoms_bohr, dtype=float)
        n = atoms.shape[0]
        grad = np.zeros((n, 3))
        e0, warm = efn(atoms, warm_state)
        d = self._FD_DELTA_BOHR
        for a in range(n):
            for k in range(3):
                up = atoms.copy(); up[a, k] += d
                dn = atoms.copy(); dn[a, k] -= d
                e_up, warm = efn(up, warm)
                e_dn, warm = efn(dn, warm)
                grad[a, k] = (e_up - e_dn) / (2 * d)
        return GradientResult(gradient=grad, forces=-grad, energy=float(e0),
                              warm_state=warm, method="finite_difference",
                              valid_off_equilibrium=True)


class FiniteDifferenceHessianMixin:
    """Default ``hessian`` via central FD over ``nuclear_gradient`` (itself FD over the
    quantum ``energy_fn``) → a **quantum** Cartesian Hessian, then the shared harmonic
    analysis (``core.harmonic``) fills frequencies / normal modes / reduced masses / ZPE.

    Cost: O((3N)²) gradient evaluations, each itself O(3N) energy re-solves — feasible for
    small molecules only; gate on system size upstream. Frequencies are physically
    meaningful only at a stationary point (∇E ≈ 0); off a minimum the Hessian matrix is
    still valid but the projected spectrum is not a true harmonic spectrum.

    Masses come from :meth:`BaseSolver._hessian_masses_amu` in the SAME atom order as
    ``atoms_bohr`` (the ``energy_fn`` rebuild order). If masses are unavailable the raw
    Hessian is returned with an empty spectrum rather than a fabricated one (honesty)."""
    _FD_DELTA_BOHR = 0.01

    def hessian(self, atoms_bohr: np.ndarray, *,
                warm_state: Optional[Any] = None) -> HessianResult:
        atoms = np.asarray(atoms_bohr, dtype=float)
        n = atoms.shape[0]
        dim = 3 * n
        H = np.zeros((dim, dim))
        d = self._FD_DELTA_BOHR
        for i in range(n):
            for ki in range(3):
                up = atoms.copy(); up[i, ki] += d
                dn = atoms.copy(); dn[i, ki] -= d
                g_up = self.nuclear_gradient(up, warm_state=warm_state).gradient.ravel()  # type: ignore[attr-defined]
                g_dn = self.nuclear_gradient(dn, warm_state=warm_state).gradient.ravel()  # type: ignore[attr-defined]
                H[3 * i + ki, :] = (g_up - g_dn) / (2 * d)
        H = 0.5 * (H + H.T)  # symmetrize

        masses = None
        get_masses = getattr(self, "_hessian_masses_amu", None)
        if callable(get_masses):
            masses = get_masses()  # type: ignore[misc]
        if masses is None:
            # No mass information → return the raw Hessian without fabricating a spectrum.
            return HessianResult(hessian=H, frequencies_cm=np.array([]),
                                 normal_modes=np.zeros((dim, 0)),
                                 reduced_masses=np.array([]), n_imaginary=0, zpe_ha=0.0)

        from kanad.core.harmonic import harmonic_analysis
        ha = harmonic_analysis(H, atoms, np.asarray(masses, dtype=float))
        return HessianResult(hessian=H, frequencies_cm=ha['frequencies_cm'],
                             normal_modes=ha['normal_modes'],
                             reduced_masses=ha['reduced_masses_amu'],
                             n_imaginary=ha['n_imaginary'], zpe_ha=ha['zpe_ha'])


__all__ = [
    "HamiltonianLike",
    "GradientResult", "HessianResult", "ExcitedStateData", "BandStructureResult",
    "EnergyProvider", "PropertyProvider", "ForceProvider", "HessianProvider",
    "ExcitedStatesProvider", "CouplingProvider", "FieldResponseProvider", "MaterialsProvider",
    "FiniteDifferenceForceMixin", "FiniteDifferenceHessianMixin",
]
