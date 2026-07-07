"""Solver metadata — the declarative contract the app/registry route on.

Part of the capability + domain solver protocol (Stage 1, additive). Every solver
class carries a class-level ``META: SolverMeta`` declaring which *domains* it serves
(which lab it belongs to) and which *capabilities* it provides beyond ``energy``.
Consumers (labs, dynamics/reactions drivers, analysis, workshop) query meta +
capabilities instead of importing concrete solver classes.

Design rules (see ``docs/design/SOLVER_PROTOCOL_PLAN.md``):
- ``solve() -> SolverResult`` stays the only *required* method.
- Defaulting ``capabilities={"energy"}`` / ``domains={"ground_state"}`` makes every
  existing solver instantly conformant — this layer is purely additive.
- Vocabularies are CLOSED: a capability/domain string outside the set is a bug,
  caught at class-definition time by ``SolverMeta.__post_init__``.
- A declared capability must actually *work and be numerically honest* — enforced
  by ``tests/unit/test_capability_conformance.py`` (value checks, not shape-only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ── Closed capability vocabulary ────────────────────────────────────────────
# New capabilities are added HERE only, and only when a real consumer lands
# (no speculative entries). 'two_rdm' and 'measurement_telemetry' are
# deliberately NOT in v1 — no shipped calculator consumes them; they remain
# reserved ``SolverResult.extra`` keys until a consumer arrives.
CAPABILITY_NAMES = frozenset({
    "energy",                  # solve() -> SolverResult with finite .energy        [REQUIRED]
    "one_rdm",                 # get_one_rdm() -> (n_orb, n_orb) MO 1-RDM (real density)
    "dipole",                  # get_dipole() -> (3,) Debye, from the REAL 1-RDM
    "orbital_energies",        # get_orbital_energies() -> {homo_ev, lumo_ev, eps, occ, source}
    "nuclear_gradient",        # ANALYTIC nuclear gradient advertised (FD floor always present)
    "hessian",                 # hessian() -> HessianResult
    "excited_states",          # solve_excited_states(n) -> SolverResult (+ typed payload)
    "transition_properties",   # excited payload carries oscillator strengths + transition dipoles
    "nonadiabatic_couplings",  # nonadiabatic_coupling(atoms,i,j); needs MEASURED state continuity
    "field_response",          # energy_under_field(atoms, E, B) -> polarizability / Raman / NMR
    "band_structure",          # materials: band_energies(k) + gap + DOS
})

# ── Closed domain vocabulary (one lab each) ─────────────────────────────────
DOMAIN_NAMES = frozenset({
    "ground_state",    # Schrödinger lab
    "md",              # molecular dynamics (Prigogine)
    "reaction",        # PES / TS / IRC / rates (Prigogine)
    "photochemistry",  # UV-Vis, NAMD, photodynamics (Prigogine)
    "materials",       # periodic / DOS / band structure
})


@dataclass(frozen=True)
class SolverMeta:
    """Declarative description the app/registry route on. One per solver class.

    Attributes:
        name: Stable id used in ``run(solver=...)`` and the registry.
        domains: Subset of :data:`DOMAIN_NAMES`. Which labs may offer this solver.
        capabilities: Subset of :data:`CAPABILITY_NAMES`. MUST include ``"energy"``.
        max_qubits: Statevector/VQE feasibility ceiling (None = unbounded). The app
            disables the solver pre-run when ``2*n_active_orbitals > max_qubits``
            instead of letting it raise mid-call.
        max_determinants: CI/SQD subspace ceiling (None = unbounded).
        supports_open_shell: Whether non-singlet systems are handled.
        analytic_gradient: Optimizer hint ONLY — a finite-difference floor is always
            present via the ForceProvider mixin, so consumers never branch on this.
        consistent_state_tracking: Whether excited states are followed continuously
            across geometry. This is a *measured* contract (verified via
            ``state_overlap`` in the conformance test), NOT a trusted flag — FSSH
            gates on it.
        backends: Backend names this solver supports.
        author: ``"kanad"`` (reference set) | ``"<user>@workshop"`` (community).
        version: Solver version string.
        description: One-line human description (shown in the UI).
        citation: DOI/arXiv if battle-tested; community solvers stay "user-defined"
            until set.
    """

    name: str
    domains: frozenset[str]
    capabilities: frozenset[str]
    # Pre-run feasibility hints (avoid mid-call NotImplementedError):
    max_qubits: Optional[int] = None
    max_determinants: Optional[int] = None
    supports_open_shell: bool = True
    analytic_gradient: bool = False
    consistent_state_tracking: bool = False
    backends: frozenset[str] = frozenset({"statevector"})
    # Provenance / discovery:
    author: str = "kanad"
    version: str = "0.0.0"
    description: str = ""
    citation: Optional[str] = None

    def __post_init__(self):
        # Normalize loose iterables (lists/sets) into frozensets so authors can
        # write `capabilities={"energy", "one_rdm"}` without ceremony.
        object.__setattr__(self, "domains", frozenset(self.domains))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))
        object.__setattr__(self, "backends", frozenset(self.backends))

        if "energy" not in self.capabilities:
            raise ValueError(
                f"SolverMeta(name={self.name!r}): every solver must declare the "
                f"'energy' capability"
            )
        unknown_caps = self.capabilities - CAPABILITY_NAMES
        if unknown_caps:
            raise ValueError(
                f"SolverMeta(name={self.name!r}): unknown capabilities "
                f"{sorted(unknown_caps)}; allowed: {sorted(CAPABILITY_NAMES)}"
            )
        if not self.domains:
            raise ValueError(
                f"SolverMeta(name={self.name!r}): must declare at least one domain"
            )
        unknown_domains = self.domains - DOMAIN_NAMES
        if unknown_domains:
            raise ValueError(
                f"SolverMeta(name={self.name!r}): unknown domains "
                f"{sorted(unknown_domains)}; allowed: {sorted(DOMAIN_NAMES)}"
            )
        if self.consistent_state_tracking and "excited_states" not in self.capabilities:
            raise ValueError(
                f"SolverMeta(name={self.name!r}): consistent_state_tracking=True "
                f"requires the 'excited_states' capability"
            )

    def has_capability(self, cap: str) -> bool:
        return cap in self.capabilities

    def serves_domain(self, domain: str) -> bool:
        return domain in self.domains
