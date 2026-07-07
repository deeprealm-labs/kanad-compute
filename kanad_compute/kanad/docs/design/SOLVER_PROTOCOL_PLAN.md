# Kanad Solver Protocol — Capability + Domain Development Plan

**Status:** final, decision-ready · **Audience:** framework + app architects · **Scope:** `kanad/solvers`, `kanad/core`, `kanad/dynamics`, `kanad/reactions`, `kanad/analysis`, `kanad/builder`, and the app registry/labs.

**Vocabulary constraint (load-bearing, from the recovered `PLAN.md` glossary):** this is an *architectural property*, not a product. Use plain words in code/docs/UI — **solver**, **capability**, **observables**, **user-defined solvers**, **bond-physics rules**. Do **not** brand it ("Solver Lane", "Plugin Protocol", "Algorithm Protocol", "Observable Plate" are banned). "Governance" is retired → "bond-physics rules" (optional hint only, never a hard dependency — verified numerically inert, `max|h_gov − h_no| = 0.0`).

**Honesty rule (PLAN.md culture #1, non-negotiable):** a capability that cannot resolve the *real* wavefunction density **raises**; it never silently substitutes HF and calls it quantum. A declared capability that does not work — or works but lies (HF-in-disguise dipole, swapped-state NAC, undifferentiable "adjoint" gradient) — is a protocol violation caught by a *value-checking* conformance test, not a shape-only one.

---

## 0. TL;DR VERDICT

**Does the protocol need upgrading?** Yes — but additively, not by rewrite. The 2026-06-12 migration already shipped the *envelope* (`BaseSolver` + frozen `SolverResult` + `BaseBackend` + `make_backend`). What is missing is the **capability layer** (`PLAN.md` M3 #5: a declared, queryable contract for 1-RDM / forces / excited states) and the **domain + registry layer** (`PLAN.md` M13: `register_solver` / capability-driven lab routing). Today drivers hard-import specific solver classes, observables silently fall back to HF, excited states surface three incompatible ways, and there is no nuclear-gradient, NAC, or domain-routing contract at all.

**The shape of the new protocol (one paragraph):** Keep `solve() -> SolverResult` as the only required method. Add a class-level `META: SolverMeta` declaring the solver's `domains`, `capabilities` (a closed vocabulary), and feasibility hints. Each capability beyond `energy` is an opt-in `typing.Protocol` + thin default mixin with a fixed units/shape contract: `get_one_rdm()` (the universal observables channel, raises on trace mismatch), `nuclear_gradient()` (with a universal finite-difference floor over the existing `energy_fn` closure), `solve_excited_states()` (one shape, typed accessor), `nonadiabatic_coupling()`, `band_structure()` (materials), and `energy_under_field()` (the honest superset for polarizability/Raman). A `register_solver()` registry lets the app and the workshop discover solvers and route labs **purely from meta** — no solver imports in the app. A parametrized conformance test asserts every declared capability is not just callable but *numerically honest* (1-RDM trace, dipole ≠ HF on a polar molecule, gradient matches FD, excited[0] == ground). Everything is purely additive: defaulting `CAPABILITIES = {"energy"}` makes all existing solvers instantly conformant.

**Build verdict:** Stage 1 (meta + Protocol scaffolding + value-checking conformance test) is low-risk and ready to start — every building block it leans on verifiably exists in-tree. **Three claims from the draft were wrong and are corrected below:** (1) qEOM does **not** track states across geometry, so FSSH must gate on a *measured* overlap-continuity check, not a meta flag; (2) polarizability/Raman are **not** covered by the geometry closure — they need a real `field_response` capability or PySCF-island fencing; (3) PhysicsVQE warm-start for reactions is a genuine multi-part task (no warm-state today, `enable_analysis` forced off, MP2-reselected excitations break warm-start shape), not a one-liner.

---

## 1. THE PROTOCOL

### 1.0 Design rules

1. **`solve()` stays the only required method.** Defaulting `META.capabilities = frozenset({"energy"})` and `META.domains = frozenset({"ground_state"})` makes every existing solver instantly conformant.
2. **Capabilities are opt-in (`typing.Protocol` + paired ABC mixin).** A solver gains a capability by (a) implementing the method (or inheriting the mixin) **and** (b) listing the capability string in `META.capabilities`. The conformance test asserts the two agree *and produce honest values*.
3. **Truthful declaration is enforced by value, not trusted.** `test_capability_conformance` runs each declared capability on H₂/LiH/a polar molecule and asserts the *physics* (not just that a method exists and returns the right shape). This is the explicit guard against the documented "capability that lies" failures (`LUCJ._supports_adjoint_gradient=True` but undifferentiable; HF-density silent fallback; overlap-NAC zeroing).
4. **Geometry/units contract fixed once:** positions **Bohr**, energy **Hartree**, gradient/forces **(n_atoms, 3) Ha/Bohr** with `forces = −gradient` (documented sign, matches `quantum_forces.compute_numerical_forces`), hessian **(3N, 3N) Ha/Bohr²**, frequencies **cm⁻¹** (negative = imaginary), masses **amu**, 1-RDM **MO basis (active-MO for active spaces), trace == n_electrons within 1e-4**. Field strengths **a.u.** This kills the legacy Å/Bohr confusion.
5. **`warm_state` is an opaque, solver-defined, round-trippable token.** The solver both produces it (`SolverResult.extra["warm_state"]`) and accepts it (`solve(warm_state=...)`). Drivers never inspect it. Matches `builder/quantum_system.py:energy_fn` exactly.
6. **`energy_at_geometry` is not separately declared — it is derived.** Per the critique, a re-solvable solver in `{md, reaction}` has the closure almost by definition. The registry **auto-derives** `energy_at_geometry` from `domain ∈ {md, reaction}` + presence of `energy_fn()`; authors never hand-declare it.
7. **`nuclear_gradient` is always callable.** The `ForceProvider` default mixin always presents `nuclear_gradient()` (FD over `energy_fn` when no analytic gradient exists), so consumers call **one** method and never branch on capability presence. Meta carries an `analytic_gradient: bool` hint *for the optimizer only*, not for dispatch.

### 1.1 Solver metadata (`SolverMeta`)

```python
# kanad/solvers/meta.py
from __future__ import annotations
from dataclasses import dataclass

# Closed capability vocabulary. New capabilities are added HERE only, and
# only when a real consumer lands (no speculative entries — see §7 decision 6).
CAPABILITY_NAMES = frozenset({
    "energy",                  # solve() -> SolverResult with finite .energy   [REQUIRED]
    "one_rdm",                 # get_one_rdm() -> (n_orb, n_orb) MO 1-RDM (real density)
    "dipole",                  # get_dipole() -> (3,) Debye, from the REAL 1-RDM
    "orbital_energies",        # get_orbital_energies() -> {homo_ev, lumo_ev, eps, occ, source}
    "nuclear_gradient",        # ANALYTIC gradient advertised (FD floor is always present)
    "hessian",                 # hessian() -> HessianResult
    "excited_states",          # solve_excited_states(n) -> SolverResult (+ typed payload)
    "transition_properties",   # excited payload carries oscillator strengths + transition dipoles
    "nonadiabatic_couplings",  # nonadiabatic_coupling(atoms,i,j); needs MEASURED state continuity
    "field_response",          # energy_under_field(atoms, E, B) closure -> polarizability/Raman/NMR
    "band_structure",          # materials: band_energies(k) + gap + DOS
})
# NOTE (critique #6): "two_rdm" and "measurement_telemetry" are DELIBERATELY NOT in v1.
# Only SamplingSQD produces a 2-RDM and no shipped calculator consumes it; telemetry is
# caller-fed generic statistics. Both remain documented reserved extra-keys until a real
# consumer arrives, at which point they graduate into CAPABILITY_NAMES.

DOMAIN_NAMES = frozenset({
    "ground_state",   # Schrodinger lab
    "md",             # molecular dynamics
    "reaction",       # PES / TS / IRC / rates
    "photochemistry", # UV-Vis, NAMD, photodynamics
    "materials",      # periodic / DOS / band structure
})

@dataclass(frozen=True)
class SolverMeta:
    """Declarative description the app/registry route on. One per solver class."""
    name: str                                    # stable id used in run(solver=...)
    domains: frozenset[str]                       # subset of DOMAIN_NAMES
    capabilities: frozenset[str]                  # subset of CAPABILITY_NAMES; MUST include "energy"
    # Pre-run feasibility hints (avoid mid-call NotImplementedError):
    max_qubits: int | None = None                # statevector/VQE feasibility ceiling
    max_determinants: int | None = None          # CI/SQD subspace ceiling
    supports_open_shell: bool = True
    analytic_gradient: bool = False              # optimizer hint ONLY (FD floor always present)
    consistent_state_tracking: bool = False      # MEASURED overlap-continuity (see §1.3 NAC); NOT trusted
    backends: frozenset[str] = frozenset({"statevector"})
    # Provenance / discovery:
    author: str = "kanad"                        # "kanad" (reference set) | "<user>@workshop"
    version: str = "0.0.0"
    description: str = ""
    citation: str | None = None                  # DOI/arXiv if battle-tested (PLAN culture #8)

    def __post_init__(self):
        assert "energy" in self.capabilities, "every solver must declare 'energy'"
        assert self.capabilities <= CAPABILITY_NAMES, f"unknown capability: {self.capabilities}"
        assert self.domains <= DOMAIN_NAMES, f"unknown domain: {self.domains}"
```

> **Note on `supports_active_space_scan` (critique #4):** this hint was **removed from `SolverMeta`**. The documented mid-scan crash (`materialize_at` raising under displaced geometry with `active_space='auto'→mp2no`) lives in `SystemSpec`/the builder, which owns the active-space policy — the solver never sees it. A solver-level flag cannot gate a builder-level reselection. Instead, **`QuantumSystem` exposes a `scan_safe() -> bool` predicate** the app checks *alongside* solver meta before enabling MD/reaction runs (see §5.2).

### 1.2 Minimal `BaseSolver` additions (back-compat)

`BaseSolver` keeps its current constructor (`__init__(self, system, *, backend='statevector', enable_analysis=True, enable_optimization=True, **kw)`, confirmed at `solvers/base_solver.py:29`) and the single abstract `solve()`. We add a class-level `META` and introspection helpers. No existing solver breaks.

```python
class BaseSolver(ABC):
    # NEW: every solver class declares this. Default keeps legacy solvers valid.
    META: SolverMeta = SolverMeta(
        name="base", domains=frozenset({"ground_state"}),
        capabilities=frozenset({"energy"}),
    )

    @classmethod
    def capabilities(cls) -> frozenset[str]: return cls.META.capabilities
    @classmethod
    def has_capability(cls, cap: str) -> bool: return cap in cls.META.capabilities
    @classmethod
    def supports_domain(cls, domain: str) -> bool: return domain in cls.META.domains

    # FIX (verified base_solver.py:164): abstract signature currently lies — it declares
    # `-> Dict[str, Any]` with a dict-shaped docstring, a trap for external implementers.
    @abstractmethod
    def solve(self, *, warm_state=None, **kwargs) -> "SolverResult":
        ...
```

### 1.3 Capability interfaces (opt-in `Protocol`s + mixins)

```python
# kanad/solvers/capabilities.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Any, Callable, Optional
from dataclasses import dataclass
import numpy as np
from kanad.core.solver_result import SolverResult
```

#### `HamiltonianLike` — ship the duck-type as an explicit Protocol in Stage 1 (critique #5)

External authors only get a working solver because `BaseSolver._resolve_system` duck-types the system into a Hamiltonian exposing an exact, un-versioned surface. We formalize it **now**, not "later", and the public example type-annotates against it.

```python
@runtime_checkable
class HamiltonianLike(Protocol):
    n_electrons: int
    n_orbitals: int
    def to_sparse_hamiltonian(self, mapper: str = "jordan_wigner") -> Any: ...
    # MUST NOT itself carry a .hamiltonian attribute (the two-classes-named-
    # MolecularHamiltonian hazard the resolver guards against).
```

`ActiveHamiltonian` and `PeriodicHamiltonian` declare conformance explicitly (kills the H15 attribute-sniffing).

#### `EnergyProvider` (required — already universal)

```python
@runtime_checkable
class EnergyProvider(Protocol):
    META: Any
    def solve(self, *, warm_state: Optional[Any] = None, **kwargs) -> SolverResult:
        """SolverResult.energy finite (Ha). extra['warm_state'] set iff warm-startable."""
```

Capability `"energy"`. Every conformant solver already satisfies this.

#### `PropertyProvider` — the real 1-RDM / dipole / orbital energies (most-used contract)

Every wavefunction-derived observable flows through the 1-RDM. Today it travels out-of-band (VQE writes `hamiltonian.set_quantum_density_matrix`; analysis reads `hamiltonian.get_density_matrix('ao')`; SQD exposes `get_1rdm_active_mo()`; the builder VQE-route *raises*). We standardize **one** accessor and keep the Hamiltonian write as an **adapter** so `PropertyCalculator`/`BondingAnalyzer`/`EnergyAnalyzer` consume it unchanged.

```python
@runtime_checkable
class PropertyProvider(Protocol):
    def get_one_rdm(self, *, basis: str = "mo") -> np.ndarray:
        """1-particle RDM. basis='mo' -> (n_active_orb, n_active_orb) active-MO, spin-summed;
        basis='ao' -> full AO density (frozen core + active embedded). Hermitian.
        trace == n_electrons within 1e-4, else RAISE (NO HF fallback — honesty rule)."""

    def get_dipole(self) -> np.ndarray:        # capability "dipole"
        """(3,) Debye from get_one_rdm('ao'). Never HF unless the solver IS HF."""

    def get_orbital_energies(self) -> dict:    # capability "orbital_energies"
        """{'eps':(n,) Ha,'occ':(n,),'homo_ev':float,'lumo_ev':float,
            'source':'koopmans_hf'|'delta_scf'|'natural_orbital'}"""
```

**Default mixin (`StatevectorPropertyMixin`):** for any solver exposing a converged statevector, `get_one_rdm` delegates to `QuantumRDMExtractor.extract_1rdm` (already built, `core/density/quantum_rdm.py`) and pushes the result into the Hamiltonian via `set_quantum_density_matrix` so the existing analysis path is fed. **Documented constraint (critique #5):** `QuantumRDMExtractor` is **Jordan-Wigner only** and assumes `n_qubits = 2·n_orbitals` with the JW even/odd spin convention. The mixin docstring states this; solvers using other mappers (e.g. Bravyi-Kitaev) **must override**. CI-vector solvers (CISolver/DeterministicCI) and selected-CI (SamplingSQD via `get_1rdm_active_mo`) override.

#### `ForceProvider` — nuclear gradient (the biggest missing capability)

Forces are NOT a solver capability today; they live in `dynamics/quantum_forces` (FD over the builder closure, verified) and `core/gradients` (PySCF HF/MP2 only). Two tiers, matching what `compute_numerical_forces`/`MDSimulator` already consume:

```python
@dataclass
class GradientResult:
    gradient: np.ndarray   # (n_atoms, 3) Ha/Bohr  (dE/dR)
    forces: np.ndarray     # (n_atoms, 3) Ha/Bohr  (= -gradient)
    energy: float          # Ha at this geometry
    warm_state: Any = None
    method: str = ""       # 'analytic' | 'finite_difference' | 'hf_analytic'
    valid_off_equilibrium: bool = True  # frozen-θ HF forces => False (honesty, idea 11)

@runtime_checkable
class ForceProvider(Protocol):
    # TIER 1 (universal floor): geometry-parametric energy closure.
    # Auto-derives the 'energy_at_geometry' routing flag — NOT separately declared.
    def energy_fn(self) -> Callable[[np.ndarray, Optional[Any]], tuple[float, Any]]:
        """(atoms_bohr (n,3), warm_state) -> (energy_Ha, new_warm_state). EXACTLY the
        builder QuantumSystem.energy_fn contract (verified). Bare electronic energy."""

    # ALWAYS present (mixin provides FD). Consumers call this and never branch.
    def nuclear_gradient(self, atoms_bohr: np.ndarray, *,
                         warm_state: Optional[Any] = None) -> GradientResult:
        """method='analytic' if the solver overrides; else 'finite_difference' via the
        default mixin over energy_fn (central diff, delta=0.01 Bohr — matches
        compute_numerical_forces). Capability 'nuclear_gradient' advertises ANALYTIC only."""
```

> **Why the FD floor is correct, not a cop-out:** FD over the *total* converged energy captures Hellmann-Feynman **and** Pulay automatically (the basis-set R-dependence is already in E(R)). This is the validated path — `quantum_forces.py` docstring and reactions N₂(10,10) matching CASCI to 0.0 mHa. `valid_off_equilibrium=False` is reserved for the legacy frozen-θ analytic path so drivers can refuse it.

#### `HessianProvider`

```python
@dataclass
class HessianResult:
    hessian: np.ndarray         # (3N, 3N) Ha/Bohr^2
    frequencies_cm: np.ndarray  # (n_modes,) cm^-1, negative = imaginary
    normal_modes: np.ndarray    # (3N, n_modes)
    reduced_masses: np.ndarray  # (n_modes,) amu
    n_imaginary: int
    zpe_ha: float

@runtime_checkable
class HessianProvider(Protocol):
    def hessian(self, atoms_bohr: np.ndarray, *,
                warm_state: Optional[Any] = None) -> HessianResult: ...
```

Capability `"hessian"`. **Default mixin** synthesizes the Hessian by FD over `nuclear_gradient` (or double-FD over `energy_fn`), giving `FrequencyCalculator`, `ThermochemistryCalculator`, `RamanIRCalculator`, and reaction TS-confirmation a uniform source. `FrequencyCalculator.compute_frequencies(hessian=...)` is the ready-made injection point.

#### `ExcitedStatesProvider` — energies + transition properties (one shape)

Today excited states surface three inconsistent ways. We standardize on **`solve_excited_states(n) -> SolverResult`** with `.states` populated and a **typed accessor** (so `dynamics/quantum_nac.py`'s broken `.ground_energy`/`.eigenvectors` *attribute* reads on the frozen `SolverResult` become a stable method).

```python
@dataclass
class ExcitedStateData:
    state_energies_ha: np.ndarray       # (n,) ABSOLUTE, ascending, [0] = ground
    excitation_energies_ev: np.ndarray  # (n-1,)
    oscillator_strengths: np.ndarray    # (n-1,)  [cap "transition_properties"]
    transition_dipoles: np.ndarray      # (n-1, 3) a.u. [cap "transition_properties"]
    eigenvectors: list[np.ndarray]      # per-state CI vectors (for NAC + tracking)
    spin_multiplicities: np.ndarray | None = None
    state_extent_r2: np.ndarray | None = None  # Rydberg diagnostic (builder computes)

@runtime_checkable
class ExcitedStatesProvider(Protocol):
    def solve_excited_states(self, n_states: int, *, spin: Optional[float] = None,
                             warm_state: Optional[Any] = None) -> SolverResult:
        """SolverResult.states = state_energies_ha[1:] (excited-only, matching core field)."""
    def get_excited_state_data(self) -> ExcitedStateData:
        """Stable typed accessor. Replaces .extra['excitation_energies'] attribute reads."""
```

Capability `"excited_states"`; add `"transition_properties"` **only** if the solver fills oscillator strengths + transition dipoles (today only `ExcitedStatesSolver` CIS branch — SQD/qEOM declare `excited_states` but **not** `transition_properties`, the honest state that lets `UVVisCalculator`/`absorption_spectrum` *refuse* rather than emit zeros).

#### `CouplingProvider` — nonadiabatic couplings (corrected per critique #2)

No solver or core path produces NACs today; `dynamics/quantum_nac.py` reconstructs them externally. **Verified: qEOM has zero state-tracking code** (`grep track|follow|reorder|overlap` over `solvers/qeom_vqe.py` finds only loop `continue`s; it solves the EOM eigenproblem at *one* geometry — root order can swap between adjacent geometries near avoided crossings, exactly where NAC matters). Therefore:

```python
@runtime_checkable
class CouplingProvider(Protocol):
    def nonadiabatic_coupling(self, atoms_bohr: np.ndarray, state_i: int, state_j: int,
                              *, warm_state: Optional[Any] = None) -> np.ndarray:
        """d_ij = <psi_i| d/dR |psi_j>, (n_atoms,3), units 1/Bohr.
        Antisymmetric (d_ij = -d_ji); translation-invariant (sum_A d_ij[A] ~ 0, ETF)."""
    def excited_state_gradient(self, atoms_bohr: np.ndarray, state: int,
                               *, warm_state: Optional[Any] = None) -> GradientResult:
        """Per-surface gradient for FSSH nuclear propagation."""
    def state_overlap(self, atoms_a: np.ndarray, atoms_b: np.ndarray) -> np.ndarray:
        """<psi_i(R_a)|psi_j(R_b)> overlap MATRIX (n_states, n_states) between two geometries.
        This is the OBSERVABLE that defines consistent_state_tracking — not a meta flag."""
```

**`consistent_state_tracking` is defined by an observable contract, not trusted as a flag.** A solver may set `META.consistent_state_tracking=True` only if it implements `state_overlap`, and the conformance test **verifies continuity on a 2-geometry probe** (overlap matrix ≈ identity after a small displacement, off-diagonal below threshold). **Until qEOM implements overlap-based root following, its flag stays `False`** ("single-geometry only"), and **`NonAdiabaticMD`/FSSH hard-refuse** any solver whose flag is False rather than silently producing swapped-state NACs. The default `nonadiabatic_coupling` computes `d_ij` via FD of tracked excited-state energies + eigenvector-overlap tracking, and **must flag** the documented overlap-NAC ground↔excited zeroing limitation instead of fabricating.

#### `FieldResponseProvider` — applied-field response (NEW; closes critique #1)

The geometry closure `energy_fn(atoms_bohr, warm_state)` has **no field argument** (verified: no `e_field`/`finite_field`/`apply_field` closure exists anywhere in `solvers/`, `builder/`, `core/gradients`, `dynamics/`). Polarizability is `d²E/dF²` under an **electric field**; Raman needs the polarizability **derivative** along normal modes. Neither can come from the geometry closure. We add the honest superset capability that polarizability + Raman + (future) NMR all need:

```python
@runtime_checkable
class FieldResponseProvider(Protocol):
    def energy_under_field(self, atoms_bohr: np.ndarray,
                           e_field: np.ndarray = np.zeros(3),   # a.u.
                           b_field: np.ndarray = np.zeros(3),   # a.u.
                           *, warm_state: Optional[Any] = None) -> tuple[float, Any]:
        """(energy_Ha, warm_state) with a static applied field. Enables finite-field
        polarizability (E-field FD), Raman (polarizability deriv along modes), and is
        the natural seam for magnetic response. RAISES if the solver cannot apply a field."""
```

Capability `"field_response"`. **If absent, polarizability and Raman are explicitly NON-capabilities** — the app must not route a quantum solver to them, exactly as NMR is fenced (§4). No solver ships `field_response` in v1; PySCF finite-field is the v1 island. This replaces the draft's incorrect claim that the geometry closure covers them.

#### `MaterialsProvider` — distinct output shape

Periodic systems don't fit `energy()->SolverResult`; `PeriodicHamiltonian` returns bands/DOS/gap.

```python
@dataclass
class BandStructureResult:
    band_energies: np.ndarray  # (n_k, n_bands) Ha
    k_points: np.ndarray       # (n_k, 3)
    fermi_energy: float
    band_gap: dict             # {'gap','vbm','cbm','type':'direct'|'indirect'}

@runtime_checkable
class MaterialsProvider(Protocol):
    def band_structure(self, k_path=None) -> BandStructureResult: ...
    def density_of_states(self, energy_grid=None) -> dict:
        """{'energies':(M,),'dos_total':(M,),'integrated':(M,),'fermi_energy':float}"""
```

Capability `"band_structure"`, domain `"materials"`. The Materials lab consumes `BandStructureResult`, not `energy()`.

### 1.4 How results relate to `SolverResult` (additive; serialization footgun fixed per critique #6)

`SolverResult` stays the universal frozen envelope (verified fields at `core/solver_result.py:58-68`). **Capability accessors (`get_one_rdm`, `get_excited_state_data`, `nuclear_gradient`) are the sole primary API.** The typed slots below are **optional convenience only and are EXCLUDED from `to_dict()`** (or emitted under a versioned `capabilities` subkey) — because `ExcitedStateData` holds `list[ndarray]` eigenvectors that are not trivially JSON-able and the app already re-flattens `to_dict()` in `QuantumSystem`.

```python
@dataclass(frozen=True)
class SolverResult:
    # ... existing fields unchanged (energy, converged, solver, backend, iterations,
    #     hf_energy, correlation_energy, energy_history, states, analysis, extra) ...

    # NEW optional in-memory convenience (None unless filled; NOT serialized by to_dict):
    one_rdm_mo: np.ndarray | None = None
    gradient: np.ndarray | None = None
    excited: "ExcitedStateData | None" = None
    # warm_state lives in extra['warm_state'] (opaque, round-trippable)
```

**Required migration BEFORE consumers switch (critique #6):** a one-time de-duplication of the fragmented legacy keys `extra['quantum_1rdm' | 'quantum_rdm1' | 'rdm1']` → a single canonical key, so the API serialization layer never double-carries RDMs and `to_dict` stays stable.

| Concern | Stable surface | Notes |
|---|---|---|
| Ground energy | `result.energy` (Ha) | unchanged, universal |
| Excited energies | `result.states` (Ha, excited-only) | unchanged meaning |
| Excited spectroscopy | `solver.get_excited_state_data()` (primary) / `result.excited` (convenience, not serialized) | replaces `.extra['oscillator_strengths']` |
| 1-RDM | `solver.get_one_rdm()` (primary) / `result.one_rdm_mo` (convenience) | replaces fragmented `quantum_1rdm/rdm1` keys |
| Gradient | `solver.nuclear_gradient()` (primary) / `result.gradient` (convenience) | new; always callable |
| Warm-start | `result.extra['warm_state']` | opaque, round-trippable |
| Everything else | `result.extra` | telemetry, determinants, h/s matrices |

---

## 2. SOLVER × CAPABILITY × LAB MATRIX

Capabilities reverse-engineered from each solver's verified surface. `energy` omitted (universal). Solver set confirmed from `solvers/__init__.py:__all__`.

| Solver (file) | Domains | Capabilities **today** | Meta hints | Notes |
|---|---|---|---|---|
| `VQESolver` | ground_state, md, reaction | `one_rdm` (statevector), `dipole`, `nuclear_gradient`(FD floor) | `max_qubits≈12`, `analytic_gradient=False` | warm_state = θ vector; `compute_energy(params)` exists |
| `PhysicsVQE` | ground_state, reaction | (energy only **today**) | `enable_analysis` forced **False** → no 1-RDM; **no warm_state** | reactions hardcode it; warm-start is a real task (§3, critique #3) |
| `HardwareVQE` | ground_state | (energy only) | `backends={statevector,ibm}` | normalize `solve_*` variants behind `solve()` |
| `CISolver` | ground_state | (energy; `one_rdm` via subspace vec possible) | classical near-FCI | subclass of VQESolver(mode=hivqe) |
| `DeterministicCI` (`SQDSolver` alias) | ground_state, photochemistry, materials | `excited_states`, `one_rdm`, `dipole`, `orbital_energies` | `consistent_state_tracking=False` | richest classical solver; `orbital_energies`→DOS |
| `SamplingSQDSolver` | ground_state, md, reaction, photochemistry | `one_rdm`, `excited_states` (separate dict today) | sampling backends; `max_determinants` | also has 2-RDM (reserved, not a v1 capability); ctor `hamiltonian`→ normalize to `system` |
| `LanczosSolver` (`KrylovSQDSolver` alias) | ground_state, photochemistry | `excited_states` | properties = HF fallback only | classical Krylov |
| `qEOMVQE` | photochemistry | `excited_states` | **`consistent_state_tracking=False`** (verified no tracking), `max_qubits≈8` | NAC-grade *operators* but single-geometry; needs `state_overlap` before FSSH use |
| `ExcitedStatesSolver` | photochemistry | `excited_states`, **`transition_properties`** | mostly classical CIS/TDDFT | the spectroscopy/UV-Vis solver |
| `VarQITESolver` | ground_state | (energy; QFI metric internal) | experimental (A-matrix unverified) | do not entrench |
| `SampledSubspaceVQE` | ground_state | (energy only) | superseded by SamplingSQD | slated for retirement |
| `SmartSolver` (`solve_smart`) | ground_state | (energy; meta-router) | unify with `SolverRouter` | becomes a thin function over the router (idea 15) |

**Lab availability (derived from `domain ∈ META.domains` AND lab-required capabilities `⊆ META.capabilities`):**

| Lab | Domain flag | Required capabilities | Optional (gates UI features) |
|---|---|---|---|
| Schrödinger | `ground_state` | `energy` | `one_rdm`, `dipole`, `orbital_energies`, `excited_states` |
| Prigogine · MD | `md` | `energy_fn` (auto-derived) | `nuclear_gradient`(analytic), `hessian` |
| Prigogine · Reactions | `reaction` | `energy_fn` (auto-derived) | `nuclear_gradient`, `hessian`, `orbital_energies` |
| Photodynamics | `photochemistry` | `excited_states` | `transition_properties` (spectrum), `nonadiabatic_couplings`+`state_overlap` (FSSH), `excited_state_gradient` |
| Materials | `materials` | `band_structure` | — |

**Cross-cutting fixes this matrix forces:** unify `SamplingSQDSolver`/`DeterministicCI` constructors to positional `system`; `HybridSubspaceVQE` already retired (absent from `__all__`, confirmed) — conformance test prevents recurrence; `PhysicsVQE` gets the three-part warm-start retrofit (§3).

---

## 3. ANALYSIS COVERAGE TABLE (the proof — ~15 calculators)

"Today" = current data source; "Protocol capability" = the declared capability that supplies it. Honesty flags called out explicitly.

| # | Calculator | Needs | Today | Protocol capability |
|---|---|---|---|---|
| 1 | `EnergyAnalyzer.decompose_energy` | 1-RDM (basis-matched) + `h_core/eri/E_nn` | density passed in; ham fields | `one_rdm` + Hamiltonian fields |
| 2 | `BondingAnalyzer` (Mayer/Mulliken/HOMO-LUMO) | AO 1-RDM + overlap S + `mo_energy/occ` | ham.mf | `one_rdm('ao')` + `orbital_energies` |
| 3 | `CorrelationAnalyzer` | `vqe_energy`, `hf_energy`, FCI | scalars | `energy` (`hf_energy`/`correlation_energy` fields) |
| 4 | `PropertyCalculator` · **dipole/quadrupole** | AO 1-RDM; `int1e_r` | `_quantum_density_matrix_ao` or HF | `one_rdm`/`dipole` ✅ |
| 4b | `PropertyCalculator` · **polarizability** | `d²E/dF²` under E-field | **PySCF finite-field island** | **`field_response`** (else NON-capability, app must not route quantum) ⚠️ |
| 5 | `UVVisCalculator` | excitation energies + oscillator strengths + transition dipoles | TDDFT or SQD (f=None) | `excited_states` + `transition_properties` (refuses if absent) |
| 6 | `VibronicCalculator` | ground+excited frequencies + displacement | heuristic 0.88–0.95 | `hessian` + excited-state `hessian` (via `excited_state_gradient`→FD) |
| 7 | `BondLengthScanner` | per-geometry energy | HF/MP2 only | `energy_fn` (auto-derived) |
| 8 | `ConfigurationExplorer` | per-point `solve().energy` | self-builds solver | `energy_fn`; NEB images via `nuclear_gradient` |
| 9 | `DOSCalculator` | periodic bands **or** molecular multi-state spectrum | PeriodicHam / DeterministicCI | `band_structure` (materials) / `excited_states` |
| 10 | `molecular_descriptors` (reactivity) | HOMO/LUMO (eV) + source | mean-field MO energies | `orbital_energies` |
| 11 | `RamanIRCalculator` | normal modes + dipole derivs + **polarizability derivs** | `FrequencyCalculator` + ham density | IR: `hessian`+`dipole` ✅ · **Raman: `field_response` (else NON-capability)** ⚠️ |
| 12 | `ThermochemistryCalculator` | vibrational frequencies + geometry/masses | hardcoded table / re-solve | `hessian` (frequencies) + `energy` |
| 13 | `FrequencyCalculator` | Hessian (or gradient to FD it) | PySCF HF/MP2 FD | `hessian` (or `nuclear_gradient`→FD) |
| 14 | `NMRCalculator` (GIAO) | magnetic shielding tensors | PySCF GIAO (recompute) | **NON-capability — PySCF island.** App never routes a quantum solver here (prevents fake-quantum-NMR). Future: `field_response` (B-field) |
| 15 | `UncertaintyAnalyzer` | per-Pauli expectations/coeffs + shots | caller-fed | **reserved extra-key** (`measurement_telemetry` not a v1 capability — critique #6) |
| 16 | `TrajectoryAnalyzer` / `ReactionAnalyzer` / `NAMDAnalyzer` | driver-result objects | dynamics output | consume **driver output**, not a raw `SolverResult` |

**Coverage verdict:** every calculator's input maps to a declared capability **or is explicitly fenced as a recompute-classically island** (NMR; polarizability/Raman until `field_response` ships). The honesty-critical gaps the survey flagged are closed: silent-HF-fallback dies because `get_one_rdm` *raises* on trace mismatch and `has_capability('one_rdm')` lets analysis refuse/label; quantum UV-Vis zeros die because `transition_properties` is declared only when real; **and the draft's polarizability/Raman over-claim is corrected** — they are now honestly either `field_response` or a PySCF island, never "covered by the geometry closure."

---

## 4. WHAT'S MISSING TODAY → HOW THE PROTOCOL FILLS IT

| Gap | Evidence (verified) | Protocol fill |
|---|---|---|
| **Nuclear gradient is not a solver capability** | forces in `core/gradients` (PySCF only) + `dynamics/quantum_forces` (FD over builder closure) | `ForceProvider`: Tier-1 `energy_fn()` (universal, Pulay-complete) + always-present `nuclear_gradient()` (FD floor); `MDSimulator`/reactions stop hard-importing solver classes |
| **Hessian has no solver path** | `FrequencyCalculator` FDs PySCF gradients | `HessianProvider` + default FD mixin; `compute_frequencies(hessian=...)` injection point |
| **NACs have no path; `quantum_nac` reads dead attributes** | `qeom_result.ground_energy/.eigenvectors` now live in `.extra` on the frozen `SolverResult` | `CouplingProvider` + `get_excited_state_data()` typed accessor + `state_overlap` **observable** gating FSSH |
| **qEOM advertised as state-tracking but isn't** | verified: zero tracking/overlap/reorder code in `qeom_vqe.py` | `consistent_state_tracking` redefined as a *measured* contract; qEOM flag stays **False**; FSSH refuses it |
| **Excited states surfaced 3 ways; SQD returns a bare dict** | `states` vs `solve_excited_states` dict vs `n_states` ctor kwarg | one `solve_excited_states()->SolverResult` + `get_excited_state_data()` |
| **Polarizability / Raman have no honest path** | verified: no field closure anywhere | **`field_response` capability** (E/B-field energy closure) or explicit PySCF-island fence |
| **Observables silently fall back to HF (M3)** | `compute_quantum_X` returns HF unless solver wrote 1-RDM | `get_one_rdm` *raises* on trace mismatch; `has_capability` lets analysis refuse/label |
| **1-RDM accessor fragmented** | `quantum_1rdm`/`quantum_rdm1`/`rdm1` keys | single `get_one_rdm()`; one-time key de-dup migration |
| **No capability/domain declaration → routing by name/threshold** | `SolverRouter` size-only; `quantum_nac`/reactions hardcode classes | `SolverMeta{domains, capabilities, limits}` + registry query |
| **Active-space mid-scan crash** | `materialize_at` raises under displaced geom (builder, not solver) | builder-owned `QuantumSystem.scan_safe()` predicate, checked alongside solver meta |
| **`ActiveHamiltonian` conforms by attribute-sniffing (H15)** | duck-typed in `_resolve_system` | `HamiltonianLike` Protocol shipped in Stage 1; `Active`/`Periodic` declare conformance |
| **`solve()` abstract type lies** | `base_solver.py:164` declares `-> Dict[str, Any]` | corrected to `SolverResult` in the ABC |
| **Capability flags can lie** | `LUCJ._supports_adjoint_gradient=True` undifferentiable | `test_capability_conformance` asserts declared == callable **and numerically honest** |

---

## 5. PUBLIC / WORKSHOP AUTHORING STORY

### 5.1 Smallest working external solver (honest surface)

Minimum: subclass `BaseSolver`, set `META`, implement `solve()`, register. The example type-annotates against the **shipped** `HamiltonianLike` Protocol and the docstring states the JW constraint inherited from the property mixin.

```python
from kanad import register_solver
from kanad.solvers import BaseSolver, SolverMeta, SolverResult
from kanad.solvers.capabilities import HamiltonianLike  # shipped in Stage 1

class MyAFQMC(BaseSolver):
    META = SolverMeta(
        name="my_afqmc",
        domains=frozenset({"ground_state"}),
        capabilities=frozenset({"energy"}),
        author="ada@workshop", version="0.1.0",
        description="Toy AFQMC over the active-space Hamiltonian.",
        max_qubits=24,
    )
    def solve(self, *, warm_state=None, **kw) -> SolverResult:
        H: HamiltonianLike = self.hamiltonian              # explicit, versioned surface
        e = my_afqmc_energy(H.to_sparse_hamiltonian("jordan_wigner"), H.n_electrons)
        return SolverResult(energy=float(e), converged=True,
                            solver="my_afqmc", backend=self.backend_name)

register_solver(MyAFQMC)   # reads MyAFQMC.META.name
```

`run(molecule, solver="my_afqmc")` now works. To unlock more labs, add a capability + its method:
- `"one_rdm"` + `get_one_rdm()` (JW-only via the mixin, or override for other mappers) → observables tab;
- `energy_fn()` + `domains ⊇ {md, reaction}` → MD + Reactions labs (router auto-derives `energy_at_geometry`);
- `"excited_states"` + `solve_excited_states()` (+ `state_overlap`, `"nonadiabatic_couplings"`) → Photochemistry + FSSH;
- `"field_response"` + `energy_under_field()` → polarizability/Raman.

### 5.2 Registry + app lab routing by domain flag (no solver imports in the app)

```python
# kanad/solvers/registry.py
_REGISTRY: dict[str, type[BaseSolver]] = {}

def register_solver(cls: type[BaseSolver]) -> type[BaseSolver]:
    assert issubclass(cls, BaseSolver) and cls.META.name not in _REGISTRY
    _REGISTRY[cls.META.name] = cls
    return cls

def get_solver(name: str) -> type[BaseSolver]: return _REGISTRY[name]
def list_solvers(*, domain=None, capability=None) -> list[SolverMeta]:
    return [c.META for c in _REGISTRY.values()
            if (domain is None or domain in c.META.domains)
            and (capability is None or capability in c.META.capabilities)]
```

The app routes **purely from meta + the builder predicate**:
- **Lab availability** — show a solver iff `domain ∈ META.domains` and the lab's required capabilities `⊆ META.capabilities`. The Photochemistry FSSH toggle appears only when `nonadiabatic_couplings ∈ capabilities` **and** the solver passes the `state_overlap` continuity check.
- **Pre-run feasibility** — compute `n_qubits = 2·n_active_orbitals`; disable when `> META.max_qubits` (replaces mid-call `NotImplementedError`; qEOM > 8, statevector ≥ 16). **For MD/reaction, also require `QuantumSystem.scan_safe()`** — the builder-owned guard against the active-space reselection crash (critique #4); solver meta alone is insufficient.
- **Feature gating** — observables/spectrum panels render only the fields whose capabilities are present.
- **Provenance** — `author`/`version`/`citation` distinguish the **kanad reference set** from **workshop/community** solvers; community solvers carry a "user-defined" badge until `citation` is set (PLAN culture #8).

### 5.3 Workshop benchmarking (capability-driven, value-gated)

Because every solver shares the contract + fixed capability surface, the workshop runs a **standard benchmark** vs the reference set on identical `system`: `energy` vs CASCI/FCI, `get_one_rdm` trace, `nuclear_gradient` vs FD, excited energies vs reference, plus wall-time + `iterations`. It is capability-driven (only runs the checks the candidate declares) and **publish is gated on `test_capability_conformance`** — a solver cannot publish a capability it fails *numerically*, not merely structurally.

---

## 6. STAGED PLAN

**Stage 0 — Recover design intent (½ day).** `git show 031a6ad^:PLAN.md` and `ideas/{07,08,11,13,15,16,17,20}.md` → restore under `docs/design/`. They contain the M3 #5 contract, M13 SDK, and per-capability signatures this plan formalizes. (`ideas/` exists in-tree.)

**Stage 1 — Protocol spec, no behavior change (framework). ← FIRST.**
1. `kanad/solvers/meta.py` (`SolverMeta`, `CAPABILITY_NAMES`, `DOMAIN_NAMES`).
2. `kanad/solvers/capabilities.py` — the Protocols (`HamiltonianLike`, `EnergyProvider`, `PropertyProvider`, `ForceProvider`, `HessianProvider`, `ExcitedStatesProvider`, `CouplingProvider`, `FieldResponseProvider`, `MaterialsProvider`) + `GradientResult`/`HessianResult`/`ExcitedStateData`/`BandStructureResult` + default mixins (`StatevectorPropertyMixin` w/ documented JW-only constraint; FD `ForceProvider`/`HessianProvider` mixins).
3. `BaseSolver`: add `META`, `capabilities()`, `has_capability()`, `supports_domain()`; **fix `solve()` return type/docstring to `SolverResult`** (the `base_solver.py:164` lie).
4. `SolverResult`: add `one_rdm_mo`/`gradient`/`excited` optional convenience slots **excluded from `to_dict()`**; ship the `quantum_1rdm/quantum_rdm1/rdm1` → single-key de-dup migration.
5. Annotate existing solvers with `META` per §2 (defaults keep untouched ones valid). qEOM `consistent_state_tracking=False`; PhysicsVQE `one_rdm` absent.
6. **`test_capability_conformance`** (registry-parametrized) — for each declared capability assert it is callable **and numerically honest** on H₂/LiH/a polar molecule: `one_rdm` trace == n_electrons (1e-4) **and** dipole differs from HF by >1% on a polar molecule (proves not silent-HF); `nuclear_gradient` matches FD over `energy_fn` within tol; `excited_states[0]` == ground energy; `consistent_state_tracking=True` ⇒ `state_overlap` ≈ identity on a 2-geometry probe; undeclared capabilities raise `NotImplementedError`.

**Stage 2 — Prove with one reference solver per domain.**
- **ground_state:** `VQESolver` (`energy`, `one_rdm`, `dipole`, FD `nuclear_gradient`).
- **md/reaction:** `SamplingSQDSolver` (`energy_fn`, `one_rdm`) as primary reference. **`PhysicsVQE` warm-start is its OWN scoped task (critique #3):** add `energy_fn` closure + `warm_state` produce/accept + the invariant that the **MP2-ranked excitation set is frozen along a scan** (mirror the builder freeze policy). If excitations are re-ranked per geometry, `warm_state` is invalid → declare *not warm-startable* for that path (cold FD fallback). Regression test: PES scan with frozen vs reselected excitations.
- **photochemistry:** `ExcitedStatesSolver` (`transition_properties`) for spectra; `qEOMVQE` for excitation energies but **FSSH-blocked** until it implements `state_overlap` root-following.
- **materials:** a `PeriodicSolver` wrapper over `PeriodicHamiltonian` (`band_structure`).
- Rewire `dynamics`/`reactions` to consume via the capability Protocols (drop hard imports). Validate vs known-good numbers (H₂ PES 0.0 mHa; N₂(10,10) vs CASCI).

**Stage 3 — App registry + capability-driven lab routing.** `registry.py` + a stable `kanad/api.py` surface (idea 17; does not exist yet). App reads `list_solvers(domain, capability)`; labs gate on capabilities + `scan_safe()`; pre-run feasibility from `max_qubits`/`max_determinants`. `SolverRouter` becomes the capability+domain+size router (thresholds as fallback); `SmartSolver` becomes a thin function over it.

**Stage 4 — Workshop authoring / benchmark / publish.** `register_solver` plugin loader (`~/.kanad/plugins/`); workshop editor stub for a `BaseSolver` subclass; capability-driven benchmark vs reference set; publish gated on the value-checking conformance test; provenance/citation badge.

**Where solvers should live (recommended):**
- **Framework reference set** (`kanad/solvers`, `author="kanad"`, citation-backed): `VQESolver, PhysicsVQE, HardwareVQE, CISolver, DeterministicCI, SamplingSQDSolver, LanczosSolver, qEOMVQE, ExcitedStatesSolver, PeriodicSolver, SmartSolver→router`. These define each domain's reference capabilities and back the workshop benchmark.
- **App/community** (`~/.kanad/plugins/`, `author="<user>@workshop"`): user-defined solvers via `register_solver`, badged "user-defined" until cited. Framework ships *contract + reference set + benchmark*; app ships *discovery + routing + publish*. (PLAN culture #4: no workflow imports a specific solver class.)

### Concrete FIRST PR (Stage 1, framework, additive, no runtime change)

**Title:** `feat(solvers): capability + domain protocol scaffolding (SolverMeta, capability Protocols, value-checking conformance test)`

- **New:** `kanad/solvers/meta.py`, `kanad/solvers/capabilities.py`.
- **Edit:** `solvers/base_solver.py` — add `META` + 3 classmethods; fix `solve()` return type/docstring to `SolverResult`.
- **Edit:** `core/solver_result.py` — add `one_rdm_mo`/`gradient`/`excited` optional fields (excluded from `to_dict`); add the legacy-RDM-key de-dup.
- **Edit:** each solver in `solvers/__init__.__all__` — add a `META` reflecting **today's real** capabilities (§2). No method bodies change. qEOM `consistent_state_tracking=False`; PhysicsVQE no `one_rdm`.
- **New test:** `tests/unit/test_capability_conformance.py` — registry-parametrized; declared == callable **and value-honest**; undeclared raises.
- **Done criteria:** existing suite green (defaults preserve behavior); conformance test green; `VQESolver.has_capability("one_rdm")` True, `PhysicsVQE.has_capability("one_rdm")` False, `qEOMVQE.META.consistent_state_tracking` False; no consumer behavior changed.

---

## 7. OPEN DECISIONS FOR THE USER TO CONFIRM

1. **`field_response` capability vs PySCF-island for polarizability/Raman.** Recommendation: **ship `field_response` in the vocabulary now** (so the seam exists and the app can fence honestly), but **implement no quantum `energy_under_field` in v1** — polarizability/Raman stay a PySCF finite-field island until a solver opts in. Confirm you want the capability declared-but-unimplemented vs omitted entirely.
2. **qEOM + FSSH.** Recommendation: ship qEOM with `consistent_state_tracking=False`; **FSSH hard-refuses** it (no warn-and-degrade — swapped-state NACs are physically wrong). A separate task adds overlap-based root following + flips the flag once the 2-geometry probe passes. Confirm hard-refuse is acceptable (Photodynamics FSSH unavailable until then).
3. **PhysicsVQE warm-start scope.** Recommendation: treat as its own Stage-2 task with the frozen-excitation-set invariant; declare *not warm-startable* when excitations re-rank. Confirm reactions can use `SamplingSQDSolver` as the primary warm-startable reference in the meantime.
4. **`SolverResult` slots excluded from `to_dict()`.** Recommendation: capability accessors are primary; slots are in-memory convenience only, never serialized (avoids the `ExcitedStateData`/numpy JSON footgun and the app re-flatten impedance). Confirm.
5. **Drop `two_rdm` + `measurement_telemetry` from v1 vocabulary.** Recommendation: yes — no shipped consumer; keep as reserved extra-keys, graduate on demand. Confirm.
6. **`energy_at_geometry` auto-derived, not declared.** Recommendation: registry derives it from `domain ∈ {md,reaction}` + `energy_fn()` presence; `nuclear_gradient` always callable (FD floor) so consumers never branch. Confirm authors should not hand-declare it.
7. **`consistent_state_tracking` enforcement.** Recommendation: defined by the `state_overlap` observable + verified in conformance; FSSH gates on it. Confirm the observable-contract approach over a trusted flag.
8. **Scan-feasibility lives in the builder, not solver meta.** Recommendation: `QuantumSystem.scan_safe()` checked alongside solver meta for MD/reaction. Confirm.
9. **`max_qubits` is a feasibility hint, not a correctness guarantee.** VQE under-convergence past ~12 qubits is fuzzy; pair the UI disable with a convergence warning in the result. Confirm.
10. **NMR / magnetic response out of scope v1.** Stays a PySCF island; explicitly *not* a capability so the app never routes a quantum solver to it (prevents fake-quantum-NMR). Future `field_response` B-field path only. Confirm.
11. **App-side copy drift.** Land the protocol in the framework (this repo, submodule `kanad` at `kanad-app/kanad`), then re-sync the submodule/vendored copy. Do **not** fork the contract app-side. Confirm the framework is the single source.

---

**Key files this design touches (verified, absolute, this repo = framework submodule root `/home/mk/deeprealm/kanad-app/kanad`):**
- Contract: `/home/mk/deeprealm/kanad-app/kanad/solvers/base_solver.py` (fix `solve()` return at line 164), `/home/mk/deeprealm/kanad-app/kanad/core/solver_result.py` (fields at 58-68, `_CORE_KEYS` at 18)
- New: `/home/mk/deeprealm/kanad-app/kanad/solvers/meta.py`, `/home/mk/deeprealm/kanad-app/kanad/solvers/capabilities.py`, `/home/mk/deeprealm/kanad-app/kanad/solvers/registry.py`, `/home/mk/deeprealm/kanad-app/kanad/api.py`
- Routing: `/home/mk/deeprealm/kanad-app/kanad/solvers/solver_router.py`
- Capability sources already built: `/home/mk/deeprealm/kanad-app/kanad/core/density/quantum_rdm.py` (JW-only `extract_1rdm`), `/home/mk/deeprealm/kanad-app/kanad/dynamics/quantum_forces.py` (`ForceResult`, `compute_numerical_forces`, central FD delta=0.01 Bohr), `/home/mk/deeprealm/kanad-app/kanad/builder/quantum_system.py` (`energy_fn` at 1163, freeze policy, no field arg)
- Solvers to annotate: `/home/mk/deeprealm/kanad-app/kanad/solvers/{vqe_solver,physics_vqe,hardware_vqe,ci_solver,deterministic_ci,sampling_sqd,lanczos_solver,excited_states_solver,qeom_vqe,varqite_solver,sampled_subspace_vqe,smart_solver}.py` (qEOM verified no state-tracking; PhysicsVQE verified `enable_analysis=False` at 151, no warm_state)
- Consumers to rewire: `/home/mk/deeprealm/kanad-app/kanad/dynamics/{md_simulator,quantum_nac,nonadiabatic,photodynamics}.py`, `/home/mk/deeprealm/kanad-app/kanad/reactions/*.py`, `/home/mk/deeprealm/kanad-app/kanad/analysis/*.py`
- Recovered design intent: `git show 031a6ad^:PLAN.md` (M3 #5, M13, glossary), `git show 031a6ad^:ideas/{07,08,11,13,15,17}.md`; in-tree audits: `/home/mk/deeprealm/kanad-app/kanad/PLANCK_SOLVER_PROTOCOL_AUDIT.md`, `/home/mk/deeprealm/kanad-app/kanad/AUDIT_WORKLIST.md`