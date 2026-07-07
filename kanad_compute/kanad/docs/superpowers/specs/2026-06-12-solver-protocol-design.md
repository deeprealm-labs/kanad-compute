# Unified Solver Protocol — Design Spec

**Date:** 2026-06-12
**Branch:** `feat/solver-protocol`
**Status:** Approved design, pre-implementation

## Problem

Kanad has 14 solver classes whose *methodology* legitimately differs (VQE, Lanczos,
SQD, CI, qEOM, etc.), but whose *envelope* — how you construct them, how you call
them, what they return, and how they reach a backend — diverges with no shared
protocol. This is the root cause of the "Solver API Gotchas" accumulated in project
memory.

Concretely, today:

- **Constructors disagree.** `VQESolver(bond, ...)` supports three APIs; `LanczosSolver(bond_or_molecule)`;
  `SamplingSQDSolver(hamiltonian)` takes no bond at all; `PhysicsVQE(bond, molecule, hamiltonian, pyscf_mol)`
  uses priority-ordered kwargs; `ExcitedStatesSolver` catches `bond=`/`molecule=` via `kwargs.pop`.
- **Return types disagree.** Most return `dict` with `energy`/`converged`, but `PhysicsVQE` →
  `PhysicsVQEResult`, `VarQITE` → `VarQITEResult`, `qEOMVQE` → `qEOMResult` (whose energy key is
  `ground_energy`, not `energy`), `HardwareVQE` → `HardwareVQEResult`, `SampledSubspaceVQE` → its
  own dataclass. Callers must handle `result['energy']`, `result.energy`, and `result.ground_energy`.
- **Base class is fragmented.** Only 5 of 14 inherit `BaseSolver` (VQE, CI, DeterministicCI,
  Lanczos, ExcitedStates). The other 7 standalones re-implement backend wiring and miss the shared
  analysis tools and `print_summary()`.
- **Backends are ad-hoc.** No `BaseBackend`. `statevector` is a boolean flag (`_use_statevector`);
  cloud backends are objects with disagreeing methods (`run_circuit` vs `run_batch` vs `run_session`).
  `BaseSolver._init_backend()` is a string-dispatch `if/elif`.

What is **already clean** and is NOT the problem: the chemistry primitives the solvers *consume*
all have proper ABCs — `MolecularHamiltonian`, `BaseMapper`, `BaseAnsatz`. This effort does not
touch them.

## Goal

Standardize the solver *envelope* — `in → solve → out → backend` — behind one protocol, while
leaving each solver's algorithm 100% private to its subclass. "Same protocol, different
methodology."

## Non-goals

- No change to solver mathematics or numerical results (energies must be unchanged).
- No change to `MolecularHamiltonian` / `BaseMapper` / `BaseAnsatz`.
- `SubspaceHamiltonianBuilder` (a projection utility taking a raw `SparsePauliOp`) and the
  deprecated `SQDSolver` alias are out of scope.

## Rollout decision

**Clean break on `feat/solver-protocol`.** No compatibility shims. The full migration lands on the
branch, is validated against a golden-value regression suite, then merges to `main` accompanied by
a migration document for the three external consumers: **kanad-app, compute, rocm-planck**.

## Scope

In scope (all classes that produce an energy/states result):

| Solver | Current base | Current return |
|---|---|---|
| VQESolver | BaseSolver | dict |
| CISolver | VQESolver (wrapper) | dict |
| DeterministicCI | BaseSolver | dict |
| SamplingSQDSolver | standalone | dict |
| PhysicsVQE | standalone | `PhysicsVQEResult` |
| LanczosSolver | BaseSolver | dict |
| ExcitedStatesSolver | BaseSolver | dict |
| SmartSolver | standalone | dict |
| VarQITESolver | standalone | `VarQITEResult` |
| qEOMVQE | standalone | `qEOMResult` |
| HardwareVQE | standalone | `HardwareVQEResult` |
| SampledSubspaceVQE | standalone | `SSVQEResult` |

`CISolver` stays a thin wrapper and conforms automatically via its `VQESolver` parent.

Out of scope: `SubspaceHamiltonianBuilder`, deprecated `SQDSolver` alias.

## Design

### 1. `BaseSolver` — the protocol contract

Every in-scope solver inherits one ABC. The contract fixes exactly four things and nothing about
methodology.

```python
class BaseSolver(ABC):
    def __init__(self, system, *, backend="statevector", enable_analysis=True, **method_kwargs):
        self._resolve_system(system)        # -> self.hamiltonian, self.molecule, self.bond
        self.backend = make_backend(backend, **method_kwargs)   # BaseBackend instance
        # remaining method_kwargs consumed by the subclass

    @classmethod
    def from_hamiltonian(cls, hamiltonian, **kw) -> "BaseSolver": ...

    @classmethod
    def from_bond(cls, bond, **kw) -> "BaseSolver": ...

    @abstractmethod
    def solve(self, **kwargs) -> "SolverResult": ...
```

- `system` accepts `Bond | Molecule | Hamiltonian`. A single normalizer `_resolve_system` replaces
  the ~14 hand-rolled type-detection blocks, setting `self.hamiltonian`, `self.molecule`,
  `self.bond` (latter two `None` when not derivable).
- `from_hamiltonian` / `from_bond` express intent explicitly for call sites that prefer it.
- Shared services (analysis-tool init, `print_summary()`) live in the base, so the 7 standalones
  gain them on migration.
- Each subclass's algorithm stays entirely private; the ABC only governs the envelope.

### 2. `SolverResult` — the return contract

```python
@dataclass(frozen=True)
class SolverResult:
    energy: float                          # canonical energy key (resolves ground_energy/['energy'])
    converged: bool
    solver: str                            # e.g. "vqe", "physics_vqe"
    backend: str
    iterations: int | None = None
    hf_energy: float | None = None
    correlation_energy: float | None = None
    energy_history: list[float] | None = None
    states: list[float] | None = None      # excited-state energies, if any
    analysis: dict | None = None
    extra: dict = field(default_factory=dict)  # solver-specific: eigenvectors, determinants,
                                               # excitations, telemetry, penalty_history, etc.
    def to_dict(self) -> dict: ...         # JSON-safe; used by the API serializer
```

- The 5 bespoke result dataclasses (`PhysicsVQEResult`, `VarQITEResult`, `qEOMResult`,
  `HardwareVQEResult`, `SSVQEResult`) are deleted. Their core fields map onto the stable fields
  above; their unique fields move into `extra`.
- `to_dict()` returns a JSON-serializable dict. The API layer calls it instead of hand-assembling
  response dicts.

`extra` field mapping (non-exhaustive):
- VQE: `parameters`, `penalty_history`, `loss_history`, `telemetry`, `mode`
- DeterministicCI / Lanczos: `eigenvectors`, `quantum_rdm1`, `subspace_dim`
- SamplingSQD: `determinants`, `n_recovered`, `recovery_rate`
- PhysicsVQE: `excitations`, `n_evaluations`, `correlation_captured`, `cloud_job_id`
- qEOMVQE: `excitation_energies`, `h_matrix`, `s_matrix`, `eigenvectors`
- ExcitedStates: `oscillator_strengths`, `transition_dipoles`, `uv_vis_spectrum`

### 3. `BaseBackend` — the backend contract

Solvers need exactly two operations from any backend: expectation values (VQE-family) and bitstring
sampling (SQD-family).

```python
class BaseBackend(ABC):
    @abstractmethod
    def estimate_expectation(self, circuit, observable, shots=None) -> float: ...

    @abstractmethod
    def sample(self, circuit, shots) -> dict[str, int]:   # bitstring -> counts
        ...
```

- `StatevectorBackend` becomes a real `BaseBackend` class implementing exact expectation and
  exact-probability sampling. The `_use_statevector` boolean flag is removed.
- `BlueQubitBackend`, `IBMBackend`, `IonQBackend`, `PlanckBackend` each implement the two methods by
  wrapping their existing `run_*` internals. Internals stay; only a thin conformance layer is added.
- `make_backend(name, **kw)` is the single construction point (replaces the string-dispatch in
  `_init_backend`).
- VQE-family solvers call `estimate_expectation`; SQD-family call `sample`. No solver branches on
  backend type anymore.

## Migration sequence

Ordered so each step is independently testable on the branch:

1. **Primitives.** Add `SolverResult`, `BaseBackend` + `StatevectorBackend`, `make_backend()`, and
   `_resolve_system()` on `BaseSolver`. No solver behavior changed yet.
2. **Wrap cloud backends** to `BaseBackend` (bluequbit, ibm, ionq, planck).
3. **Migrate the five BaseSolver-inheriting solvers** (VQE, CI, DeterministicCI, Lanczos,
   ExcitedStates) — closest to the target shape.
4. **Migrate the seven standalones** (PhysicsVQE, HardwareVQE, SamplingSQD, VarQITE, qEOMVQE,
   SampledSubspaceVQE, SmartSolver) — each gains the base class and loses its bespoke result
   dataclass.
5. **Rewire dispatch.** Update `builder/quantum_system.py` (`SolverRouter`) and the API
   `calculations.py` call sites (`track_solver` / `SOLVER_MAP`) to construct via the uniform
   signature and serialize via `.to_dict()`.
6. **Delete dead code.** Remove the 5 result dataclasses, the `_use_statevector` flag, and per-solver
   type-detection blocks.

## Testing strategy

- **Golden-value regression (safety net for the clean break).** Capture current energies for the
  existing matrix (H₂, LiH, H₂O, HeH⁺, BeH₂ across all in-scope solvers; baseline: 10 solvers pass on
  H₂, 205 analysis cases pass) *before* refactor. Assert numerically-equivalent energies after.
- **Protocol conformance test (parametrized over every in-scope solver):** accepts `system`; returns
  a `SolverResult`; `.to_dict()` is JSON-serializable; works through `from_hamiltonian`.
- **Backend conformance test:** every `BaseBackend` implements both operations; `StatevectorBackend`
  reproduces the previous statevector energies.

## Migration document for dependent projects

On merge, commit `docs/MIGRATION-solver-protocol.md` targeting **kanad-app, compute, rocm-planck**,
covering:

- Old → new constructor calls (per solver), including the `system` positional and `from_*`
  classmethods.
- Result access change: `result['energy']` / `result.ground_energy` → `result.energy`, or
  `result.to_dict()['energy']` for dict consumers.
- Removed dataclasses and the `extra` field mapping for previously top-level fields.
- New backend selection model (`make_backend`, `BaseBackend` ops, removal of `_use_statevector`).

## Risks

- **Numerical drift.** Mitigated by the golden-value regression gate — energies must match within
  tolerance before merge.
- **Hidden result-key consumers.** The API serializer and frontend read specific keys; the migration
  doc plus `to_dict()` parity test cover the known surface. Grep the three dependent repos' contract
  during the migration-doc step.
- **Cloud backend behavioral parity.** Wrapping `run_*` behind `estimate_expectation`/`sample` must
  preserve async/job-id semantics (BlueQubit job_id, IBM session). Conformance tests assert the wrap
  returns the same values the solvers previously consumed.
