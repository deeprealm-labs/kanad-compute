# Unified Solver Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all 12 Kanad solvers one shared envelope — `Solver(system, *, backend, **kw)` → `solve() -> SolverResult` → `BaseBackend` — without changing any solver's mathematics or numerical results.

**Architecture:** Introduce three primitives (`SolverResult`, `BaseBackend`/`StatevectorBackend`, a system normalizer on `BaseSolver`), wrap the cloud backends to the new backend interface, migrate each solver to the uniform constructor and return type using a shared `SolverResult.from_mapping()` adapter, then rewire the in-repo dispatch (`builder/quantum_system.py`, `SolverRouter` consumers) and delete the dead per-solver result dataclasses and the `_use_statevector` flag. A golden-value regression suite captured up front guarantees energies are unchanged.

**Tech Stack:** Python 3.12, pytest (+ xdist), Qiskit 2.x, PySCF, numpy. Package imports as `kanad.*` (resolved via `tests/conftest.py` symlink). Run tests with `python -m pytest`.

---

## File Structure

**New files:**
- `core/solver_result.py` — `SolverResult` dataclass + `from_mapping()` + `to_dict()`.
- `backends/base_backend.py` — `BaseBackend` ABC.
- `backends/statevector_backend.py` — `StatevectorBackend` (real class replacing the `_use_statevector` flag).
- `backends/factory.py` — `make_backend(name, **kw)`.
- `tests/unit/test_solver_result.py` — result contract tests.
- `tests/unit/test_backend_protocol.py` — backend conformance tests.
- `tests/integration/test_solver_protocol_conformance.py` — parametrized per-solver protocol tests.
- `tests/integration/test_golden_energies.py` — golden-value regression (generated in Task 1).
- `docs/MIGRATION-solver-protocol.md` — external-consumer migration doc (kanad-app, compute, rocm-planck).

**Modified files (one responsibility each):**
- `solvers/base_solver.py` — new `__init__(system, *, backend, ...)`, `_resolve_system()`, `from_hamiltonian()`/`from_bond()`, backend built via `make_backend`; `_init_backend` removed.
- `solvers/{vqe_solver,deterministic_ci,lanczos_solver,excited_states_solver,smart_solver,physics_vqe,hardware_vqe,sampling_sqd,varqite_solver,qeom_vqe,sampled_subspace_vqe}.py` — uniform constructor + `solve() -> SolverResult`.
- `solvers/ci_solver.py` — inherits VQESolver; conforms automatically (verify only).
- `backends/{bluequbit,ibm,ionq}/backend.py`, `backends/planck_adapter.py` — implement `BaseBackend`.
- `builder/quantum_system.py` — construct solvers via uniform signature, read `result.energy`/`.to_dict()`.
- `solvers/__init__.py` — drop exports of deleted result dataclasses.

---

## Conventions for every task

- Run a single test: `python -m pytest tests/path::test_name -v`
- Run the fast guard suite (used as the per-phase gate): `python -m pytest tests/unit tests/smoke -q`
- Energies are compared at **1e-6 Ha** tolerance (floating-point safe; the spec's "numerically-equivalent" gate).
- Commit after every green task. Commit message footer is the project default.

---

## Phase 0 — Baseline safety net

### Task 0: Capture golden-value energy baseline

**Files:**
- Create: `tests/integration/test_golden_energies.py`
- Create (generated artifact): `tests/integration/golden_energies.json`

- [ ] **Step 1: Write the baseline generator + regression test**

This test runs the in-scope solvers on small molecules and compares to a frozen JSON. On first run (no JSON) it writes the baseline and xfails; on later runs it asserts equality. This captures *current* numbers BEFORE any refactor.

```python
# tests/integration/test_golden_energies.py
import json, os
import pytest
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden_energies.json"
TOL = 1e-6

def _h2_bond():
    from kanad.bonds.bond_factory import BondFactory
    return BondFactory.create_bond('H', 'H', distance=0.74)

# (solver_label, callable -> energy). Kept tiny + statevector-only for speed.
def _cases():
    from kanad.solvers import (
        VQESolver, CISolver, DeterministicCI, LanczosSolver, SmartSolver,
    )
    b = _h2_bond()
    return {
        "vqe_h2": lambda: VQESolver(b, ansatz_type='hardware_efficient',
                                    optimizer='COBYLA', max_iterations=200,
                                    use_cache=False).solve()['energy'],
        "ci_h2": lambda: CISolver(b).solve()['energy'],
        "detci_h2": lambda: DeterministicCI(b).solve()['energy'],
        "lanczos_h2": lambda: LanczosSolver(b).solve()['energy'],
        "smart_h2": lambda: SmartSolver(bond=b).solve()['energy'],
    }

def _measure():
    return {k: float(fn()) for k, fn in _cases().items()}

@pytest.mark.slow
def test_golden_energies():
    measured = _measure()
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(measured, indent=2, sort_keys=True))
        pytest.xfail("Baseline written; re-run to assert against it.")
    golden = json.loads(GOLDEN.read_text())
    for k, v in golden.items():
        assert k in measured, f"missing case {k}"
        assert abs(measured[k] - v) < TOL, f"{k}: {measured[k]} vs golden {v}"
```

- [ ] **Step 2: Generate the baseline (run twice)**

Run: `python -m pytest tests/integration/test_golden_energies.py -q` (writes JSON, xfails)
Then: `python -m pytest tests/integration/test_golden_energies.py -q`
Expected: second run PASSES.

- [ ] **Step 3: Commit the baseline**

```bash
git add tests/integration/test_golden_energies.py tests/integration/golden_energies.json
git commit -m "test: golden-value energy baseline before solver-protocol refactor"
```

> NOTE: This baseline list is intentionally minimal (the 5 statevector solvers that run in seconds). After Phase 4, extend `_cases()` to cover PhysicsVQE/VarQITE/qEOM/HardwareVQE/SampledSubspaceVQE/ExcitedStates/SamplingSQD and regenerate against a saved copy of the pre-refactor numbers (capture those now in Step 2's JSON by adding the extra cases if their runtime is acceptable on this machine).

---

## Phase 1 — Primitives

### Task 1: `SolverResult`

**Files:**
- Create: `core/solver_result.py`
- Test: `tests/unit/test_solver_result.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_solver_result.py
import json
import numpy as np
import pytest
from kanad.core.solver_result import SolverResult

def test_core_fields_and_to_dict():
    r = SolverResult(energy=-1.137, converged=True, solver="vqe", backend="statevector",
                     iterations=30, hf_energy=-1.117, correlation_energy=-0.02)
    assert r.energy == -1.137
    d = r.to_dict()
    assert d["energy"] == -1.137 and d["solver"] == "vqe"
    json.dumps(d)  # must be JSON-serializable

def test_to_dict_is_numpy_safe():
    r = SolverResult(energy=-1.0, converged=True, solver="detci", backend="statevector",
                     extra={"eigenvectors": np.zeros((2, 2)), "states": np.array([-1.0, -0.5])})
    json.dumps(r.to_dict())  # numpy arrays -> lists

def test_from_mapping_splits_core_and_extra():
    legacy = {"energy": -7.88, "converged": True, "iterations": 12,
              "hf_energy": -7.86, "determinants": ["11", "10"], "recovery_rate": 0.9}
    r = SolverResult.from_mapping(legacy, solver="sqd", backend="statevector")
    assert r.energy == -7.88 and r.iterations == 12
    assert r.extra["determinants"] == ["11", "10"]
    assert "energy" not in r.extra  # core keys not duplicated into extra

def test_from_mapping_alt_energy_key():
    legacy = {"ground_energy": -2.1, "excited_energies": [-1.0]}
    r = SolverResult.from_mapping(legacy, solver="qeom", backend="statevector",
                                  energy_key="ground_energy")
    assert r.energy == -2.1
    assert r.extra["excited_energies"] == [-1.0]

def test_frozen():
    r = SolverResult(energy=-1.0, converged=True, solver="x", backend="statevector")
    with pytest.raises(Exception):
        r.energy = 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_solver_result.py -q`
Expected: FAIL (`ModuleNotFoundError: kanad.core.solver_result`).

- [ ] **Step 3: Implement `SolverResult`**

```python
# core/solver_result.py
"""Unified return type for all Kanad solvers (solver-protocol refactor, 2026-06-12)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np

# Core fields pulled out of a legacy dict by from_mapping(); everything else -> extra.
_CORE_KEYS = {
    "energy", "converged", "solver", "backend", "iterations",
    "hf_energy", "correlation_energy", "energy_history", "states", "analysis",
}

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj

@dataclass(frozen=True)
class SolverResult:
    energy: float
    converged: bool
    solver: str
    backend: str
    iterations: int | None = None
    hf_energy: float | None = None
    correlation_energy: float | None = None
    energy_history: list[float] | None = None
    states: list[float] | None = None
    analysis: dict | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data, *, solver: str, backend: str,
                     energy_key: str = "energy") -> "SolverResult":
        d = dict(data)
        energy = d.pop(energy_key)
        core = {k: d.pop(k) for k in list(d.keys()) if k in _CORE_KEYS}
        return cls(
            energy=float(energy),
            converged=bool(core.pop("converged", True)),
            solver=solver,
            backend=backend,
            iterations=core.pop("iterations", None),
            hf_energy=core.pop("hf_energy", None),
            correlation_energy=core.pop("correlation_energy", None),
            energy_history=core.pop("energy_history", None),
            states=core.pop("states", None),
            analysis=core.pop("analysis", None),
            extra=d,  # whatever remains
        )

    def to_dict(self) -> dict:
        return _jsonable(asdict(self))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_solver_result.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add core/solver_result.py tests/unit/test_solver_result.py
git commit -m "feat: SolverResult unified return type + from_mapping adapter"
```

### Task 2: `BaseBackend`, `StatevectorBackend`, `make_backend`

**Files:**
- Create: `backends/base_backend.py`, `backends/statevector_backend.py`, `backends/factory.py`
- Test: `tests/unit/test_backend_protocol.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_backend_protocol.py
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from kanad.backends.base_backend import BaseBackend
from kanad.backends.statevector_backend import StatevectorBackend
from kanad.backends.factory import make_backend

def test_factory_returns_statevector():
    be = make_backend("statevector")
    assert isinstance(be, StatevectorBackend) and isinstance(be, BaseBackend)
    assert be.name == "statevector"

def test_planck_falls_back_when_absent():
    be = make_backend("planck")          # planck optional; must not raise
    assert isinstance(be, BaseBackend)

def test_statevector_expectation_zz():
    qc = QuantumCircuit(1)               # |0> -> <Z> = +1
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    assert abs(StatevectorBackend().estimate_expectation(qc, obs) - 1.0) < 1e-9

def test_statevector_expectation_x_basis():
    qc = QuantumCircuit(1); qc.h(0)      # |+> -> <Z> = 0, <X> = 1
    z = SparsePauliOp.from_list([("Z", 1.0)])
    x = SparsePauliOp.from_list([("X", 1.0)])
    sb = StatevectorBackend()
    assert abs(sb.estimate_expectation(qc, z)) < 1e-9
    assert abs(sb.estimate_expectation(qc, x) - 1.0) < 1e-9

def test_statevector_sample_shapes():
    qc = QuantumCircuit(2); qc.h(0); qc.cx(0, 1)   # Bell -> only 00 and 11
    counts = StatevectorBackend(seed=7).sample(qc, shots=2000)
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 2000

def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        make_backend("does_not_exist")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_backend_protocol.py -q`
Expected: FAIL (`ModuleNotFoundError: kanad.backends.base_backend`).

- [ ] **Step 3: Implement the ABC**

```python
# backends/base_backend.py
"""Backend protocol: the two operations every solver needs."""
from __future__ import annotations
from abc import ABC, abstractmethod

class BaseBackend(ABC):
    name: str = "base"

    @abstractmethod
    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        """Return <circuit| observable |circuit>. shots=None means exact when supported."""

    @abstractmethod
    def sample(self, circuit, shots: int) -> dict[str, int]:
        """Return {bitstring: count}. Bitstrings are little-endian Qiskit order."""
```

- [ ] **Step 4: Implement `StatevectorBackend`**

```python
# backends/statevector_backend.py
"""Exact statevector backend (replaces the legacy `_use_statevector` bool flag)."""
from __future__ import annotations
import numpy as np
from qiskit.quantum_info import Statevector
from kanad.backends.base_backend import BaseBackend

class StatevectorBackend(BaseBackend):
    name = "statevector"

    def __init__(self, seed: int | None = None, **_ignored):
        self._rng = np.random.default_rng(seed)

    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        sv = Statevector(circuit)
        return float(np.real_if_close(sv.expectation_value(observable)))

    def sample(self, circuit, shots: int) -> dict[str, int]:
        sv = Statevector(circuit)
        probs = sv.probabilities()
        n = circuit.num_qubits
        idx = self._rng.choice(len(probs), size=shots, p=probs)
        counts: dict[str, int] = {}
        for i in idx:
            key = format(int(i), f"0{n}b")
            counts[key] = counts.get(key, 0) + 1
        return counts
```

- [ ] **Step 5: Implement `make_backend`**

```python
# backends/factory.py
"""Single construction point for backends (replaces BaseSolver._init_backend dispatch)."""
from __future__ import annotations
from kanad.backends.base_backend import BaseBackend
from kanad.backends.statevector_backend import StatevectorBackend

def make_backend(name: str, **kwargs) -> BaseBackend:
    name = (name or "statevector").lower()
    if name == "statevector":
        return StatevectorBackend(**kwargs)
    if name == "planck":
        try:
            import planck  # noqa: F401
            from kanad.backends.planck_adapter import PlanckBackend
            return PlanckBackend(**kwargs)
        except ImportError:
            return StatevectorBackend(**kwargs)  # graceful fallback (matches legacy behavior)
    if name == "bluequbit":
        from kanad.backends.bluequbit import BlueQubitBackend
        return BlueQubitBackend(**kwargs)
    if name == "ibm":
        from kanad.backends.ibm import IBMBackend
        return IBMBackend(**kwargs)
    if name == "ionq":
        from kanad.backends.ionq import IonQBackend
        return IonQBackend(**kwargs)
    raise ValueError(f"Unknown backend: {name!r}")
```

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/unit/test_backend_protocol.py -q`
Expected: PASS (6 tests). `test_planck_falls_back_when_absent` and the bluequbit/ibm/ionq imports inside `make_backend` are lazy, so this passes without those packages.

- [ ] **Step 7: Commit**

```bash
git add backends/base_backend.py backends/statevector_backend.py backends/factory.py tests/unit/test_backend_protocol.py
git commit -m "feat: BaseBackend protocol + StatevectorBackend + make_backend factory"
```

---

## Phase 2 — Conform cloud backends to `BaseBackend`

Each cloud backend keeps its existing `run_*` internals and gains the two protocol methods that delegate to them. These cannot be behaviorally tested without credentials, so tests assert the interface (subclass + methods present + signature) only, marked `hardware`.

### Task 3: BlueQubit, IBM, IonQ, Planck implement `BaseBackend`

**Files:**
- Modify: `backends/bluequbit/backend.py`, `backends/ibm/backend.py`, `backends/ionq/backend.py`, `backends/planck_adapter.py`
- Test: `tests/backends/test_cloud_backend_interface.py` (create)

- [ ] **Step 1: Write the interface test**

```python
# tests/backends/test_cloud_backend_interface.py
import inspect
import pytest
from kanad.backends.base_backend import BaseBackend

@pytest.mark.parametrize("modpath,cls", [
    ("kanad.backends.bluequbit", "BlueQubitBackend"),
    ("kanad.backends.ibm", "IBMBackend"),
    ("kanad.backends.ionq", "IonQBackend"),
    ("kanad.backends.planck_adapter", "PlanckBackend"),
])
def test_cloud_backend_conforms(modpath, cls):
    mod = __import__(modpath, fromlist=[cls])
    klass = getattr(mod, cls)
    assert issubclass(klass, BaseBackend)
    for method in ("estimate_expectation", "sample"):
        sig = inspect.signature(getattr(klass, method))
        assert "circuit" in sig.parameters
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/backends/test_cloud_backend_interface.py -q`
Expected: FAIL (`PlanckBackend` not defined / not a `BaseBackend` subclass).

- [ ] **Step 3: Make each backend subclass `BaseBackend` and add the two methods**

For `backends/bluequbit/backend.py` — add the import and base class, and implement the two methods by delegating to the existing `run_circuit`. Example shape (adapt the body to the existing `run_circuit` return contract in that file):

```python
from kanad.backends.base_backend import BaseBackend

class BlueQubitBackend(BaseBackend):   # was: class BlueQubitBackend:
    name = "bluequbit"
    # ... existing __init__ / run_circuit unchanged ...

    def estimate_expectation(self, circuit, observable, shots=None):
        # Delegate to existing execution path; compute <O> from returned statevector
        # or counts. Reuse whatever run_circuit already returns.
        result = self.run_circuit(circuit, shots=shots or 4096)
        sv = result.get("statevector")
        if sv is not None:
            from qiskit.quantum_info import Statevector
            return float(Statevector(sv).expectation_value(observable).real)
        return self._expectation_from_counts(result["counts"], observable)

    def sample(self, circuit, shots):
        result = self.run_circuit(circuit, shots=shots)
        return result["counts"]

    @staticmethod
    def _expectation_from_counts(counts, observable):
        # Diagonal (Z-basis) expectation; raise for non-diagonal observables so the
        # caller knows to provide a basis-rotated circuit.
        import numpy as np
        total = sum(counts.values())
        exp = 0.0
        for pauli, coeff in zip(observable.paulis, observable.coeffs):
            if set(pauli.to_label()) - {"I", "Z"}:
                raise NotImplementedError("counts-based expectation needs Z-basis observable")
            val = 0.0
            for bits, c in counts.items():
                parity = sum(int(b) for b, p in zip(bits[::-1], pauli.to_label()[::-1]) if p == "Z")
                val += c * (-1) ** parity
            exp += float(np.real(coeff)) * val / total
        return exp
```

Apply the same two-method pattern to `IBMBackend` (delegate to its `run_batch`/`run_session`), `IonQBackend` (its run path), and create `PlanckBackend(BaseBackend)` in `planck_adapter.py` delegating to the planck statevector core. Each sets a `name` class attribute.

> The `_expectation_from_counts` helper is identical across cloud backends — to stay DRY, put it as a module-level function in `backends/base_backend.py` named `expectation_from_counts(counts, observable)` and call it from each. (Add it there; it has no state.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/backends/test_cloud_backend_interface.py -q`
Expected: PASS (4 params). Import-time errors here mean a missing optional dep — guard the import inside the test with `pytest.importorskip` for that backend's SDK if needed.

- [ ] **Step 5: Commit**

```bash
git add backends/ tests/backends/test_cloud_backend_interface.py
git commit -m "feat: cloud backends (bluequbit/ibm/ionq/planck) implement BaseBackend"
```

---

## Phase 3 — `BaseSolver` adopts the protocol

### Task 4: New `BaseSolver` constructor + `_resolve_system` + `from_*` + backend wiring

**Files:**
- Modify: `solvers/base_solver.py:29-211` (replace `__init__` body region and delete `_init_backend`)
- Test: `tests/unit/test_base_solver_protocol.py` (create)

- [ ] **Step 1: Write the failing test (use a trivial concrete subclass)**

```python
# tests/unit/test_base_solver_protocol.py
import pytest
from kanad.solvers.base_solver import BaseSolver
from kanad.backends.base_backend import BaseBackend
from kanad.bonds.bond_factory import BondFactory

class _Dummy(BaseSolver):
    def solve(self, **kw):
        from kanad.core.solver_result import SolverResult
        return SolverResult(energy=0.0, converged=True, solver="dummy", backend=self.backend.name)

def test_accepts_bond_and_builds_backend():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = _Dummy(b)
    assert s.hamiltonian is not None
    assert isinstance(s.backend, BaseBackend) and s.backend.name == "statevector"
    assert s.bond is b

def test_accepts_hamiltonian():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = _Dummy(b.hamiltonian)
    assert s.hamiltonian is b.hamiltonian and s.bond is None

def test_from_hamiltonian_classmethod():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = _Dummy.from_hamiltonian(b.hamiltonian)
    assert s.hamiltonian is b.hamiltonian

def test_from_bond_classmethod():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = _Dummy.from_bond(b)
    assert s.bond is b

def test_backend_name_passthrough():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = _Dummy(b, backend="planck")          # planck optional -> may fall back
    assert isinstance(s.backend, BaseBackend)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_base_solver_protocol.py -q`
Expected: FAIL (`_Dummy(b)` — current base requires `bond_or_molecule` positional but does not build `self.backend`; `from_hamiltonian` missing).

- [ ] **Step 3: Replace `BaseSolver.__init__` and add the new methods**

Replace lines 29–211 (`__init__` through the end of `_init_backend`) with:

```python
    def __init__(
        self,
        system,
        *,
        backend: str = "statevector",
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        **method_kwargs,
    ):
        """Unified solver constructor.

        Args:
            system: Bond, Molecule, or MolecularHamiltonian.
            backend: backend name resolved via kanad.backends.factory.make_backend.
            method_kwargs: solver-specific params consumed by the subclass.
        """
        self.enable_analysis = enable_analysis
        self.enable_optimization = enable_optimization
        self._resolve_system(system)

        from kanad.backends.factory import make_backend
        self.backend = make_backend(backend, **method_kwargs)
        self.backend_name = self.backend.name

        if enable_analysis:
            self._init_analysis_tools()
        if enable_optimization:
            self._init_optimization_tools()
        self.results = {}
        logger.info(f"Initialized {self.__class__.__name__} for {self._bond_type} system")

    def _resolve_system(self, system):
        """Normalize Bond | Molecule | MolecularHamiltonian -> hamiltonian/molecule/bond."""
        from kanad.core.molecule import Molecule, MolecularHamiltonian
        if isinstance(system, Molecule):
            self.bond, self.molecule = None, system
            self.hamiltonian, self.atoms = system.hamiltonian, system.atoms
            self._bond_type = "molecular"
        elif isinstance(system, MolecularHamiltonian):
            self.bond, self.molecule = None, None
            self.hamiltonian, self.atoms = system, system.atoms
            self._bond_type = "molecular"
        elif hasattr(system, "hamiltonian"):
            self.bond = system
            self.hamiltonian = system.hamiltonian
            self.molecule = getattr(system, "molecule", None)
            self.atoms = getattr(system, "atoms", [])
            self._bond_type = getattr(system, "bond_type", "unknown")
        else:
            raise TypeError(
                f"Expected Bond, Molecule, or MolecularHamiltonian, got {type(system).__name__}"
            )

    @classmethod
    def from_hamiltonian(cls, hamiltonian, **kw):
        return cls(hamiltonian, **kw)

    @classmethod
    def from_bond(cls, bond, **kw):
        return cls(bond, **kw)
```

Keep `_init_analysis_tools`, `_init_optimization_tools`, and everything from `_add_analysis_to_results` onward unchanged. **Delete** the old `_init_backend` method entirely.

> `StatevectorBackend.__init__` accepts `**_ignored`, so solver-specific `method_kwargs` (e.g. `device=`, `shots=`) passed through to `make_backend` won't crash the statevector path. Cloud backends consume their own kwargs.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/unit/test_base_solver_protocol.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add solvers/base_solver.py tests/unit/test_base_solver_protocol.py
git commit -m "feat: BaseSolver adopts (system, *, backend) protocol + from_* + make_backend"
```

> After this task the 5 BaseSolver-inheriting solvers' `__init__`s break (they call `super().__init__(bond, enable_analysis, enable_optimization)` positionally and/or `self._init_backend(...)`). Tasks 5–9 fix them. Do not run the full suite green until Phase 4 completes; use each task's targeted test as the gate.

---

## Phase 4 — Migrate the 12 solvers

Each solver task does three things: (a) adapt `__init__` to call the new `super().__init__(system, *, backend=..., enable_analysis=...)` and stop calling `_init_backend`, (b) route its energy-eval/sampling through `self.backend` where it previously branched on `_use_statevector`, (c) make `solve()` return `SolverResult.from_mapping(<existing dict>, solver=..., backend=self.backend.name)`. The existing result dict is preserved verbatim and fed to the adapter, so no field is lost.

### Task 5: VQESolver (pattern-setter)

**Files:**
- Modify: `solvers/vqe_solver.py` (`__init__` ~122, the `_init_backend` call ~330, `_use_statevector` branches in `_compute_energy*`, `solve` return ~1878)
- Test: `tests/integration/test_solver_protocol_conformance.py::test_vqe` (added in Task 17; here use a focused test)
- Test: `tests/unit/test_vqe_returns_solver_result.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_vqe_returns_solver_result.py
from kanad.solvers import VQESolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_vqe_returns_solver_result_and_matches_golden():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = VQESolver(b, ansatz_type='hardware_efficient', optimizer='COBYLA',
                  max_iterations=200, use_cache=False)
    r = s.solve()
    assert isinstance(r, SolverResult)
    assert abs(r.energy - (-1.137)) < 5e-3      # H2 STO-3G ground state
    assert "parameters" in r.extra
    import json; json.dumps(r.to_dict())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_vqe_returns_solver_result.py -q`
Expected: FAIL (VQESolver `__init__` passes positional args to the new base / returns a dict).

- [ ] **Step 3: Adapt VQESolver `__init__`**

In the bond-mode branch, replace the legacy `super().__init__(bond, enable_analysis, enable_optimization)` + later `self._init_backend(backend, **kwargs)` with a single call:

```python
        # bond mode
        super().__init__(
            bond,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **backend_kwargs,   # device/shots/etc. destined for the backend
        )
```

Remove the `self._init_backend(...)` line. Keep `_init_ansatz()`, `_init_mapper()`, `_build_circuit()` calls. For the low-level components mode (`hamiltonian=`/`ansatz=`/`mapper=`), call `super().__init__(hamiltonian, backend=backend, ...)` instead of setting attributes by hand (the normalizer handles a bare hamiltonian).

- [ ] **Step 4: Replace `_use_statevector` branches**

Anywhere `_compute_energy` / `_compute_energy_quantum` branched on `if self._use_statevector:`, replace the quantum-path branch with `self.backend.estimate_expectation(bound_circuit, pauli_observable)`. The statevector path is now `StatevectorBackend.estimate_expectation`. Keep the existing exact-statevector fast path by checking `isinstance(self.backend, StatevectorBackend)` only where a perf shortcut (adjoint gradient on the raw statevector) requires it; otherwise call `self.backend.estimate_expectation`.

- [ ] **Step 5: Wrap the `solve()` return**

At the end of `solve()` (~line 1878 region), the method currently builds and returns a dict `result`. Wrap it:

```python
        return SolverResult.from_mapping(result, solver="vqe", backend=self.backend.name)
```

Add `from kanad.core.solver_result import SolverResult` at the top of the file. Keep `self.results = result` (the pre-wrap dict) if downstream code reads `self.results`.

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest tests/unit/test_vqe_returns_solver_result.py -q`
Expected: PASS.

- [ ] **Step 7: Update the golden baseline reader**

The golden test (Task 0) reads `result['energy']`. Update its `_cases()` to read `.energy`:

```python
"vqe_h2": lambda: VQESolver(b, ansatz_type='hardware_efficient', optimizer='COBYLA',
                            max_iterations=200, use_cache=False).solve().energy,
```

Run: `python -m pytest tests/integration/test_golden_energies.py -q`
Expected: PASS (energy unchanged within 1e-6).

- [ ] **Step 8: Commit**

```bash
git add solvers/vqe_solver.py tests/unit/test_vqe_returns_solver_result.py tests/integration/test_golden_energies.py
git commit -m "feat: VQESolver on solver protocol (SolverResult + backend object)"
```

### Task 6: CISolver (verify-only)

**Files:**
- Modify (if needed): `solvers/ci_solver.py`
- Test: `tests/unit/test_ci_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_ci_returns_solver_result.py
from kanad.solvers import CISolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_ci_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = CISolver(b).solve()
    assert isinstance(r, SolverResult)
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_ci_returns_solver_result.py -q`
Expected: PASS if CISolver's `super().__init__(*args, mode='hivqe', **kwargs)` forwards cleanly to the new VQESolver signature. If it FAILS because it passes `bond` positionally plus disallowed kwargs, change its `__init__` to forward `system` and `backend` explicitly:

```python
    def __init__(self, system=None, **kwargs):
        kwargs.pop('mode', None)
        super().__init__(system, mode='hivqe', **kwargs)
```

- [ ] **Step 3: Commit**

```bash
git add solvers/ci_solver.py tests/unit/test_ci_returns_solver_result.py
git commit -m "test: CISolver conforms to solver protocol via VQESolver parent"
```

### Task 7: DeterministicCI

**Files:**
- Modify: `solvers/deterministic_ci.py` (`__init__` ~22, `_init_backend` call ~132, `solve` return ~1082)
- Test: `tests/unit/test_detci_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_detci_returns_solver_result.py
from kanad.solvers import DeterministicCI
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_detci_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = DeterministicCI(b).solve(n_states=2)
    assert isinstance(r, SolverResult)
    assert "eigenvectors" in r.extra
    assert r.states is not None        # excited-state energies surfaced
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_detci_returns_solver_result.py -q`
Expected: FAIL.

- [ ] **Step 3: Adapt `__init__`**

Replace the three input paths with one `super().__init__` call. The current class accepts `bond_or_molecule` plus `hamiltonian=`/`molecule=` aliases; collapse to:

```python
    def __init__(self, system=None, *, subspace_dim=10, circuit_depth=3,
                 backend="statevector", enable_analysis=True, enable_optimization=True,
                 random_seed=None, experiment_id=None, hamiltonian=None, molecule=None, **kwargs):
        system = system or hamiltonian or molecule
        super().__init__(system, backend=backend, enable_analysis=enable_analysis,
                         enable_optimization=enable_optimization)
        self.subspace_dim = subspace_dim
        self.circuit_depth = circuit_depth
        self.random_seed = random_seed
        self.experiment_id = experiment_id
```

Remove the `self._init_backend(self.backend, **kwargs)` line.

- [ ] **Step 4: Wrap the `solve()` return**

The current `solve()` returns a dict with `energies`, `eigenvectors`, `ground_state_energy`, `energy`, etc. The dict already has `energy`. Surface excited states into `states`:

```python
        result["states"] = list(result.get("excited_state_energies", []))
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="deterministic_ci",
                                         backend=self.backend.name)
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/unit/test_detci_returns_solver_result.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add solvers/deterministic_ci.py tests/unit/test_detci_returns_solver_result.py
git commit -m "feat: DeterministicCI on solver protocol"
```

### Task 8: LanczosSolver

**Files:**
- Modify: `solvers/lanczos_solver.py` (`__init__` ~32, `_init_backend` call ~112, `solve` return ~329)
- Test: `tests/unit/test_lanczos_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_lanczos_returns_solver_result.py
from kanad.solvers import LanczosSolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_lanczos_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = LanczosSolver(b).solve()
    assert isinstance(r, SolverResult)
    assert "eigenvectors" in r.extra
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_lanczos_returns_solver_result.py -q`
Expected: FAIL.

- [ ] **Step 3: Adapt `__init__`** — change `super().__init__(bond_or_molecule, ...)` positional + `self._init_backend(**kwargs)` to:

```python
    def __init__(self, system, *, krylov_dim=15, n_states=3, initial_state=None,
                 backend="statevector", enable_analysis=True, enable_optimization=True,
                 random_seed=None, reorthogonalize=True, experiment_id=None, **kwargs):
        super().__init__(system, backend=backend, enable_analysis=enable_analysis,
                         enable_optimization=enable_optimization)
        self.krylov_dim = krylov_dim
        self.n_states = n_states
        self.initial_state = initial_state
        self.random_seed = random_seed
        self.reorthogonalize = reorthogonalize
        self.experiment_id = experiment_id
```

Remove the `_init_backend` call.

- [ ] **Step 4: Wrap the `solve()` return**

```python
        result["states"] = list(result.get("excited_state_energies",
                                            result.get("energies", []))[1:])
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="lanczos", backend=self.backend.name)
```

- [ ] **Step 5: Run / Step 6: Commit**

Run: `python -m pytest tests/unit/test_lanczos_returns_solver_result.py -q` → PASS
```bash
git add solvers/lanczos_solver.py tests/unit/test_lanczos_returns_solver_result.py
git commit -m "feat: LanczosSolver on solver protocol"
```

### Task 9: ExcitedStatesSolver

**Files:**
- Modify: `solvers/excited_states_solver.py` (`__init__` ~17, `solve` return ~106)
- Test: `tests/unit/test_excited_states_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_excited_states_returns_solver_result.py
from kanad.solvers import ExcitedStatesSolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_excited_states_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = ExcitedStatesSolver(b, method='cis', n_states=3).solve()
    assert isinstance(r, SolverResult)
    assert "oscillator_strengths" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Adapt `__init__`** — this class catches `bond=`/`molecule=` via `kwargs.pop`. Collapse to the `system` arg:

```python
    def __init__(self, system=None, *, method='cis', n_states=5,
                 enable_analysis=True, enable_optimization=False,
                 experiment_id=None, vqe_callback=None, **kwargs):
        system = system or kwargs.pop('bond', None) or kwargs.pop('molecule', None)
        super().__init__(system, enable_analysis=enable_analysis,
                         enable_optimization=enable_optimization)
        self.method = method
        self.n_states = n_states
        self.experiment_id = experiment_id
        self.vqe_callback = vqe_callback
```

(This solver is classical — no backend kwarg needed; base defaults to statevector.)

- [ ] **Step 4: Wrap the `solve()` return**

```python
        result["states"] = list(result.get("excitation_energies", []))
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="excited_states",
                                         backend=self.backend.name)
```

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/excited_states_solver.py tests/unit/test_excited_states_returns_solver_result.py
git commit -m "feat: ExcitedStatesSolver on solver protocol"
```

### Task 10: SmartSolver

**Files:**
- Modify: `solvers/smart_solver.py` (`__init__` ~35, `solve` return ~129)
- Test: `tests/unit/test_smart_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_smart_returns_solver_result.py
from kanad.solvers import SmartSolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_smart_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = SmartSolver(bond=b).solve()
    assert isinstance(r, SolverResult)
    assert r.extra.get("method") in {"classical_fci", "classical_approx", "vqe"}
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Adapt `__init__`** — SmartSolver is standalone (no BaseSolver). Make it inherit BaseSolver:

```python
class SmartSolver(BaseSolver):
    def __init__(self, system=None, *, force_method=None, verbose=True,
                 bond=None, hamiltonian=None, **kwargs):
        system = system or bond or hamiltonian
        super().__init__(system, enable_analysis=False, enable_optimization=False)
        self.force_method = force_method
        self.verbose = verbose
        self.n_qubits = 2 * self.hamiltonian.n_orbitals
```

Add `from kanad.solvers.base_solver import BaseSolver` import. Keep the dispatch logic in `solve()` but, where it currently constructs an inner `VQESolver`, pass `self.bond or self.hamiltonian` and read the inner result via `.energy` (it now returns a `SolverResult`).

- [ ] **Step 4: Wrap the `solve()` return**

```python
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="smart", backend=self.backend.name)
```

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/smart_solver.py tests/unit/test_smart_returns_solver_result.py
git commit -m "feat: SmartSolver inherits BaseSolver + returns SolverResult"
```

### Task 11: PhysicsVQE (delete `PhysicsVQEResult`)

**Files:**
- Modify: `solvers/physics_vqe.py` (`@dataclass PhysicsVQEResult` ~40, `class PhysicsVQE` ~55, `solve` ~943, `solve_physics_vqe` ~1084)
- Test: `tests/unit/test_physics_vqe_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_physics_vqe_returns_solver_result.py
from kanad.solvers import PhysicsVQE
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_physics_vqe_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = PhysicsVQE(bond=b, max_excitations=5).solve()
    assert isinstance(r, SolverResult)
    assert abs(r.energy - (-1.137)) < 5e-3
    assert "excitations" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Make PhysicsVQE inherit BaseSolver**

```python
class PhysicsVQE(BaseSolver):
    def __init__(self, system=None, *, bond=None, molecule=None, hamiltonian=None,
                 pyscf_mol=None, max_excitations=5, frozen_core=True, triple_bond_mode=False,
                 amplitude_threshold=None, backend="statevector", include_singles=None,
                 cloud_credentials=None, **kwargs):
        resolved = system or bond or molecule or hamiltonian
        if resolved is not None:
            super().__init__(resolved, backend=backend, enable_analysis=False,
                             enable_optimization=False)
        else:
            # pyscf-only path: keep existing manual setup, set a backend object
            from kanad.backends.factory import make_backend
            self.backend = make_backend(backend)
            self.backend_name = self.backend.name
        # ... preserve existing pyscf_mol/pyscf_mf handling and the rest of __init__ ...
        self.max_excitations = max_excitations
        # (retain all other existing attribute assignments)
```

Add `from kanad.solvers.base_solver import BaseSolver`.

- [ ] **Step 4: Convert the result**

`solve()` currently returns a `PhysicsVQEResult`. Build the same data as a dict and wrap it; delete the `@dataclass PhysicsVQEResult`:

```python
        from dataclasses import asdict
        from kanad.core.solver_result import SolverResult
        payload = {
            "energy": energy, "converged": converged, "iterations": n_evaluations,
            "hf_energy": hf_energy, "energy_history": energy_history,
            "excitations": excitations, "n_evaluations": n_evaluations,
            "correlation_captured": correlation_captured, "fci_energy": fci_energy,
            "parameters": parameters, "cloud_job_id": cloud_job_id,
        }
        return SolverResult.from_mapping(payload, solver="physics_vqe", backend=self.backend.name)
```

Update the `solve_physics_vqe(...)` module function's return annotation to `SolverResult` and have it return `PhysicsVQE(...).solve()`.

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/physics_vqe.py tests/unit/test_physics_vqe_returns_solver_result.py
git commit -m "feat: PhysicsVQE on solver protocol; remove PhysicsVQEResult"
```

### Task 12: HardwareVQE (delete `HardwareVQEResult`, add canonical `solve()`)

**Files:**
- Modify: `solvers/hardware_vqe.py` (`@dataclass HardwareVQEResult` ~34, class ~48, `solve_local` ~521, `solve_hardware` ~633)
- Test: `tests/unit/test_hardware_vqe_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_hardware_vqe_returns_solver_result.py
from kanad.solvers import HardwareVQE
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_hardware_vqe_local_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = HardwareVQE(bond=b, n_layers=2).solve()      # solve() == local path
    assert isinstance(r, SolverResult)
```

- [ ] **Step 2: Run to verify it fails** → FAIL (no `solve()`; returns dataclass).

- [ ] **Step 3: Inherit BaseSolver, add `solve()` dispatch**

```python
class HardwareVQE(BaseSolver):
    def __init__(self, system=None, *, bond=None, molecule=None, hamiltonian=None,
                 pyscf_mol=None, circuit_type='auto', n_layers=2, max_excitations=5,
                 frozen_core=True, optimizer='cobyla', shots=4096, backend="statevector",
                 **kwargs):
        resolved = system or bond or molecule or hamiltonian
        super().__init__(resolved, backend=backend, enable_analysis=False,
                         enable_optimization=False)
        # ... preserve existing attribute setup ...

    def solve(self, **kwargs):
        result = self._solve_local_impl(**kwargs)   # rename old solve_local body
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="hardware_vqe",
                                         backend=self.backend.name)
```

Rename the existing `solve_local` body to `_solve_local_impl` returning a **dict** (convert the `HardwareVQEResult` construction to a dict literal with the same fields). Keep `solve_hardware(backend=...)` but have it also return a `SolverResult` via `from_mapping`. Delete `@dataclass HardwareVQEResult`.

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add solvers/hardware_vqe.py tests/unit/test_hardware_vqe_returns_solver_result.py
git commit -m "feat: HardwareVQE on solver protocol; canonical solve(); remove HardwareVQEResult"
```

### Task 13: SamplingSQDSolver (sampling backend path)

**Files:**
- Modify: `solvers/sampling_sqd.py` (`class SamplingSQDSolver` ~307, `solve` ~556)
- Test: `tests/unit/test_sampling_sqd_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_sampling_sqd_returns_solver_result.py
from kanad.solvers import SamplingSQDSolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_sampling_sqd_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    s = SamplingSQDSolver(b.hamiltonian, n_samples=2000)
    r = s.solve()           # ansatz_circuit optional; uses internal default/HF
    assert isinstance(r, SolverResult)
    assert "determinants" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Inherit BaseSolver, route sampling through `self.backend`**

SamplingSQD is standalone and takes `hamiltonian` positionally. Make it conform while keeping `hamiltonian` as the accepted system:

```python
class SamplingSQDSolver(BaseSolver):
    def __init__(self, hamiltonian, *, n_samples=10000, backend="statevector",
                 target_sz=None, random_seed=None, recover_configurations=True,
                 ci_backend="pyscf", **kwargs):
        super().__init__(hamiltonian, backend=backend, enable_analysis=False,
                         enable_optimization=False)
        self.n_samples = n_samples
        # ... preserve remaining attribute setup ...
```

Where `solve()` currently samples the circuit (statevector probabilities or shot counts), replace the sampling call with `self.backend.sample(ansatz_circuit, shots=self.n_samples)`. This unifies the statevector/qasm split behind the backend.

- [ ] **Step 4: Wrap the `solve()` return**

```python
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="sampling_sqd",
                                         backend=self.backend.name)
```

Do the same `from_mapping` wrap at the end of `solve_iterative` and `solve_excited_states` (with `solver="sampling_sqd"`), surfacing excited energies into `states`.

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/sampling_sqd.py tests/unit/test_sampling_sqd_returns_solver_result.py
git commit -m "feat: SamplingSQDSolver on solver protocol; sampling via backend.sample"
```

### Task 14: VarQITESolver (delete `VarQITEResult`/`VarQRTEResult`)

**Files:**
- Modify: `solvers/varqite_solver.py` (dataclasses ~41/55, class ~65, `solve` ~304)
- Test: `tests/unit/test_varqite_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_varqite_returns_solver_result.py
from kanad.solvers import VarQITESolver
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_varqite_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = VarQITESolver(b, ansatz_type='hardware_efficient').solve()
    assert isinstance(r, SolverResult)
    assert "tau_final" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Inherit BaseSolver**

```python
class VarQITESolver(BaseSolver):
    def __init__(self, system, *, ansatz_type='hardware_efficient', backend="statevector",
                 regularization=1e-2, convergence_threshold=1e-6, **kwargs):
        super().__init__(system, backend=backend, enable_analysis=False,
                         enable_optimization=False)
        self.ansatz_type = ansatz_type
        self.regularization = regularization
        self.convergence_threshold = convergence_threshold
```

- [ ] **Step 4: Convert the result** — build a dict in `solve()` mirroring `VarQITEResult` fields and wrap; delete both dataclasses:

```python
        payload = {
            "energy": energy, "converged": converged, "iterations": n_steps,
            "energy_history": energy_history, "parameters": optimal_parameters,
            "tau_final": tau_final, "energy_variance": energy_variance, "method": "varqite",
        }
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(payload, solver="varqite", backend=self.backend.name)
```

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/varqite_solver.py tests/unit/test_varqite_returns_solver_result.py
git commit -m "feat: VarQITESolver on solver protocol; remove VarQITEResult/VarQRTEResult"
```

### Task 15: qEOMVQE (delete `qEOMResult`; energy key is `ground_energy`)

**Files:**
- Modify: `solvers/qeom_vqe.py` (dataclass ~41, class ~55, `solve` ~390)
- Test: `tests/unit/test_qeom_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_qeom_returns_solver_result.py
from kanad.solvers import qEOMVQE
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_qeom_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = qEOMVQE(b, n_states=3).solve()
    assert isinstance(r, SolverResult)
    assert r.states is not None and "h_matrix" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Inherit BaseSolver**

```python
class qEOMVQE(BaseSolver):
    def __init__(self, system, *, n_states=3, include_singles=True, include_doubles=True,
                 backend="statevector", vqe_max_iterations=500, **kwargs):
        super().__init__(system, backend=backend, enable_analysis=False,
                         enable_optimization=False)
        self.n_states = n_states
        self.include_singles = include_singles
        self.include_doubles = include_doubles
        self.vqe_max_iterations = vqe_max_iterations
```

- [ ] **Step 4: Convert the result** — note the energy lives in `ground_energy`; use `energy_key`:

```python
        payload = {
            "ground_energy": ground_energy, "converged": True,
            "states": list(excited_energies), "excitation_energies": list(excitation_energies),
            "eigenvectors": eigenvectors, "h_matrix": h_matrix, "s_matrix": s_matrix,
            "n_excitations": n_excitations, "method": "qeom-vqe",
        }
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(payload, solver="qeom_vqe",
                                         backend=self.backend.name, energy_key="ground_energy")
```

Delete `@dataclass qEOMResult`.

- [ ] **Step 5: Run** → PASS. **Step 6: Commit**

```bash
git add solvers/qeom_vqe.py tests/unit/test_qeom_returns_solver_result.py
git commit -m "feat: qEOMVQE on solver protocol; remove qEOMResult"
```

### Task 16: SampledSubspaceVQE (delete `SSVQEResult`, canonical `solve()`)

**Files:**
- Modify: `solvers/sampled_subspace_vqe.py` (dataclass ~30, class ~43, `solve_local` ~465, `solve_ibm` ~566; `HybridSubspaceVQE` ~669 keep)
- Test: `tests/unit/test_ssvqe_returns_solver_result.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/unit/test_ssvqe_returns_solver_result.py
from kanad.solvers import SampledSubspaceVQE
from kanad.core.solver_result import SolverResult
from kanad.bonds.bond_factory import BondFactory

def test_ssvqe_conforms():
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    r = SampledSubspaceVQE(b, n_layers=2, n_shots=2000, max_configs=10).solve()
    assert isinstance(r, SolverResult)
    assert "configurations" in r.extra
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Inherit BaseSolver, add `solve()`**

```python
class SampledSubspaceVQE(BaseSolver):
    def __init__(self, system, *, n_layers=2, n_shots=10000, max_configs=20,
                 backend="statevector", **kwargs):
        super().__init__(system, backend=backend, enable_analysis=False,
                         enable_optimization=False)
        self.n_layers = n_layers
        self.n_shots = n_shots
        self.max_configs = max_configs

    def solve(self, **kwargs):
        result = self._solve_local_impl(**kwargs)   # rename old solve_local body -> dict
        from kanad.core.solver_result import SolverResult
        return SolverResult.from_mapping(result, solver="sampled_subspace_vqe",
                                         backend=self.backend.name)
```

Rename `solve_local`'s body to `_solve_local_impl` returning a dict (convert the `SSVQEResult` literal to a dict). Delete `@dataclass SSVQEResult`. Leave `HybridSubspaceVQE` untouched beyond imports (it has its own `solve()`; if a downstream consumer needs it conformed, that's a follow-up — note it in the migration doc).

- [ ] **Step 4: Run** → PASS. **Step 5: Commit**

```bash
git add solvers/sampled_subspace_vqe.py tests/unit/test_ssvqe_returns_solver_result.py
git commit -m "feat: SampledSubspaceVQE on solver protocol; remove SSVQEResult"
```

### Task 17: Parametrized protocol conformance test

**Files:**
- Create: `tests/integration/test_solver_protocol_conformance.py`

- [ ] **Step 1: Write the conformance matrix**

```python
# tests/integration/test_solver_protocol_conformance.py
import json
import pytest
from kanad.core.solver_result import SolverResult
from kanad.backends.base_backend import BaseBackend
from kanad.bonds.bond_factory import BondFactory
from kanad.solvers import (
    VQESolver, CISolver, DeterministicCI, LanczosSolver, ExcitedStatesSolver,
    SmartSolver, PhysicsVQE, HardwareVQE, SamplingSQDSolver, VarQITESolver,
    qEOMVQE, SampledSubspaceVQE,
)

def _b():
    return BondFactory.create_bond('H', 'H', distance=0.74)

# (label, constructor thunk, solve thunk)
CASES = [
    ("vqe", lambda b: VQESolver(b, optimizer='COBYLA', max_iterations=100, use_cache=False)),
    ("ci", lambda b: CISolver(b)),
    ("detci", lambda b: DeterministicCI(b)),
    ("lanczos", lambda b: LanczosSolver(b)),
    ("excited", lambda b: ExcitedStatesSolver(b, method='cis', n_states=3)),
    ("smart", lambda b: SmartSolver(bond=b)),
    ("physics", lambda b: PhysicsVQE(bond=b)),
    ("hardware", lambda b: HardwareVQE(bond=b)),
    ("varqite", lambda b: VarQITESolver(b)),
    ("qeom", lambda b: qEOMVQE(b, n_states=3)),
    ("ssvqe", lambda b: SampledSubspaceVQE(b, n_shots=2000, max_configs=10)),
]

@pytest.mark.parametrize("label,ctor", CASES, ids=[c[0] for c in CASES])
def test_solver_conforms(label, ctor):
    s = ctor(_b())
    assert isinstance(s.backend, BaseBackend)
    r = s.solve()
    assert isinstance(r, SolverResult), f"{label} did not return SolverResult"
    assert r.solver and r.backend == "statevector"
    json.dumps(r.to_dict())                 # JSON-serializable
    assert isinstance(r.energy, float)

def test_sampling_sqd_conforms():
    s = SamplingSQDSolver(_b().hamiltonian, n_samples=2000)
    r = s.solve()
    assert isinstance(r, SolverResult)

@pytest.mark.parametrize("label,ctor", [("vqe", lambda b: VQESolver(b, use_cache=False))],
                         ids=["vqe"])
def test_from_hamiltonian_entrypoint(label, ctor):
    b = _b()
    s = VQESolver.from_hamiltonian(b.hamiltonian, use_cache=False)
    assert s.hamiltonian is b.hamiltonian
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/integration/test_solver_protocol_conformance.py -q`
Expected: PASS (all solvers conform). Fix any solver that fails here before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_solver_protocol_conformance.py
git commit -m "test: parametrized solver-protocol conformance across all 12 solvers"
```

---

## Phase 5 — Rewire in-repo consumers + cleanup

### Task 18: Update `builder/quantum_system.py` dispatch

**Files:**
- Modify: `builder/quantum_system.py` (`_solve_vqe` ~302-320, `_solve_sqd` ~334-357, `_solve_ci` consumers, and `res[...]` reads)
- Test: `tests/integration/test_builder_dispatch.py` (create)

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_builder_dispatch.py
from kanad.builder.quantum_system import QuantumSystem   # adjust import to the real entry
from kanad.bonds.bond_factory import BondFactory

def test_builder_auto_solves_h2():
    # Build via the same path the builder uses; assert an energy comes back.
    b = BondFactory.create_bond('H', 'H', distance=0.74)
    # Minimal smoke: the VQE route returns a result the builder can read.
    from kanad.solvers import VQESolver
    r = VQESolver(b, optimizer='COBYLA', max_iterations=100, use_cache=False).solve()
    assert abs(r.energy - (-1.137)) < 5e-3
```

> The exact `QuantumSystem` construction differs per builder API; if the lightweight smoke above is insufficient, replace it with the builder's real entry once confirmed by reading `builder/quantum_system.py:47-90`.

- [ ] **Step 2: Update `_solve_vqe` / `_solve_sqd` reads**

In `_solve_vqe` (~320): `res = solver.solve(**solve_kwargs)` now returns a `SolverResult`. Replace every subsequent `res['energy']` / `res[...]` with `res.energy` / `res.extra[...]`, or convert once: `res = solver.solve(**solve_kwargs).to_dict()` if the surrounding code expects a dict. Choose `.to_dict()` for the smallest diff. Apply the same to `_solve_sqd` (~357) and any `res[...]` consumers in `solve()` (~64-90) and `reactions`/`dynamics` call sites found via grep.

- [ ] **Step 3: Grep for stale result-dict access and fix**

Run: `grep -rn "\.solve(" --include=*.py builder reactions dynamics spectroscopy | grep -v test`
For each call site, ensure it reads `.energy`/`.to_dict()` rather than subscripting the `SolverResult`. Fix each.

- [ ] **Step 4: Run the smoke + builder test**

Run: `python -m pytest tests/integration/test_builder_dispatch.py tests/smoke -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add builder/quantum_system.py tests/integration/test_builder_dispatch.py reactions dynamics spectroscopy
git commit -m "refactor: in-repo consumers read SolverResult (.energy/.to_dict)"
```

### Task 19: Purge dead code + exports

**Files:**
- Modify: `solvers/__init__.py`, any module still importing the deleted dataclasses
- Modify: `solvers/base_solver.py` (confirm `_init_backend` and `_use_statevector` fully gone)

- [ ] **Step 1: Grep for the deleted symbols**

Run:
```bash
grep -rn "PhysicsVQEResult\|VarQITEResult\|VarQRTEResult\|qEOMResult\|HardwareVQEResult\|SSVQEResult\|_use_statevector\|_init_backend" --include=*.py . | grep -v '\.venv'
```
Expected: only matches inside tests you intend to keep, or none. Remove every remaining import/reference (e.g. drop these from `solvers/__init__.py` exports).

- [ ] **Step 2: Run the unit + smoke gate**

Run: `python -m pytest tests/unit tests/smoke -q`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add solvers/__init__.py solvers/base_solver.py
git commit -m "chore: remove dead result dataclasses, _use_statevector flag, _init_backend"
```

### Task 20: Full regression + golden energies

- [ ] **Step 1: Run the golden energy gate**

Run: `python -m pytest tests/integration/test_golden_energies.py -q`
Expected: PASS — energies unchanged within 1e-6 of the Phase 0 baseline. If any drift, the refactor changed numerics; bisect the offending solver task.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -x`
Expected: PASS (excluding pre-existing `hardware`/credentialed tests, which require external accounts — deselect with `-m "not hardware"` if needed).

- [ ] **Step 3: Commit any test fixes**

```bash
git add -A && git commit -m "test: full regression green on solver protocol"
```

---

## Phase 6 — Migration document

### Task 21: Write `docs/MIGRATION-solver-protocol.md`

**Files:**
- Create: `docs/MIGRATION-solver-protocol.md`

- [ ] **Step 1: Write the migration doc**

Content (fill each section with concrete before/after from the tasks above):

```markdown
# Migration: Unified Solver Protocol

Audience: kanad-app, compute, rocm-planck.

## What changed
- Every solver: `Solver(system, *, backend="statevector", **method_kwargs)`.
  `system` is a Bond, Molecule, or Hamiltonian. Also: `Solver.from_hamiltonian(h)`,
  `Solver.from_bond(b)`.
- Every `solve()` returns a `kanad.core.solver_result.SolverResult` (frozen dataclass).
  Energy is always `result.energy`. JSON via `result.to_dict()`.
- Removed result dataclasses: PhysicsVQEResult, VarQITEResult, VarQRTEResult, qEOMResult,
  HardwareVQEResult, SSVQEResult. Their unique fields now live in `result.extra`.
- Backends are objects implementing `BaseBackend` (`estimate_expectation`, `sample`).
  `statevector` is `StatevectorBackend`; the `_use_statevector` flag is gone.

## Call-site changes (before -> after)
- `PhysicsVQE(bond=b).solve().energy` (was `.energy` on PhysicsVQEResult — unchanged attr,
  new type).
- `qEOMVQE(b).solve()` — energy moved from `.ground_energy` to `.energy`; excited
  energies now in `.states` (and `.extra["excitation_energies"]`).
- Dict consumers: `result['energy']` -> `result.energy` OR `result.to_dict()['energy']`.
- `extra` field map: <list each solver's extra keys from the tasks>.

## Backend selection
- `make_backend("bluequbit", device=...)` etc. Solvers accept `backend="bluequbit"` and
  forward kwargs.
```

- [ ] **Step 2: Commit**

```bash
git add docs/MIGRATION-solver-protocol.md
git commit -m "docs: migration guide for solver protocol (kanad-app/compute/rocm-planck)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** SolverResult (Task 1), BaseBackend/StatevectorBackend/make_backend (Task 2), cloud-backend conformance (Task 3), BaseSolver normalizer + `from_*` (Task 4), all 12 solvers migrated (Tasks 5–16), protocol conformance test (Task 17), dispatch rewire (Task 18), dead-code purge (Task 19), golden regression (Tasks 0 + 20), migration doc (Task 21). Every spec section maps to a task.
- **Placeholder scan:** Two spots intentionally defer to the engineer reading real code (builder entry construction in Task 18, exact `run_*` return contract in Task 3) — both are flagged with the exact file:line to read, not vague "handle it" instructions. No TODO/TBD left in code steps.
- **Type consistency:** `SolverResult.from_mapping(data, *, solver, backend, energy_key="energy")` and `.to_dict()` are used with that exact signature in every solver task. `make_backend(name, **kwargs)` and `BaseBackend.estimate_expectation/sample` signatures are consistent across Tasks 2, 3, 4, and the solver tasks.
```
