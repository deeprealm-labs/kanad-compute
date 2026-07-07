# Migration Guide: Unified Solver Protocol

**Branch:** `feat/solver-protocol` · **Audience:** kanad-app, compute, rocm-planck (any code that
constructs a Kanad solver or reads a solver result).

This release standardizes the *envelope* of every solver — how you construct it, how you call it,
what it returns, and how it reaches a backend. **No solver mathematics changed; energies are
identical.** Only the surface changed.

---

## TL;DR — the one change you almost certainly need

Every solver's `solve()` (and `solve_local`/`solve_iterative`/`solve_hardware`/`solve_excited_states`)
now returns a **`SolverResult`** object instead of a plain dict or a per-solver dataclass.

If your code did dict access:

```python
result = solver.solve()
e = result['energy']          # ❌ TypeError: 'SolverResult' object is not subscriptable
```

pick one of:

```python
result = solver.solve()
e = result.energy             # ✅ typed attribute access (preferred in new code)

# OR, to keep existing dict-style code working with a one-token change:
result = solver.solve().to_dict()
e = result['energy']          # ✅ to_dict() returns the legacy flat dict
```

`to_dict()` returns a **flat** dict (solver-specific fields merged to the top level), so
`solver.solve().to_dict()` is a drop-in for the old return value in almost all cases.

---

## 1. Constructors: one `system` argument

Every solver now accepts the system as the **first positional argument** (a Bond, a builder
`QuantumSystem`, a `Molecule`, or a bare `Hamiltonian`), followed by keyword-only options:

```python
Solver(system, *, backend="statevector", **method_kwargs)
```

Plus explicit classmethods when you want to be unambiguous:

```python
VQESolver.from_hamiltonian(h, ...)
VQESolver.from_bond(bond, ...)
```

Before → after:

| Before | After |
|---|---|
| `VQESolver(bond=b, ...)` | `VQESolver(b, ...)` *(keyword `bond=` still accepted)* |
| `LanczosSolver(bond_or_molecule=x)` | `LanczosSolver(x)` |
| `DeterministicCI(bond, hamiltonian=h)` | `DeterministicCI(h)` or `DeterministicCI(bond)` |
| `SamplingSQDSolver(hamiltonian)` | `SamplingSQDSolver(hamiltonian)` *(unchanged — still takes a Hamiltonian)* |
| `PhysicsVQE(bond=b, molecule=m, ...)` | `PhysicsVQE(b, ...)` *(b/molecule/hamiltonian kwargs still accepted)* |

Note: arguments after `system` are now **keyword-only**. `VQESolver(bond, 'hardware_efficient')`
(second positional) no longer works — use `VQESolver(bond, ansatz_type='hardware_efficient')`.

## 2. `SolverResult`

`kanad.core.solver_result.SolverResult` — a frozen dataclass:

```python
SolverResult(
    energy,            # float, Hartree — the canonical energy (always `.energy`)
    converged,         # bool
    solver,            # str tag, e.g. "vqe", "physics_vqe", "deterministic_ci"
    backend,           # str, e.g. "statevector"
    iterations=None,
    hf_energy=None,
    correlation_energy=None,
    energy_history=None,
    states=None,       # excited-state energies (Hartree) for multi-state solvers
    analysis=None,
    extra={},          # solver-specific fields: parameters, determinants, eigenvectors,
                       # excitations, telemetry, h_matrix/s_matrix, tau_final, ...
)
```

- **Canonical energy is always `result.energy`.** In particular qEOMVQE's old `.ground_energy`
  is now `.energy` (its excited-state energies are in `.states` / `.extra['excitation_energies']`).
- Solver-specific fields live in `result.extra` (typed access) **and** at the top level of
  `result.to_dict()` (flat dict, for legacy consumers).
- `result.to_dict()` is JSON-serializable (numpy scalars/arrays/complex are coerced).

## 3. Removed result dataclasses (kept as aliases)

The per-solver result dataclasses were removed; every entry point returns a `SolverResult`. For
back-compat, the old names are kept as **aliases of `SolverResult`** so imports don't break:

`PhysicsVQEResult`, `VarQITEResult`, `VarQRTEResult`, `qEOMResult`, `HardwareVQEResult`,
`SSVQEResult` → all `= SolverResult`.

If you accessed dataclass-specific attributes, find them in `.extra`:

| Old (dataclass attribute) | New |
|---|---|
| `PhysicsVQEResult.n_evaluations` | `result.extra['n_evaluations']` (or `.to_dict()['n_evaluations']`) |
| `PhysicsVQEResult.excitations` | `result.extra['excitations']` |
| `VarQITEResult.tau_final` | `result.extra['tau_final']` |
| `qEOMResult.ground_energy` | `result.energy` |
| `qEOMResult.excited_energies` | `result.states` |
| `qEOMResult.h_matrix` / `.s_matrix` | `result.extra['h_matrix']` / `result.extra['s_matrix']` |
| `HardwareVQEResult.energy_std` | `result.extra['energy_std']` |
| `SSVQEResult.configurations` | `result.extra['configurations']` |

## 4. Backends

`statevector` is now a real backend object, not a boolean flag. Construction is centralized:

```python
from kanad.backends.factory import make_backend
be = make_backend("statevector")              # or "planck" / "bluequbit" / "ibm" / "ionq"
```

Solvers still take `backend="..."` by name and build the object internally; `solver.backend` is
now a `BaseBackend` instance and `solver.backend_name` is the string. `BaseBackend` exposes two
operations: `estimate_expectation(circuit, observable, shots=None)` and `sample(circuit, shots)`.
The legacy `_use_statevector` attribute is gone.

## 5. What did NOT change

- Numerical results (energies, convergence behavior, analysis values).
- `Hamiltonian` / `Mapper` / `Ansatz` interfaces.
- Solver methodology — UCC is still UCC, Lanczos is still Lanczos, SQD is still SQD.
- `SamplingSQDSolver` still takes a `Hamiltonian` (not a bond) as its system argument.
- The builder `QuantumSystem.solve()` still returns a plain dict (it already flattens internally).

## 6. Quick checklist for a consumer repo

1. Search for `.solve()` followed by `[`, `.get(`, or `in result` → add `.to_dict()` or switch to
   attribute access.
2. Search for `XxxResult` dataclass imports / attribute access → use `SolverResult` + `.extra`.
3. Search for second-positional constructor args → make them keyword.
4. Search for `_use_statevector` → replace with `isinstance(solver.backend, StatevectorBackend)`
   or just check `solver.backend_name == "statevector"`.
