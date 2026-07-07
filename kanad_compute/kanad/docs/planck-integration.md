# Planck GPU backend integration (DeepRealm-internal)

How kanad's VQE hot path offloads to the **public** `rocm-planck` core via
`backend='planck'`. This document is the private counterpart to rocm-planck's
Phase 0 plan; it captures kanad-side facts so the public repo never has to quote
kanad internals.

> **IP boundary.** `rocm-planck` is public and must never import `kanad`. The
> dependency is one-directional: `kanad` → `planck`. All glue lives here in the
> kanad repo (`kanad/backends/planck_adapter.py`, the `backend='planck'` branches
> in `solvers/vqe_solver.py` + `solvers/base_solver.py`, and
> `tests/test_planck_backend.py`).

## What planck accelerates (the three hot-path ops)

1. **Gate apply** — `Statevector.from_instruction(bound_circuit)` →
   `planck.Circuit.from_qiskit(bound_circuit).run()` (builds |ψ⟩ on GPU).
2. **Expectation** — `statevector.expectation_value(sparse_pauli_op).real` →
   `planck.PauliSum.from_qiskit(spo).expectation(planck_sv)` (term-by-term, H never
   materialized).
3. **Adjoint gradient** — `AdjointGradientCalculator(qc, H)` →
   `PlanckAdjointGradient(qc, H)` (same `.parameters` / `.gradient(param_dict)`
   interface; one forward + one backward sweep on GPU, H seeded as Σ cₖ Pₖ|ψ⟩, never
   `H.to_matrix(sparse=True)`).

## Integration points (verified 2026-06-11, branch feat/planck-backend)

- `solvers/base_solver.py::_init_backend` — added `elif backend == 'planck':` that
  sets `_use_statevector = True` (planck rides the statevector code path) and falls
  back to Qiskit statevector if `import planck` fails.
- `solvers/vqe_solver.py::_compute_energy_statevector`
  - statevector construction site: when `backend_name == 'planck'`, build |ψ⟩ via
    `planck_statevector(bound_circuit)`, stash on `self._planck_sv`, and wrap in a
    Qiskit `Statevector(...)` so penalties / RDMs are byte-identical.
  - non-padded expectation site: when planck, `energy = expectation(self._planck_sv,
    self._sparse_pauli_op)`; else the original Qiskit call. (The rare padded branch
    stays on Qiskit — operating on the planck-built amplitudes — which is correct.)
- `solvers/vqe_solver.py::_build_adjoint_calculator` — returns
  `PlanckAdjointGradient(qc, combined)` when planck, else `AdjointGradientCalculator`.

All branches are additive and gated on `backend_name == 'planck'`: with the default
`backend='statevector'` the diffs are unreachable and the CPU path is byte-for-byte
unchanged.

## Validation (local GTX 1650 via rocm-planck's NVIDIA build)

- **Gradient parity vs kanad's own adjoint:** on the basis kanad's adjoint supports
  (rx/ry/rz + cx, incl. shared parameter + linear `ParameterExpression`), planck and
  `AdjointGradientCalculator` agree to **1.1e-16** and both match central
  finite-difference to ~1e-10.
- **Energy parity vs Qiskit:** 5.6e-17.
- **End-to-end H₂/STO-3G VQE:** `backend='planck'` = −1.13728380 Ha vs
  `backend='statevector'` = −1.13728379 Ha (|Δ| = 1.3e-8 Ha; the physical FCI energy).

### Known difference (planck is a superset)

kanad's `AdjointGradientCalculator` only differentiates single-Pauli rotations; it
**silently drops** the gradient of a parameterized 2q rotation (`rzz`/`rxx`/`ryy`).
planck transpiles to {rx,ry,rz,h,x,cx} and captures those gradients correctly
(validated vs finite-difference in the rocm-planck suite). So on the default HEA
ansatz (ry + cx) they are identical; on ansätze with parameterized entanglers planck
is *more* correct. The parity test therefore compares on the mutually-supported basis.

## Running the parity tests

```bash
# kanad's own 3.12 venv with rocm-planck installed editable
source .venv/bin/activate
export PYTHONPATH=$(cd .. && pwd)         # kanad importable
pytest tests/test_planck_backend.py -v    # adapter parity (no pyscf); molecule tests need pyscf
```

Adapter-level tests need only qiskit + planck; the molecule VQE tests
(`test_molecule_vqe_planck_matches_cpu`) additionally need pyscf and are skipped
otherwise.

## Phase 1 (MI300X)

Rebuild rocm-planck with `-DPLANCK_GPU_PLATFORM=amd -DCMAKE_HIP_ARCHITECTURES=gfx942`
in this venv; the adapter is unchanged. The end-to-end VQE wall-clock (CPU vs MI300X)
on H₂/LiH is the kanad-side benchmark referenced in
`rocm-planck/docs/benchmarks/phase1-handoff.md`.
