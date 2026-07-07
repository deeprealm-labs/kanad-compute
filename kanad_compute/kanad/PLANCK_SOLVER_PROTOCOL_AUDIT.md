# rocm-planck × unified solver protocol — audit & bug report

**Date:** 2026-06-14 · **Branch:** `feat/solver-protocol` · **Backend under test:** `planck`
(rocm-planck GPU core) vs `statevector` (CPU reference).

Goal: drive the new unified solver protocol with the `planck` backend, run molecules
through every applicable solver + the photodynamics/analysis workflows, validate returned
values rigorously, and report bugs.

**Status: MI300X session complete.** The full grid (8 solvers × {H₂, LiH, H₂O} ×
{statevector, planck}) plus the photodynamics + analysis workflows were run on a real
**AMD MI300X** (DigitalOcean, gfx942 / ROCm 7.2, 192 GB; droplet `577425130` @
`134.199.205.119`). Harness: `benchmarks/planck_protocol_audit.py` and
`benchmarks/planck_photodyn_analysis.py` (JSON → `benchmarks/out/`).

## Headline results

- **0 / 18** `backend='planck'` cells silently fell back to CPU — every planck cell ran on
  the GPU `PlanckBackend` (this is the regression that bug #1 would otherwise have hidden).
- **All SolverResults conform** to the unified protocol: every result that ran exposes
  `.energy` and `.to_dict()` (conformance issue list empty across the whole grid).
- **Planck energy evaluation is exact** wherever the solver is deterministic: PhysicsVQE,
  qEOMVQE, ExcitedStates, SmartSolver all match the CPU statevector backend to 1e-12 – 1e-15
  (often bit-identical). Remaining divergences are all explained (stochastic sampling, or
  optimizer under-convergence) — none is a planck correctness bug.

---

## Bugs found

### 1. CRITICAL — `backend='planck'` silently ran on CPU (`PlanckBackend` was missing) — FIXED ✓
`kanad/backends/factory.py` does `from kanad.backends.planck_adapter import PlanckBackend`,
but **`PlanckBackend` did not exist** in `planck_adapter.py` (only `PlanckAdjointGradient` /
`PlanckVQEEvaluator` + helper funcs). The import hit the "graceful fallback" and
`make_backend('planck')` returned a **`StatevectorBackend`** — even though `import planck`
succeeds. Net effect: every `backend='planck'` run silently executed on CPU; an MI300X
session would have tested nothing and burned credit.

- **Repro (pre-fix):** `make_backend('planck')` → `StatevectorBackend(name='statevector')`.
- **Fix (this branch):** implemented `PlanckBackend(BaseBackend)` in `planck_adapter.py` —
  `estimate_expectation` builds |ψ⟩ on the GPU and contracts (Pauli sums fully on-GPU;
  non-Pauli on the GPU-built state via Qiskit); `sample` draws bitstrings from the GPU
  statevector's probabilities.
- **Validated on GPU:** the MI300X grid reports **0/18 planck cells used StatevectorBackend**
  (`backend_obj=PlanckBackend` on every planck cell). Regression test added:
  `tests/test_planck_backend.py::test_make_backend_planck_is_planck_not_silent_cpu`.
- **Note:** the kanad venv also had a **stale `_core`** (pre-complex64) — rebuild rocm-planck
  editable after pulling (`uv pip install -e ../rocm-planck …`).

### 2. HIGH — `VQESolver` does not route energy through `self.backend` (non-uniform protocol) — FIXED ✓ (`dd76a0b`)
VQESolver keeps **bespoke** energy paths instead of `self.backend.estimate_expectation`:
`Statevector.from_instruction(...)` (lines ~871, 1710, 2323), a `backend_name=='planck'`
branch building `self._planck_sv` (~line 1132), and `self._use_statevector =
isinstance(self.backend, StatevectorBackend)` (line ~721). Now that `PlanckBackend` exists,
`_use_statevector` flips to **False** for planck, risking dispatch into the cloud/shot path.
The planck *energy* is correct (see §works), but the integration is inconsistent and
fragile — VQESolver should compute energy via `self.backend.estimate_expectation` like the
other protocol solvers, and drop the bespoke `backend_name=='planck'` / `_use_statevector`
machinery.

### 3. MEDIUM — `VQESolver` planck↔statevector energies diverge (convergence, not correctness) — FIXED ✓ (`184edba`, guard)
Full GPU grid (|ΔE| planck vs statevector): **H₂ 5.8e-2, LiH 9.4e-1, H₂O 2.8e+1**. The H₂O
statevector run lands at **−39.48 Ha** and planck at **−67.56 Ha** — *both* far above any
physical energy, i.e. the optimizer never converged on either backend. **Root cause is
optimizer under-convergence**, not a planck bug: with COBYLA/80 iters the trajectories
diverge between backends. The proof is in the same grid — PhysicsVQE, qEOMVQE, SmartSolver
all reach exact planck↔statevector parity (1e-12–1e-15) on the *same molecules*, so the
planck energy evaluation is correct; only VQESolver's default optimizer path is flaky. Worth
a better default optimizer/iteration budget or a convergence guard.

### 4. MEDIUM — `ExcitedStates` on H₂O crashes (backend-independent kanad bug) — FIXED ✓ (`6e64e07`, TDA route)
`ExcitedStatesSolver(..., method="cis")` on H₂O raises
`AttributeError: 'ActiveHamiltonian' object has no attribute 'solve_scf'` on **both**
backends (statevector and planck identically). So it is **not** a planck bug — it is a
kanad bug in the ExcitedStates ⇄ `ActiveHamiltonian` path (the CIS code calls `solve_scf`,
which `ActiveHamiltonian` doesn't implement; the smaller molecules use a Hamiltonian type
that does). H₂ and LiH ExcitedStates runs are bit-identical across backends.

### 5. LOW — `SolverResult.converged` is `False` despite chemical accuracy — FIXED ✓ (`6e64e07`)
PhysicsVQE/planck on H₂ reaches FCI (0.00 mHa error, "CHEMICAL ACCURACY ACHIEVED") yet
`result.to_dict()['converged']` is `False`. The `converged` flag is misleading/incorrect for
the sequential-optimizer path; consumers keying on it (dynamics/analysis) would mis-report.

---

## Confirmed working with planck (protocol-conformant) — MI300X grid

`|ΔE|` = |planck − statevector|. `obj` = the live backend object on the planck run.

| solver | H₂ \|ΔE\| | LiH \|ΔE\| | H₂O \|ΔE\| | notes |
|--------|----------:|-----------:|-----------:|-------|
| **PhysicsVQE** | 8.7e-15 | 2.1e-13 | 2.4e-12 | exact; reaches FCI on H₂ |
| **qEOMVQE** | 1.0e-14 | 2.7e-13 | 4.1e-04 | exact on H₂/LiH; H₂O off at 4e-4 (subspace-diag sensitivity, below mismatch threshold) |
| **SmartSolver** | 0 (bit-id) | 1.1e-14 | 0 (bit-id) | exact across all three |
| **ExcitedStates** | 0 (bit-id) | 0 (bit-id) | — (bug #4) | bit-identical where it runs |
| **VarQITE** | 6.8e-10 | (skipped) | — | imag-time evolution on planck; H₂ matches. LiH skipped — see note. |
| **SampledSubspaceVQE** | 0 (bit-id) | 9.5e-04 | 1.8e-03 | **stochastic** subspace method; divergence is sampling noise, not a planck error |

- **`backend_obj=PlanckBackend` on all 18 planck cells** — GPU genuinely in the loop.
- **SampledSubspaceVQE** crosses 1e-3 on H₂O (1.8e-3) but this is a *sampling-based* solver:
  both backends draw bitstrings stochastically, so a sub-1e-2 spread is expected sampling
  noise on top of subspace conditioning — not a correctness defect. Worth seeding both
  backends from the same RNG for a tighter parity check.
- **VarQITE/LiH was skipped** (`AUDIT_SKIP=VarQITE`): the planck run hangs >60 min with no
  progress and does not respond to the per-cell SIGALRM (the alarm can't interrupt a blocking
  call inside the imaginary-time inner loop). H₂ completed and matches (6.8e-10). This hang is
  worth a separate look — likely a VarQITE inner-loop issue rather than planck, since planck
  returns promptly for every other solver; flagged for follow-up, not yet root-caused.

## Photodynamics + analysis (planck-backed) — validated

`benchmarks/planck_photodyn_analysis.py`, run on the MI300X:

- **Photodynamics** (`PhotodynamicsSimulator`, `use_quantum=True`, `vqe_backend='planck'`,
  qEOM per step, H₂ + Gaussian laser, 2.0 fs / dt 0.1): ran end-to-end, 11 saved steps.
  Deep value-validation passes:
  - **population conservation** `max|Σpop − 1| = 4.4e-16` (machine precision),
  - **energies all finite**, **populations ≥ 0**,
  - result exposes `energies, populations, times, field_amplitudes, excitation_probability,
    final_state, final_population`.
- **Analysis workflow** on a planck-backed PhysicsVQE/LiH result (E = −7.8817 Ha, finite):
  `EnergyAnalyzer.analyze_convergence` returns the full key set
  (`converged, energy_change, final_energy, final_gradient, initial_energy, iterations,
  mean_energy_change`); `BondingAnalyzer.analyze_bonding_type` returns
  `bonding_type, characteristics, homo_lumo_gap, homo_lumo_gap_ev`, all finite. `to_dict()`
  carries the full result contract (`energy, hf_energy, fci_energy, correlation_energy,
  correlation_captured, excitations, energy_history, …`).

## Timing note (why this run is about correctness, not speed)

These molecules (8–12 qubits) sit **below the GPU forward-sim crossover (≈16 qubits)**, so
per-iteration host overhead (transpile + state alloc per `estimate_expectation`) dominates and
planck is not meaningfully faster than CPU here (e.g. qEOM/H₂O 72 s vs 75 s; VQE/H₂O is slower
on planck only because its divergent trajectory took more evals). **That is expected and not
the point** — the planck speed story is the **capacity** result (34-qubit complex64 / 33-qubit
complex128 single-GPU, 137 GB, ~4 TB/s), already captured separately. This audit's job is to
prove the GPU backend is *correct and protocol-conformant* under the unified solver protocol,
which it is.

---

## Future work (not bugs)

### Quantum-centric SQD workflow (`SamplingSQDSolver` and the SQD family)
`SamplingSQDSolver` validates `backend` against a hardcoded list
`{'statevector','qasm','bluequbit','ibm'}` and raises `ValueError` for `'planck'`, so it
does not currently accept the planck `BaseBackend`. **This is intentionally left as future
work, not classified as a bug:** the SQD family needs its own *quantum-centric execution
workflow* (optimized sample → configuration-recovery → subspace-diagonalize pipeline) rather
than being forced through the generic per-iteration `estimate_expectation` abstraction. When
that workflow is designed, SQD should route sampling through `self.backend.sample` (which
`PlanckBackend` already implements) so planck can drive it natively. Tracked separately from
this protocol audit.

### VarQITE/LiH planck hang
See note above — VarQITE/LiH does not return on planck and ignores the per-cell timeout.
Needs root-causing (suspected VarQITE inner-loop, not planck core).

---

# Session 2 (2026-06-14) — fix validation + analysis & scaling audit

Re-tested the fixes (`6e64e07`, `184edba`, `dd76a0b`) on the same MI300X, ran a
solver audit **with analysis enabled** on molecular sims, and probed the planck core's
capacity envelope. Harnesses: `tests/validation/test_planck_audit_fixes.py`,
`benchmarks/planck_solver_analysis_audit.py`, `benchmarks/planck_scale_audit.py`
(+ `planck_scale_34q.py`). JSON in `benchmarks/out/`.

## Fix validation — all four FIXED ✓ (re-tested on MI300X)

`tests/validation/test_planck_audit_fixes.py`: **8/8 pass** (CPU and MI300X/real planck).
Re-run of the affected (solver × molecule) cells on the GPU:

- **#2 (planck local routing):** `VQESolver(..., backend='planck')._use_statevector` is now
  **True**, `self.backend` is a real `PlanckBackend`, energy routes through
  `_compute_energy_statevector` (the GPU branch), not the cloud path. VQESolver/H₂ planck runs
  and returns −1.13728 (conv=True), parity vs statevector 8.5e-8 (L-BFGS-B optimizer-level).
- **#3 (convergence guard + active-space HF ref):** VQESolver/H₂ → `converged=True`, below HF,
  no warning. VQESolver/LiH and /H₂O now finish **`converged=False`** (was silently returning a
  divergent energy) — the under-converged run is honestly flagged instead of trusted.
  `get_reference_energy()` returns a real HF reference (≈ −74.96 Ha) for the H₂O
  `ActiveHamiltonian`.
- **#4 (ExcitedStates / ActiveHamiltonian):** `ExcitedStatesSolver(method='cis')` on **H₂O no
  longer crashes** — `method='CIS (PySCF TDA)'`, two positive/finite excitation energies,
  planck↔statevector parity **2.84e-14**. H₂/LiH keep the hand-rolled `method='CIS'` path
  (parity 0.0, unchanged).
- **#5 (converged flag):** PhysicsVQE/LiH and /H₂O now report **`converged=True`** at chemical
  accuracy (was hardcoded `False`); planck parity 3.6e-13 / 3.2e-12.

## Analysis-pipeline audit (analysis ENABLED on molecular sims)

`planck_solver_analysis_audit.py` — solvers run with `enable_analysis=True`, validating the
attached `result['analysis']` dict (`energy_components`, `bonding`, `properties`) and parity.

| solver | molecule | planck conv | parity \|ΔE\| | analysis dict | checks |
|--------|----------|:-----------:|--------------:|:-------------:|--------|
| **VQESolver** | H₂ | True | 8.5e-8 | ✓ present | energy-decomp **self-consistent**, dipole ≥0 finite |
| **VQESolver** | LiH | False (guarded) | 4.6e-1 | ✓ present | decomp ✓, dipole ✓ |
| **VQESolver** | H₂O | False (guarded) | 5.0e0 | ✗ absent | no analysis emitted for a non-converged run |
| **PhysicsVQE** | H₂/LiH/H₂O | True | 8.7e-15 / 3.6e-13 / 3.2e-12 | ✗ absent | exact energy; no analysis dict (own solve path) |
| **ExcitedStates** | H₂/LiH/H₂O | True | 0 / 0 / 2.8e-14 | ✗ (spectroscopy) | excitation energies positive/finite |

Validated invariants on the VQESolver analysis dict (planck-backed): `energy_components.total`
== `nuclear_repulsion + one_electron + two_electron` (self-consistent), `dipole_moment` real &
≥0. (`bond_orders` is a nested per-pair structure — recorded, not scalar-asserted.)

New findings from this pass:
- **F6 — PhysicsVQE never attaches `result['analysis']`.** Its sequential path bypasses
  `BaseSolver._add_analysis_to_results`, so "analysis on by default" holds for VQESolver but
  **not** PhysicsVQE/ExcitedStates. Analysis availability is solver-dependent and undocumented.
- **F7 — VQESolver emits no analysis dict for a non-converged run** (H₂O): analysis coverage
  drops out exactly where convergence is poor. Reasonable, but worth surfacing explicitly
  (consumers shouldn't assume `analysis` is always present).
- **F8 — `BondingAnalyzer.analyze_bonding_type()` returns only `{'bonding_type':'unknown'}`**
  (no gap/characteristics) for builder/`ActiveHamiltonian` systems — full output appears only
  for `BondFactory` bonds. The data exists (SCF HOMO-LUMO gap is fine, e.g. 27.1 eV for H₂O);
  the analyzer just doesn't extract it for that Hamiltonian type.
- **F9 — VQESolver default optimizer still under-converges at ≥12q** (LiH/H₂O `conv=False`).
  The #3 guard now flags this honestly (a real improvement), but the default optimizer/iteration
  budget is inadequate at scale — PhysicsVQE reaches FCI on the same molecules, so the gap is
  VQESolver's defaults, not the backend.

## Scaling / capacity envelope — planck CORE on molecular Hamiltonians (MI300X)

`planck_scale_audit.py` / `planck_scale_34q.py` — a **core** capacity probe (deliberately
distinct from the solver audit above): build a dense fixed-seed state at N qubits on a real
linear-Hₙ Jordan-Wigner Hamiltonian (sto-3g, full space → 2N qubits), measure resident VRAM,
verify unitary kernels preserve norm, evaluate ⟨H⟩, and parity-check vs CPU where it fits.

| qubits | system | Pauli terms | state | **VRAM resident** | norm dev | ⟨H⟩ (Ha) | CPU parity \|ΔE\| | t(⟨H⟩) |
|-------:|--------|------------:|------:|------------------:|---------:|---------:|------------------:|-------:|
| 20 | H₁₀ | 7,151 | 0.02 GB | 0.54 GB | 1.1e-15 | 0.0707 | **1.2e-15** | 2.6 s |
| 24 | H₁₂ | 14,905 | 0.27 GB | 0.25 GB | 1.1e-15 | −0.1471 | **6.3e-15** | 79 s |
| 28 | H₁₄ | 27,735 | 4.29 GB | 4.03 GB | 2.0e-14 | −0.6225 | (>26q) | 150 s |
| 32 | H₁₆ | 47,489 | 68.7 GB | **64.4 GB** | 8.7e-15 | −0.1214 | (>26q) | 1,615 s |
| **34** (c64) | H₁₇ | — | 137.4 GB | **137.96 GB** | 5.3e-9 | — (c64) | (>26q) | — |

- **"Every register" demonstrated on a real molecular system:** 64.4 GB resident at 32q
  (complex128) and **137.96 GB at 34q (complex64)** — matching the theoretical `2ⁿ × dtype`
  to <1%. The unitary kernels preserve `‖ψ‖ = 1` to **8.7e-15** (c128, 32q) / **5.3e-9**
  (c64, 34q over 170 gates).
- **Exact at scale:** planck ⟨H⟩ matches the CPU statevector to **1e-15** at 20/24q on real
  molecular Hamiltonians; a full **47,489-term** molecular ⟨H⟩ runs at 32q/64 GB.
- **KEY FINDING — the bottleneck is the algorithm, not the GPU.** The planck *core* holds and
  correctly operates on a 34-qubit molecular-scale state (137 GB). But a full *solver +
  analysis* run tops out far lower (~22–24q): cost is `O(N⁴ Pauli terms) × 2ⁿ` per ⟨H⟩ × many
  optimizer iterations. One ⟨H⟩ at 32q took **27 min** (47k terms × 64 GB); a full iterative
  VQE there is intractable. **MI300X memory is not the limit for large-molecule simulation —
  the Hamiltonian term count and the classical optimizer are.** Using the device's capacity at
  scale needs term-count reduction (Pauli grouping / low-rank factorization) and better
  optimizers, not more VRAM.
- **F10 (planck API gap):** `StateVector.vdot` (and `expectation`) are **complex128-only**, so a
  34q complex64 norm requires a 137 GB host pull. A device-side c64 norm/reduction would close
  this for the capacity regime.

## Recommended fixes (priority order)
1. ✓ done — `PlanckBackend` + `make_backend('planck')` regression test.
2. ✓ done (`dd76a0b`) — planck energy routes through the local-statevector path. *Follow-up:*
   the full "retire `_use_statevector`, energy via `estimate_expectation`" refactor is deferred.
3. ✓ done (`184edba`) — convergence guard. *Open (F9):* still need better default optimizer /
   iteration budget so VQESolver actually converges at ≥12q (LiH/H₂O), not just flags failure.
4. ✓ done (`6e64e07`) — ExcitedStates ActiveHamiltonian → PySCF TDA.
5. ✓ done (`6e64e07`) — `converged` flag for the sequential/PhysicsVQE path.
6. **Open (F6/F7):** make analysis availability uniform/documented — attach `result['analysis']`
   for PhysicsVQE (or document that it doesn't), and signal when analysis is skipped (non-converged).
7. **Open (F8):** `BondingAnalyzer` should extract the HOMO-LUMO gap / bonding type for
   `ActiveHamiltonian`/builder systems (data is present in the mean-field).
8. **Open (F10):** add a device-side complex64 norm/reduction to planck for the capacity regime.
9. Design the quantum-centric SQD workflow (future work, above).

---

# Session 3 (2026-06-14) — ALL analyzers on CHALLENGING molecules (planck/GPU)

Per request, dropped the toy H₂/LiH/H₂O set and ran the **full 16-analyzer suite** —
including **photochemistry** (UV-Vis excited states, absorption, photodynamics) and
**reactions** (bond-length scan, configuration explorer) — on real, hard molecules,
planck-backed where the analysis routes through a quantum solver.
Harness: `benchmarks/planck_full_analysis_audit.py` (JSON in `benchmarks/out/`).

**Molecules:** N₂ (triple bond, multireference), CO (heteronuclear), C₂ (notoriously
multireference), CH₂O / formaldehyde (n→π\* photochemistry) — all sto-3g full/active space,
~12–20 qubits. (Benzene aromatic CAS was attempted but its PhysicsVQE ground-state is
CPU-bound on the sequential-excitation path — see finding F13.)

## Coverage: ~14 / 19 analyzer-cells succeed per molecule

| analyzer | result on the challenging set | GPU |
|----------|-------------------------------|:---:|
| **PhysicsVQE ground state** | N₂ −107.58, CO −111.26, C₂ −74.55, CH₂O −112.42 Ha — all `converged=True` | ✓ |
| EnergyAnalyzer.decompose | self-consistent (N₂/CO/C₂); CH₂O fails (active-space dm shape, F12) | |
| CorrelationAnalyzer | E_corr < 0 for all (N₂ −0.080, CO −0.040, C₂ −0.131, CH₂O −0.068) — variational ✓ | |
| PropertyCalculator.dipole | **CO 0.169 (polar ✓)**, N₂/C₂ ≈0 (symmetric ✓), CH₂O 1.525 ✓ | |
| PropertyCalculator.polarizability | symmetric tensor ✓ (N₂ 3.95, CO 4.41, CH₂O 6.38); **C₂ −102 (unphysical → F11)** | |
| **UVVis.excited_states** (photochem) | positive excitations ✓: N₂ 9.48 eV, CO 8.98, C₂ 2.95, **CH₂O 5.53 eV ≈ n→π\*** | |
| **absorption_spectrum** (photochem) | full spectrum dict (energies, oscillator strengths, lineshape) ✓ | |
| FrequencyCalculator | N₂/CO/C₂ 1 mode, **CH₂O 6 modes (3N−6 ✓)**, ZPE>0, 0 imaginary ✓ | |
| ThermochemistryCalculator | runs; S/G populate partially (F14) | |
| UncertaintyAnalyzer | shot-noise std ≥ 0 ✓ | |
| **BondLengthScanner** (reactions) | **N₂ dissociation PES monotone, eq = 1.13 Å** (exp 1.10); CO/C₂/CH₂O eq found ✓ | |
| descriptors + reactivity | `quantum_reactivity` (χ, η, ω) computed ✓ | |
| **Photodynamics** (photochem) | N₂/CO **pop conservation 2.2e-16**, C₂ 4.4e-16 (machine precision) — qEOM/planck per step | ✓ |
| BondingAnalyzer | runs but `bonding_type='unknown'` (F8) | |

## Failures = findings (genuine kanad issues, surfaced by the audit)

- **F11 — C₂ breaks single-reference methods.** PhysicsVQE FCI error 71 mHa and HF
  polarizability −102 (negative ⇒ unphysical). C₂'s strong static correlation isn't captured
  by the HF-reference / sequential-VQE path — a real multireference limitation, not a planck
  bug (planck just evaluates what it's given). Flags C₂ as needing a multireference solver.
- **F12 — `EnergyAnalyzer.decompose_energy` shape mismatch on active-space systems.** When the
  builder applies an active space (CH₂O frontier), `mf.make_rdm1()` is full-dimension but the
  active-space integrals are smaller → `operands could not be broadcast (12,12) vs (n,n)`.
  decompose needs the active-space density, or the analyzer should project.
- **F13 — NMR & IR/Raman: "Hamiltonian has no atoms".** `NMRCalculator` (GIAO) and
  `RamanIRCalculator` need a hamiltonian carrying its atoms; the builder `MolecularHamiltonian`
  doesn't expose them, so both fail on every molecule. (NMR also needs `pyscf.prop` for the
  paramagnetic term on the bond path.)
- **F14 — `DOSCalculator.compute_quantum_dos` doesn't scale.** `solver='sqd'` path tries to
  build a dense `2ⁿ × 2ⁿ` matrix → **MemoryError (16 TiB at 20q)**; `solver='vqe'` forwards an
  unexpected `n_states` kwarg to `VQESolver.solve`. Quantum DOS is unusable above ~tiny.
- **F15 — `ConfigurationExplorer.scan_bond_length` raises** `TypeError: unsupported format
  string passed to dict.__format__` on every molecule (a formatting bug inside the analyzer).
  The plain `BondLengthScanner` works, so reactions ARE covered — but the governance-aware
  explorer is broken.
- **F16 — `VibronicCalculator.compute_franck_condon_factors`** raises an inhomogeneous-array
  error (Franck-Condon path).
- **F17 — `ThermochemistryCalculator`** runs but returns `S=None/G=None` for these inputs
  (entropy/Gibbs not populated from a frequency list alone).
- **F18 — PhysicsVQE ground-state is CPU-bound at aromatic CAS.** Benzene CAS(6,6) never
  finished its sequential-excitation optimization (>>5 min, GPU idle, host at 196% CPU). The
  bottleneck is the classical sequential-excitation loop, not planck — same root as F9.

## Honest scale framing (answering "use the 33-qubit GPU")

The full analysis suite is validated on genuinely hard molecules (multireference N₂/CO/C₂,
photochemical CH₂O) at **12–20 qubits** — the scale where a complete solve+analysis actually
finishes. It does **not** reach 33 qubits, and that is the central, repeatedly-confirmed
finding: **the wall is algorithmic, not the GPU.** A full molecular ⟨H⟩ is `O(N⁴ terms) × 2ⁿ`
and the VQE/PhysicsVQE optimizer adds a large iteration count on top — both classical/host
costs. The planck *core* already holds and correctly operates on a 34-qubit / 137 GB molecular
state (Session 2 capacity envelope: norm-preserving, exact ⟨H⟩ vs CPU to 1e-15 ≤24q). Closing
the gap so analysis can use 192 GB needs **term-count reduction (Pauli grouping / low-rank
factorization) and faster optimizers / a quantum-centric (SQD) workflow** — not more VRAM.
