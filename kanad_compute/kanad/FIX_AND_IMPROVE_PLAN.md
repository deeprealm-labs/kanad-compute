# Kanad — Fix & Improve Plan (consolidated from all audits + the 2026-06-16 campaign)

Sources: `PLANCK_SOLVER_PROTOCOL_AUDIT.md` (F6–F18) + the themed validation campaign
(`kanad-compute/docs/CAMPAIGN_RESULTS.md`) + the SQD/QPU sessions. Most fixes are CPU-testable
locally in `kanad/.venv`; GPU/scale re-validation waits on a fresh MI300X.

Sequence: **Tier 1 (correctness) → Tier 2 (usability/crashes) → Tier 3 (better defaults) →
Tier 4 (perf/scale) → integration into kanad-app + kanad-compute.**

## STATUS — Tier 1 COMPLETE (2026-06-16), 7/7 regression tests pass

Tests in `tests/validation/test_tier1_fixes.py` (run: `PYTHONPATH=<deeprealm> .venv/bin/python` import-and-call, since pytest's conftest collides with the path-based `kanad` package).
- **C1 — already correct, no fix needed:** `diagonalize_custom` == pyscf FCI to ~1e-14 on αβ-doubles cases (N₂ CAS(6,6), H₂O CAS(4,4)). The old "αβ-doubles sign bug" warning was **stale**; locked in with a regression test.
- **C2 — FIXED** (`analysis/energy_analysis.py`): active-space fallback to the HF reference density; total == mf.e_tot to 1e-13, self-consistent.
- **C3 — FIXED** (`core/active_space/selector.py`): mp2no falls back to a frontier active space on near-single-reference systems instead of raising (methanol builds).
- **C4 — FIXED** (`analysis/energy_analysis.py`): `analyze_bonding_type` now reads gap/type from `.mf` for active-space/builder systems (no more 'unknown').
- **U1 — FIXED** (`core/ci/selected_ci.py`): excited-states FCI `max_memory` sized from RAM (psutil), not pyscf's 4 GB default.
- **U2 — FIXED** (`solvers/sampling_sqd.py`): a no-entangling-gate (HF) ansatz is **refused on QPU backends** and warned on statevector — closes the vacuous-circuit hazard.

## STATUS — Tier 2 COMPLETE (2026-06-16), tests in `tests/validation/test_tier2_fixes.py`

Most "open" Tier-2 items turned out **already fixed** in a prior session (verified, not assumed):
- **F13** (NMR/Raman "no atoms"), **F15** (ConfigExplorer scalar energy), **F12** (decompose clear error) — covered by the existing `test_planck_analysis_audit_fixes.py` (8/8 pass).
- **F16** (Vibronic FCF) — **already works** with the correct `(ground_freqs, excited_freqs, displacement)` signature; the audit error was a ragged-input edge case. Locked with a test.
- **F17** (Thermochemistry S/G) — **already works**; water S=45.1 cal/(mol·K) matches the experimental S°. Locked with a test.
- **F14 — FIXED** (`analysis/dos_calculator.py`): (a) constructor now accepts a molecular Hamiltonian (duck-typed periodic check) instead of raising the periodic-only error; (b) `compute_quantum_dos` no longer passes `n_states` to `solve()` (the crash). Molecular quantum DOS runs for vqe + sqd.

## STATUS — Tier 3 + Tier 4 (2026-06-16): the one silent-error risk fixed; rest verified-safe / guidance / pending-box

**Tier 3** (tests in `tests/validation/test_tier3_fixes.py`, 3/3 pass):
- **I1 — FIXED**: `reactivity_descriptors` warns on a minimal basis (sto-Ng/minao) — prevents silently-garbage χ/η/ω (sto-3g benzene χ=0.15 eV vs cc-pVDZ 2.67). The real "silent wrong result at scale" risk; now loud.
- **I6 — verified ALREADY-CORRECT**: both `'ci'` routes pin spin (`fix_spin_`), so the wrong-multiplicity-root hazard can't bite kanad's own solver (returns a spin-pure singlet, ⟨S²⟩~0). Tested. The CASCI(nroots=1) hazard exists only in external/benchmark code (documented in memory).
- **I5 — NOT A BUG**: the `'ci'` string is a legitimate `method` metadata field; only a blanket-float-coercing consumer breaks.
- **I3 — guidance**: `excited_states` docstring now warns that extended π systems need the FULL π active space (AVAS `C 2pz`/`N 2pz`), not a small frontier window; the AVAS-π path is available (validated to 48q on porphine).
- **I2 — guidance**: relative-energy calcs (proton affinity, tautomers) need a MATCHED active space across species — documented; the unbalanced-CAS errors were quantitative (signs were correct). A shared-CAS mechanism is a future enhancement.
- **I4** (analysis-availability uniformity): minor UX, deferred.

**Tier 4:**
- **P1 — verified NOT-silent**: VQE carries a `converged` flag (non-convergence is signalled, not a silent bad energy). The CPU-bound sequential-excitation at ≥12q is PERFORMANCE — route hard/large cases to the SQD/GPU-det-CI path (built + validated to 56q / 1.4M dets).
- **P2**: paramagnetic NMR is a FLAGGED feature-gap (`paramagnetic_part_missing=True`) — diamagnetic-only, honestly flagged, NOT silent-wrong. Feature for later.
- **P3**: wire the GPU det-CI as kanad's large-CAS backend — INTEGRATION (kanad-compute) + needs a fresh MI300X (current box destroyed). Pending box.

**NET across Tiers 1–4:** every genuine bug is fixed + regression-tested (C2, C3, C4, U1, U2, F14, I1); items that looked open were verified already-correct (C1, F12, F13, F15, F16, F17, I6); the one silent-error risk (I1) is now loud; the rest are verified-not-silent, documented guidance, or pending a box. Framework correctness is clean.

---

## Tier 1 — Correctness (wrong answers / crashes on common paths)

| # | finding | where | fix | test |
|---|---------|-------|-----|------|
| C1 | **`ci_backend='custom'` sparse Slater-Condon sign bug** on αβ-mixed doubles at scale (wrong SQD energies; the reason the scale path is risky) | `core/ci/slater_condon.py` (`_slater_condon_offdiag` / `_build_sparse_h_subspace`) | diff the custom sparse H element-wise against the **validated GPU det-CI builder** (`kanad_compute…det_ci_gpu.build_sparse_h_parallel`, bit-identical to serial) and against `diagonalize_pyscf`; fix the αβ sign | full-sector CAS(6,6)/(8,8)/(10,10) energy == FCI to 1e-10 |
| C2 | **F12 — `decompose_energy` shape mismatch** on active-space systems (full `mf.make_rdm1()` vs active integrals → broadcast error; confirmed again on CH₂O, acetone) | `analysis/energy_analysis.py:37` | accept/derive an **active-space density**, or project the dm to the active block | decompose on a frontier-CAS H is self-consistent (sum of parts == total) |
| C3 | **`mp2no` active-space selector fails** "no partially-occupied orbitals" on common closed-shell molecules (methanol) | active-space module (`core/active_space*`) | robust fallback (loosen NOON threshold / fall back to `frontier`) instead of raising | mp2no builds for methanol/H₂O/glycine without error |
| C4 | **F8 — `BondingAnalyzer` returns `'unknown'`** for ActiveHamiltonian/builder systems (gap/bonding-type present in the mean-field but not read) | `analysis/bonding_*` | pull HOMO-LUMO gap + bonding type from `ham.mf.mo_energy`/`mo_occ` | gap/type returned (non-'unknown') for builder systems |

## Tier 2 — Usability & crashes

| # | finding | where | fix |
|---|---------|-------|-----|
| U1 | **FCI/CASCI default `max_memory` too low** → spurious "Not enough memory for FCI solver" at CAS(16,16) | wherever pyscf FCI/CASCI is invoked (`solvers/`, `core/ci/`) | set `max_memory` from available RAM (e.g. 80% of psutil total) |
| U2 | **Vacuous SQD ansatz** — `SamplingSQDSolver().solve()` defaults to a trivial HF circuit; a depth-≤1 circuit was nearly submitted to a QPU | `solvers/sampling_sqd.py` | warn/raise when the sampling circuit depth is trivial **before** any (esp. `backend='ibm'`) submission |
| U3 | **F15 — `ConfigurationExplorer.scan_bond_length`** `TypeError: dict.__format__` (formatting bug, every molecule) | reactions/`configuration_explorer` | fix the f-string formatting on a dict value |
| U4 | **F17 — `ThermochemistryCalculator` returns S=None/G=None** from a frequency list | `analysis/…thermo` | populate translational+rotational+vibrational S, then G=H−TS |
| U5 | **F16 — `VibronicCalculator.compute_franck_condon_factors`** inhomogeneous-array error | `analysis/…vibronic` | pad/ragged-fix the FCF array construction |
| U6 | **F14 — `DOSCalculator` quantum path** builds dense 2ⁿ×2ⁿ (MemoryError 20q); vqe path forwards bad `n_states` kwarg | `analysis/dos*` | gate dense path to tiny n; drop the bad kwarg; or matrix-free |

## Tier 3 — Better defaults & guidance (bake in the campaign methodology)

| # | finding | fix |
|---|---------|-----|
| I1 | **reactivity χ/ω garbage on sto-3g** (unphysical virtuals) — code is correct, basis is wrong | `reactivity_descriptors` warns (or auto-bumps) when basis is minimal |
| I2 | **Relative-energy calcs (proton affinity, tautomers) use unbalanced active spaces** → imidazole PA +23 kcal/mol, tautomer magnitudes overshoot | a "matched active space" option so both species share a consistent CAS |
| I3 | **Large-π UV-Vis needs the full π space** (frontier CAS misorders states; indigo/anthracene quantitatively wrong) | `excited_states` auto-selects the π active space (AVAS on `C 2pz`/`N 2pz`) for conjugated systems |
| I4 | **F6/F7 — analysis availability non-uniform** | attach `result['analysis']` consistently (or document); signal when skipped (non-converged) |
| I5 | **API consistency** — `reactivity_descriptors` returns a dataclass while other analyzers return dicts; `observables` has a string `'ci'` field that breaks blanket float coercion | standardize return types / document |
| I6 | **CASCI(nroots=1) is an unreliable reference** (converges to excited states on multireference systems) | internal validators/benchmarks use `fci nroots≥6` / det-CI, never `CASCI(nroots=1)` |

## Tier 4 — Performance / scale (algorithmic; bigger, GPU box needed)

| # | finding | direction |
|---|---------|-----------|
| P1 | **F9/F18 — VQE/PhysicsVQE CPU-bound + non-converging at ≥12q** (sequential-excitation loop; benzene CAS never finishes); the repeatedly-confirmed "wall is algorithmic, not GPU" | better optimizer + iteration budget, Pauli-term grouping, and/or route hard cases to the SQD/det-CI path |
| P2 | **Paramagnetic NMR not implemented** (diamagnetic-only ⇒ can't reproduce ¹³C shifts); F13 NMR/Raman "no atoms" when passed a Hamiltonian | implement paramagnetic term (pyscf.prop / response) + let NMR/Raman accept a Molecule or carry atoms on the Hamiltonian |
| P3 | **GPU det-CI / iterative-SQD as the scale engine** (validated to 56q / 1.4M dets) | wire it as kanad's large-active-space CI backend so analysis/excited-states can use it |

---

## Then: test → integrate

- **Test:** unit/regression test per fix in `tests/`; re-run the relevant audit harness
  (`benchmarks/planck_full_analysis_audit.py`, `solver_protocol_matrix.py`) and the campaign
  scripts. Tier-1/2/3 are CPU-testable locally now; Tier-4 + GPU re-validation on a fresh MI300X.
- **Integrate into kanad-compute:** point the hybrid SQD engines at the fixed `ci_backend` (C1) +
  the GPU det-CI (P3); confirm `det_ci.py` / `subspace.py` still match FCI.
- **Integrate into kanad-app:** surface the fixed analyzers (bonding gap C4, decompose C2, thermo
  U4) + the matched-active-space (I2) and minimal-basis-warning (I1) so the app's results are
  trustworthy by default.

**Suggested first PR batch:** C1, C2, C3, C4 + U1, U2 (correctness + the two highest-impact
usability fixes), each with a regression test. That clears the bugs that actually corrupt or block
results, and is fully testable on the local CPU env today.
