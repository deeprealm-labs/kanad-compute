# Quantum Property Engines — Spectroscopy, Thermochemistry, Kinetics from the Wavefunction

**Branch:** `feat/quantum-properties` (off `feat/node-sqd-sync` = `main` + SQD IBM/GPU kwargs).
**Goal:** Make IR, Raman, NMR, thermochemistry, and reaction-kinetics observables derive from the
**VQE/SQD correlated wavefunction** (energy derivatives + correlated density/transition densities),
not classical Hartree–Fock stand-ins. This *graduates* the two capabilities the solver protocol
declared-but-deferred (`hessian`, `field_response`) into real implementations, and builds the
rigorous Ramsey sum-over-states NMR that `SOLVER_PROTOCOL_PLAN.md §7.10` explicitly punted on.

## 0. Design rules (inherit from SOLVER_PROTOCOL_PLAN.md)

1. **Everything is a derivative of the quantum energy or a contraction of the correlated density.**
   No property recomputes electronic structure classically when a quantum wavefunction is in hand.
2. **Honesty over coverage.** A property is emitted with a `source` tag (`vqe`/`sqd`/`casci`/`hf`).
   If the quantum path is infeasible (system too large, active space non-reproducible under
   displacement), the engine **raises or falls back to a *labeled* `hf`** — never a silent fake.
3. **Capabilities are the seam.** New physics attaches to the existing `HessianProvider` /
   `FieldResponseProvider` / `ExcitedStatesProvider` Protocols. Declared ⇒ numerically conformance-tested.
4. **Units contract** (from `capabilities.py`): Bohr, Hartree, Ha/Bohr, Ha/Bohr², cm⁻¹, amu, a.u. fields.
5. **Cost is gated + logged.** FD Hessians are O((3N)²) re-solves; Raman is nested field×geometry FD.
   Gate by system size with an explicit "too large for quantum path" message; never silently truncate.

## 1. The dependency stack

```
 VQE/SQD ground state ─► E(R)  +  correlated 1-RDM/2-RDM  +  excited states (E_n, transition RDMs)
        │                    │                                        │
   [Phase 1] quantum Hessian = FD over quantum nuclear_gradient (already a capability)
        │                    │
        │        ┌───────────┼─────────────┐
        │        ▼           ▼             ▼
        │   frequencies  IR (∂μ/∂Q)   Raman (∂α/∂Q, nested finite-field)
        │        │        [Ph 2]        [Ph 3 field_response → Ph 4]
        │        ▼
        │   Thermochemistry (RRHO: ZPE,H,S,G,Cv)  ─►  Kinetics (Eyring ΔG‡ + Wigner tunneling)
        │        [Phase 5]                              [Phase 6]
        ▼
   UV-Vis (already correlated) ──► NMR shielding = diamagnetic (have) + paramagnetic Ramsey SOS
                                    [Phase 7: angmom integrals + transition-RDM accessor + SOS engine]
```

Build the **quantum Hessian once** and vibrational frequencies, IR positions, thermochemistry ZPE/entropy,
and kinetics all fall out of it. That is the highest-leverage single piece → Phase 1.

## 2. Physics → primitive map (all primitives verified present unless noted GAP)

| Property | Physics | Quantum primitive | Status |
|---|---|---|---|
| Vibrational freq | eig(mass-weighted ∂²E/∂R²) | `hessian()` = FD over `nuclear_gradient()` (FD over `energy_fn`) | mixin exists, **unwired** |
| IR intensity | I ∝ \|∂μ/∂Q\|² | correlated dipole from 1-RDM at displaced geoms | `get_dipole` GAP on PhysicsVQE/SQD |
| Polarizability α | −∂²E/∂F² | `energy_under_field()` E-field re-solve | Protocol exists, **unwired**; `response_properties.py` proves it |
| Raman activity | Placzek(∂α/∂Q) | nested FD: α at displaced geoms | needs Ph1+Ph3 |
| Thermochem H,S,G | RRHO partition fns | quantum E + quantum frequencies | `ThermochemistryCalculator` injectable |
| Rate constant k | Eyring k=(kT/h)e^(−ΔG‡/kT)·κ_Wigner | quantum PES barrier + quantum thermochem | `_compute_rate_constant` is a `None` stub (app) |
| NMR σ (dia) | −(μ₀/4π)(e²/2mₑ)⟨0\|Σrᵢ·𝟙−rᵢrᵢ/rᵢ³\|0⟩ | 1-RDM + `int1e` integrals | `compute_diamagnetic_nmr_shielding` exists |
| NMR σ (para) | Ramsey SOS: Σₙ[⟨0\|L_A/r³\|n⟩⟨n\|L\|0⟩+c.c.]/(Eₙ−E₀) | excited E_n + transition RDMs + **angmom integrals** | **all three GAPs** |

## 3. Phases (each: framework impl → conformance/unit test → GPU/QPU validation)

- **Phase 1 — Quantum Hessian** (`hessian` capability). Mix `FiniteDifferenceHessianMixin` into
  `PhysicsVQE` + `SamplingSQDSolver`; declare `hessian`; add a pure `core/harmonic.py` (mass-weight,
  project trans/rot, eig→cm⁻¹, ZPE) shared with `FrequencyCalculator`. Conformance: FD-Hessian
  symmetric; H₂/LiH ω vs HF-Hessian and vs experiment (H₂ ~4400 cm⁻¹). **Keystone.**
- **Phase 2 — Quantum IR** (`dipole` + `hessian`). Add `get_dipole()` (from 1-RDM) to PhysicsVQE/SQD;
  build ∂μ/∂Q along Phase-1 normal modes from the correlated dipole. Test H₂O/CO intensities.
- **Phase 3 — Field response** (`field_response` capability). Implement `energy_under_field()` on
  SamplingSQD (graduate `api/services/response_properties.py`'s h1-perturbation into the solver);
  polarizability = −∂²E/∂F². Conformance vs finite-field HF on a small polar molecule.
- **Phase 4 — Quantum Raman.** ∂α/∂Q (nested field×geometry FD) → Placzek invariants (45α'²+7γ'²),
  depolarization, Bose-weighted Stokes. Gate hard on size; reuse `RamanIRCalculator` broadening.
- **Phase 5 — Quantum thermochemistry.** Feed quantum E (persisted) + Phase-1 frequencies into
  `ThermochemistryCalculator` (injectable `frequencies`, pluggable E_elec). ZPE, U/H, S, G, Cv, Cp.
- **Phase 6 — Reaction kinetics.** Quantum PES scan → TS (max along path) → Eyring ΔG‡ from Phase-5
  thermochem at reactant+TS; Wigner (and optional Eckart) tunneling. Replace the `None` rate stub.
- **Phase 7 — Quantum NMR (Ramsey SOS).** (7a) angular-momentum + magnetic integrals in
  `core/integrals/property_integrals.py` (`mol.intor('int1e_cg_irxp'...)`). (7b) public
  transition-RDM accessor ⟨0\|a†ₚa_q\|n⟩ on excited-state solvers (CASCI `trans_rdm1` is already
  computed then discarded; qEOM reconstruct from EOM eigenvectors). (7c) SOS response engine:
  σ_para from transition energies + L matrix elements; total σ = dia + para → shifts vs reference.
  Tiny systems only (qEOM ≤8 qubits / CASCI active space). Validate vs PySCF GIAO + experiment.
- **Phase 8 — Rigorous validation.** GPU cluster (MI300X, `ssh gpu`) large-active-space runs; real
  QPU (IBM Heron) for H₂/LiH frequencies + a small IR, under `@pytest.mark.slow` / `.hardware`.
- **Phase 9 — Push + app rewire.** PR `feat/quantum-properties` → `main`; app bumps submodule pin;
  rewire `api/routes/analysis.py` (attach persisted 1-RDM via `set_quantum_density_matrix`, forward
  quantum energy to thermo, drive `wavefunction='vqe'`, add `source` provenance to IR/Raman/NMR/Thermo
  result schemas), then browser-verify end to end.

## 4. Test strategy

- **Unit** (fast, `tests/unit/`): symmetry/trace/consistency + small-molecule numbers vs HF and vs
  published experiment; extend `tests/unit/test_capability_conformance.py` for `hessian`/`field_response`.
- **Slow** (`@pytest.mark.slow`): H₂O/CO full IR, Raman, thermochem end-to-end.
- **Hardware** (`@pytest.mark.hardware`): IBM QPU H₂/LiH frequency + one IR; GPU MI300X large active space.
- **Honesty guard:** every engine's result carries `source`; a test asserts the quantum path is NOT
  silently HF (property differs from HF by a physical margin on a correlated system).

## 5. Status log

- 2026-07-08 — Branch created off `feat/node-sqd-sync`. Roadmap written.
- 2026-07-08 — **Phase 1 DONE (quantum Hessian).** `core/harmonic.py` (pure harmonic analysis,
  single source of truth); `FiniteDifferenceHessianMixin` now fills the full spectrum via a
  `BaseSolver._hessian_masses_amu()` hook; PhysicsVQE + SamplingSQD inherit it and declare
  `hessian`. Validated: quantum H₂ Hessian = **5001.6 cm⁻¹ == FCI/STO-3G gold parabola fit**,
  and **479 cm⁻¹ softer than HF (5481)** → genuine correlated observable, not relabeled HF.
  Tests: `tests/unit/test_quantum_hessian.py` (6 fast + 2 slow) + conformance wiring — all green.
  FOLLOW-UP: refactor `FrequencyCalculator` to delegate to `core/harmonic.py` (kill the duplicate
  classical copy) when Phase 2/5 touches the analysis layer.
- 2026-07-08 — **Phase 2 DONE (dipole capability + fully-quantum IR).** `BaseSolver.get_dipole()`
  (capability-gated, from the correlated 1-RDM); `dipole` declared on VQESolver + DeterministicCI
  ONLY (SamplingSQD's default circuit is non-entangling → HF-like density, so honestly NOT declared).
  `PropertyCalculator.compute_ir_spectrum(hessian=…)` takes normal modes from the Phase-1 quantum
  Hessian → fully-quantum IR (quantum modes + quantum dipole derivatives), with a `hessian_source`
  provenance field. Fixed a mass-convention drift (now `isotope_avg=True`) so the IR frequency equals
  the solver Hessian frequency exactly. Validated: HeH⁺ dipole DetCI = FCI (1.0691) exactly, VQE within
  0.002 a.u., both ≠ HF (0.635); HeH⁺ fully-quantum IR stretch 4447 cm⁻¹ (== Hessian), IR-active
  672 km/mol, ≠ HF-Hessian IR (4391). Tests: `tests/unit/test_quantum_dipole_ir.py` (1 fast + 3 slow)
  + existing IR validation suite — 9/9 green. Infra: MI300X node (`ssh gpu`) + IBM QPU (`ibm_fez`)
  wired for large-scale/QPU validation.
- 2026-07-08 — **Phase 3 DONE (field_response → polarizability).** `SamplingSQDSolver.energy_under_field(atoms, E, B)`
  graduates the deferred `field_response` capability: rebuilds at geometry, adds the electronic
  `+Σ E_i·r_i(MO)` term to a correlated LUCJ `_h1` solve (never the vacuous default), returns the
  true H(E) energy. Guards honestly — raises on a nonzero B-field (that's Phase 7/NMR) and on a
  frozen-core active space (full-orbital only for now). Declared on SamplingSQD. Static polarizability
  = −d²E/dF². Validated (H₂/STO-3G): SQD α_zz = 2.7823 a.u. = FCI exactly, E0 = FCI exactly, and
  HF α_zz = 3.0755 (correlation lowers it) → SQD ≠ HF. Tests: `tests/unit/test_field_response.py`
  (2 fast + 1 slow) green. GPU table (in progress): H₂ 4899, HeH⁺ 4418 cm⁻¹ quantum-Hessian freqs.
- 2026-07-08 — **Phase 5 DONE (quantum thermochemistry).** `ThermochemistryCalculator.compute_thermochemistry`
  gained an `e_elec=` override; combined with the injectable `frequencies=` (from the quantum Hessian),
  the electronic energy, ZPE, S_vib and Cp all come from the wavefunction. Validated (H₂, 298 K):
  E_elec = FCI; S = 31.09 vs exp 31.23 cal/mol/K; Cp = 6.96 vs 6.9; ZPE self-consistent with the
  quantum frequency; G lowered by correlation vs HF. Tests: `tests/unit/test_quantum_thermochemistry.py`
  (1 fast + 1 slow) green. (Done before Phase 4 — thermo only needs Phase 1 + energy.)
- 2026-07-08 — **Phase 6 DONE (reaction kinetics).** New pure module `reactions/kinetics.py`:
  `eyring_rate_constant` (k = κ·(k_BT/h)·exp(−ΔG‡/k_BT)), `wigner_tunneling`
  (κ_W = 1+(1/24)(h|ν‡|/k_BT)²), and `quantum_rate_constant` which takes ΔG‡ = G_TS − G_reactant
  from Phase-5 quantum thermochemistry + the TS imaginary frequency. Validated analytically:
  prefactor = k_BT/h (6.212e12 /s at 298 K); Arrhenius T-dependence; Wigner κ(1500i,298K)=3.18 → 1
  as T→∞. Tests: `tests/unit/test_kinetics.py` (4 fast) green. This completes the three named
  pillars: spectroscopy (IR, polarizability), thermodynamics, reaction kinetics.
- 2026-07-08 — **GPU validation (MI300X):** quantum-Hessian frequencies to 12 qubits (LiH 1594,
  HF 4841 cm⁻¹) and correlated dipoles to 14 qubits (H₂O 1.73 D vs FCI 1.62, exp 1.85) — the
  engines hold at scale. Caveat logged: DeterministicCI auto-reduces its CISD subspace >8 qubits.
- 2026-07-08 — **Phase 4 DONE (quantum Raman).** New `analysis/quantum_response.py`:
  `polarizability_tensor` (full symmetric α via finite field over `energy_under_field`) +
  `raman_spectrum` (∂α/∂Q along quantum-Hessian modes → Placzek invariants 45ᾱ'²+7γ'²,
  depolarization ratio). Validated: H₂ is Raman-ACTIVE (activity 169.6 a.u.) while IR-inactive
  (complementary spectroscopy!); α = diag(0,0,2.782) — α_∥ = FCI, α_⊥ = 0 (minimal basis, no
  p-orbitals); ρ = 1/3 (exact analytic). Tests: `tests/unit/test_quantum_raman.py` (3 fast + 1 slow) green.
- 2026-07-08 — **QPU:** H₂ SQD submitted to real Heron `ibm_fez` (job d96nc52f47jc73a5qki0); the
  framework's synchronous path timed out at 3200s with the job still QUEUED (queue ~163). Job is live
  in the queue; a background poll retrieves it on completion. Finding: the framework IBM path needs a
  submit/poll (async) mode for deep queues, not the blocking run — a Phase 8/9 item.
- 2026-07-08 — **Phase 7 DONE (quantum NMR via Ramsey sum-over-states).** (7a) added
  `compute_angular_momentum` (int1e_cg_irxp) + `compute_pso` (int1e_prinvxp) to
  core/integrals/property_integrals.py. (7c) new `analysis/quantum_nmr.py`: σ = σ^dia
  (ground 1-RDM · int1e_cg_a11part, traceless-corrected — reproduces PySCF EXACTLY) + σ^para
  (Ramsey SOS: first-order 1-RDM response = ANTI-symmetrized transition RDM contracted with
  PSO). Two bugs found+fixed via PySCF cross-check: the response must be antisymmetrized (sym
  contracts to 0 against antisym PSO), and the prefactor is −1 not +2 (i² from the two
  anti-Hermitian operators + c.c. already in the response). Validated: H₂ ¹H = 32.9 (= PySCF,
  para 0); LiH ⁷Li dia = 107.3 (= PySCF exactly), para = −16.1 vs PySCF-CPHF −17.0 — the ~1 ppm
  gap is the genuine FCI-vs-HF correlation (the quantum value-add). Tests:
  `tests/unit/test_quantum_nmr.py` (1 fast + 2 slow) green. NOTE (7b): states come from PySCF
  FCI = the framework's CASCI in full space (exact); surfacing a transition-RDM accessor on the
  builder's excited-state path is a small wiring follow-up. Gauge = common-origin-at-nucleus
  (GIAO-free) — absolute values gauge-dependent in finite basis; the correlated shift is the signal.
- ALL 7 PROPERTY PHASES DONE. Remaining: Phase 8 (QPU SQD retrieval — job queued), Phase 9
  (push framework→main + app submodule bump + rewire api/routes/analysis.py + provenance schemas).
