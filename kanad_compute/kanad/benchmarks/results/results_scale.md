# Kanad Large-Scale Benchmarks — SQD on real IBM Heron + Phase-1 observables/dynamics

**Captured:** 2026-05-28. These are the high-level (20–40 qubit) runs that the
Tier-1 table in [`results.md`](results.md) does not cover. Several were executed
on **real IBM Heron (`ibm_marrakesh`)** hardware via sample-based quantum
diagonalization (SQD); the raw job results live on IBM's side under the job IDs
below. **Provenance matters here** — this file exists because the original run
logs were ephemeral (`/tmp` on the GPU cluster) and M11a/M11b were already lost.

Method legend:
- **SQD** = sample bitstrings from a (noisy) LUCJ circuit → configuration
  recovery → selected-CI diagonalization → iterative subspace expansion
  (Robledo-Moreno 2025). Energy is variational (≥ exact in the active space).
- **CASCI** = classical exact diagonalization in the same active space (the
  reference SQD targets). Infeasible past ~(18,18) — that's the point of SQD.
- Backend `ibm_marrakesh` = IBM Heron r2, 156 qubits.

---

## M11 — SQD champions on real Heron

| ID | System | Active space | Qubits | Backend | Shots | Job ID | E_HF (Ha) | E_CASCI (Ha) | **E_SQD (Ha)** | Gap to CASCI |
|----|--------|--------------|:------:|---------|:-----:|--------|----------:|-------------:|---------------:|-------------:|
| M11c | naphthalene (C₁₀H₈) | (16e,16o)/cc-pVDZ | **32** | ibm_marrakesh | 30 000 | `d8bk0bb8amns73bia0dg` | −382.728660 | −382.780015 | **−382.779771** | **+0.244 mHa** |
| M11d | [Fe₂S₂Cl₄]²⁻ | (20e,20o), Fe ECP+cc-pVDZ | **40** | ibm_marrakesh | 40 000 | `d8bkloajki0s73aq2ong` | −2878.081946 | *infeasible* | **−2878.201451**† | n/a (beyond CASCI) |

† M11d iterative expansion **converged** at iteration 3 (138 749 dets, 743 s; iter 3
added no new determinants). N_FCI for (20,20) = C(20,10)² ≈ **34.1 billion**
determinants — classical CASCI is intractable, so SQD recovered **+119.5 mHa** of
correlation over HF with **no classical reference possible**. This is the headline
"beyond-classical-CASCI" result. The recovery ladder also shows *why* recovery is
non-negotiable on noisy hardware: **drop-only** (discard invalid bitstrings, no
recovery) gave −2877.443 Ha — **639 mHa _above_ HF** (1.8% of shots survived);
**single-shot recovery** → −2878.199 (+116.6 mHa); **iterative** → −2878.201 (+119.5 mHa).

### M11c recovery ladder (why iterative SQD matters)

The same 30 000 Heron shots, three post-processing levels:

| Level | E_SQD (Ha) | Gap to CASCI | # determinants |
|-------|-----------:|-------------:|---------------:|
| drop-only (no recovery) | −382.729966 | 50.05 mHa | 1 424 |
| single-shot recovery | −382.761794 | 18.22 mHa | 18 323 |
| **iterative expansion** | **−382.779771** | **0.244 mHa** | 94 004 (609 s) |

Configuration recovery + iterative subspace expansion closes a 50 mHa raw-hardware
gap to **chemical accuracy** (0.244 mHa) on a 32-qubit system. Circuit: LUCJ
1-layer, 91 two-qubit gates pre-transpile → 542 post-transpile, depth 1098.

### M11a / M11b — hardware logs lost; spin-correct reference targets recovered

M11a (N₂(10e,10o)/cc-pVDZ, 20q) and M11b (C₂ at rₑ, 24q) ran on Heron earlier but
their **hardware SQD result logs were lost** from `/tmp`. The classical CASCI
targets they aimed for are reproducible and recomputed here via the unified
builder's spin-correct CI route (`fix_spin_(ss=0)`):

| ID | System | Active space | Qubits | **CASCI(singlet)** | source |
|----|--------|--------------|:------:|-------------------:|--------|
| M11a | N₂ @ 1.10 Å | (10e,10o)/cc-pVDZ | 20 | **−109.048232 Ha** | `MolecularBuilder…solver('ci')` |
| M11b | C₂ @ 1.243 Å | (12e,12o)/cc-pVDZ | 24 | **−75.567884 Ha** | `MolecularBuilder…solver('ci')` |

**Note (SQD<CASCI investigation):** an apparent "SQD below CASCI" on these was
traced to the *classical* CASCI reference, not SQD — PySCF's default Davidson
converged to a triplet on C₂. Trustworthy anchors require `conv_tol=1e-12,
fix_spin_(ss=0)` — now built into the builder's CI route (`builder/quantum_system.py`).
At 20–24 qubits the builder **auto-routes to exact CI, not SQD** (CASCI is only
63 504 / 853 776 determinants — trivial); SQD earns its keep at 32q+ (M11c/M11d)
where CASCI is infeasible.

---

## M4 — observables at scale (SQD 1-RDM)

Validates the SQD → 1-RDM → property pipeline on the 32-qubit naphthalene
wavefunction (M11c Heron samples, single-shot recovery, E = −382.761794).

| Observable | System | HF | CCSD | SQD(Heron) | Check |
|------------|--------|---:|-----:|-----------:|-------|
| Dipole \|μ\| | naphthalene (D₂h) | 2.74e-12 D | 2.43e-12 D | 8.21e-08 D | ✓ respects centrosymmetry (<1 mD) |

The SQD 1-RDM is extracted from the embedded selected-CI eigenvector (25.5 s at
32q) and the resulting dipole vanishes by D₂h symmetry, matching HF/CCSD — i.e.
the quantum density is physically correct, not an HF fallback.

---

## M5 — forces & dynamics at scale

Numerical nuclear forces (central finite-difference of total SQD energy → captures
Pulay automatically) and a short quantum MD, on H₂O CAS(8e,6o)/cc-pVDZ (12 qubits),
statevector SQD.

| Quantity | Result | Threshold | Reference |
|----------|-------:|----------:|-----------|
| Max \|ΔF\| (all 9 components) vs CASCI analytic gradient | **13 µHa/Bohr** | <2 mHa/Bohr | PySCF CASCI(8,6) `nuc_grad_method` |
| 5-step velocity-Verlet MD energy drift | **0.0068 mHa** | <1 mHa | NVE conservation |

The MD shows correct E_pot↔E_kin exchange (E_tot flat at −76.0313 Ha across 5
steps). Forces are consistent enough across geometries to run credible quantum MD.

### Reaction on real hardware — N₂ dissociation on Heron (COMPLETE)

N₂(10e,10o)/cc-pVDZ dissociation scan, **20 qubits, 5 geometries in one
`ibm_marrakesh` allocation** (job `d8bu8pajki0s73aqfdvg`, 20 000 shots each).
SQD post-processing of the hardware bitstrings vs classical CASCI(10,10):

| r (Å) | E_SQD (Heron) | CASCI(10,10) | Δ (mHa) |
|------:|--------------:|-------------:|--------:|
| 1.00 | −109.003756 | −109.003781 | 0.025 |
| 1.10 (min) | −109.048190 | −109.048232 | 0.042 |
| 1.25 | −109.018313 | −109.018434 | 0.121 |
| 1.50 | −108.922514 | −108.923020 | **0.51** |
| 2.00 | −108.775587 | −108.775827 | 0.24 |

A full **bond-breaking profile on real quantum hardware** reproduces CASCI(10,10)
to **≤0.51 mHa across the whole curve** — sub-chemical-accuracy even at the
stretched, most multireference point (r=1.5 Å). This is the M7 "one Heron-routed
reaction" deliverable: a reaction PES computed on a QPU, validated against the
exact classical reference. Reproduce:
`python -m benchmarks.m5_heron_reactions_dynamics --reaction --poll d8bu8pajki0s73aqfdvg`.

---

## M7 — reactions on SQD (via the builder)

`QuantumReactionSimulator.from_system(builder_system)` makes every reaction
routine (TS / IRC / dissociation) **solver-agnostic**: it consumes the builder's
geometry-parametric `energy_fn`, so the reaction runs on whatever solver the
builder routes to — SQD on Heron for large active spaces, exact CI/VQE otherwise
— with warm-starting threaded between geometries. Provenance is captured via
`to_provenance()` → JSON.

Validation:

| Test | System | Result |
|------|--------|--------|
| SQD lane == CI lane | H₂ dissociation (3 pts) | **0.0 mHa** max diff — reactions on SQD are exact-equivalent to CI |
| Reaction scan at scale | N₂(10,10)/cc-pVDZ dissociation (20q, 5 pts) | matches PySCF **CASCI(10,10) to 0.0 mHa** at every point |
| Physics | N₂ 1.0 → 2.2 Å | minimum at 1.10 Å; +178.8 kcal/mol on bond-breaking |

At 20q the builder correctly routes to **exact CI** (CASCI is trivial); the SQD
lane is proven exact-equivalent on H₂, and the scale demonstration on hardware is
the queued Heron N₂ dissociation (`d8bu8pajki0s73aqfdvg`, parked on QPU
maintenance). Tests: `tests/validation/test_reactions_sqd.py`.

## M8 — excited states at scale (finding)

Naphthalene S₁/T₁ at **32 qubits** from the existing M11c Heron data (job
`d8bk0bb8amns73bia0dg`): rebuilt the selected-CI subspace (→128k dets, E₀ within
0.27 mHa of CASCI) and extracted the low-lying spectrum via
`solve_excited_states`. Result:

| state | ΔE (eV) | note |
|---|---:|---|
| S0 | 0.000 | ground (−382.7799 Ha, 0.27 mHa from CASCI) |
| 1 | 4.268 | — |
| 2 | 5.082 | — |
| lit. | T1 ~2.6, S1 ~4.0, 1La ~4.5 | experiment / CASPT2 |

**Finding — algorithm resolved; scale/triplet sampling is the open frontier (2026-05-28).**

*Algorithm fix (resolved + validated):* the original expansion grew only from the
ground eigenvector → too ground-biased. **State-averaged selected CI**
(`solve_excited_states_iterative`) expands from the top determinants of *all* target
states each round. On **H₂O CAS(8,6)/12q** (where the subspace reaches full CASCI) SQD
excited energies now equal PySCF CASCI `nroots` to **0.0 mHa** (vs 2.1 eV with the
ground-only subspace). Algorithm is correct.

*Scale result (naphthalene 32q, state-averaged):* singlet excited manifold roughly
recovered (state 1 = 4.22 eV, in the S1/1La 4.0–4.5 eV range), but **T1 (~2.6 eV) is
still missing**. Root cause: the subspace is built from a **singlet ground-state**
Heron sample, which under-represents the open-shell determinants the lowest triplet
needs — and 250k of the 165M-determinant FCI space is too sparse to reach them by
expansion alone. **Excited-state SQD at scale — especially low triplets — needs
excited-state-/spin-targeted *sampling*** (new circuits biased toward the target
manifold), not just post-processing ground-state samples. This is a research
direction, not a post-processing fix. The **CI route gives exact excited states +
oscillator strengths ≤30 qubits** — the production path for chromophores at that scale.

---

## M10 — observables, density grids, active-space picker, honest descriptors

All four pieces drive off the unified builder's solved wavefunction (CI/SQD 1-RDM),
not an HF fallback.

| Test | System | Scale | Result |
|---|---|---|---|
| **.cube density + ESP** | benzene / cc-pVDZ | 114 AOs, frontier CAS, 12 q | E = −230.756 Ha; ∫ρ dV = **41.7 ≈ 42 e⁻**; \|μ\| = 0.000 D (D₆ₕ ✓); M-diag 0.21 (π multireference). `M10_CUBE_SCALE_OK` |
| **AVAS picker** (atom-targeted) | benzene / cc-pVDZ | `ao_labels=['C 2pz']` | auto-selects the exact textbook **CAS(6,6)** π space (6 orb / 6 e⁻); E = −230.791 Ha |
| **AVAS picker at scale** | naphthalene / cc-pVDZ | **180 AOs**, 20 q | recovers the **10-orbital π-space dimension** from `['C 2pz']` alone; default threshold 0.2 partitions it CAS(8,10) (occ/virt split is threshold-tunable, *not* forced to the textbook CAS(10,10)); E = −383.123 Ha. `M10_PICKER_SCALE_OK` |
| **conceptual-DFT reactivity** | H₂O (χ/η/ω); naphthalene | — | H₂O gap 18.48 eV (= pyscf HF), χ 4.18, η 9.24, ω 0.95 eV; naphthalene gap 7.32, η 3.66, ω 2.31 eV (softer/smaller-gap aromatic — physical) |

**Active-space picker** now offers two automated methods, both returning the
AVAS/NO rotation as `mo_coeff` so they compose with the integral transform: **NO-MP2**
(occupation-based, general) and **AVAS** (atom/orbital-targeted via `ao_labels` — the
"bond-physics-aligned" picker for π systems / metal centers). Both are geometry-
dependent, so they raise on a scan (continuity guard) rather than silently flipping.

**Honesty pass (A7 closed).** The old `adme_calculator.py` — logP from molecular
weight, Caco-2/PAMPA/BBB/PPB from hand-tuned step functions, all labelled "quantum
ML" — was **deleted**. Replaced by `analysis/molecular_descriptors.py`, which keeps
two families strictly separate: validated **RDKit physicochemical** descriptors
(Crippen logP, Ertl TPSA, H-bond counts; raises ImportError with no fake fallback)
and genuinely-quantum **conceptual-DFT reactivity** indices (χ, η, S, ω in Koopmans'
approximation, `source`-labelled) from the solved frontier orbitals, plus
Lipinski/Veber/Ghose rule filters. Fixing this surfaced and fixed a latent bug in
`observables()['homo_lumo_gap_ev']` (it indexed the full `mo_energy` with the
*active* electron count → wrong frontier gap for any active-space system).

## Reproduction

All scripts submit/poll via env vars `IBM_QUANTUM_TOKEN` + `IBM_QUANTUM_CRN`:

```
python -m benchmarks.m11c_naphthalene_heron --poll d8bk0bb8amns73bia0dg
python -m benchmarks.m11d_fe2s2_heron       --poll d8bkloajki0s73aq2ong
python -m benchmarks.m5_heron_reactions_dynamics --reaction --poll d8bu8pajki0s73aqfdvg
```

Statevector cross-checks (no hardware/credentials needed) can now be driven
through the unified builder, e.g.:

```python
from kanad import MolecularBuilder
qs = (MolecularBuilder.from_atoms([('N',(0,0,0)),('N',(0,0,1.10))]).basis('cc-pvdz')
        .active_space('manual', frozen=[0,1], active=list(range(2,12)))
        .solver('sqd').build())
qs.solve()['energy']
```

M10 scale tests (cluster):

```
python -m benchmarks.m10_cube_scale      # benzene 114 AOs → density + ESP cubes
python -m benchmarks.m10_picker_scale    # naphthalene 180 AOs → AVAS π picker + reactivity
```

M10 builder surface (statevector):

```python
# AVAS atom-targeted active-space picker
qs = (MolecularBuilder.from_atoms(benzene).basis('cc-pvdz')
        .active_space('avas', ao_labels=['C 2pz']).build())   # → CAS(6,6) π
qs.solve()
qs.export_cube('rho.cube', kind='density')                    # quantum density → VMD
obs = qs.observables('core')                                  # dipole, NOONs, M-diag, gap
react = qs.reactivity_descriptors()['quantum_reactivity']     # χ, η, S, ω (conceptual DFT)
```
