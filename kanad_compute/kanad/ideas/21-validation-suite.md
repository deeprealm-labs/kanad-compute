# Idea 21 — Research-validation suite (15 molecules across research domains)

**Lands in:** M9 (Phase 2 quality-control deliverable)
**Effort:** ~600 lines new + 15 molecule notebooks, 3–4 weeks
**Renamed from:** "DFT-fails benchmark" (2026-05-13) — the comparison framing was wrong; this is a research-validation suite, not a benchmarks-vs-DFT exercise.
**Prerequisites:** M0 (truth pass), M1 (active space + symmetry penalty), M2 (Tier-1 chemistry), M7 (2-RDM), M8 (observables plate)

---

## Problem

A framework that ships claims of "quantum-correlated chemistry on small-to-medium molecules" needs an evidence corpus. Not a comparison-to-DFT benchmark (per [`18-differentiation.md`](18-differentiation.md), we don't define ourselves by beating DFT). A **research-validation suite**: a curated set of 15 molecules that span the research questions academic and industrial users actually bring to a quantum-chemistry tool, with documented Kanad results validated against experiment and high-level classical references.

The suite's job: prove the framework works on the kinds of molecules real users care about. The DFT functional values are *included* in each report — but as context for the user (what would they have gotten from DFT?) — not as the headline-vs-Kanad comparison.

---

## The 15-molecule catalog (organized by research relevance)

Each entry: molecule, why it's research-relevant (academic + industrial), active space, qubit count, properties to validate, reference.

### Tier A — Foundational + mechanism chemistry (5 molecules)

| # | Molecule | Why it matters | Active | Qubits | Validation properties |
|---|---|---|---|---|---|
| 1 | **C₂** | Strong correlation in carbon chemistry; relevant to materials (graphene defects, carbon clusters) and combustion intermediates | (8,8) | 16 | r_eq, D_e, NO occupations (the fractional 4th bond), bond order |
| 2 | **F₂** | Halogen chemistry; bond-breaking reactions; basic photodissociation | (10,10) | 20 | E(R) curve R=1.0–4.0 Å, NO occupations along R, multireference diagnostic |
| 3 | **N₂** stretched | Triple-bond breaking; relevant to nitrogen-fixation chemistry, catalytic ammonia synthesis | (6,6) | 12 | E(R) for R=1.0–3.0 Å vs reference DMRG |
| 4 | **H₂O** | The benchmark molecule; spectroscopic validation; ubiquitous reference | (10,7) | 14 | Energy, dipole, HOMO-LUMO, vibrational frequencies, NMR shielding, IR intensities |
| 5 | **HCN** | Small organic; π-system; relevant to atmospheric and prebiotic chemistry | (10,9) | 18 | Energy, excitation energies (UV-Vis), dipole, IR |

### Tier B — Open-shell + biradical chemistry (4 molecules)

| # | Molecule | Why it matters | Active | Qubits | Validation properties |
|---|---|---|---|---|---|
| 6 | **CH₂ (methylene)** | Smallest carbene; reactive intermediate in synthesis; spin-state ordering question | (6,6) | 12 | ΔE_ST = 9.0 kcal/mol (experimental), spin densities |
| 7 | **O₂** | Atmospheric chemistry; spin-triplet ground state; oxidation reactions | (8,6) | 12 | Triplet ground, spin density per O atom, magnetic moment |
| 8 | **m-benzyne** | Diradical organic; relevant to DNA-damage chemistry, polymer chemistry, drug metabolism | (8,8) | 16 | ΔE_ST = 21 ± 1 kcal/mol, spin density, J coupling |
| 9 | **NO (radical)** | Atmospheric chemistry; biological signaling molecule; spin density on N vs O contested | (8,5) | 10 | Dipole = 0.158 D, spin density, hyperfine A |

### Tier C — Transition-metal chemistry (3 molecules)

| # | Molecule | Why it matters | Active | Qubits | Validation properties |
|---|---|---|---|---|---|
| 10 | **FeO⁺** | Iron-oxygen chemistry; catalysis intermediate; spin-state ordering classically contested | (10,10) | 20 | ⁶Σ⁺ vs ⁴Δ ordering (experimental ⁶Σ⁺), spin density on Fe |
| 11 | **NiH** | Catalysis (hydrogenation); transition-metal-hydride bonding | (10,7) | 14 | r_eq, D_e |
| 12 | **CrO** | Materials (chromium-oxide phases), catalysis | (12,10) | 20 | Ground state, spin density |

### Tier D — Excited-state + photochemistry (3 molecules)

| # | Molecule | Why it matters | Active | Qubits | Validation properties |
|---|---|---|---|---|---|
| 13 | **HCHO (formaldehyde)** | Smallest molecule with rich photochemistry; carbonyl photoexcitation | (10,8) | 16 | S₀, S₁, T₁ energies; oscillator strengths |
| 14 | **m-xylylene** | Larger biradical; ferromagnetic coupling; relevant to organic-magnet design | (10,10) | 20 | ΔE_ST (triplet ground), spin density, J coupling |
| 15 | **HF (hydrogen fluoride)** | Hydrogen-bonding reference; spectroscopic benchmark; ionic-character study | (10,5) | 10 | Dipole, polarizability, NMR shielding, E(R) for bond-breaking |

Three are radicals/open-shell; six are biradicals or strongly correlated; three are transition-metal; the rest span typical organic chemistry. **Every molecule is researched daily across academic and industrial labs.**

---

## What each validation report contains

For each molecule, the runner produces a one-page Markdown report:

```markdown
# {molecule} — Kanad research validation

## Molecule
{name, basis, active space, qubit count, why it matters in research}

## Kanad results (Phase 1+2 framework)
| Property | Value | Reference | Δ |
|---|---|---|---|
| Energy / D_e | -X.XXXXXX | -X.XXXXXX (exp/CCSD(T)/DMRG) | ±0.XX |
| Dipole | X.XX D | X.XX D (exp) | ±0.XX |
| ΔE_ST | X.X kcal/mol | X.X kcal/mol (exp) | ±0.X |
| ... | ... | ... | ... |

## Multireference signature
| Observable | Value | Interpretation |
|---|---|---|
| NO occupations | [1.99, 1.98, ..., 0.18] | Fractional 4th-bond NO present |
| Mayer bond order | 3.42 | Bonding analysis |
| M-diagnostic | 0.14 | Strong multireference character |
| Effective unpaired electrons | 0.62 | Diradical signature |

## Context: DFT functional values
| Method | Value |
|---|---|
| B3LYP | X.XX |
| ωB97X-V | X.XX |
| M06-2X | X.XX |
| **Kanad VQE** | **X.XX** |

(DFT values shown for context. Where they disagree among themselves by > 5 kcal/mol, that's a signal the molecule is in a regime where Kanad's wavefunction-correlated answer adds value.)

## Visualization
- density.cube (electron density grid)
- spin_density.cube
- elf.cube
- fukui_plus.cube

## How to reproduce
python benchmarks/validation/run.py --molecule {name}
```

The DFT values are *context*, not the headline. The headline is the Kanad result + the multireference signature + the validation against experiment / high-level reference.

---

## Solution

```
benchmarks/validation/
├── molecules/
│   ├── 01_c2.py
│   ├── 02_f2.py
│   ├── ...
│   └── 15_hf.py
├── references.py              # exp + DMRG/CCSD(T)/CASPT2 values
├── run.py                     # runs Kanad + (for context) DFT
├── reports.py                 # generates per-molecule Markdown
└── README.md                  # how to reproduce
```

One command — `python benchmarks/validation/run.py --all` — runs all 15 molecules, generates 15 per-molecule reports + one summary table.

---

## Done criteria

- **Tier A (5 molecules):** energy / r_eq / D_e match reference within stated tolerance (1 mHa / 0.005 Å / 0.05 eV). Observable plate exports for each.
- **Tier B (4 biradicals + radicals):** ΔE_ST within ±2 kcal/mol; spin densities qualitatively correct.
- **Tier C (3 transition-metal):** right spin-state ordering; gap magnitude within ±0.3 eV of MR-CI reference (these are hard).
- **Tier D (3 excited-state / photochemistry):** S₀ + S₁ + T₁ energies within ±0.3 eV; oscillator strengths within 30%.
- Per-molecule reports rendered as Markdown + exported as .cube files.
- Summary README in `benchmarks/validation/README.md` linking to all 15.
- CI runs Tier A nightly.
- The README's lead becomes a one-line link to the validation suite: *"15 molecules validated against experiment and reference methods; full per-molecule reports here."*

---

## Dependencies

- M0 + M1 + M2 (Phase 1 foundation).
- M7 (2-RDM) — for the multireference-signature observables.
- M8 (observables plate) — `obs.everything()` is the per-molecule report generator.
- M10 (NO-driven active space) — for the Tier C transition-metals.
- M11 (excited states) — for Tier D.

---

## Test files to add

- `benchmarks/validation/test_runner.py`
- `tests/integration/test_validation_suite_smoke.py`
- `tests/integration/test_validation_reports_render.py`

---

## Notes for the executor

- **Frame the reports positively.** The header is *"Kanad's quantum-correlated answer for {molecule}"* — not *"How Kanad beats DFT on {molecule}."* The DFT functional values are a context-paragraph, not the headline.
- **One reference per number.** Each validation value (experiment, DMRG, CCSD(T), CASPT2) needs a single citable source with explicit method/basis. Don't mix references mid-table.
- **Choose 3 DFT functionals max** for the context column: B3LYP (industry default), ωB97X-V (range-separated; usually best), M06-2X (Truhlar). Don't show 10 functionals — that's a comparison-tool feature; we're a validation suite.
- **Tier C is the hardest.** If FeO⁺ / CrO won't converge cleanly with the NO-driven active space, document the limitation in the per-molecule report — don't pretend it's solved.
- **The validation suite is a *living* artifact.** New molecules added as user demand surfaces. The 15 here are the founding set; the structure is designed for growth.
- **The kanad-app integration:** every per-molecule report becomes a UI page in kanad-app — "Validated molecules" gallery, with the report rendered + the .cube grids loaded into the 3D viewer. The validation suite is also the seed for kanad-app's "example molecules" library.
- **Avoid the temptation to brand this as 'we beat DFT.'** Some Tier B and C molecules genuinely show DFT's limits, but framing it as a beat-down distracts from what the framework *is for*: useful exploration. The DFT columns are context for the reader; nothing more.
