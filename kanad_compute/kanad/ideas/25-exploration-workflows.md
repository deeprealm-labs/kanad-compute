# Idea 25 — Exploration workflows (one-liners for what researchers actually do)

**Lands in:** M8/M9 alongside the observables plate (Phase 2)
**Effort:** ~600 lines new + workflow notebooks, 3–4 weeks
**Blocks:** kanad-app's user-facing workflows; industrial deployment usability
**Prerequisites:** M2 (Tier-1 VQE), M5 (forces + reactions), M7 (2-RDM), M8 (observables plate)

---

## Problem

Researchers don't think in terms of *"run VQESolver"*. They think:

- *"What does the potential-energy surface look like for this molecule's bond?"*
- *"What's the transition state for this reaction?"*
- *"What's the UV-Vis spectrum of this chromophore?"*
- *"Rank these 50 drug fragments by binding-affinity proxy."*
- *"Compare these three conformers' energies and dipoles."*
- *"Plot the density evolution along this reaction coordinate."*

Each of these is a *workflow* — a recipe combining several primitive operations (VQE, observables, geometry generation, plotting). Today, doing any of them in Kanad requires writing ~50 lines of glue. **A useful framework ships the workflows as one-liners.** That's the difference between *"toolkit"* and *"developer kit."*

This idea catalogues the 8-10 most common chemistry-research workflows and specs them as Kanad one-liners. Each one is itself a Python function plus a CLI entry plus a kanad-app API route.

---

## The eight workflows

### W1 — Potential-energy-surface (PES) scan

```python
from kanad.workflows import pes_scan

results = pes_scan(
    molecule="H 0 0 0; H 0 0 R",
    parameter='R',
    values=np.linspace(0.5, 4.0, 30),
    method='vqe',                       # or 'vqe+observables' for full plate per point
)
results.plot()                          # energy curve
results.observable('dipole').plot()     # any observable as function of R
results.to_csv('pes.csv')
```

**Use cases:** bond-breaking studies (academic + industrial), reactivity exploration, equilibrium-geometry refinement.

### W2 — Transition-state search + IRC

```python
from kanad.workflows import find_ts_and_irc

ts = find_ts_and_irc(
    reactant_xyz='reactant.xyz',
    product_xyz='product.xyz',
    method='neb',                       # or 'dimer'
    n_images=8,
)
ts.barrier_height                       # ΔE_TS (in kcal/mol)
ts.irc_forward.plot()                   # forward IRC trajectory
ts.irc_backward.plot()
ts.eyring_rate(temperature=298)         # rate constant at T
```

**Use cases:** mechanism studies (academic synthesis, catalysis), kinetics prediction for industrial chemistry, drug-synthesis route planning.

### W3 — UV-Vis spectrum prediction

```python
from kanad.workflows import predict_uv_vis

spec = predict_uv_vis(
    molecule="H2CO",
    n_states=10,
    method='qeom-vqe',
)
spec.plot()                             # broadened Gaussian/Lorentzian spectrum
spec.peaks_table                        # table of λ_max, oscillator strengths, transitions
spec.nto_for_state(1)                   # natural transition orbital for S₁
```

**Use cases:** chromophore design (academic photochemistry, industrial OLEDs/solar-cells/dyes), photocatalysis screening.

### W4 — IR / Raman / NMR prediction

```python
from kanad.workflows import predict_spectra

result = predict_spectra(
    molecule="caffeine",
    types=['ir', 'raman', 'nmr_h', 'nmr_c'],
    method='vqe',
)
result.ir.plot()                        # IR spectrum
result.raman.plot()
result.nmr_h.peaks_table                # H NMR chemical shifts + couplings
```

**Use cases:** experimental-spectrum assignment (academic + industrial NMR for QA), product confirmation in synthesis, drug-impurity identification.

### W5 — Conformer comparison

```python
from kanad.workflows import compare_conformers

conformers = compare_conformers(
    molecule_xyz_list=['conf1.xyz', 'conf2.xyz', 'conf3.xyz'],
    properties=['energy', 'dipole', 'homo_lumo_gap', 'polarizability'],
)
conformers.relative_energies            # in kcal/mol
conformers.ranking('dipole')            # sorted by dipole magnitude
```

**Use cases:** drug-conformer ranking (industrial pharma), materials polymorph energetics, solvation-mode comparison.

### W6 — Fragment screening (industrial throughput)

```python
from kanad.workflows import screen_fragments

results = screen_fragments(
    smiles_list=['CC(=O)O', 'C1=CC=CC=C1', ...],  # 50 SMILES strings
    properties=['homo', 'lumo', 'dipole', 'polarizability', 'pka_proxy'],
    parallel=8,
)
results.to_dataframe()                  # pandas DataFrame for pipeline integration
results.filter(homo_lumo_gap=(3.0, 6.0)).export('hits.sdf')
```

**Use cases:** drug-fragment library screening, materials cluster screening, catalyst-ligand library evaluation.

### W7 — Density / observable difference along reaction

```python
from kanad.workflows import reaction_observable_movie

movie = reaction_observable_movie(
    reactant_xyz='reactant.xyz',
    product_xyz='product.xyz',
    observable='density',                # or 'spin_density', 'elf', 'fukui'
    n_frames=20,
)
movie.save('reaction.mp4')              # animated grid evolution
```

**Use cases:** reaction-mechanism visualization (papers, presentations), teaching tools, photochemistry pathway visualization.

### W8 — Bonding analysis (single molecule, deep)

```python
from kanad.workflows import bonding_analysis

report = bonding_analysis(
    molecule="benzene",
    method='vqe',
)
report.bond_orders                      # Mayer bond orders for every atom pair
report.aromaticity                      # NICS, ELF-π
report.lone_pairs                       # ELF basin centers
report.print_summary()                  # 1-page summary
```

**Use cases:** bonding-character studies, new-ligand design, organometallic mechanism (relevant academically + in catalysis R&D).

---

## What each workflow provides

| Aspect | Spec |
|---|---|
| **One-line invocation** | `from kanad.workflows import X; X(...)` |
| **Reasonable defaults** | The user doesn't have to know which solver / ansatz / active-space picker to use |
| **Progress tracking** | Logs to console + writes JSON intermediate state every step |
| **Reproducibility** | Output includes git commit, Kanad version, parameters in JSON header |
| **Visualization** | `.plot()` method returns a matplotlib figure (or savable .png/.svg) |
| **Export** | `.to_csv()`, `.to_json()`, `.to_sdf()` for pipeline integration |
| **Programmatic + CLI** | Each workflow has a Python API + a CLI entry (`kanad pes-scan --molecule H2 --range 0.5,4.0,30`) |
| **kanad-app integration** | Each workflow has a corresponding API route in kanad-app |

---

## Solution

```
kanad/workflows/
├── __init__.py                # exports the 8 workflow functions
├── pes_scan.py
├── ts_and_irc.py
├── uv_vis.py
├── spectra.py                 # IR / Raman / NMR
├── conformers.py
├── fragment_screen.py
├── reaction_movie.py
├── bonding_analysis.py
└── _shared/
    ├── results.py             # WorkflowResult base class with .plot/.export
    ├── progress.py            # progress logging / checkpointing
    └── parallel.py            # threading / multiprocessing utilities
```

Each workflow ~150 LOC. Each composes the existing primitives (VQESolver, observables plate, density analysis grids, reaction primitives, qEOM excited states). No new physics — just well-designed composition.

---

## Done criteria

- All 8 workflows have a Python API + CLI + kanad-app route + example notebook.
- **W1 (PES scan):** H₂ scan from 0.5 to 4.0 Å in 30 points runs in <2 minutes on a laptop. Includes energy + dipole + NO-occupation curves.
- **W2 (TS+IRC):** H + H₂ → H₂ + H reaction with NEB(8 images) + IRC produces TS within 1 kcal/mol of literature 9.7 kcal/mol.
- **W3 (UV-Vis):** H₂CO π→π* and n→π* peaks within 0.3 eV of CASPT2 reference.
- **W4 (spectra):** H₂O IR frequencies + 5 NMR proton shifts within 5% of experimental.
- **W5 (conformers):** butane gauche-vs-anti energy difference matches CCSD(T) within 0.2 kcal/mol.
- **W6 (fragment screen):** 20-fragment drug-fragment screen runs in <10 minutes (parallelized).
- **W7 (reaction movie):** H + H₂ density-evolution movie renders cleanly.
- **W8 (bonding analysis):** benzene bonding analysis report runs in <1 minute, shows aromaticity + bond orders + π-system character.

## Dependencies

- All Phase 1 milestones (foundation).
- M7 (2-RDM) — for bonding analysis, observable plate.
- M8 (observables plate) — workflows pipe `result.observables.everything()` into reports.
- M11 (excited states) — for W3 UV-Vis and W7 reaction-state-tracking.
- M13 (density grids) — for W7 reaction movie.

## Test files to add

- `tests/integration/test_workflow_pes_scan.py`
- `tests/integration/test_workflow_ts_irc.py`
- `tests/integration/test_workflow_uv_vis.py`
- `tests/integration/test_workflow_spectra.py`
- `tests/integration/test_workflow_conformers.py`
- `tests/integration/test_workflow_fragment_screen.py`
- `tests/integration/test_workflow_bonding.py`
- `examples/workflows/*.ipynb` — one notebook per workflow

## Notes for the executor

- **Workflows are the user-facing surface.** Even the `VQESolver` is too low-level for most users. The workflow API is what kanad-app exposes and what new users hit first.
- **Reasonable defaults > flexibility.** Each workflow has 1-3 required parameters and the rest defaulted. Power users can override; new users don't have to know what an ansatz is.
- **Progress + checkpointing matter.** A 30-point PES scan or 50-fragment screen takes 30 min. Users want to see progress, want to resume if interrupted, want intermediate output. Build it in.
- **Output format = pipeline contract.** `.to_csv()`, `.to_json()`, `.to_sdf()` are how industrial users plug Kanad into existing pipelines (KNIME, RDKit, internal data lakes). Make sure these work.
- **CLI mirrors Python.** `kanad pes-scan --molecule H2 --range 0.5,4.0,30 --output pes.csv` does the same as the Python call. Industrial users sometimes prefer shell.
- **kanad-app contract:** each workflow has a JSON-schema-versioned input + a JSON-schema-versioned output. kanad-app's UI is a thin renderer of these. Idea 17 (workflow API) extends to cover this.
- **Don't pre-build all 8 at once.** Pareto-priority: W1 (PES) + W4 (spectra) + W5 (conformers) are the most common; ship those first. W2/W3/W6/W7/W8 follow as Phase 2 finishes.
- **Avoid the temptation to over-design.** Each workflow is "compose existing primitives, ship the right defaults." If a workflow needs a *new* primitive (e.g., a new ansatz or a new observable), build the primitive first, then compose. Don't bake one-off physics into a workflow.
