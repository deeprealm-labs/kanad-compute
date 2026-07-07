# Atlas v0 — Seed Design

**Date:** 2026-05-27
**Status:** design spec; layers under the parent strategy at [`2026-05-25-discovery-fuel-phase2-design.md`](2026-05-25-discovery-fuel-phase2-design.md)
**Author:** mk0dz + Claude
**Audience:** the executors who will build Atlas v0; future curators; the operations + scoreboard sub-projects that will consume the atlas.

---

## 0. TL;DR

The parent discovery-fuel spec defines **what** the Fuel Atlas is — a live catalog of publishable open research questions Kanad can answer, indexed by milestone / tier / cluster. This spec defines **how** to actually seed v0: ~250 entries in ~3 weeks via a three-phase hybrid pipeline (benchmark conversion → LLM-assisted lit mining with mechanical anti-fabrication guards → manual cluster-fill), backed by schema validation, an indexer, and CI that mechanically regenerates PLAN.md's per-milestone fuel ledger.

This spec is the implementation contract for the parent's "Atlas v0 seeded — 200+ entries committed to `atlas/`, indexed by milestone + tier + cluster" done criterion.

---

## 1. Relationship to parent spec

| Parent spec defines | This spec defines |
|---|---|
| Discovery-fuel concept (4 axes, 5 tiers, 13 accelerators, scoreboard) | Concrete sourcing pipeline + tooling for Atlas v0 |
| What an atlas entry looks like (template) | Production-ready YAML schema with anti-fabrication fields + machine-parseable capability refs |
| Where atlas lives (`atlas/` at repo root) | Directory structure: `kanad/atlas/entries/`, `kanad/atlas/sourcing/`, `kanad/atlas/validators/`, generated `index.json` |
| That milestones get fuel-ledger sections | Mechanical ledger generator + CI workflow + idea-file injection points |
| Scoreboard exists | Boundary contract (`index.json` schema) so scoreboard work isn't blocked |

This spec does **not** change anything in the parent. It implements one of the parent's done criteria.

---

## 2. Goals and non-goals

### 2.1 Goals (v0 ships when these are true)

1. **~250 atlas YAML entries** distributed proportionally to the parent spec's Appendix C cluster sizes
2. **Zero fabricated literature references** — every `doi` resolves via Crossref; every `excerpt` byte-matches the source abstract/paper
3. **Schema validation in CI** — invalid entries fail PR builds
4. **Mechanical fuel-ledger generation** — `atlas/indexer.py` + `atlas/ledger_generator.py` regenerate `index.json` and idea-file ledger blocks deterministically from atlas entries; CI enforces freshness
5. **Boundary contract documented** — `index.json` schema versioned so operations + scoreboard sub-projects can build against it
6. **No manual milestone-status sync** — entries' `status` and `unlocking_milestone` are computed fields, not stored in YAML

### 2.2 Non-goals (deferred to v1 or downstream specs)

- **Scoreboard dashboard** — needs `index.json`; v0 produces it but doesn't render
- **Operations researcher-scoring integration** (`operations/pipeline/atlas_link.py`) — separate spec; v0 just commits to the contract
- **User-contributed entries / contribution workflow** — v0 is curator-seeded; v1 opens to outside contributors
- **Runtime-result tracking** (user runs an entry, atlas records the result) — atlas is a *question* catalog in v0; results live elsewhere
- **Cluster-recruitment campaigns** — outreach is the operations side and follows v0
- **Sub-discipline tagging beyond the 8 clusters** — v0 has free-form `tags` but no enforced taxonomy

---

## 3. Entry schema

Every atlas entry is one YAML file at `kanad/atlas/entries/ATLAS-NNNN.yaml`. The filename mirrors the `id` field (CI enforces).

### 3.1 Schema (canonical)

```yaml
# kanad/atlas/entries/ATLAS-0042.yaml
id: ATLAS-0042                              # zero-padded, monotonic; filename must match
title: "Spin-state ordering of FeO+ under DFT functional disagreement"
created: 2026-05-27
last_reviewed: 2026-05-27
curator: mk0dz
tags: [biradical, transition-metal, dft-disagreement]

molecule:
  formula: "FeO+"
  geometry: "equilibrium (re-optimized at recommended level)"
  charge: 1
  multiplicity_to_test: [2, 4, 6]
  basis_minimum_recommended: "def2-SVP"
  active_space_recipe: "MP2-NO (8e, 8o)"

question: |
  DFT B3LYP gives sextet ground state; BP86 gives quartet.
  CASPT2 (Hoyer 2014) disagrees with both. What does VQE +
  2-RDM with MP2-NO active space say, and how does the
  answer depend on basis set (STO-3G vs def2-SVP)?

question_type: discovery                    # discovery | methodology | verification | benchmark
expected_observables:                       # which M7+M8 outputs answer this
  - ground_state_energy_per_multiplicity
  - natural_orbital_occupations
  - mayer_bond_order
  - spin_density_grid

success_criterion: |
  Definitive ordering of (2, 4, 6) multiplicities with ΔE
  separations consistent with CASPT2 (±2 kcal/mol).
  Bonus: NO occupations distinguish bonding character.

capability_required:
  milestones: [M7, M10]                     # machine-parseable; CI validates against PLAN.md
  accelerators: [A1]                        # validates against parent spec A1–A13

ships_when: "M10 + A1 land"                 # human-readable, derived from above

scale_tier_recommended: T2                  # T1 | T2 | T3 | T4 | T5
scale_tier_minimum: T2
estimated_runtime:
  T2: "4 hours"
  T3: "25 minutes"

novelty_score: high                         # low | medium | high
novelty_justification: "DFT B3LYP vs BP86 ordering contradicts CASPT2; no VQE answer in literature"

target_researcher_cluster: organometallic-spin-state
secondary_clusters: [academic-small-mol-methodology]

literature_context:
  - doi: "10.1002/cphc.200200006"
    citation: "Reiher 2002 ChemPhysChem 3, 565"
    role: "DFT functional disagreement reported"
    excerpt: "B3LYP and BP86 yield different ground-state multiplicities for FeO+..."
  - doi: "10.1021/ct5006693"
    citation: "Hoyer 2014 JCTC 10, 5377"
    role: "CASPT2 reference"
    excerpt: "Multireference CASPT2 calculations resolve the FeO+ ground-state controversy..."
```

### 3.2 Field reference

| Field | Required | Type | Notes |
|---|---|---|---|
| `id` | Y | string | Format `ATLAS-NNNN`, zero-padded to 4 digits, monotonic |
| `title` | Y | string | One-line research-question summary |
| `created` | Y | ISO-8601 date | Set once, never edited |
| `last_reviewed` | Y | ISO-8601 date | Updated on any content edit |
| `curator` | Y | string | Username of original author |
| `tags` | N | list[string] | Free-form; for filtering / search |
| `molecule.formula` | Y | string | Hill-notation chemical formula |
| `molecule.geometry` | Y | string | Free-form prose; XYZ coords optional in v1 |
| `molecule.charge` | Y | int | Net charge |
| `molecule.multiplicity_to_test` | N | list[int] | If multireference, all relevant 2S+1 values |
| `molecule.basis_minimum_recommended` | Y | string | Anything below this likely fails success_criterion |
| `molecule.active_space_recipe` | N | string | Free-form recipe; populated if M10 needed |
| `question` | Y | multiline | The actual research question, 2–5 sentences |
| `question_type` | Y | enum | `discovery`, `methodology`, `verification`, `benchmark` |
| `expected_observables` | Y | list[string] | Names matching M8 observables-plate keys |
| `success_criterion` | Y | multiline | What a "publishable answer" looks like, with tolerance |
| `capability_required.milestones` | Y | list[string] | Subset of PLAN.md milestone IDs (M0–M19, M9.5, etc.) |
| `capability_required.accelerators` | N | list[string] | Subset of A1–A13 from parent spec |
| `ships_when` | Y | string | Human-readable derived statement |
| `scale_tier_recommended` | Y | enum | `T1`–`T5` |
| `scale_tier_minimum` | Y | enum | Must be ≤ recommended |
| `estimated_runtime` | Y | dict | Keys are tier IDs; values are human-readable durations |
| `novelty_score` | Y | enum | `low`, `medium`, `high` (definitions in §3.3) |
| `novelty_justification` | Y | string | 1-sentence reason for the score |
| `target_researcher_cluster` | Y | enum | One of 8 cluster IDs (§3.4) |
| `secondary_clusters` | N | list[enum] | Other clusters this entry serves |
| `literature_context` | Y | list[dict] | At least 1; each entry needs `doi`, `citation`, `role`, `excerpt` |

### 3.3 Novelty score definitions

- **low** — answer is settled in published literature; entry useful for user onboarding, benchmarking, or replication only
- **medium** — partial answer exists but is methodology-dependent (basis-set, functional, or active-space sensitivity); contributes to a documented debate
- **high** — open question OR contested across published methods OR no quantum-correlated (post-HF wavefunction) answer exists; paper-shaped opportunity

### 3.4 Cluster enum (validated)

Exactly these 8 values, matching parent spec Appendix C:

- `academic-small-mol-methodology`
- `organometallic-spin-state`
- `photochemistry-pv`
- `drug-fragment-screening`
- `materials-small-cell`
- `hardware-paper-authors`
- `industrial-pharma-rd`
- `industrial-materials-rd`

### 3.5 Computed fields (NOT stored — derived by indexer)

- `status` — `open` if all `capability_required.milestones` are done (per `MILESTONES.json`); else `blocked`
- `is_reachable_today` — boolean from milestone progress
- `unlocking_milestone` — the latest milestone in `capability_required.milestones` (the one whose completion flips `status` to `open`)

Storing these would create sync drift; computing them keeps atlas honest.

---

## 4. Directory layout

```
kanad/atlas/
├── README.md                       # ~50 lines; entry for curators
├── SCHEMA.md                       # human-readable schema (this spec §3 restated)
├── MILESTONES.json                 # single source of truth: which milestones are "done"
├── entries/                        # one YAML per atlas entry
│   ├── ATLAS-0001.yaml
│   ├── ATLAS-0002.yaml
│   └── ...
├── index.json                      # generated; commits to repo for offline consumers
├── by-milestone/                   # generated
│   ├── M7.md
│   ├── M9.5.md
│   └── ...
├── by-cluster/                     # generated
│   ├── organometallic-spin-state.md
│   └── ...
├── by-tier/                        # generated
│   └── T2.md
├── sourcing/                       # tools that produced entries; checked in
│   ├── benchmarks/                 # P1 converters
│   │   ├── gmtkn55_to_atlas.py
│   │   ├── w4_11_to_atlas.py
│   │   ├── mor41_to_atlas.py
│   │   ├── truhlar_tm_to_atlas.py
│   │   ├── reiher_spin_to_atlas.py
│   │   ├── head_gordon_mr_to_atlas.py
│   │   └── classic_challenging_to_atlas.py
│   ├── lit_mining/                 # P2 pipeline
│   │   ├── discover.py             # query S2 / Crossref / OpenAlex
│   │   ├── filter.py               # abstract heuristic filter
│   │   ├── extract.py              # LLM extraction (parallel agents)
│   │   ├── verify.py               # DOI + excerpt verification
│   │   ├── review_cli.py           # batch review TUI
│   │   ├── emit_yaml.py            # approved → atlas YAML
│   │   └── cache/                  # gitignored; raw API responses + LLM outputs
│   └── cluster_fill/               # P3 working files
│       ├── hardware-paper-authors.md
│       ├── industrial-materials-rd.md
│       └── ...
├── validators/
│   ├── schema.py                   # Pydantic models + JSONSchema
│   └── doi_resolver.py             # Crossref API wrapper
├── indexer.py                      # builds index.json + by-* views
├── ledger_generator.py             # writes fuel-ledger blocks into idea files + PLAN.md
└── stats.py                        # cluster/novelty/milestone distribution report
```

---

## 5. Sourcing pipeline

Three phases, each with concrete inputs, tools, and output targets.

### 5.1 Phase 1 — Benchmark conversion (days 1–3, ~100 entries)

**Inputs:** published benchmark datasets, all openly available.

| Source | Estimated entries | Cluster bias | Novelty bias |
|---|---|---|---|
| GMTKN55 (filtered to ≤30 atoms) | ~30 | academic-small-mol-methodology | medium |
| W4-11 small-molecule thermochemistry | ~15 | academic-small-mol-methodology | low–medium |
| MOR41 open-shell organic radicals | ~12 | academic-small-mol-methodology, organometallic | medium |
| Truhlar TM monoxide / nitride benchmark | ~10 | organometallic-spin-state | medium |
| Reiher spin-state collection | ~10 | organometallic-spin-state | medium–high |
| Head-Gordon multireference set | ~10 | academic-small-mol-methodology | high |
| Classic challenging-molecules family (C₂, F₂, m-benzyne, N₂, O₃, Cr₂, ...) | ~10 | academic-small-mol-methodology, materials | high |

**Mechanism:** one converter script per benchmark in `atlas/sourcing/benchmarks/`. Each ~80 LOC: parses the published table → emits one atlas YAML per molecule. Capability requirements auto-assigned based on molecule size + observable type. Novelty score defaulted per source bias above, with manual override during P3 review.

**Output:** ~100 entries in `atlas/entries/`, sequential IDs ATLAS-0001 through ATLAS-0100.

### 5.2 Phase 2 — LLM-assisted literature mining (days 4–14, ~120 entries)

**Inputs:** open metadata APIs (Semantic Scholar, Crossref, OpenAlex) + open-access paper PDFs where available.

**Six-step pipeline** (parallelizable per cluster):

1. **`discover.py`** — Query S2/Crossref/OpenAlex for papers in JCTC, JPCA, JCP, JACS, PCCP, ChemRxiv from 2021–2026. ~2000–5000 candidates per cluster.
2. **`filter.py`** — Regex + keyword filter on abstracts for contested-claim signals: `"DFT vs"`, `"remains unclear"`, `"we benchmark"`, `"contradictory"`, `"open question"`, `"controversy"`, `"disagreement"`. Cuts to ~200–500 per cluster.
3. **`extract.py`** — Dispatch parallel LLM agents (one per ~50 candidate papers). Each agent emits structured candidate: `{molecule, contested_claim, soa_answer, kanad_capability_fit, doi, citation, excerpt}`.
4. **`verify.py`** — Mechanical guards:
   - DOI must return HTTP 200 from Crossref → auto-reject otherwise
   - `excerpt` must appear byte-for-byte in the abstract returned by the API → auto-reject otherwise
   - Molecule fits ≤32-qubit envelope (after frozen-core estimation) → auto-reject otherwise
   - Reduces ~500 to ~200–300 verified candidates.
5. **`review_cli.py`** — Batch review TUI. Curator (you) sees each candidate (molecule + question + literature context + capability fit) → approve / reject / edit. Target throughput: 3 min/entry × 300 candidates = ~15 hours over the week.
6. **`emit_yaml.py`** — Approved candidates → atlas YAML, IDs continuing from P1 (ATLAS-0101+).

**Output:** ~120 entries in `atlas/entries/`, IDs ATLAS-0101 through ATLAS-0220.

**Critical anti-fabrication guards (mechanical, not advisory):**
- DOI hard-check via Crossref API
- Excerpt byte-match against API-returned abstract text
- Capability cross-ref against PLAN.md milestones + parent spec accelerators
- Unknown enum values → auto-reject
- All verification logs committed to `atlas/sourcing/lit_mining/cache/verify-log.jsonl` (gitignored body; summary stats checked in)

### 5.3 Phase 3 — Manual cluster-fill (days 15–21, ~50 entries)

**Inputs:** the P1+P2 distribution report from `stats.py` + curator (you) judgment.

**Process:**
1. Run `atlas/stats.py` → cluster-distribution histogram vs. Appendix C targets
2. Identify under-represented clusters (likely `hardware-paper-authors`, `industrial-materials-rd`, possibly `materials-small-cell`)
3. For each, open `atlas/sourcing/cluster_fill/<cluster>.md` — pre-populated with candidate question seeds drawn from cluster-specific expert sources:
   - `hardware-paper-authors`: IBM Quantum benchmark papers, IonQ application notes, BlueQubit case studies
   - `industrial-materials-rd`: Materials Project case studies, polymorph energetics benchmarks
   - `materials-small-cell`: Hubbard-U fitting collections, small-cell catalyst-screening data
4. Hand-write YAML per the schema; ~30 min/entry

**Output:** ~50 entries in `atlas/entries/`, IDs ATLAS-0221 through ATLAS-0270.

**Target final distribution** (proportional to parent spec Appendix C, ±10%):

```
Academic small-mol & methodology  ~63   (25%)
Drug-fragment screening           ~50   (20%)
Photochemistry & PV               ~38   (15%)
Organometallic spin-state         ~25   (10%)
Materials small-cell              ~25   (10%)
Industrial pharma R&D             ~25   (10%)
Hardware-paper authors            ~13    (5%)
Industrial materials R&D          ~13    (5%)
─────────────────────────────────────────
Total                            ~252
```

---

## 6. Tooling

### 6.1 Schema validator (`atlas/validators/schema.py`)

- Pydantic models matching §3.1 schema
- JSONSchema export for downstream consumers
- DOI resolution via `validators/doi_resolver.py` (cached Crossref API client)
- Cluster enum check
- Milestone ID cross-ref against `MILESTONES.json`
- Accelerator ID cross-ref against parent spec (parsed once from the markdown's vector matrix table)
- Field-presence + type checks
- Output: per-entry error report; exit 1 on any failure

**Estimated LOC:** ~200

### 6.2 Indexer (`atlas/indexer.py`)

- Walks `atlas/entries/`, parses each YAML
- Computes `status`, `is_reachable_today`, `unlocking_milestone` from `MILESTONES.json`
- Emits `atlas/index.json` (schema in §7.3 below)
- Emits `atlas/by-milestone/<M*>.md` — for each milestone, a markdown table of unlocked entries
- Emits `atlas/by-cluster/<cluster>.md` — analogous for each of the 8 clusters
- Emits `atlas/by-tier/<T*>.md` — analogous per tier
- Idempotent; CI fails if regenerated output drifts from committed

**Estimated LOC:** ~300

### 6.3 Fuel-ledger generator (`atlas/ledger_generator.py`)

- Reads `index.json` (run indexer first if needed)
- For each milestone in PLAN.md:
  - Computes ledger block per parent spec §6.1
  - Writes block into the corresponding `kanad/ideas/NN-*.md` between `<!-- FUEL-LEDGER: auto -->` and `<!-- FUEL-LEDGER: /auto -->` markers
  - For milestones without idea files (M9.5, M10.5, M11.5, M14.5, M14.9), writes to `atlas/by-milestone/<M*>.md` instead
- Updates a summary block in `kanad/PLAN.md`
- Idempotent

**Estimated LOC:** ~250

### 6.4 Stats reporter (`atlas/stats.py`)

- Cluster distribution histogram + delta vs. Appendix C targets
- Novelty distribution
- Milestone-unlock counts (open vs. blocked)
- Tier distribution
- Curator + creation-date metadata for staleness tracking
- Output: markdown report to stdout; `atlas/STATS.md` snapshot per CI run

**Estimated LOC:** ~150

### 6.5 CI workflow (`.github/workflows/atlas-validate.yml`)

On any PR touching `atlas/`, `PLAN.md`, or `MILESTONES.json`:
1. Run `atlas/validators/schema.py` over `atlas/entries/`
2. Regenerate `index.json` and compare to committed
3. Regenerate idea-file ledger blocks and compare to committed
4. Run `atlas/stats.py` and post summary as PR comment
5. Fail PR if any check fails

**Estimated config:** ~50 lines YAML

---

## 7. Integration with parent strategy

### 7.1 PLAN.md fuel ledger

Each Phase 2 milestone idea file gets a `<!-- FUEL-LEDGER: auto -->` block managed by `ledger_generator.py`. Contents per parent spec §6.1:

```markdown
<!-- FUEL-LEDGER: auto -->
## Fuel ledger

- **Atlas cells unlocked when this milestone lands:** 84
- **Novelty-weighted score:** 47.2
- **Median time-to-result on unlocked cells (T2):** 4 hours
- **Tier impact:** T1+ (all tiers; T3 dramatically faster)
- **Target clusters opened/deepened:** academic-small-mol-methodology, organometallic-spin-state
- **Entry index:** [`atlas/by-milestone/M8.md`](../atlas/by-milestone/M8.md)
<!-- FUEL-LEDGER: /auto -->
```

`PLAN.md` gets a similar summary block aggregating M0–M19.

### 7.2 New-milestone idea files

M9.5, M10.5, M11.5, M14.5, M14.9 don't have idea files yet. For these, the ledger lives in `atlas/by-milestone/<M*>.md` until their idea files are written (each as part of its writing-plans cycle).

### 7.3 Operations + scoreboard boundary contract

The single artifact downstream consumers read is `atlas/index.json`:

```json
{
  "version": "v0",
  "generated_at": "2026-05-27T14:30:00Z",
  "entries": [
    {
      "id": "ATLAS-0042",
      "title": "...",
      "molecule_formula": "FeO+",
      "novelty_score": "high",
      "scale_tier_recommended": "T2",
      "target_cluster": "organometallic-spin-state",
      "secondary_clusters": [],
      "capability_required": {"milestones": ["M7", "M10"], "accelerators": ["A1"]},
      "status": "blocked",
      "unlocking_milestone": "M10",
      "tags": ["biradical", "transition-metal", "dft-disagreement"],
      "ships_when": "M10 + A1 land"
    }
  ],
  "stats": {
    "total": 252,
    "by_cluster": {"organometallic-spin-state": 25, ...},
    "by_tier": {"T1": 30, "T2": 110, "T3": 80, "T4": 17, "T5": 15},
    "by_status": {"open": 84, "blocked": 168},
    "by_novelty": {"high": 95, "medium": 130, "low": 27}
  }
}
```

**Future operations work** (separate spec, not v0): `operations/pipeline/atlas_link.py` consumes this index for the `discovery_fuel_match` scoring axis.

**Future scoreboard work** (separate spec, not v0): the scoreboard frontend reads this index for progress visualization.

### 7.4 Milestone-status update workflow

`atlas/MILESTONES.json` is the single source of truth for milestone done-status:

```json
{
  "M0": {"status": "done", "done_at": "2026-05-15"},
  "M1": {"status": "in_progress"},
  "M7": {"status": "blocked"},
  "M9.5": {"status": "not_started"},
  ...
}
```

When a PR closes a milestone:
1. PR includes update to `MILESTONES.json` flipping the status
2. CI runs indexer → entries with `capability_required.milestones=[M*]` (and no other unfinished blockers) flip from `blocked` to `open`
3. CI regenerates idea-file ledger blocks
4. CI commits the regenerated artifacts (or fails the PR if author didn't run locally)

Self-maintaining once wired.

---

## 8. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| LLM fabricates DOIs / excerpts | High | High | Mechanical DOI resolution + excerpt byte-match; auto-reject. Verification stats committed to repo. If rejection rate exceeds 40%, fall back to "human-led search, LLM only for capability mapping" mode. |
| Cluster imbalance after P1+P2 (e.g., materials underrepresented) | High | Medium | P3 explicitly designed for cluster-fill. If a single cluster needs >20 manual entries, v0 ships with that gap documented and a follow-up sprint scheduled. |
| Schema drift during P2 (curator wants to add fields mid-build) | Medium | Medium | Schema locked at start of P1. New fields wait for v1. Curator can use the `tags` field as escape hatch. |
| Curator burnout during P2 review (~15 hrs of review) | Medium | High | Batch the review into 3-hour sessions; ensure review_cli.py supports keyboard navigation; auto-defer obviously-borderline entries to P3. |
| Novelty score inflation (everything tagged "high") | Medium | Medium | P3 review pass spot-checks novelty scores; the `novelty_justification` field forces explicit reasoning per entry. CI may emit a warning if any cluster has >70% high-novelty (statistically improbable). |
| Atlas stales out after v0 | Medium | High | `last_reviewed` field + `stats.py` flag entries older than 12 months. Atlas-curator role formally assigned at v0 ship time (per parent spec §11 Risks). |
| Capability cross-ref breaks when PLAN.md changes | Medium | Low | CI catches this — schema validator dies if any `capability_required.milestones` references a non-existent milestone. Forces explicit handling. |

---

## 9. Done criteria

This spec is implemented (Atlas v0 ships) when:

- [ ] `kanad/atlas/entries/` contains ≥200 valid YAML entries
- [ ] `atlas/stats.py` reports cluster distribution within ±20% of Appendix C targets for every cluster
- [ ] All entries pass `atlas/validators/schema.py` (CI green)
- [ ] No entry has a fabricated DOI or excerpt (verification log shows zero rejections in committed entries)
- [ ] `atlas/index.json` regenerates deterministically and is committed
- [ ] `atlas/by-milestone/M7.md` through `atlas/by-milestone/M14.md` exist and render correctly
- [ ] At least one idea file (`kanad/ideas/19-quantum-2rdm.md` recommended as proof) has a populated `<!-- FUEL-LEDGER: auto -->` block
- [ ] `PLAN.md` has a populated `<!-- FUEL-LEDGER-SUMMARY: auto -->` block
- [ ] CI workflow `.github/workflows/atlas-validate.yml` runs on PR and enforces schema + freshness
- [ ] `MILESTONES.json` exists with M0–M19 + M9.5/M10.5/M11.5/M14.5/M14.9 listed
- [ ] This spec's curator (the owner) signs off in a follow-up review

---

## 10. Out of scope for v0 (deferred)

- Scoreboard dashboard rendering
- `operations/pipeline/atlas_link.py` — researcher-scoring integration
- Outside-contributor workflow (PR template, contribution guide, curator triage queue)
- Runtime-result tracking (user runs an entry → atlas records the result)
- Sub-discipline taxonomy beyond the 8 clusters
- Multi-language atlas (e.g., Chinese, Japanese researchers)
- Atlas search UI / faceted filtering (the markdown views suffice for v0)

Each of these earns its own spec when prioritized.

---

## 11. Transition to writing-plans

After this spec is approved, the implementation plan should split the work into the following parallelizable tracks for the executors:

- **Track A — Tooling first:** schema validator, indexer, ledger generator, stats reporter, CI workflow. Estimate: 4–5 days, 1 developer.
- **Track B — P1 (benchmarks):** 7 converter scripts + entry emission. Estimate: 3 days, 1 developer.
- **Track C — P2 (lit mining):** 6-step pipeline + parallel agent dispatch + review TUI. Estimate: 7 days, 1 developer + curator time.
- **Track D — P3 (cluster fill):** manual curation per under-represented cluster. Estimate: 5 days, curator only.
- **Track E — Integration:** populating ledger blocks in idea files, updating PLAN.md, smoke-testing CI. Estimate: 2 days, 1 developer.

Tracks A, B, and C-pipeline-build can run in parallel. C-review-pass and D wait on output from B+C. E waits on A finishing.

The writing-plans skill should:
1. Convert these tracks into ordered task lists with dependencies
2. Identify the developer + curator split clearly
3. Specify done-criteria checks per track
4. Define progress checkpoints (end-of-week atlas count, schema validation pass rate, cluster distribution snapshot)

---

## Appendix A — file map

| Path | Purpose | New in this spec? |
|---|---|---|
| `kanad/atlas/README.md` | Curator entry point | Yes |
| `kanad/atlas/SCHEMA.md` | Human schema doc | Yes |
| `kanad/atlas/MILESTONES.json` | Source of truth: milestone done-status | Yes |
| `kanad/atlas/entries/` | All entry YAML files | Yes |
| `kanad/atlas/index.json` | Generated index | Yes |
| `kanad/atlas/by-milestone/` | Generated milestone views | Yes |
| `kanad/atlas/by-cluster/` | Generated cluster views | Yes |
| `kanad/atlas/by-tier/` | Generated tier views | Yes |
| `kanad/atlas/sourcing/benchmarks/` | P1 converters | Yes |
| `kanad/atlas/sourcing/lit_mining/` | P2 pipeline | Yes |
| `kanad/atlas/sourcing/cluster_fill/` | P3 working files | Yes |
| `kanad/atlas/validators/schema.py` | Pydantic + JSONSchema validation | Yes |
| `kanad/atlas/indexer.py` | Build index.json + by-* views | Yes |
| `kanad/atlas/ledger_generator.py` | Write fuel-ledger blocks | Yes |
| `kanad/atlas/stats.py` | Distribution report | Yes |
| `.github/workflows/atlas-validate.yml` | CI | Yes |
| `kanad/PLAN.md` | Add fuel-ledger summary block | Edit |
| `kanad/ideas/19-quantum-2rdm.md` through `26-industrial-deployment.md` | Add fuel-ledger block | Edit |

## Appendix B — cross-references

- Parent strategy: [`2026-05-25-discovery-fuel-phase2-design.md`](2026-05-25-discovery-fuel-phase2-design.md)
- Phase 2 milestone plan: [`../../../PLAN.md`](../../../PLAN.md)
- Strategic identity: [`../../../ideas/18-differentiation.md`](../../../ideas/18-differentiation.md)
- Per-milestone idea files: [`../../../ideas/19-quantum-2rdm.md`](../../../ideas/19-quantum-2rdm.md) through [`../../../ideas/26-industrial-deployment.md`](../../../ideas/26-industrial-deployment.md)
