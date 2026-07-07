# Atlas v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Kanad's Atlas v0 — ~8000 publishable open-research-question YAML entries, validated by CI, indexed mechanically, and wired into PLAN.md's per-milestone fuel ledger.

**Architecture:** Three-phase hybrid sourcing pipeline (mechanical benchmark conversion → LLM-assisted literature mining with anti-fabrication guards → manual cluster-fill) backed by a strict Pydantic schema, deterministic indexer, and CI workflow. All artifacts live at `kanad/atlas/`; the single boundary contract is `atlas/index.json` consumed by future scoreboard + operations integrations.

**Tech Stack:** Python 3.11+, Pydantic v2 for schema, `requests`/`httpx` for Crossref/Semantic Scholar/OpenAlex APIs, `pyyaml` for YAML I/O, `rich` for the review TUI, GitHub Actions for CI.

**Spec:** [`docs/superpowers/specs/2026-05-27-atlas-v0-seed-design.md`](../specs/2026-05-27-atlas-v0-seed-design.md)
**Parent strategy:** [`docs/superpowers/specs/2026-05-25-discovery-fuel-phase2-design.md`](../specs/2026-05-25-discovery-fuel-phase2-design.md)

---

## Execution shape

Six phases. Phases 2, 3, and 6 can start in parallel as soon as Phase 1 lands. Phase 4 depends on Phase 3. Phase 5 depends on Phase 2 + 4 outputs.

```
Phase 1 (Foundation tooling, ~4 days)
  ├─→ Phase 2 (P1: Benchmark conversion, ~3 days)  ──┐
  ├─→ Phase 3 (P2: Lit-mining pipeline build, ~3 days) ──┐
  └─→ Phase 6 (Integration: idea-file ledgers, ~2 days)  │
                                                          │
       Phase 4 (P2: Run pipeline + curator review, ~7d) ←─┘ (needs Phase 3)
                                                          │
       Phase 5 (P3: Manual cluster fill, ~5d) ←───────────┘ (needs Phase 2 + 4)
```

**Total wall-clock:** ~3 weeks. **Curator time:** ~25 hours (P2 review + P3 cluster fill).

---

## File structure overview

All new files at `kanad/atlas/`:

```
kanad/atlas/
├── README.md                                    # curator entry point
├── SCHEMA.md                                    # human-readable schema doc
├── MILESTONES.json                              # source of truth for milestone done-status
├── entries/                                     # ATLAS-NNNN.yaml × ~250
├── index.json                                   # generated; committed
├── by-milestone/                                # generated markdown views × ~25
├── by-cluster/                                  # generated × 8
├── by-tier/                                     # generated × 5
├── sourcing/
│   ├── benchmarks/                              # 7 converter scripts (~80 LOC each)
│   ├── lit_mining/                              # 6-step pipeline
│   │   └── cache/                               # gitignored API/LLM responses
│   └── cluster_fill/                            # curator working files
├── validators/
│   ├── __init__.py
│   ├── schema.py                                # Pydantic v2 models
│   └── doi_resolver.py                          # Crossref client
├── indexer.py
├── ledger_generator.py
└── stats.py
```

Plus:
- `.github/workflows/atlas-validate.yml` — CI
- `kanad/ideas/19-quantum-2rdm.md` through `kanad/ideas/26-industrial-deployment.md` — add `<!-- FUEL-LEDGER: auto -->` block
- `kanad/PLAN.md` — add `<!-- FUEL-LEDGER-SUMMARY: auto -->` block
- `kanad/atlas/tests/` — test suite for all atlas tooling

---

# Phase 1 — Foundation tooling

**Goal:** all atlas tooling (schema, validator, indexer, ledger generator, stats, CI) works end-to-end against a single hand-written test entry.

**Owner:** developer. **Estimated duration:** 4 days.

---

### Task 1.1: Scaffold atlas directory and config

**Files:**
- Create: `kanad/atlas/README.md`
- Create: `kanad/atlas/SCHEMA.md`
- Create: `kanad/atlas/MILESTONES.json`
- Create: `kanad/atlas/__init__.py`
- Create: `kanad/atlas/.gitignore`
- Create: `kanad/atlas/sourcing/lit_mining/cache/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd /home/mk/deeprealm/kanad
mkdir -p atlas/{entries,by-milestone,by-cluster,by-tier,sourcing/{benchmarks,lit_mining/cache,cluster_fill},validators,tests}
touch atlas/__init__.py atlas/validators/__init__.py atlas/tests/__init__.py
```

- [ ] **Step 2: Write `kanad/atlas/.gitignore`**

```
sourcing/lit_mining/cache/*
!sourcing/lit_mining/cache/.gitkeep
```

- [ ] **Step 3: Write `kanad/atlas/MILESTONES.json`**

Source content from `PLAN.md` §"What 'done' looks like — milestone summary". All milestones start `not_started` except M0 which is the current frontier:

```json
{
  "schema_version": "v0",
  "updated_at": "2026-05-27",
  "milestones": {
    "M0":   {"title": "Truth pass",                  "phase": 1, "status": "in_progress"},
    "M1":   {"title": "Foundation fixes",            "phase": 1, "status": "not_started"},
    "M2":   {"title": "Tier-1 chemistry",            "phase": 1, "status": "not_started"},
    "M3":   {"title": "Real observables (1-RDM)",    "phase": 1, "status": "not_started"},
    "M4":   {"title": "Hardware-grade SQD",          "phase": 1, "status": "not_started"},
    "M5":   {"title": "Real reactions + dynamics",   "phase": 1, "status": "not_started"},
    "M6":   {"title": "First champion (p-benzyne)",  "phase": 1, "status": "not_started"},
    "M7":   {"title": "Quantum 2-RDM extraction",    "phase": 2, "status": "not_started"},
    "M8":   {"title": "Observables plate (40+)",     "phase": 2, "status": "not_started"},
    "M9":   {"title": "Research-validation suite",   "phase": 2, "status": "not_started"},
    "M9.5": {"title": "Molecular Workbench (A1)",    "phase": 2, "status": "not_started"},
    "M10":  {"title": "NO-driven active space",      "phase": 2, "status": "not_started"},
    "M10.5":{"title": "Cache-memory ML optimizer (A10)", "phase": 2, "status": "not_started"},
    "M11":  {"title": "Excited states + transition", "phase": 2, "status": "not_started"},
    "M11.5":{"title": "GPU + autodiff (A11)",        "phase": 2, "status": "not_started"},
    "M12":  {"title": "Spin-coupling J extraction",  "phase": 2, "status": "not_started"},
    "M13":  {"title": "Density analysis grids",      "phase": 2, "status": "not_started"},
    "M14":  {"title": "Exploration workflows + industrial deployment", "phase": 2, "status": "not_started"},
    "M14.5":{"title": "Algorithm Development Protocol (A9)", "phase": 2, "status": "not_started"},
    "M14.9":{"title": "Hardware Report Card (A7)",   "phase": 2, "status": "not_started"},
    "M15":  {"title": "C₂ champion",                 "phase": 3, "status": "not_started"},
    "M16":  {"title": "F₂ champion",                 "phase": 3, "status": "not_started"},
    "M17":  {"title": "m-benzyne champion",          "phase": 3, "status": "not_started"},
    "M18":  {"title": "FeO+ champion",               "phase": 3, "status": "not_started"},
    "M19":  {"title": "Pentacene champion",          "phase": 3, "status": "not_started"}
  }
}
```

- [ ] **Step 4: Write `kanad/atlas/README.md`** (curator entry point, ~30 lines)

```markdown
# Kanad Fuel Atlas

Live catalog of publishable open research questions Kanad can answer.

- **Schema:** [`SCHEMA.md`](SCHEMA.md)
- **Strategy:** [`../docs/superpowers/specs/2026-05-25-discovery-fuel-phase2-design.md`](../docs/superpowers/specs/2026-05-25-discovery-fuel-phase2-design.md)
- **Implementation spec:** [`../docs/superpowers/specs/2026-05-27-atlas-v0-seed-design.md`](../docs/superpowers/specs/2026-05-27-atlas-v0-seed-design.md)

## Layout

- `entries/` — one YAML per question (ATLAS-NNNN.yaml)
- `index.json` — generated index (do not hand-edit)
- `by-milestone/`, `by-cluster/`, `by-tier/` — generated markdown views
- `sourcing/` — converter scripts + lit-mining pipeline
- `validators/`, `indexer.py`, `ledger_generator.py`, `stats.py` — tooling

## Adding an entry

1. Pick the next free ATLAS-NNNN
2. Copy an existing entry as template
3. Fill in all required fields (see SCHEMA.md)
4. Run `python -m kanad.atlas.validators.schema <path>`
5. Run `python -m kanad.atlas.indexer` to regenerate index
6. Commit
```

- [ ] **Step 5: Write `kanad/atlas/SCHEMA.md`** — copy field reference table from spec §3.2 verbatim.

- [ ] **Step 6: Commit**

```bash
git add atlas/
git commit -m "feat(atlas): scaffold v0 directory structure and milestone manifest"
```

---

### Task 1.2: Pydantic schema models

**Files:**
- Create: `kanad/atlas/validators/schema.py`
- Test: `kanad/atlas/tests/test_schema.py`

- [ ] **Step 1: Write failing test** at `kanad/atlas/tests/test_schema.py`

```python
import pytest
from pathlib import Path
import yaml
from kanad.atlas.validators.schema import AtlasEntry, NoveltyScore, Cluster, Tier

VALID_ENTRY = {
    "id": "ATLAS-0001",
    "title": "Test entry",
    "created": "2026-05-27",
    "last_reviewed": "2026-05-27",
    "curator": "test",
    "tags": ["test"],
    "molecule": {
        "formula": "H2",
        "geometry": "equilibrium",
        "charge": 0,
        "basis_minimum_recommended": "STO-3G",
    },
    "question": "Trivial test question.",
    "question_type": "verification",
    "expected_observables": ["ground_state_energy"],
    "success_criterion": "Energy within 1 mHa of FCI.",
    "capability_required": {"milestones": ["M2"], "accelerators": []},
    "ships_when": "M2 lands",
    "scale_tier_recommended": "T1",
    "scale_tier_minimum": "T1",
    "estimated_runtime": {"T1": "5 minutes"},
    "novelty_score": "low",
    "novelty_justification": "Settled in literature.",
    "target_researcher_cluster": "academic-small-mol-methodology",
    "literature_context": [
        {
            "doi": "10.1063/1.4869536",
            "citation": "Test 2014 J. Chem. Phys. 140",
            "role": "Reference",
            "excerpt": "Some excerpt text.",
        }
    ],
}


def test_valid_entry_parses():
    entry = AtlasEntry(**VALID_ENTRY)
    assert entry.id == "ATLAS-0001"
    assert entry.novelty_score == NoveltyScore.LOW
    assert entry.target_researcher_cluster == Cluster.ACADEMIC_SMALL_MOL_METHODOLOGY


def test_invalid_id_format_rejected():
    bad = {**VALID_ENTRY, "id": "ATLAS-1"}  # not zero-padded to 4
    with pytest.raises(ValueError, match="zero-padded"):
        AtlasEntry(**bad)


def test_invalid_cluster_rejected():
    bad = {**VALID_ENTRY, "target_researcher_cluster": "nonexistent-cluster"}
    with pytest.raises(ValueError):
        AtlasEntry(**bad)


def test_tier_minimum_must_be_le_recommended():
    bad = {**VALID_ENTRY, "scale_tier_recommended": "T1", "scale_tier_minimum": "T3"}
    with pytest.raises(ValueError, match="tier_minimum"):
        AtlasEntry(**bad)


def test_literature_context_requires_doi():
    bad = {**VALID_ENTRY}
    bad["literature_context"] = [{"citation": "x", "role": "y", "excerpt": "z"}]
    with pytest.raises(ValueError):
        AtlasEntry(**bad)
```

- [ ] **Step 2: Run tests, confirm failure**

```bash
cd /home/mk/deeprealm/kanad
pytest atlas/tests/test_schema.py -v
```

Expected: ImportError — `AtlasEntry` doesn't exist yet.

- [ ] **Step 3: Implement `kanad/atlas/validators/schema.py`**

```python
"""Pydantic v2 schema for atlas entries. Spec §3."""
from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


ID_PATTERN = re.compile(r"^ATLAS-\d{4}$")


class NoveltyScore(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Tier(str, Enum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"

    @property
    def order(self) -> int:
        return int(self.value[1])


class Cluster(str, Enum):
    ACADEMIC_SMALL_MOL_METHODOLOGY = "academic-small-mol-methodology"
    ORGANOMETALLIC_SPIN_STATE = "organometallic-spin-state"
    PHOTOCHEMISTRY_PV = "photochemistry-pv"
    DRUG_FRAGMENT_SCREENING = "drug-fragment-screening"
    MATERIALS_SMALL_CELL = "materials-small-cell"
    HARDWARE_PAPER_AUTHORS = "hardware-paper-authors"
    INDUSTRIAL_PHARMA_RD = "industrial-pharma-rd"
    INDUSTRIAL_MATERIALS_RD = "industrial-materials-rd"


class QuestionType(str, Enum):
    DISCOVERY = "discovery"
    METHODOLOGY = "methodology"
    VERIFICATION = "verification"
    BENCHMARK = "benchmark"


class Molecule(BaseModel):
    formula: str
    geometry: str
    charge: int
    multiplicity_to_test: Optional[list[int]] = None
    basis_minimum_recommended: str
    active_space_recipe: Optional[str] = None


class CapabilityRequired(BaseModel):
    milestones: list[str] = Field(default_factory=list)
    accelerators: list[str] = Field(default_factory=list)


class LiteratureRef(BaseModel):
    doi: str
    citation: str
    role: str
    excerpt: str

    @field_validator("doi")
    @classmethod
    def doi_format(cls, v: str) -> str:
        if not v.startswith("10."):
            raise ValueError(f"DOI must start with '10.': {v!r}")
        return v


class AtlasEntry(BaseModel):
    id: str
    title: str
    created: date
    last_reviewed: date
    curator: str
    tags: list[str] = Field(default_factory=list)
    molecule: Molecule
    question: str
    question_type: QuestionType
    expected_observables: list[str]
    success_criterion: str
    capability_required: CapabilityRequired
    ships_when: str
    scale_tier_recommended: Tier
    scale_tier_minimum: Tier
    estimated_runtime: dict[str, str]
    novelty_score: NoveltyScore
    novelty_justification: str
    target_researcher_cluster: Cluster
    secondary_clusters: list[Cluster] = Field(default_factory=list)
    literature_context: list[LiteratureRef] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def id_zero_padded(cls, v: str) -> str:
        if not ID_PATTERN.match(v):
            raise ValueError(
                f"id must be zero-padded ATLAS-NNNN format (4 digits), got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def tier_minimum_le_recommended(self) -> "AtlasEntry":
        if self.scale_tier_minimum.order > self.scale_tier_recommended.order:
            raise ValueError(
                f"scale_tier_minimum {self.scale_tier_minimum} > "
                f"scale_tier_recommended {self.scale_tier_recommended}"
            )
        return self

    @model_validator(mode="after")
    def runtime_keys_are_tiers(self) -> "AtlasEntry":
        valid = {t.value for t in Tier}
        for k in self.estimated_runtime:
            if k not in valid:
                raise ValueError(f"estimated_runtime key {k!r} is not a valid tier")
        return self
```

- [ ] **Step 4: Run tests, confirm pass**

```bash
pytest atlas/tests/test_schema.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add atlas/validators/schema.py atlas/tests/test_schema.py
git commit -m "feat(atlas): Pydantic schema for atlas entries"
```

---

### Task 1.3: DOI resolver via Crossref

**Files:**
- Create: `kanad/atlas/validators/doi_resolver.py`
- Test: `kanad/atlas/tests/test_doi_resolver.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_doi_resolver.py
import pytest
from unittest.mock import patch, MagicMock
from kanad.atlas.validators.doi_resolver import resolve_doi, DOIResolutionError


def test_resolves_valid_doi():
    with patch("kanad.atlas.validators.doi_resolver.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "message": {
                "title": ["Test title"],
                "abstract": "Some abstract excerpt here",
            }
        }
        result = resolve_doi("10.1063/1.4869536")
        assert result.title == "Test title"
        assert "Some abstract excerpt" in result.abstract


def test_rejects_unresolvable_doi():
    with patch("kanad.atlas.validators.doi_resolver.httpx.get") as mock_get:
        mock_get.return_value.status_code = 404
        with pytest.raises(DOIResolutionError, match="404"):
            resolve_doi("10.9999/nonexistent")


def test_excerpt_byte_match_succeeds():
    from kanad.atlas.validators.doi_resolver import verify_excerpt
    abstract = "This paper reports that B3LYP gives sextet ground state."
    assert verify_excerpt("B3LYP gives sextet ground state", abstract)


def test_excerpt_byte_match_fails_on_paraphrase():
    from kanad.atlas.validators.doi_resolver import verify_excerpt
    abstract = "This paper reports that B3LYP gives sextet ground state."
    assert not verify_excerpt("B3LYP yields sextet ground state", abstract)
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_doi_resolver.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `kanad/atlas/validators/doi_resolver.py`**

```python
"""Crossref DOI resolver + excerpt byte-match verifier.

Used by both the schema validator (when validating committed entries)
and the lit-mining pipeline (when verifying LLM-extracted candidates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx


CROSSREF_BASE = "https://api.crossref.org/works"
USER_AGENT = "Kanad-Atlas/0.1 (mailto:31vivekpal@gmail.com)"
TIMEOUT_SECONDS = 10.0


class DOIResolutionError(Exception):
    """Raised when a DOI cannot be resolved against Crossref."""


@dataclass(frozen=True)
class CrossrefRecord:
    doi: str
    title: str
    abstract: Optional[str]


def resolve_doi(doi: str) -> CrossrefRecord:
    url = f"{CROSSREF_BASE}/{doi}"
    response = httpx.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        raise DOIResolutionError(
            f"DOI {doi!r} returned HTTP {response.status_code}"
        )
    message = response.json().get("message", {})
    title_list = message.get("title") or [""]
    return CrossrefRecord(
        doi=doi,
        title=title_list[0],
        abstract=message.get("abstract"),
    )


def verify_excerpt(excerpt: str, source_text: Optional[str]) -> bool:
    """Byte-match verification: excerpt must appear verbatim in source.

    No fuzzy matching, no paraphrase tolerance — paraphrase is a fabrication
    risk and must auto-reject.
    """
    if not source_text:
        return False
    return excerpt.strip() in source_text
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_doi_resolver.py -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add atlas/validators/doi_resolver.py atlas/tests/test_doi_resolver.py
git commit -m "feat(atlas): DOI resolver with Crossref + byte-match excerpt verification"
```

---

### Task 1.4: Schema validator entry point with capability cross-ref

**Files:**
- Modify: `kanad/atlas/validators/schema.py` (add validator entry point)
- Create: `kanad/atlas/validators/__main__.py`
- Test: `kanad/atlas/tests/test_validator_cli.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_validator_cli.py
import json
import pytest
from pathlib import Path
from kanad.atlas.validators.schema import validate_entry_file, load_milestone_ids, load_accelerator_ids


def test_validates_good_entry(tmp_path: Path, monkeypatch):
    milestones_path = tmp_path / "MILESTONES.json"
    milestones_path.write_text(json.dumps({"milestones": {"M2": {}}}))

    entry_path = tmp_path / "ATLAS-0001.yaml"
    entry_path.write_text(_VALID_YAML)

    errors = validate_entry_file(
        entry_path, milestones={"M2"}, accelerators={"A1"}
    )
    assert errors == []


def test_rejects_unknown_milestone(tmp_path: Path):
    entry_path = tmp_path / "ATLAS-0002.yaml"
    entry_path.write_text(_VALID_YAML.replace("M2", "M999"))
    errors = validate_entry_file(entry_path, milestones={"M2"}, accelerators={"A1"})
    assert any("M999" in e for e in errors)


def test_loads_milestone_ids(tmp_path: Path):
    p = tmp_path / "MILESTONES.json"
    p.write_text(json.dumps({"milestones": {"M0": {}, "M1": {}, "M9.5": {}}}))
    assert load_milestone_ids(p) == {"M0", "M1", "M9.5"}


_VALID_YAML = """
id: ATLAS-0001
title: Test
created: 2026-05-27
last_reviewed: 2026-05-27
curator: test
tags: []
molecule:
  formula: H2
  geometry: equilibrium
  charge: 0
  basis_minimum_recommended: STO-3G
question: Test.
question_type: verification
expected_observables: [ground_state_energy]
success_criterion: Test.
capability_required:
  milestones: [M2]
  accelerators: []
ships_when: M2 lands
scale_tier_recommended: T1
scale_tier_minimum: T1
estimated_runtime:
  T1: 5 minutes
novelty_score: low
novelty_justification: Test.
target_researcher_cluster: academic-small-mol-methodology
secondary_clusters: []
literature_context:
  - doi: "10.1063/1.4869536"
    citation: Test
    role: reference
    excerpt: Test excerpt.
"""
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_validator_cli.py -v
```

- [ ] **Step 3: Extend `kanad/atlas/validators/schema.py`** with validator entry point

Append to the existing file:

```python
import json
from pathlib import Path

import yaml


ACCELERATOR_IDS = {f"A{n}" for n in range(1, 14)}  # A1..A13 from parent spec


def load_milestone_ids(milestones_json: Path) -> set[str]:
    data = json.loads(milestones_json.read_text())
    return set(data.get("milestones", {}).keys())


def load_accelerator_ids() -> set[str]:
    return set(ACCELERATOR_IDS)


def validate_entry_file(
    path: Path,
    milestones: set[str],
    accelerators: set[str],
) -> list[str]:
    """Validate one entry YAML. Returns list of error strings (empty = valid)."""
    errors: list[str] = []
    try:
        data = yaml.safe_load(path.read_text())
        entry = AtlasEntry(**data)
    except Exception as e:
        return [f"{path.name}: schema error: {e}"]

    if entry.id != path.stem:
        errors.append(
            f"{path.name}: filename does not match id field {entry.id!r}"
        )

    for m in entry.capability_required.milestones:
        if m not in milestones:
            errors.append(
                f"{path.name}: unknown milestone {m!r} (not in MILESTONES.json)"
            )

    for a in entry.capability_required.accelerators:
        if a not in accelerators:
            errors.append(
                f"{path.name}: unknown accelerator {a!r} (not A1..A13)"
            )

    return errors
```

- [ ] **Step 4: Write `kanad/atlas/validators/__main__.py`** — CLI entry point

```python
"""CLI: python -m kanad.atlas.validators [PATH]

Validates entries against schema, milestone cross-ref, and accelerator cross-ref.
With no PATH, validates every file in atlas/entries/.
Exit 0 = all valid; exit 1 = at least one error.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .schema import load_milestone_ids, load_accelerator_ids, validate_entry_file


REPO_ROOT = Path(__file__).resolve().parents[2]
ATLAS_ROOT = REPO_ROOT / "atlas"
MILESTONES = ATLAS_ROOT / "MILESTONES.json"
ENTRIES_DIR = ATLAS_ROOT / "entries"


def main(argv: list[str]) -> int:
    targets: list[Path]
    if len(argv) > 1:
        targets = [Path(argv[1])]
    else:
        targets = sorted(ENTRIES_DIR.glob("ATLAS-*.yaml"))

    milestones = load_milestone_ids(MILESTONES)
    accelerators = load_accelerator_ids()

    all_errors: list[str] = []
    for p in targets:
        all_errors.extend(validate_entry_file(p, milestones, accelerators))

    if all_errors:
        for e in all_errors:
            print(e, file=sys.stderr)
        print(f"\n{len(all_errors)} error(s) in {len(targets)} entries", file=sys.stderr)
        return 1

    print(f"OK: {len(targets)} entries valid")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 5: Run, confirm pass**

```bash
pytest atlas/tests/test_validator_cli.py -v
```

- [ ] **Step 6: Commit**

```bash
git add atlas/validators/schema.py atlas/validators/__main__.py atlas/tests/test_validator_cli.py
git commit -m "feat(atlas): schema validator CLI with milestone+accelerator cross-ref"
```

---

### Task 1.5: Indexer with computed status fields

**Files:**
- Create: `kanad/atlas/indexer.py`
- Test: `kanad/atlas/tests/test_indexer.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_indexer.py
import json
from pathlib import Path
import yaml

from kanad.atlas.indexer import build_index, compute_status, IndexBuilder


def test_compute_status_open_when_all_milestones_done():
    milestones = {"M2": {"status": "done"}, "M3": {"status": "done"}}
    required = ["M2", "M3"]
    assert compute_status(required, milestones) == ("open", None)


def test_compute_status_blocked_when_any_pending():
    milestones = {"M2": {"status": "done"}, "M3": {"status": "not_started"}}
    required = ["M2", "M3"]
    status, unlocking = compute_status(required, milestones)
    assert status == "blocked"
    assert unlocking == "M3"


def test_build_index_writes_json_and_views(tmp_path: Path):
    entries_dir = tmp_path / "entries"
    entries_dir.mkdir()
    (entries_dir / "ATLAS-0001.yaml").write_text(_VALID_YAML)

    milestones_path = tmp_path / "MILESTONES.json"
    milestones_path.write_text(json.dumps({
        "milestones": {"M2": {"status": "done", "title": "Tier-1 chemistry"}}
    }))

    out_dir = tmp_path
    builder = IndexBuilder(
        entries_dir=entries_dir,
        milestones_path=milestones_path,
        atlas_root=out_dir,
    )
    builder.build()

    index = json.loads((out_dir / "index.json").read_text())
    assert index["stats"]["total"] == 1
    assert index["entries"][0]["status"] == "open"

    by_m2 = (out_dir / "by-milestone" / "M2.md").read_text()
    assert "ATLAS-0001" in by_m2


_VALID_YAML = """
id: ATLAS-0001
title: Test entry
created: 2026-05-27
last_reviewed: 2026-05-27
curator: test
tags: []
molecule:
  formula: H2
  geometry: equilibrium
  charge: 0
  basis_minimum_recommended: STO-3G
question: Test.
question_type: verification
expected_observables: [ground_state_energy]
success_criterion: Test.
capability_required:
  milestones: [M2]
  accelerators: []
ships_when: M2 lands
scale_tier_recommended: T1
scale_tier_minimum: T1
estimated_runtime:
  T1: 5 minutes
novelty_score: low
novelty_justification: Test.
target_researcher_cluster: academic-small-mol-methodology
secondary_clusters: []
literature_context:
  - doi: "10.1063/1.4869536"
    citation: Test
    role: reference
    excerpt: Test.
"""
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_indexer.py -v
```

- [ ] **Step 3: Implement `kanad/atlas/indexer.py`**

```python
"""Atlas indexer — reads entries/, computes status, emits index.json + views.

Idempotent: running twice on the same input produces byte-identical output.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from .validators.schema import AtlasEntry


def compute_status(
    required_milestones: list[str],
    milestones: dict[str, dict],
) -> tuple[str, Optional[str]]:
    """Returns (status, unlocking_milestone).

    status = 'open' if every required milestone is done; else 'blocked'.
    unlocking_milestone = the latest required milestone that is NOT done
    (the one whose completion would flip status to 'open'); None if open.
    """
    pending: list[str] = []
    for m in required_milestones:
        m_info = milestones.get(m, {})
        if m_info.get("status") != "done":
            pending.append(m)
    if not pending:
        return ("open", None)
    return ("blocked", pending[-1])


def _milestone_sort_key(m: str) -> tuple[int, float]:
    """Sort M0, M1, ... M9, M9.5, M10, ... M14, M14.5, M14.9, M15, ..."""
    rest = m[1:]
    if "." in rest:
        major, minor = rest.split(".")
        return (int(major), float("0." + minor))
    return (int(rest), 0.0)


@dataclass
class IndexBuilder:
    entries_dir: Path
    milestones_path: Path
    atlas_root: Path

    def build(self) -> None:
        milestones = json.loads(self.milestones_path.read_text())["milestones"]
        entries: list[dict] = []

        for entry_path in sorted(self.entries_dir.glob("ATLAS-*.yaml")):
            data = yaml.safe_load(entry_path.read_text())
            entry = AtlasEntry(**data)
            status, unlocking = compute_status(
                entry.capability_required.milestones, milestones
            )
            entries.append({
                "id": entry.id,
                "title": entry.title,
                "molecule_formula": entry.molecule.formula,
                "novelty_score": entry.novelty_score.value,
                "scale_tier_recommended": entry.scale_tier_recommended.value,
                "target_cluster": entry.target_researcher_cluster.value,
                "secondary_clusters": [c.value for c in entry.secondary_clusters],
                "capability_required": {
                    "milestones": entry.capability_required.milestones,
                    "accelerators": entry.capability_required.accelerators,
                },
                "status": status,
                "unlocking_milestone": unlocking,
                "tags": entry.tags,
                "ships_when": entry.ships_when,
            })

        stats = self._compute_stats(entries)
        index = {
            "version": "v0",
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "entries": entries,
            "stats": stats,
        }

        (self.atlas_root / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n"
        )
        self._write_views(entries, milestones)

    def _compute_stats(self, entries: list[dict]) -> dict:
        return {
            "total": len(entries),
            "by_cluster": dict(Counter(e["target_cluster"] for e in entries)),
            "by_tier": dict(Counter(e["scale_tier_recommended"] for e in entries)),
            "by_status": dict(Counter(e["status"] for e in entries)),
            "by_novelty": dict(Counter(e["novelty_score"] for e in entries)),
        }

    def _write_views(self, entries: list[dict], milestones: dict) -> None:
        by_milestone_dir = self.atlas_root / "by-milestone"
        by_cluster_dir = self.atlas_root / "by-cluster"
        by_tier_dir = self.atlas_root / "by-tier"
        for d in (by_milestone_dir, by_cluster_dir, by_tier_dir):
            d.mkdir(parents=True, exist_ok=True)

        # by-milestone: every milestone that any entry references
        ms_to_entries: dict[str, list[dict]] = {}
        for e in entries:
            for m in e["capability_required"]["milestones"]:
                ms_to_entries.setdefault(m, []).append(e)
        for m, es in sorted(ms_to_entries.items(), key=lambda kv: _milestone_sort_key(kv[0])):
            self._write_view_md(
                by_milestone_dir / f"{m}.md",
                f"Atlas entries requiring {m}",
                f"Milestone: **{m}** — {milestones.get(m, {}).get('title', '')}",
                es,
            )

        # by-cluster: 8 fixed clusters
        cluster_to_entries: dict[str, list[dict]] = {}
        for e in entries:
            cluster_to_entries.setdefault(e["target_cluster"], []).append(e)
        for c, es in sorted(cluster_to_entries.items()):
            self._write_view_md(
                by_cluster_dir / f"{c}.md",
                f"Atlas entries targeting {c}",
                f"Cluster: **{c}**",
                es,
            )

        # by-tier: 5 tiers
        tier_to_entries: dict[str, list[dict]] = {}
        for e in entries:
            tier_to_entries.setdefault(e["scale_tier_recommended"], []).append(e)
        for t, es in sorted(tier_to_entries.items()):
            self._write_view_md(
                by_tier_dir / f"{t}.md",
                f"Atlas entries recommended for {t}",
                f"Tier: **{t}**",
                es,
            )

    @staticmethod
    def _write_view_md(path: Path, title: str, subtitle: str, entries: list[dict]) -> None:
        lines = [f"# {title}", "", subtitle, ""]
        lines.append("| ID | Title | Molecule | Novelty | Status |")
        lines.append("|---|---|---|---|---|")
        for e in sorted(entries, key=lambda x: x["id"]):
            lines.append(
                f"| [{e['id']}](../entries/{e['id']}.yaml) "
                f"| {e['title']} | {e['molecule_formula']} "
                f"| {e['novelty_score']} | {e['status']} |"
            )
        path.write_text("\n".join(lines) + "\n")


def build_index(atlas_root: Optional[Path] = None) -> None:
    root = atlas_root or Path(__file__).resolve().parent
    builder = IndexBuilder(
        entries_dir=root / "entries",
        milestones_path=root / "MILESTONES.json",
        atlas_root=root,
    )
    builder.build()


if __name__ == "__main__":
    build_index()
    print("Index regenerated.", file=sys.stderr)
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_indexer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/indexer.py atlas/tests/test_indexer.py
git commit -m "feat(atlas): indexer computes status from MILESTONES.json and emits index.json + views"
```

---

### Task 1.6: Fuel-ledger generator

**Files:**
- Create: `kanad/atlas/ledger_generator.py`
- Test: `kanad/atlas/tests/test_ledger_generator.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_ledger_generator.py
import json
from pathlib import Path

from kanad.atlas.ledger_generator import (
    generate_milestone_ledger,
    inject_ledger_block,
    LEDGER_START,
    LEDGER_END,
)


def test_generate_ledger_counts_unlocked():
    entries = [
        {"id": "ATLAS-0001", "novelty_score": "high",
         "estimated_runtime": {"T1": "5 minutes"},
         "scale_tier_recommended": "T1",
         "target_cluster": "academic-small-mol-methodology",
         "capability_required": {"milestones": ["M2"], "accelerators": []},
         "status": "open"},
        {"id": "ATLAS-0002", "novelty_score": "medium",
         "estimated_runtime": {"T2": "1 hour"},
         "scale_tier_recommended": "T2",
         "target_cluster": "drug-fragment-screening",
         "capability_required": {"milestones": ["M2", "M3"], "accelerators": []},
         "status": "blocked"},
    ]
    ledger = generate_milestone_ledger("M2", entries)
    assert ledger["cells_unlocked"] == ["ATLAS-0001"]
    assert ledger["novelty_weighted_score"] == 3  # high=3
    assert "academic-small-mol-methodology" in ledger["clusters"]


def test_inject_ledger_replaces_block(tmp_path: Path):
    idea = tmp_path / "20-test.md"
    idea.write_text(
        "# Existing content\n\nSome text.\n\n"
        f"{LEDGER_START}\nstale content\n{LEDGER_END}\n\n"
        "More content.\n"
    )
    inject_ledger_block(idea, "## Fuel ledger\n\nfresh content")

    result = idea.read_text()
    assert "stale content" not in result
    assert "fresh content" in result
    assert "Existing content" in result
    assert "More content." in result


def test_inject_ledger_appends_if_no_block(tmp_path: Path):
    idea = tmp_path / "21-test.md"
    idea.write_text("# Content without ledger block\n")
    inject_ledger_block(idea, "## Fuel ledger\n\nfresh content")

    result = idea.read_text()
    assert LEDGER_START in result
    assert LEDGER_END in result
    assert "fresh content" in result
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_ledger_generator.py -v
```

- [ ] **Step 3: Implement `kanad/atlas/ledger_generator.py`**

```python
"""Generate per-milestone fuel-ledger blocks; inject into idea files + PLAN.md.

Parent spec §6 defines what each ledger contains.
This module reads the indexed entries (atlas/index.json) and produces:
- A ledger block per milestone (returned as dict or rendered markdown)
- Injection into kanad/ideas/NN-*.md between FUEL-LEDGER markers
- A summary block injected into kanad/PLAN.md
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

LEDGER_START = "<!-- FUEL-LEDGER: auto -->"
LEDGER_END = "<!-- FUEL-LEDGER: /auto -->"
SUMMARY_START = "<!-- FUEL-LEDGER-SUMMARY: auto -->"
SUMMARY_END = "<!-- FUEL-LEDGER-SUMMARY: /auto -->"

NOVELTY_WEIGHT = {"low": 1, "medium": 2, "high": 3}


def generate_milestone_ledger(milestone: str, entries: list[dict]) -> dict:
    """Compute ledger block for one milestone."""
    unlocked = [
        e for e in entries
        if milestone in e["capability_required"]["milestones"]
        and e["status"] == "open"
    ]
    referenced = [
        e for e in entries
        if milestone in e["capability_required"]["milestones"]
    ]
    novelty_score = sum(NOVELTY_WEIGHT[e["novelty_score"]] for e in unlocked)
    tier = next(
        (e["scale_tier_recommended"] for e in unlocked),
        next((e["scale_tier_recommended"] for e in referenced), "T1"),
    )
    runtimes = [
        e["estimated_runtime"].get(tier)
        for e in unlocked if tier in e["estimated_runtime"]
    ]
    median_runtime = _median_or_none(runtimes)

    return {
        "milestone": milestone,
        "cells_unlocked": [e["id"] for e in unlocked],
        "cells_referenced": [e["id"] for e in referenced],
        "novelty_weighted_score": novelty_score,
        "median_time_to_result": median_runtime,
        "tier_impact": sorted(set(e["scale_tier_recommended"] for e in referenced)),
        "clusters": dict(Counter(e["target_cluster"] for e in unlocked)),
    }


def _median_or_none(values: list[Optional[str]]) -> Optional[str]:
    nums: list[float] = []
    units: list[str] = []
    for v in values:
        if not v:
            continue
        parts = v.split()
        if len(parts) >= 2:
            try:
                nums.append(float(parts[0]))
                units.append(parts[1])
            except ValueError:
                continue
    if not nums or len(set(units)) != 1:
        return values[0] if values else None
    return f"{statistics.median(nums)} {units[0]}"


def render_ledger_block(ledger: dict) -> str:
    n_unlocked = len(ledger["cells_unlocked"])
    n_referenced = len(ledger["cells_referenced"])
    clusters_str = ", ".join(
        f"{c}({n})" for c, n in sorted(ledger["clusters"].items())
    ) or "(none unlocked yet)"
    tier_str = ", ".join(ledger["tier_impact"]) or "—"
    runtime_str = ledger["median_time_to_result"] or "—"
    cells_link = f"`atlas/by-milestone/{ledger['milestone']}.md`"
    return (
        "## Fuel ledger\n\n"
        f"- **Atlas cells unlocked by this milestone:** {n_unlocked} (of {n_referenced} referenced)\n"
        f"- **Novelty-weighted score:** {ledger['novelty_weighted_score']}\n"
        f"- **Median time-to-result:** {runtime_str}\n"
        f"- **Tier impact:** {tier_str}\n"
        f"- **Target clusters:** {clusters_str}\n"
        f"- **Entry index:** [{cells_link}](../{cells_link})\n"
    )


def inject_ledger_block(path: Path, content: str) -> None:
    """Replace the FUEL-LEDGER block in `path` with `content`. Append if missing."""
    text = path.read_text()
    block = f"{LEDGER_START}\n{content}\n{LEDGER_END}"

    pattern = re.compile(
        rf"{re.escape(LEDGER_START)}.*?{re.escape(LEDGER_END)}",
        re.DOTALL,
    )
    if pattern.search(text):
        new_text = pattern.sub(block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + block + "\n"
    path.write_text(new_text)


def inject_summary_block(path: Path, content: str) -> None:
    """Like inject_ledger_block but uses SUMMARY markers."""
    text = path.read_text()
    block = f"{SUMMARY_START}\n{content}\n{SUMMARY_END}"
    pattern = re.compile(
        rf"{re.escape(SUMMARY_START)}.*?{re.escape(SUMMARY_END)}",
        re.DOTALL,
    )
    if pattern.search(text):
        new_text = pattern.sub(block, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        new_text = text + "\n" + block + "\n"
    path.write_text(new_text)


# Map milestone → idea file that hosts its ledger block.
# For new milestones without idea files (M9.5, M10.5, M11.5, M14.5, M14.9),
# value is None → ledger written to atlas/by-milestone/M*.md only.
IDEA_FILE_MAP: dict[str, Optional[str]] = {
    "M7":  "ideas/19-quantum-2rdm.md",
    "M8":  "ideas/20-observables-plate.md",
    "M9":  "ideas/21-validation-suite.md",
    "M10": "ideas/22-natural-orbital-active-space.md",
    "M11": None,  # no dedicated idea file yet; lives in atlas/by-milestone only
    "M12": "ideas/23-spin-coupling-J.md",
    "M13": "ideas/24-density-analysis.md",
    "M14": "ideas/25-exploration-workflows.md",
    "M9.5":  None,
    "M10.5": None,
    "M11.5": None,
    "M14.5": None,
    "M14.9": None,
}


def regenerate_all(repo_root: Path) -> None:
    index = json.loads((repo_root / "atlas" / "index.json").read_text())
    entries = index["entries"]
    milestones_data = json.loads(
        (repo_root / "atlas" / "MILESTONES.json").read_text()
    )["milestones"]

    summary_rows: list[str] = ["| Milestone | Cells unlocked | Novelty score | Tier |", "|---|---|---|---|"]

    for milestone in sorted(milestones_data.keys()):
        ledger = generate_milestone_ledger(milestone, entries)
        block = render_ledger_block(ledger)

        idea_rel = IDEA_FILE_MAP.get(milestone)
        if idea_rel:
            idea_path = repo_root / idea_rel
            if idea_path.exists():
                inject_ledger_block(idea_path, block)

        # Always write the standalone milestone view in atlas/by-milestone/
        # (indexer already writes a table view; ledger_generator overwrites with block prepended)
        by_m = repo_root / "atlas" / "by-milestone" / f"{milestone}.md"
        by_m_existing = by_m.read_text() if by_m.exists() else ""
        by_m.write_text(block + "\n" + by_m_existing)

        n = len(ledger["cells_unlocked"])
        tier_s = ", ".join(ledger["tier_impact"]) or "—"
        summary_rows.append(
            f"| {milestone} | {n} | {ledger['novelty_weighted_score']} | {tier_s} |"
        )

    summary_content = "## Fuel-ledger summary\n\n" + "\n".join(summary_rows) + "\n"
    plan = repo_root / "PLAN.md"
    if plan.exists():
        inject_summary_block(plan, summary_content)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    regenerate_all(repo_root)
    print("Fuel-ledger blocks regenerated.", file=sys.stderr)
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_ledger_generator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/ledger_generator.py atlas/tests/test_ledger_generator.py
git commit -m "feat(atlas): fuel-ledger generator with idea-file block injection"
```

---

### Task 1.7: Stats reporter

**Files:**
- Create: `kanad/atlas/stats.py`
- Test: `kanad/atlas/tests/test_stats.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_stats.py
from kanad.atlas.stats import (
    cluster_distribution,
    CLUSTER_TARGETS_PCT,
    health_report,
)


def test_cluster_distribution_with_deltas():
    entries = [
        {"target_cluster": "academic-small-mol-methodology"} for _ in range(60)
    ] + [
        {"target_cluster": "drug-fragment-screening"} for _ in range(40)
    ]
    dist = cluster_distribution(entries)
    assert dist["academic-small-mol-methodology"]["count"] == 60
    assert dist["academic-small-mol-methodology"]["pct_actual"] == 60.0
    assert dist["academic-small-mol-methodology"]["pct_target"] == CLUSTER_TARGETS_PCT["academic-small-mol-methodology"]


def test_health_report_flags_imbalance():
    entries = [{"target_cluster": "academic-small-mol-methodology"} for _ in range(100)]
    report = health_report(entries)
    assert "imbalance" in report.lower() or "under" in report.lower()
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_stats.py -v
```

- [ ] **Step 3: Implement `kanad/atlas/stats.py`**

```python
"""Atlas v0 health + distribution reporter.

Targets from parent spec Appendix C:
  academic-small-mol-methodology  25%
  drug-fragment-screening         20%
  photochemistry-pv               15%
  organometallic-spin-state       10%
  materials-small-cell            10%
  industrial-pharma-rd            10%
  hardware-paper-authors           5%
  industrial-materials-rd          5%
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


CLUSTER_TARGETS_PCT = {
    "academic-small-mol-methodology": 25.0,
    "drug-fragment-screening": 20.0,
    "photochemistry-pv": 15.0,
    "organometallic-spin-state": 10.0,
    "materials-small-cell": 10.0,
    "industrial-pharma-rd": 10.0,
    "hardware-paper-authors": 5.0,
    "industrial-materials-rd": 5.0,
}

TOLERANCE_PCT = 5.0  # ±5pp before we flag it


def cluster_distribution(entries: list[dict]) -> dict[str, dict]:
    """Per-cluster: count, actual %, target %, delta_pp."""
    total = len(entries) or 1
    counts = Counter(e["target_cluster"] for e in entries)
    result: dict[str, dict] = {}
    for cluster, target_pct in CLUSTER_TARGETS_PCT.items():
        count = counts.get(cluster, 0)
        actual_pct = round(count / total * 100, 1)
        result[cluster] = {
            "count": count,
            "pct_actual": actual_pct,
            "pct_target": target_pct,
            "delta_pp": round(actual_pct - target_pct, 1),
        }
    return result


def health_report(entries: list[dict]) -> str:
    total = len(entries)
    dist = cluster_distribution(entries)
    by_novelty = Counter(e.get("novelty_score") for e in entries if "novelty_score" in e)
    by_tier = Counter(e.get("scale_tier_recommended") for e in entries if "scale_tier_recommended" in e)

    lines = [
        "# Atlas v0 health report",
        "",
        f"**Total entries:** {total}",
        "",
        "## Cluster distribution",
        "",
        "| Cluster | Count | Actual % | Target % | Δ pp | Status |",
        "|---|---|---|---|---|---|",
    ]
    for cluster, info in sorted(dist.items()):
        status = "OK"
        if info["delta_pp"] < -TOLERANCE_PCT:
            status = "under-represented"
        elif info["delta_pp"] > TOLERANCE_PCT:
            status = "over-represented"
        if total > 0 and status != "OK":
            lines[0] = lines[0]  # touch
        lines.append(
            f"| {cluster} | {info['count']} | {info['pct_actual']} "
            f"| {info['pct_target']} | {info['delta_pp']:+.1f} | {status} |"
        )

    if total > 0:
        lines.extend(["", "## Novelty distribution", ""])
        for k, v in sorted(by_novelty.items()):
            lines.append(f"- **{k}:** {v}")
        lines.extend(["", "## Tier distribution", ""])
        for k, v in sorted(by_tier.items()):
            lines.append(f"- **{k}:** {v}")

    has_imbalance = any(
        abs(info["delta_pp"]) > TOLERANCE_PCT for info in dist.values()
    )
    if has_imbalance:
        lines.append("\n**WARNING: cluster imbalance detected.**")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    index_path = repo_root / "atlas" / "index.json"
    if not index_path.exists():
        print("No index.json — run indexer first.", file=sys.stderr)
        return 1
    entries = json.loads(index_path.read_text())["entries"]
    report = health_report(entries)
    print(report)
    (repo_root / "atlas" / "STATS.md").write_text(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_stats.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/stats.py atlas/tests/test_stats.py
git commit -m "feat(atlas): stats reporter with cluster-distribution targets vs actual"
```

---

### Task 1.8: CI workflow

**Files:**
- Create: `.github/workflows/atlas-validate.yml`

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/atlas-validate.yml
name: atlas-validate

on:
  pull_request:
    paths:
      - 'atlas/**'
      - 'PLAN.md'
      - 'ideas/**'
  push:
    branches: [main]
    paths:
      - 'atlas/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install pydantic pyyaml httpx pytest rich

      - name: Run unit tests
        run: pytest atlas/tests/ -v

      - name: Schema validation
        run: python -m kanad.atlas.validators

      - name: Indexer is idempotent
        run: |
          python -m kanad.atlas.indexer
          git diff --exit-code atlas/index.json atlas/by-milestone/ atlas/by-cluster/ atlas/by-tier/

      - name: Ledger generator is idempotent
        run: |
          python -m kanad.atlas.ledger_generator
          git diff --exit-code ideas/ PLAN.md atlas/by-milestone/

      - name: Stats report
        run: python -m kanad.atlas.stats
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/atlas-validate.yml
git commit -m "ci(atlas): schema validation, indexer idempotency, ledger freshness checks"
```

---

### Task 1.9: End-to-end smoke test with one hand-written entry

**Files:**
- Create: `kanad/atlas/entries/ATLAS-0001.yaml`

- [ ] **Step 1: Write a representative test entry**

```yaml
# kanad/atlas/entries/ATLAS-0001.yaml
id: ATLAS-0001
title: "p-Benzyne singlet-triplet gap as a methodology benchmark"
created: 2026-05-27
last_reviewed: 2026-05-27
curator: mk0dz
tags: [biradical, singlet-triplet, methodology]

molecule:
  formula: "C6H4"
  geometry: "para-benzyne equilibrium (CASSCF-optimized)"
  charge: 0
  multiplicity_to_test: [1, 3]
  basis_minimum_recommended: "cc-pVDZ"
  active_space_recipe: "valence π space (8e, 8o)"

question: |
  p-Benzyne is a textbook biradical with experimental ΔE_ST ≈ 3.8
  kcal/mol. Standard DFT functionals scatter widely (-1 to +8 kcal/mol)
  depending on exchange admixture. Does VQE + 2-RDM with the valence
  π active space reproduce experiment and yield diagnostics
  (Mayer bond order, NO occupations) that distinguish biradical
  from closed-shell character?

question_type: methodology
expected_observables:
  - ground_state_energy_per_multiplicity
  - natural_orbital_occupations
  - mayer_bond_order
  - spin_density_grid

success_criterion: |
  ΔE_ST within ±1 kcal/mol of experiment (3.8 kcal/mol). Biradical
  NO occupations of the σ_g/σ_u pair both within (0.9, 1.1).

capability_required:
  milestones: [M7]
  accelerators: []

ships_when: "M7 lands"

scale_tier_recommended: T2
scale_tier_minimum: T1

estimated_runtime:
  T1: "2 hours"
  T2: "20 minutes"

novelty_score: medium
novelty_justification: "Settled experiment, contested across DFT functionals; quantum-correlated 2-RDM answer not in literature"

target_researcher_cluster: academic-small-mol-methodology
secondary_clusters: [organometallic-spin-state]

literature_context:
  - doi: "10.1063/1.476850"
    citation: "Wenthold 1998 JACS 120, 5279"
    role: "Experimental ΔE_ST reference"
    excerpt: "We report the singlet-triplet splitting of p-benzyne measured by photoelectron spectroscopy."
  - doi: "10.1021/jp022048a"
    citation: "Crawford 2001 J. Phys. Chem. A 105, 11486"
    role: "CCSD benchmark"
    excerpt: "Coupled-cluster calculations of the singlet-triplet gap in p-benzyne reveal sensitivity to dynamic correlation."
```

- [ ] **Step 2: Run end-to-end pipeline**

```bash
cd /home/mk/deeprealm/kanad
python -m kanad.atlas.validators
python -m kanad.atlas.indexer
python -m kanad.atlas.ledger_generator
python -m kanad.atlas.stats
```

Expected output:
- `OK: 1 entries valid`
- `atlas/index.json` exists with 1 entry, status="blocked", unlocking_milestone="M7"
- `atlas/by-milestone/M7.md` lists ATLAS-0001
- `atlas/by-cluster/academic-small-mol-methodology.md` lists it
- `atlas/STATS.md` shows imbalance warning (single entry vs. distribution targets)

- [ ] **Step 3: Verify CI green locally**

```bash
pytest atlas/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add atlas/entries/ATLAS-0001.yaml atlas/index.json atlas/by-milestone/ atlas/by-cluster/ atlas/by-tier/ atlas/STATS.md
git commit -m "feat(atlas): smoke-test entry ATLAS-0001 (p-benzyne) exercises full pipeline"
```

---

**Phase 1 complete when:**
- [ ] All Phase 1 tests pass under `pytest atlas/tests/`
- [ ] `python -m kanad.atlas.validators` reports OK on ATLAS-0001
- [ ] `python -m kanad.atlas.indexer` is idempotent (second run produces zero git diff)
- [ ] `python -m kanad.atlas.ledger_generator` writes block to `ideas/19-quantum-2rdm.md`
- [ ] CI workflow passes on PR

---

# Phase 2 — P1: Benchmark conversion (~100 entries)

**Goal:** convert 7 published benchmark datasets into atlas entries via mechanical scripts, no LLM.

**Owner:** developer. **Estimated duration:** 3 days. **Depends on:** Phase 1.

Each converter follows the same shape: parse the published benchmark's molecule list, emit one YAML per molecule, defaulting novelty/cluster/tier based on the source's known character. Sequential IDs ATLAS-0002 through ATLAS-0100 (approximately — exact count depends on filtering).

---

### Task 2.1: GMTKN55 converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/gmtkn55_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/gmtkn55_molecules.json` (filtered subset)
- Test: `kanad/atlas/tests/test_gmtkn55_converter.py`

- [ ] **Step 1: Curate input data**

Download GMTKN55 subset files (publicly available via the Goerigk-Mertens paper supplementary). Filter to:
- ≤30 atoms total
- ≤16 heavy atoms
- Charge ∈ {-1, 0, +1}
- Multiplicity ∈ {1, 2, 3}

Output: `kanad/atlas/sourcing/benchmarks/data/gmtkn55_molecules.json` — list of objects:
```json
[
  {
    "id_in_source": "G2RC-01",
    "formula": "H2",
    "name": "Hydrogen",
    "charge": 0,
    "multiplicity": 1,
    "reference_method": "W1-F12",
    "n_atoms": 2,
    "doi": "10.1039/C7CP04913G",
    "subcategory": "small-molecule reactions"
  },
  ...
]
```

Target: ~30 molecules after filtering.

- [ ] **Step 2: Write failing test**

```python
# kanad/atlas/tests/test_gmtkn55_converter.py
from pathlib import Path
import yaml
from kanad.atlas.sourcing.benchmarks.gmtkn55_to_atlas import (
    convert_one,
    convert_all,
)


def test_convert_one_emits_valid_entry():
    src = {
        "id_in_source": "G2RC-01",
        "formula": "H2",
        "name": "Hydrogen",
        "charge": 0,
        "multiplicity": 1,
        "reference_method": "W1-F12",
        "n_atoms": 2,
        "doi": "10.1039/C7CP04913G",
        "subcategory": "small-molecule reactions",
    }
    entry = convert_one(src, next_id=42)
    assert entry["id"] == "ATLAS-0042"
    assert entry["molecule"]["formula"] == "H2"
    assert entry["target_researcher_cluster"] == "academic-small-mol-methodology"
    assert entry["novelty_score"] == "medium"
    assert entry["capability_required"]["milestones"] == ["M2"]
```

- [ ] **Step 3: Run, confirm fail**

```bash
pytest atlas/tests/test_gmtkn55_converter.py -v
```

- [ ] **Step 4: Implement `kanad/atlas/sourcing/benchmarks/gmtkn55_to_atlas.py`**

```python
"""Convert GMTKN55 subset → atlas YAML entries.

GMTKN55 is a 55-sub-benchmark thermochemistry/barrier-height/non-covalent
suite curated by Goerigk-Mertens-Najafzadeh-Karton (2017).
Mechanical conversion — no LLM, no human judgment per entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml


def convert_one(src: dict, next_id: int) -> dict:
    """One GMTKN55 record → atlas entry dict."""
    n_atoms = src["n_atoms"]
    tier = "T1" if n_atoms <= 6 else "T2"

    return {
        "id": f"ATLAS-{next_id:04d}",
        "title": f"{src['name']} ({src['formula']}) — GMTKN55 reference for {src['subcategory']}",
        "created": "2026-05-27",
        "last_reviewed": "2026-05-27",
        "curator": "gmtkn55_converter",
        "tags": ["gmtkn55", "benchmark", src["subcategory"].replace(" ", "-")],
        "molecule": {
            "formula": src["formula"],
            "geometry": "GMTKN55 reference geometry",
            "charge": src["charge"],
            "multiplicity_to_test": [src["multiplicity"]],
            "basis_minimum_recommended": "cc-pVDZ",
        },
        "question": (
            f"Reproduce GMTKN55 reference values for {src['formula']} "
            f"({src['subcategory']}) using VQE + correlated wavefunction, "
            f"and compare to {src['reference_method']} reference."
        ),
        "question_type": "benchmark",
        "expected_observables": ["ground_state_energy"],
        "success_criterion": "VQE energy within chemical accuracy (1.6 mHa) of W1-F12 reference",
        "capability_required": {
            "milestones": ["M2"],
            "accelerators": [],
        },
        "ships_when": "M2 lands",
        "scale_tier_recommended": tier,
        "scale_tier_minimum": "T1",
        "estimated_runtime": {
            "T1": "30 minutes" if n_atoms <= 4 else "2 hours",
            "T2": "5 minutes" if n_atoms <= 4 else "20 minutes",
        },
        "novelty_score": "medium",
        "novelty_justification": (
            "GMTKN55 reference value exists; novelty is in quantum-correlated "
            "wavefunction agreement and observable plate richness"
        ),
        "target_researcher_cluster": "academic-small-mol-methodology",
        "secondary_clusters": [],
        "literature_context": [
            {
                "doi": src["doi"],
                "citation": "Goerigk et al. 2017 PCCP 19, 32184 (GMTKN55)",
                "role": "Benchmark reference",
                "excerpt": (
                    "GMTKN55 is a benchmark database for general main-group "
                    "thermochemistry, kinetics, and noncovalent interactions."
                ),
            },
        ],
    }


def convert_all(input_path: Path, output_dir: Path, starting_id: int) -> list[Path]:
    src_list = json.loads(input_path.read_text())
    written: list[Path] = []
    for offset, src in enumerate(src_list):
        next_id = starting_id + offset
        entry = convert_one(src, next_id)
        out_path = output_dir / f"{entry['id']}.yaml"
        out_path.write_text(yaml.safe_dump(entry, sort_keys=False, default_flow_style=False))
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data" / "gmtkn55_molecules.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "entries")
    parser.add_argument("--start-id", type=int, default=2)
    args = parser.parse_args()

    written = convert_all(args.input, args.output, args.start_id)
    print(f"Wrote {len(written)} entries (IDs {args.start_id}..{args.start_id + len(written) - 1})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run test, confirm pass**

```bash
pytest atlas/tests/test_gmtkn55_converter.py -v
```

- [ ] **Step 6: Run converter against curated data**

```bash
cd /home/mk/deeprealm/kanad
python -m kanad.atlas.sourcing.benchmarks.gmtkn55_to_atlas --start-id 2
python -m kanad.atlas.validators
```

Expected: ~30 valid entries written; validator passes all.

- [ ] **Step 7: Commit**

```bash
git add atlas/sourcing/benchmarks/gmtkn55_to_atlas.py atlas/sourcing/benchmarks/data/gmtkn55_molecules.json atlas/tests/test_gmtkn55_converter.py atlas/entries/
git commit -m "feat(atlas): GMTKN55 converter — ~30 small-mol benchmark entries"
```

---

### Task 2.2: W4-11 converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/w4_11_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/w4_11_molecules.json`
- Test: `kanad/atlas/tests/test_w4_11_converter.py`

Same shape as Task 2.1 but for the W4-11 thermochemistry benchmark (140 reference small molecules with sub-mHa CCSD(T)/CBS energies). Filter to ≤14 atoms, target ~15 entries.

- [ ] **Step 1: Curate `data/w4_11_molecules.json`** from W4-11 published table (DOI: 10.1021/jp809974e). Same schema as GMTKN55 file but with `reference_method: "W4-CCSD(T)/CBS"`.

- [ ] **Step 2: Write `atlas/tests/test_w4_11_converter.py`** — copy structure from `test_gmtkn55_converter.py`, change fixture molecule.

- [ ] **Step 3: Implement converter** — copy `gmtkn55_to_atlas.py`, adjust:
  - `tags`: `["w4-11", "benchmark", "thermochemistry"]`
  - Citation: "Karton et al. 2011 JCP 136, 084110 (W4-11)"
  - DOI: `10.1063/1.3613639`
  - `novelty_score`: `low` (W4-11 is high-accuracy reference data — known values)
  - `novelty_justification`: "W4-11 has CCSD(T)/CBS reference; entry serves user-onboarding and replication benchmarking"

- [ ] **Step 4: Run tests + converter**

```bash
pytest atlas/tests/test_w4_11_converter.py -v
python -m kanad.atlas.sourcing.benchmarks.w4_11_to_atlas --start-id 32
python -m kanad.atlas.validators
```

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/benchmarks/w4_11_to_atlas.py atlas/sourcing/benchmarks/data/w4_11_molecules.json atlas/tests/test_w4_11_converter.py atlas/entries/
git commit -m "feat(atlas): W4-11 converter — ~15 high-accuracy small-mol entries"
```

---

### Task 2.3: MOR41 converter (open-shell organic radicals)

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/mor41_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/mor41_molecules.json`
- Test: `kanad/atlas/tests/test_mor41_converter.py`

Same pattern. MOR41 (Iron et al., DOI: 10.1021/acs.jctc.7b00094) is an open-shell organic radical reaction benchmark.

- [ ] **Step 1: Curate data file** — extract MOR41 reactions, keep one entry per unique radical, target ~12 entries
- [ ] **Step 2: Write test** with MOR41 fixture
- [ ] **Step 3: Implement converter** with:
  - `tags`: `["mor41", "benchmark", "open-shell", "radical"]`
  - `multiplicity_to_test`: `[2]` (doublet ground state typical)
  - `capability_required.milestones`: `["M1", "M2"]` (M1 for symmetry penalty on open-shell)
  - `novelty_score`: `medium`
  - `secondary_clusters`: `["organometallic-spin-state"]` (some entries; organic radicals connect)
- [ ] **Step 4: Run + validate**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(atlas): MOR41 converter — ~12 open-shell organic radical entries"
```

---

### Task 2.4: Truhlar TM benchmark converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/truhlar_tm_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/truhlar_tm_molecules.json`
- Test: `kanad/atlas/tests/test_truhlar_tm_converter.py`

3d transition-metal monoxides + nitrides + halides where DFT functionals disagree on spin state.

- [ ] **Step 1: Curate data file** — ~10 entries (FeO, FeO+, NiO, MnO, CrO, CoH, FeH, NiH, CrN, MnH)
- [ ] **Step 2: Write test**
- [ ] **Step 3: Implement converter** with:
  - `tags`: `["truhlar", "transition-metal", "spin-state", "dft-disagreement"]`
  - `multiplicity_to_test`: per-molecule (lookup from data)
  - `capability_required.milestones`: `["M7", "M10"]` (2-RDM + MP2-NO active space)
  - `capability_required.accelerators`: `["A1"]` (basis-set freedom for def2-SVP)
  - `scale_tier_recommended`: `T2`
  - `novelty_score`: `high` (DFT contested; quantum-correlated answer not in literature)
  - `target_researcher_cluster`: `organometallic-spin-state`
- [ ] **Step 4: Run + validate**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(atlas): Truhlar TM converter — ~10 transition-metal spin-state entries"
```

---

### Task 2.5: Reiher spin-state collection converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/reiher_spin_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/reiher_spin_molecules.json`
- Test: `kanad/atlas/tests/test_reiher_spin_converter.py`

Reiher's published TM spin-state controversies (ChemPhysChem 2002 onward).

- [ ] **Step 1: Curate data file** — ~10 entries from Reiher's published collection
- [ ] **Step 2: Write test**
- [ ] **Step 3: Implement converter** — same shape as Truhlar TM, slightly different DOI list (Reiher papers) and:
  - `tags`: `["reiher", "spin-state", "transition-metal"]`
  - `novelty_score`: `high` for contested cases, `medium` for resolved ones
- [ ] **Step 4: Run + validate**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(atlas): Reiher TM spin-state converter — ~10 entries"
```

---

### Task 2.6: Head-Gordon multireference set converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/head_gordon_mr_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/head_gordon_mr_molecules.json`
- Test: `kanad/atlas/tests/test_head_gordon_mr_converter.py`

Multireference biradical / strong-correlation collection.

- [ ] **Step 1: Curate data file** — ~10 entries (biradicals, stretched bonds, transition states with multireference character)
- [ ] **Step 2: Write test**
- [ ] **Step 3: Implement converter** with:
  - `tags`: `["head-gordon", "multireference", "biradical"]`
  - `capability_required.milestones`: `["M7"]` (2-RDM is core diagnostic)
  - `capability_required.accelerators`: `["A6"]` (multi-reference diagnostic toolkit if implemented)
  - `expected_observables`: includes `m_diagnostic`, `cumulant_norm`
  - `novelty_score`: `high`
- [ ] **Step 4: Run + validate**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(atlas): Head-Gordon MR converter — ~10 multireference biradical entries"
```

---

### Task 2.7: Classic challenging-molecules family converter

**Files:**
- Create: `kanad/atlas/sourcing/benchmarks/classic_challenging_to_atlas.py`
- Create: `kanad/atlas/sourcing/benchmarks/data/classic_challenging_molecules.json`
- Test: `kanad/atlas/tests/test_classic_challenging_converter.py`

C₂, F₂, N₂, m-benzyne, ozone, Cr₂ — the textbook hard cases with rich published references.

- [ ] **Step 1: Curate data file** — ~10 entries, each with multiple literature references (the family has decades of papers)
- [ ] **Step 2: Write test**
- [ ] **Step 3: Implement converter** with:
  - `tags`: `["classic", "challenging-molecule", "<molecule-name>"]`
  - Per-molecule custom `capability_required`, `expected_observables`, `tier`
  - `novelty_score`: `high` for entries where quantum-correlated 2-RDM richness adds publishable substance (most)
  - Multiple `literature_context` entries per molecule
- [ ] **Step 4: Run + validate**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(atlas): classic challenging-molecules converter — ~10 textbook hard cases"
```

---

### Task 2.8: Regenerate index + stats; verify P1 distribution

- [ ] **Step 1: Re-run pipeline**

```bash
cd /home/mk/deeprealm/kanad
python -m kanad.atlas.validators
python -m kanad.atlas.indexer
python -m kanad.atlas.ledger_generator
python -m kanad.atlas.stats > /tmp/atlas-p1-stats.md
cat /tmp/atlas-p1-stats.md
```

Expected: ~97 entries total (1 from Phase 1 + ~96 from Phase 2). Distribution:
- `academic-small-mol-methodology` dominant (~70%)
- `organometallic-spin-state` ~20%
- Other clusters underrepresented (expected — gets filled in P2/P3)

- [ ] **Step 2: Commit regenerated artifacts**

```bash
git add atlas/index.json atlas/by-milestone/ atlas/by-cluster/ atlas/by-tier/ atlas/STATS.md
git commit -m "feat(atlas): regenerate index + stats after Phase 2 (P1 ~100 entries)"
```

---

**Phase 2 complete when:**
- [ ] 7 benchmark converters work + tested
- [ ] ~97 valid entries in `atlas/entries/`
- [ ] `stats.py` reports cluster distribution; `academic-small-mol-methodology` and `organometallic-spin-state` dominate; other clusters flagged as under-represented (this is expected and gets resolved in P2/P3)
- [ ] CI is green

---

# Phase 3 — P2: Lit-mining pipeline build

**Goal:** build the 6-step LLM-assisted lit-mining pipeline. Does not run it — just constructs the tools.

**Owner:** developer. **Estimated duration:** 3 days. **Depends on:** Phase 1.

---

### Task 3.1: `discover.py` — query S2 / Crossref / OpenAlex

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/__init__.py`
- Create: `kanad/atlas/sourcing/lit_mining/discover.py`
- Test: `kanad/atlas/tests/test_lit_discover.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_lit_discover.py
from unittest.mock import patch
from kanad.atlas.sourcing.lit_mining.discover import (
    discover_candidates,
    CandidatePaper,
)


def test_discover_returns_paper_objects():
    fake_response = {
        "data": [
            {
                "paperId": "abc123",
                "doi": "10.1021/test",
                "title": "DFT vs CASSCF for FeO+",
                "abstract": "We compare B3LYP and CASSCF on FeO+...",
                "year": 2024,
                "venue": "JCTC",
            }
        ],
        "total": 1,
    }
    with patch("kanad.atlas.sourcing.lit_mining.discover.httpx.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_response

        papers = discover_candidates(
            query="quantum chemistry transition metal",
            year_min=2021,
            limit=10,
        )
        assert len(papers) == 1
        assert isinstance(papers[0], CandidatePaper)
        assert papers[0].doi == "10.1021/test"
```

- [ ] **Step 2: Run, confirm fail**

```bash
pytest atlas/tests/test_lit_discover.py -v
```

- [ ] **Step 3: Implement `discover.py`**

```python
"""Query Semantic Scholar (primary) + Crossref (fallback) for candidate papers.

Output: list of CandidatePaper objects, cached to disk.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx


S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
USER_AGENT = "Kanad-Atlas-LitMining/0.1 (mailto:31vivekpal@gmail.com)"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CandidatePaper:
    paper_id: str
    doi: Optional[str]
    title: str
    abstract: Optional[str]
    year: Optional[int]
    venue: Optional[str]


def discover_candidates(
    query: str,
    year_min: int = 2021,
    limit: int = 100,
) -> list[CandidatePaper]:
    cache_key = f"discover_{abs(hash(query))}_{year_min}_{limit}.json"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return [CandidatePaper(**p) for p in json.loads(cache_path.read_text())]

    params = {
        "query": query,
        "year": f"{year_min}-",
        "fields": "title,abstract,year,venue,externalIds",
        "limit": limit,
    }
    response = httpx.get(
        S2_BASE,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()

    papers: list[CandidatePaper] = []
    for item in data.get("data", []):
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI") or item.get("doi")
        papers.append(CandidatePaper(
            paper_id=item.get("paperId", ""),
            doi=doi,
            title=item.get("title", ""),
            abstract=item.get("abstract"),
            year=item.get("year"),
            venue=item.get("venue"),
        ))

    cache_path.write_text(json.dumps([asdict(p) for p in papers], indent=2))
    time.sleep(1.0)  # polite throttling for unauth S2 (1 RPS)
    return papers
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_lit_discover.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/__init__.py atlas/sourcing/lit_mining/discover.py atlas/tests/test_lit_discover.py
git commit -m "feat(atlas): lit-mining discovery via Semantic Scholar API"
```

---

### Task 3.2: `filter.py` — heuristic abstract filter

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/filter.py`
- Test: `kanad/atlas/tests/test_lit_filter.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_lit_filter.py
from kanad.atlas.sourcing.lit_mining.discover import CandidatePaper
from kanad.atlas.sourcing.lit_mining.filter import has_contested_claim_signal


def test_dft_vs_phrase_passes():
    p = CandidatePaper("x", "10.x", "DFT vs CASSCF for FeO+", "We compare DFT functionals to CASSCF...", 2024, "JCTC")
    assert has_contested_claim_signal(p)


def test_unrelated_abstract_rejected():
    p = CandidatePaper("x", "10.x", "A new catalyst", "We report a new catalyst for hydrogenation.", 2024, "JACS")
    assert not has_contested_claim_signal(p)


def test_remains_unclear_passes():
    p = CandidatePaper("x", "10.x", "Spin state of NiO", "The ground state of NiO remains unclear.", 2024, "JCP")
    assert has_contested_claim_signal(p)


def test_missing_abstract_rejected():
    p = CandidatePaper("x", "10.x", "DFT vs CASSCF", None, 2024, "JCTC")
    assert not has_contested_claim_signal(p)
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `filter.py`**

```python
"""Heuristic filter: abstract must contain a contested-claim signal."""
from __future__ import annotations

import re

from .discover import CandidatePaper


# Patterns that suggest the paper contains an open / contested research question.
SIGNAL_PATTERNS = [
    r"\bDFT\s+vs\b",
    r"\bremains?\s+unclear\b",
    r"\bwe\s+benchmark\b",
    r"\bcontradictory\b",
    r"\bopen\s+question\b",
    r"\bcontroversy\b",
    r"\bdisagreement\b",
    r"\bunresolved\b",
    r"\bin\s+contrast\s+to\s+previous\b",
    r"\bfunctional\s+dependence\b",
    r"\bsensitive\s+to\b",
    r"\bmultireference\b",
    r"\bstrongly\s+correlated\b",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in SIGNAL_PATTERNS]


def has_contested_claim_signal(paper: CandidatePaper) -> bool:
    """True if abstract contains at least one signal pattern."""
    if not paper.abstract:
        return False
    return any(rx.search(paper.abstract) for rx in COMPILED)


def filter_candidates(
    candidates: list[CandidatePaper],
) -> list[CandidatePaper]:
    return [c for c in candidates if has_contested_claim_signal(c)]
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_lit_filter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/filter.py atlas/tests/test_lit_filter.py
git commit -m "feat(atlas): lit-mining heuristic abstract filter"
```

---

### Task 3.3: `extract.py` — LLM extraction with structured prompt

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/extract.py`
- Test: `kanad/atlas/tests/test_lit_extract.py`

For v0 the LLM extraction is invoked via the user's Claude Code (or a scripted Claude API call). The module's job is to: build the structured prompt, parse the structured JSON response, and write candidates to disk for the verify+review steps.

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_lit_extract.py
import json
from kanad.atlas.sourcing.lit_mining.extract import (
    build_extraction_prompt,
    parse_extraction_response,
    ExtractedCandidate,
)
from kanad.atlas.sourcing.lit_mining.discover import CandidatePaper


def test_prompt_includes_paper_metadata():
    paper = CandidatePaper(
        "abc", "10.1021/test", "DFT vs CASSCF for FeO+",
        "We compare B3LYP and CASSCF for FeO+ spin state.", 2024, "JCTC"
    )
    prompt = build_extraction_prompt(paper)
    assert "10.1021/test" in prompt
    assert "FeO+" in prompt
    assert "Return JSON" in prompt


def test_parse_extraction_response_valid():
    resp = json.dumps({
        "is_open_question": True,
        "molecule": "FeO+",
        "contested_claim": "B3LYP vs BP86 disagree on spin state",
        "soa_answer": "CASPT2 gives sextet",
        "kanad_capability_fit": True,
        "candidate_cluster": "organometallic-spin-state",
        "excerpt": "B3LYP and BP86 yield different ground-state multiplicities for FeO+."
    })
    candidate = parse_extraction_response(resp, paper_doi="10.1021/test")
    assert isinstance(candidate, ExtractedCandidate)
    assert candidate.molecule == "FeO+"
    assert candidate.is_open_question is True


def test_parse_skips_non_open():
    resp = json.dumps({
        "is_open_question": False,
        "molecule": "H2",
        "contested_claim": None,
        "soa_answer": "Settled",
        "kanad_capability_fit": True,
        "candidate_cluster": "academic-small-mol-methodology",
        "excerpt": "..."
    })
    candidate = parse_extraction_response(resp, paper_doi="10.x")
    assert candidate is None
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `extract.py`**

```python
"""LLM extraction step: paper abstract → ExtractedCandidate.

For v0 the LLM call is performed by the operator (Claude Code session,
parallel Agent dispatch, or scripted Anthropic API call). This module
provides the prompt template + response parser; the orchestration script
(`run_extraction.py`) coordinates the actual calls.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .discover import CandidatePaper


CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class ExtractedCandidate:
    paper_doi: str
    is_open_question: bool
    molecule: str
    contested_claim: Optional[str]
    soa_answer: Optional[str]
    kanad_capability_fit: bool
    candidate_cluster: str
    excerpt: str


PROMPT_TEMPLATE = """\
You are extracting candidate research questions for a quantum-chemistry
atlas. Strict rules:

1. Output JSON only, no prose.
2. Do NOT invent excerpts; copy verbatim from the abstract below.
3. Do NOT invent DOIs.
4. If the paper is methodology/benchmark only with no open question,
   set is_open_question = false and the rest may be null.
5. kanad_capability_fit is true ONLY if the molecule fits in ≤32 qubits
   (rule of thumb: ≤16 heavy atoms, no full FeMoco/Mn12).

Paper metadata:
- DOI: {doi}
- Title: {title}
- Year: {year}
- Venue: {venue}

Abstract:
\"\"\"
{abstract}
\"\"\"

Return JSON with keys:
- is_open_question (bool)
- molecule (string; chemical formula or family name)
- contested_claim (string or null; what's contested)
- soa_answer (string or null; state-of-the-art answer or "no answer in literature")
- kanad_capability_fit (bool)
- candidate_cluster (one of: academic-small-mol-methodology, organometallic-spin-state,
    photochemistry-pv, drug-fragment-screening, materials-small-cell,
    hardware-paper-authors, industrial-pharma-rd, industrial-materials-rd)
- excerpt (string; copied verbatim from the abstract above)
"""


def build_extraction_prompt(paper: CandidatePaper) -> str:
    return PROMPT_TEMPLATE.format(
        doi=paper.doi or "(no DOI)",
        title=paper.title or "",
        year=paper.year or "",
        venue=paper.venue or "",
        abstract=paper.abstract or "(no abstract available)",
    )


def parse_extraction_response(
    response_text: str, paper_doi: str
) -> Optional[ExtractedCandidate]:
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return None
    if not data.get("is_open_question"):
        return None
    return ExtractedCandidate(
        paper_doi=paper_doi,
        is_open_question=True,
        molecule=data.get("molecule", ""),
        contested_claim=data.get("contested_claim"),
        soa_answer=data.get("soa_answer"),
        kanad_capability_fit=bool(data.get("kanad_capability_fit", False)),
        candidate_cluster=data.get("candidate_cluster", ""),
        excerpt=data.get("excerpt", ""),
    )


def save_candidates(candidates: list[ExtractedCandidate], name: str) -> Path:
    path = CACHE_DIR / f"extracted_{name}.jsonl"
    with path.open("w") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c)) + "\n")
    return path


def load_candidates(name: str) -> list[ExtractedCandidate]:
    path = CACHE_DIR / f"extracted_{name}.jsonl"
    if not path.exists():
        return []
    out: list[ExtractedCandidate] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        out.append(ExtractedCandidate(**json.loads(line)))
    return out
```

- [ ] **Step 4: Run, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/extract.py atlas/tests/test_lit_extract.py
git commit -m "feat(atlas): lit-mining LLM extraction prompt + structured-response parser"
```

---

### Task 3.4: `verify.py` — DOI + excerpt verification

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/verify.py`
- Test: `kanad/atlas/tests/test_lit_verify.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_lit_verify.py
from unittest.mock import patch
from kanad.atlas.sourcing.lit_mining.verify import (
    verify_candidate,
    VerificationResult,
)
from kanad.atlas.sourcing.lit_mining.extract import ExtractedCandidate


def _candidate(**overrides):
    base = {
        "paper_doi": "10.1021/test",
        "is_open_question": True,
        "molecule": "FeO+",
        "contested_claim": "DFT functional disagreement",
        "soa_answer": "CASPT2 gives sextet",
        "kanad_capability_fit": True,
        "candidate_cluster": "organometallic-spin-state",
        "excerpt": "B3LYP and BP86 yield different ground-state multiplicities for FeO+.",
    }
    base.update(overrides)
    return ExtractedCandidate(**base)


def test_verify_passes_when_doi_and_excerpt_match():
    candidate = _candidate()
    with patch("kanad.atlas.sourcing.lit_mining.verify.resolve_doi") as mock_resolve:
        from kanad.atlas.validators.doi_resolver import CrossrefRecord
        mock_resolve.return_value = CrossrefRecord(
            doi="10.1021/test", title="x",
            abstract="B3LYP and BP86 yield different ground-state multiplicities for FeO+. More text.",
        )
        result = verify_candidate(candidate)
        assert isinstance(result, VerificationResult)
        assert result.passed


def test_verify_fails_when_doi_unresolvable():
    from kanad.atlas.validators.doi_resolver import DOIResolutionError
    candidate = _candidate()
    with patch("kanad.atlas.sourcing.lit_mining.verify.resolve_doi") as mock_resolve:
        mock_resolve.side_effect = DOIResolutionError("404")
        result = verify_candidate(candidate)
        assert not result.passed
        assert "doi" in result.reason.lower()


def test_verify_fails_when_excerpt_not_in_abstract():
    candidate = _candidate(excerpt="This text is not in the actual abstract")
    with patch("kanad.atlas.sourcing.lit_mining.verify.resolve_doi") as mock_resolve:
        from kanad.atlas.validators.doi_resolver import CrossrefRecord
        mock_resolve.return_value = CrossrefRecord(
            doi="10.1021/test", title="x", abstract="Different abstract content."
        )
        result = verify_candidate(candidate)
        assert not result.passed
        assert "excerpt" in result.reason.lower()


def test_verify_fails_when_capability_fit_false():
    candidate = _candidate(kanad_capability_fit=False)
    result = verify_candidate(candidate)
    assert not result.passed
    assert "capability" in result.reason.lower()
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `verify.py`**

```python
"""Verify extracted candidates: DOI resolves + excerpt byte-matches + capability fits."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from kanad.atlas.validators.doi_resolver import (
    DOIResolutionError,
    resolve_doi,
    verify_excerpt,
)
from .extract import ExtractedCandidate, CACHE_DIR


@dataclass(frozen=True)
class VerificationResult:
    candidate: ExtractedCandidate
    passed: bool
    reason: str


def verify_candidate(candidate: ExtractedCandidate) -> VerificationResult:
    if not candidate.kanad_capability_fit:
        return VerificationResult(candidate, False, "capability fit declared false")

    try:
        record = resolve_doi(candidate.paper_doi)
    except DOIResolutionError as e:
        return VerificationResult(candidate, False, f"doi unresolvable: {e}")

    if not verify_excerpt(candidate.excerpt, record.abstract):
        return VerificationResult(
            candidate, False,
            f"excerpt not byte-matched in abstract for {candidate.paper_doi}",
        )

    return VerificationResult(candidate, True, "verified")


def verify_all(
    candidates: list[ExtractedCandidate],
    log_path: Optional[Path] = None,
) -> list[VerificationResult]:
    results = [verify_candidate(c) for c in candidates]
    log_path = log_path or (CACHE_DIR / "verify-log.jsonl")
    with log_path.open("w") as f:
        for r in results:
            f.write(json.dumps({
                "doi": r.candidate.paper_doi,
                "passed": r.passed,
                "reason": r.reason,
            }) + "\n")
    return results
```

- [ ] **Step 4: Run, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/verify.py atlas/tests/test_lit_verify.py
git commit -m "feat(atlas): lit-mining DOI + excerpt verification with auto-reject"
```

---

### Task 3.5: `review_cli.py` — batch curator review TUI

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/review_cli.py`
- Test: `kanad/atlas/tests/test_review_cli.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_review_cli.py
from io import StringIO
from kanad.atlas.sourcing.lit_mining.review_cli import (
    render_candidate,
    apply_decision,
    ReviewDecision,
)
from kanad.atlas.sourcing.lit_mining.verify import VerificationResult
from kanad.atlas.sourcing.lit_mining.extract import ExtractedCandidate


def _vr(passed=True):
    c = ExtractedCandidate(
        paper_doi="10.1021/test", is_open_question=True,
        molecule="FeO+", contested_claim="DFT disagreement",
        soa_answer="CASPT2 gives sextet", kanad_capability_fit=True,
        candidate_cluster="organometallic-spin-state",
        excerpt="B3LYP and BP86 yield different ground-state multiplicities for FeO+.",
    )
    return VerificationResult(c, passed, "ok" if passed else "fail")


def test_render_includes_key_fields():
    rendered = render_candidate(_vr())
    assert "FeO+" in rendered
    assert "10.1021/test" in rendered
    assert "DFT disagreement" in rendered


def test_apply_decision_approve_keeps_candidate():
    state = {"approved": [], "rejected": []}
    apply_decision(state, _vr(), ReviewDecision.APPROVE)
    assert len(state["approved"]) == 1
    assert len(state["rejected"]) == 0


def test_apply_decision_reject_buckets_candidate():
    state = {"approved": [], "rejected": []}
    apply_decision(state, _vr(), ReviewDecision.REJECT)
    assert len(state["rejected"]) == 1
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `review_cli.py`**

```python
"""Curator batch-review TUI for verified lit-mining candidates.

Reads verified candidates from cache → presents one at a time →
prompts curator for approve / reject / edit → writes decisions to
cache → emit_yaml.py consumes approved set.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .extract import CACHE_DIR
from .verify import VerificationResult


console = Console()


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


def render_candidate(vr: VerificationResult) -> str:
    c = vr.candidate
    return (
        f"DOI: {c.paper_doi}\n"
        f"Molecule: {c.molecule}\n"
        f"Cluster: {c.candidate_cluster}\n"
        f"Contested claim: {c.contested_claim}\n"
        f"SOA answer: {c.soa_answer}\n"
        f"Excerpt: {c.excerpt}\n"
        f"Verification: {'passed' if vr.passed else 'FAILED'} ({vr.reason})\n"
    )


def apply_decision(
    state: dict, vr: VerificationResult, decision: ReviewDecision
) -> None:
    if decision == ReviewDecision.APPROVE:
        state["approved"].append(asdict(vr.candidate))
    elif decision == ReviewDecision.REJECT:
        state["rejected"].append({
            "candidate": asdict(vr.candidate),
            "verification_reason": vr.reason,
        })
    # DEFER: not bucketed; revisit later


def prompt_decision() -> Optional[ReviewDecision]:
    """Read a single keystroke. a=approve, r=reject, d=defer, q=quit."""
    console.print(
        "[bold cyan]Decision: [a]pprove / [r]eject / [d]efer / [q]uit[/]",
    )
    response = input("> ").strip().lower()
    return {
        "a": ReviewDecision.APPROVE,
        "r": ReviewDecision.REJECT,
        "d": ReviewDecision.DEFER,
        "q": None,
    }.get(response)


def review_session(verification_log: Path, decisions_path: Path) -> None:
    """Walk through verified candidates, prompt curator, write decisions."""
    verified: list[VerificationResult] = []
    for line in verification_log.read_text().splitlines():
        if not line.strip():
            continue
        # NOTE: this is the verify-log.jsonl which contains pass/fail per DOI.
        # For full review, also load the cached candidates and pair by DOI.
        # (In v0 the orchestration script handles pairing.)

    state = {"approved": [], "rejected": []}
    for vr in verified:
        if not vr.passed:
            continue  # auto-rejected by verification
        console.print(Panel(render_candidate(vr), title=vr.candidate.molecule))
        decision = prompt_decision()
        if decision is None:
            break
        apply_decision(state, vr, decision)

    decisions_path.write_text(json.dumps(state, indent=2))
    console.print(f"\n[bold green]Approved: {len(state['approved'])}[/]")
    console.print(f"[bold red]Rejected: {len(state['rejected'])}[/]")
```

- [ ] **Step 4: Run, confirm pass**

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/review_cli.py atlas/tests/test_review_cli.py
git commit -m "feat(atlas): lit-mining batch-review TUI with rich rendering"
```

---

### Task 3.6: `emit_yaml.py` — approved candidates → atlas YAML

**Files:**
- Create: `kanad/atlas/sourcing/lit_mining/emit_yaml.py`
- Test: `kanad/atlas/tests/test_lit_emit_yaml.py`

- [ ] **Step 1: Write failing test**

```python
# kanad/atlas/tests/test_lit_emit_yaml.py
from pathlib import Path
import yaml
from kanad.atlas.sourcing.lit_mining.emit_yaml import candidate_to_entry
from kanad.atlas.sourcing.lit_mining.extract import ExtractedCandidate


def test_candidate_becomes_valid_entry(tmp_path: Path):
    c = ExtractedCandidate(
        paper_doi="10.1021/test", is_open_question=True,
        molecule="FeO+", contested_claim="DFT disagreement",
        soa_answer="CASPT2 gives sextet", kanad_capability_fit=True,
        candidate_cluster="organometallic-spin-state",
        excerpt="B3LYP and BP86 yield different ground-state multiplicities for FeO+.",
    )
    entry = candidate_to_entry(c, next_id=101, paper_title="DFT vs CASSCF for FeO+", paper_venue="JCTC")
    assert entry["id"] == "ATLAS-0101"
    assert entry["molecule"]["formula"] == "FeO+"
    assert entry["target_researcher_cluster"] == "organometallic-spin-state"
    assert entry["novelty_score"] == "high"
    assert entry["literature_context"][0]["doi"] == "10.1021/test"
```

- [ ] **Step 2: Run, confirm fail**

- [ ] **Step 3: Implement `emit_yaml.py`**

```python
"""Convert approved ExtractedCandidate → atlas YAML entry.

Defaults:
- novelty_score = high (lit-mining surfaces contested claims by design)
- question_type = discovery
- capability_required = inferred from cluster + heuristic
- scale_tier = T2 (lit-mining catches mid-size systems mostly)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml

from .extract import ExtractedCandidate


CLUSTER_TO_MILESTONES: dict[str, list[str]] = {
    "academic-small-mol-methodology": ["M7", "M8"],
    "organometallic-spin-state": ["M7", "M10"],
    "photochemistry-pv": ["M11"],
    "drug-fragment-screening": ["M8", "M14"],
    "materials-small-cell": ["M7"],  # M14.7 (periodic) when scoped
    "hardware-paper-authors": ["M4", "M14.9"],
    "industrial-pharma-rd": ["M8", "M14"],
    "industrial-materials-rd": ["M14"],
}

CLUSTER_TO_ACCELERATORS: dict[str, list[str]] = {
    "academic-small-mol-methodology": [],
    "organometallic-spin-state": ["A1"],
    "photochemistry-pv": ["A5"],
    "drug-fragment-screening": ["A2"],
    "materials-small-cell": ["A3"],
    "hardware-paper-authors": ["A7"],
    "industrial-pharma-rd": [],
    "industrial-materials-rd": ["A3"],
}


def candidate_to_entry(
    c: ExtractedCandidate,
    next_id: int,
    paper_title: str = "",
    paper_venue: str = "",
) -> dict:
    milestones = CLUSTER_TO_MILESTONES.get(c.candidate_cluster, ["M7"])
    accelerators = CLUSTER_TO_ACCELERATORS.get(c.candidate_cluster, [])
    ships_when = " + ".join(milestones + accelerators) + " land"

    return {
        "id": f"ATLAS-{next_id:04d}",
        "title": (paper_title or f"{c.molecule}: {c.contested_claim or 'open question'}")[:100],
        "created": "2026-05-27",
        "last_reviewed": "2026-05-27",
        "curator": "lit_mining_pipeline",
        "tags": ["lit-mining", c.candidate_cluster],
        "molecule": {
            "formula": c.molecule,
            "geometry": "to be determined per literature reference",
            "charge": 0,  # default; curator may edit during review
            "basis_minimum_recommended": "cc-pVDZ",
        },
        "question": (
            f"{c.contested_claim or 'Open research question'}. "
            f"State of the art: {c.soa_answer or 'no settled answer in literature'}. "
            f"What does Kanad's quantum-correlated 2-RDM analysis say?"
        ),
        "question_type": "discovery",
        "expected_observables": ["ground_state_energy", "natural_orbital_occupations"],
        "success_criterion": (
            "Quantum-correlated answer either resolves the contested claim "
            "or quantifies the disagreement with diagnostic observables."
        ),
        "capability_required": {
            "milestones": milestones,
            "accelerators": accelerators,
        },
        "ships_when": ships_when,
        "scale_tier_recommended": "T2",
        "scale_tier_minimum": "T1",
        "estimated_runtime": {"T1": "4 hours", "T2": "30 minutes"},
        "novelty_score": "high",
        "novelty_justification": c.contested_claim or "open question per literature",
        "target_researcher_cluster": c.candidate_cluster,
        "secondary_clusters": [],
        "literature_context": [
            {
                "doi": c.paper_doi,
                "citation": paper_title or c.paper_doi,
                "role": "Source paper from lit mining",
                "excerpt": c.excerpt,
            }
        ],
    }


def emit_all(
    approved_path: Path,
    output_dir: Path,
    starting_id: int,
) -> list[Path]:
    import json
    state = json.loads(approved_path.read_text())
    written: list[Path] = []
    for offset, c_dict in enumerate(state.get("approved", [])):
        c = ExtractedCandidate(**c_dict)
        entry = candidate_to_entry(c, next_id=starting_id + offset)
        out_path = output_dir / f"{entry['id']}.yaml"
        out_path.write_text(yaml.safe_dump(entry, sort_keys=False, default_flow_style=False))
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "entries")
    parser.add_argument("--start-id", type=int, required=True)
    args = parser.parse_args()

    written = emit_all(args.approved, args.output, args.start_id)
    print(f"Wrote {len(written)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run, confirm pass**

```bash
pytest atlas/tests/test_lit_emit_yaml.py -v
```

- [ ] **Step 5: Commit**

```bash
git add atlas/sourcing/lit_mining/emit_yaml.py atlas/tests/test_lit_emit_yaml.py
git commit -m "feat(atlas): lit-mining approved → atlas YAML emitter"
```

---

**Phase 3 complete when:**
- [ ] All 6 lit-mining modules implemented + tested
- [ ] `pytest atlas/tests/test_lit_*.py` all pass
- [ ] No actual lit-mining run yet (that's Phase 4)

---

# Phase 4 — P2: Run lit-mining + curator review

**Goal:** execute the lit-mining pipeline across all 8 clusters; curator reviews ~300 candidates; ~120 approved YAML entries written.

**Owner:** developer dispatches agents; curator reviews. **Estimated duration:** 7 days (3 days dispatch + 4 days curator review). **Depends on:** Phase 3.

Per-cluster batches let curator pause / resume between clusters.

---

### Task 4.1: Run discover + filter per cluster

For each of the 8 clusters, run the discovery + filter steps with cluster-specific queries.

- [ ] **Step 1: Write cluster query map**

Create `kanad/atlas/sourcing/lit_mining/cluster_queries.py`:

```python
CLUSTER_QUERIES: dict[str, list[str]] = {
    "academic-small-mol-methodology": [
        "quantum chemistry small molecule benchmark methodology",
        "VQE correlated wavefunction small molecule",
        "multireference small molecule disagreement",
    ],
    "organometallic-spin-state": [
        "transition metal spin state DFT functional dependence",
        "iron oxide CASSCF multireference",
        "organometallic ground state multiplicity",
    ],
    "photochemistry-pv": [
        "excited state quantum chemistry oscillator strength",
        "photochemistry conical intersection TDDFT disagreement",
        "singlet fission triplet pair character",
    ],
    "drug-fragment-screening": [
        "drug fragment quantum descriptor HOMO LUMO",
        "molecular property prediction ab initio fragment",
        "ADME quantum chemistry descriptor",
    ],
    "materials-small-cell": [
        "small unit cell band structure DFT Hubbard",
        "Mott insulator small molecule cluster",
        "materials quantum simulation periodic",
    ],
    "hardware-paper-authors": [
        "VQE noisy intermediate-scale quantum hardware",
        "error mitigation quantum chemistry hardware",
        "IBM quantum chemistry benchmark molecule",
    ],
    "industrial-pharma-rd": [
        "drug discovery quantum chemistry screening pipeline",
        "pharmaceutical molecular property quantum",
        "fragment-based drug design ab initio",
    ],
    "industrial-materials-rd": [
        "catalyst design quantum chemistry industrial",
        "polymorph energetics computational chemistry",
        "battery materials density functional",
    ],
}
```

- [ ] **Step 2: Write driver script `run_per_cluster.py`**

```python
# kanad/atlas/sourcing/lit_mining/run_per_cluster.py
"""Driver: per cluster, run discover + filter, save filtered candidates."""
import argparse
import json
from pathlib import Path

from .cluster_queries import CLUSTER_QUERIES
from .discover import discover_candidates
from .filter import filter_candidates
from .extract import CACHE_DIR


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True, choices=list(CLUSTER_QUERIES.keys()))
    parser.add_argument("--year-min", type=int, default=2021)
    parser.add_argument("--limit-per-query", type=int, default=100)
    args = parser.parse_args()

    all_filtered = []
    for query in CLUSTER_QUERIES[args.cluster]:
        candidates = discover_candidates(query, year_min=args.year_min, limit=args.limit_per_query)
        filtered = filter_candidates(candidates)
        all_filtered.extend(filtered)
        print(f"  query={query!r}: {len(candidates)} → {len(filtered)} after filter")

    seen_dois = set()
    deduped = []
    for c in all_filtered:
        if c.doi and c.doi not in seen_dois:
            seen_dois.add(c.doi)
            deduped.append(c)

    out = CACHE_DIR / f"filtered_{args.cluster}.json"
    out.write_text(json.dumps([c.__dict__ for c in deduped], indent=2))
    print(f"Wrote {len(deduped)} unique candidates to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run per cluster**

```bash
for cluster in academic-small-mol-methodology organometallic-spin-state photochemistry-pv drug-fragment-screening materials-small-cell hardware-paper-authors industrial-pharma-rd industrial-materials-rd; do
  python -m kanad.atlas.sourcing.lit_mining.run_per_cluster --cluster "$cluster"
done
```

Expected output: ~50–200 filtered candidates per cluster cached to disk.

- [ ] **Step 4: Commit driver scripts**

```bash
git add atlas/sourcing/lit_mining/cluster_queries.py atlas/sourcing/lit_mining/run_per_cluster.py
git commit -m "feat(atlas): per-cluster lit-mining driver"
```

---

### Task 4.2: Dispatch parallel LLM extraction (batches of ~50 papers per agent)

The extraction step is parallelizable. For each cluster's filtered list, dispatch parallel Claude agents (via subagent or scripted API call), each handling ~50 papers.

- [ ] **Step 1: For each cluster, dispatch parallel agents**

For each cluster, dispatch N parallel agents (where N = ceil(filtered_count / 50)). Each agent receives a chunk of papers and uses `extract.build_extraction_prompt()` to construct prompts, then runs them through Claude (or the LLM of choice). Output is appended to `cache/extracted_<cluster>.jsonl`.

The orchestration is done by running the writing-plans skill's recommended subagent-driven-development sub-skill, which dispatches one agent per chunk. Alternatively use the Anthropic SDK directly in a script.

Example agent prompt:
```
You're extracting candidate research questions for the Kanad quantum-chemistry atlas.
Read these N paper records (DOI + title + abstract). For each, call build_extraction_prompt()
and produce a JSON object following parse_extraction_response()'s expected schema.
Output to cache/extracted_<cluster>.jsonl, one JSON object per line.
Do NOT invent DOIs or excerpts. Copy excerpts verbatim from the abstracts.
```

- [ ] **Step 2: After extraction completes for a cluster, create `verify_runner.py`**

```python
# kanad/atlas/sourcing/lit_mining/verify_runner.py
"""Driver: load extracted candidates for a cluster, verify, save passing set."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .extract import CACHE_DIR, ExtractedCandidate
from .verify import verify_all


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", required=True)
    args = parser.parse_args()

    extracted_path = CACHE_DIR / f"extracted_{args.cluster}.jsonl"
    if not extracted_path.exists():
        print(f"No extracted candidates at {extracted_path}", file=sys.stderr)
        return 1

    candidates: list[ExtractedCandidate] = []
    for line in extracted_path.read_text().splitlines():
        if not line.strip():
            continue
        candidates.append(ExtractedCandidate(**json.loads(line)))

    log_path = CACHE_DIR / f"verify-log-{args.cluster}.jsonl"
    results = verify_all(candidates, log_path=log_path)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print(f"Verified {len(results)}: {len(passed)} passed, {len(failed)} failed")
    print(f"Rejection rate: {len(failed) / len(results) * 100:.1f}%")
    if len(results) and len(failed) / len(results) > 0.4:
        print(f"WARNING: rejection rate >40% for {args.cluster} — fall back to manual mode")

    verified_path = CACHE_DIR / f"verified_{args.cluster}.jsonl"
    with verified_path.open("w") as f:
        for r in passed:
            f.write(json.dumps(asdict(r.candidate)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Then run:

```bash
python -m kanad.atlas.sourcing.lit_mining.verify_runner --cluster <cluster>
```

- [ ] **Step 3: Verify rejection rate <40% per cluster**

If a cluster's verification rejection rate exceeds 40%, fall back to "human-led search + LLM for capability mapping only" (per spec §8 risks). Pause the cluster, switch to manual mode for P3.

- [ ] **Step 4: Commit per-cluster verification logs**

```bash
git add atlas/sourcing/lit_mining/cache/verify-log-summary.json
git commit -m "feat(atlas): lit-mining extraction + verification round 1"
```

(Raw caches are gitignored; only summary stats commit.)

---

### Task 4.3: Curator batch review per cluster (~15 hours total)

- [ ] **Step 1: For each cluster, run review TUI**

```bash
python -m kanad.atlas.sourcing.lit_mining.review_cli \
  --verified cache/verified_<cluster>.jsonl \
  --decisions cache/decisions_<cluster>.json
```

Curator reviews each verified candidate at ~3 minutes/each. For each: approve / reject / defer. Target: 15–20 approvals per cluster on average → ~120 across 8 clusters.

- [ ] **Step 2: Emit YAML for approved candidates**

```bash
python -m kanad.atlas.sourcing.lit_mining.emit_yaml \
  --approved cache/decisions_<cluster>.json \
  --start-id 101  # adjust per cluster, sequential
```

Each cluster emits ~15 entries starting from the next free ID.

- [ ] **Step 3: Validate emitted entries**

```bash
python -m kanad.atlas.validators
```

Expected: all emitted YAMLs pass validation.

- [ ] **Step 4: Regenerate index + stats after each cluster**

```bash
python -m kanad.atlas.indexer
python -m kanad.atlas.stats
```

- [ ] **Step 5: Commit per-cluster batches**

```bash
git add atlas/entries/ATLAS-*.yaml atlas/index.json atlas/by-*/
git commit -m "feat(atlas): P2 lit-mining batch — <cluster> (~15 entries)"
```

---

**Phase 4 complete when:**
- [ ] All 8 clusters processed
- [ ] ~120 lit-mining entries written (ATLAS-0101 to ~ATLAS-0220)
- [ ] Verification rejection rate <40% per cluster (else cluster falls back to manual P3)
- [ ] `python -m kanad.atlas.stats` shows distribution moving toward Appendix C targets

---

# Phase 5 — P3: Manual cluster fill (~50 entries)

**Goal:** close cluster gaps identified by stats.py after P1+P2; hand-write entries for under-represented clusters.

**Owner:** curator only (no developer time required after Phase 1's tooling is in place). **Estimated duration:** 5 days (~25 hours curator time). **Depends on:** Phase 2 + Phase 4.

---

### Task 5.1: Identify gaps + create cluster_fill working files

- [ ] **Step 1: Run stats**

```bash
cd /home/mk/deeprealm/kanad
python -m kanad.atlas.stats > /tmp/atlas-pre-p3-stats.md
cat /tmp/atlas-pre-p3-stats.md
```

Identify all clusters with `delta_pp < -5` (under-represented).

- [ ] **Step 2: For each under-represented cluster, create a working file**

```bash
# Per cluster:
touch kanad/atlas/sourcing/cluster_fill/<cluster>.md
```

Each file has the template:

```markdown
# Cluster fill: <cluster-name>

**Current:** N entries (X% actual vs Y% target)
**Target gap:** Z entries to add

## Candidate questions to curate

1. [Question 1 placeholder — research as needed]
2. ...

## Expert sources to mine

- [Source 1 — link or DOI]
- ...
```

Pre-populate `Expert sources to mine` from the spec §5.3:
- `hardware-paper-authors`: IBM Quantum benchmark papers (Kandala 2017, Nature 549, 242), IonQ application notes, BlueQubit case studies
- `industrial-materials-rd`: Materials Project case studies, polymorph energetics benchmarks
- `materials-small-cell`: Hubbard-U fitting collections, small-cell catalyst-screening data
- `industrial-pharma-rd`: ChEMBL examples, DrugBank reference compounds
- `photochemistry-pv`: TD-DFT vs experiment surveys (Adamo & Jacquemin), singlet-fission benchmark sets

- [ ] **Step 3: Commit cluster-fill scaffolding**

```bash
git add atlas/sourcing/cluster_fill/
git commit -m "feat(atlas): P3 cluster-fill scaffolding for under-represented clusters"
```

---

### Task 5.2: Hand-curate ~50 entries (curator work, ~25 hours)

- [ ] **Step 1: Per cluster, for each candidate question in its `cluster_fill/<cluster>.md`**

For each question:
1. Identify 1–2 published references (real DOIs only — verify via `doi_resolver`)
2. Copy verbatim excerpt from abstract
3. Fill out atlas YAML entry by hand using the schema in `SCHEMA.md`
4. Save to `atlas/entries/ATLAS-NNNN.yaml` with next sequential ID

- [ ] **Step 2: Validate each batch as you go**

```bash
python -m kanad.atlas.validators
```

- [ ] **Step 3: Commit per cluster**

```bash
git add atlas/entries/ATLAS-*.yaml
git commit -m "feat(atlas): P3 manual fill — <cluster> (~N entries)"
```

---

### Task 5.3: Final distribution check

- [ ] **Step 1: Regenerate everything**

```bash
python -m kanad.atlas.indexer
python -m kanad.atlas.ledger_generator
python -m kanad.atlas.stats > /tmp/atlas-final-stats.md
cat /tmp/atlas-final-stats.md
```

- [ ] **Step 2: Check Phase 5 done criterion**

All clusters must have `|delta_pp| ≤ 20% of their target` (per spec §9 done criteria — "within ±20% of Appendix C targets").

If any cluster still fails, do an additional fill pass.

- [ ] **Step 3: Commit final state**

```bash
git add atlas/index.json atlas/by-*/  atlas/STATS.md
git commit -m "feat(atlas): P3 complete — cluster distribution within ±20% of targets"
```

---

**Phase 5 complete when:**
- [ ] ~250 entries total in `atlas/entries/`
- [ ] `stats.py` reports every cluster within ±20% of Appendix C target
- [ ] All entries pass validator

---

# Phase 6 — Integration with PLAN.md + idea files

**Goal:** populate the fuel-ledger blocks in the existing Phase 2 idea files (M7–M14) and the PLAN.md summary. Smoke-test the CI workflow.

**Owner:** developer. **Estimated duration:** 2 days. **Depends on:** Phase 1 (tools must exist).

Note: this phase can be run in parallel with Phase 2/3 if you want to demonstrate the wiring earlier. But its final form lands after Phase 5 when atlas content is complete.

---

### Task 6.1: Add FUEL-LEDGER markers to idea files

**Files (modify):**
- `kanad/ideas/19-quantum-2rdm.md` (M7)
- `kanad/ideas/20-observables-plate.md` (M8)
- `kanad/ideas/21-validation-suite.md` (M9)
- `kanad/ideas/22-natural-orbital-active-space.md` (M10)
- `kanad/ideas/23-spin-coupling-J.md` (M12)
- `kanad/ideas/24-density-analysis.md` (M13)
- `kanad/ideas/25-exploration-workflows.md` (M14, workflows)
- `kanad/ideas/26-industrial-deployment.md` (M14, deployment)

- [ ] **Step 1: For each idea file, append the marker block at the end**

```markdown

<!-- FUEL-LEDGER: auto -->
(will be auto-generated by atlas/ledger_generator.py)
<!-- FUEL-LEDGER: /auto -->
```

- [ ] **Step 2: Run ledger generator**

```bash
cd /home/mk/deeprealm/kanad
python -m kanad.atlas.ledger_generator
```

- [ ] **Step 3: Inspect a generated block to confirm correctness**

```bash
grep -A 10 "FUEL-LEDGER: auto" ideas/19-quantum-2rdm.md
```

Expected: block contains "Atlas cells unlocked", novelty score, tier impact, cluster breakdown.

- [ ] **Step 4: Commit**

```bash
git add ideas/19-quantum-2rdm.md ideas/20-observables-plate.md ideas/21-validation-suite.md ideas/22-natural-orbital-active-space.md ideas/23-spin-coupling-J.md ideas/24-density-analysis.md ideas/25-exploration-workflows.md ideas/26-industrial-deployment.md
git commit -m "feat(atlas): inject fuel-ledger blocks into M7–M14 idea files"
```

---

### Task 6.2: Add FUEL-LEDGER-SUMMARY block to PLAN.md

- [ ] **Step 1: Append marker block to `kanad/PLAN.md`**

Just before the existing Appendix-type sections (or at the very end), add:

```markdown

<!-- FUEL-LEDGER-SUMMARY: auto -->
(auto-generated by atlas/ledger_generator.py)
<!-- FUEL-LEDGER-SUMMARY: /auto -->
```

- [ ] **Step 2: Run generator**

```bash
python -m kanad.atlas.ledger_generator
```

- [ ] **Step 3: Verify summary table populates**

```bash
grep -A 30 "FUEL-LEDGER-SUMMARY" PLAN.md
```

Expected: a table with rows for every milestone (M0–M19 + M9.5/M10.5/M11.5/M14.5/M14.9) showing cells unlocked, novelty score, tier impact.

- [ ] **Step 4: Commit**

```bash
git add PLAN.md
git commit -m "feat(atlas): inject fuel-ledger summary into PLAN.md"
```

---

### Task 6.3: End-to-end CI smoke test

- [ ] **Step 1: Verify all CI checks pass locally**

```bash
cd /home/mk/deeprealm/kanad
pytest atlas/tests/ -v
python -m kanad.atlas.validators
python -m kanad.atlas.indexer && git diff --exit-code atlas/index.json atlas/by-milestone/ atlas/by-cluster/ atlas/by-tier/
python -m kanad.atlas.ledger_generator && git diff --exit-code ideas/ PLAN.md atlas/by-milestone/
python -m kanad.atlas.stats
```

All commands should exit 0 with no diff.

- [ ] **Step 2: Push branch and verify CI passes on PR**

Open a PR to `main`. CI workflow `atlas-validate` should run and pass.

- [ ] **Step 3: Final commit / merge**

```bash
git commit --allow-empty -m "feat(atlas): v0 shipped — ~250 entries, full ledger integration"
```

---

**Phase 6 complete when:**
- [ ] All M7–M14 idea files contain populated FUEL-LEDGER blocks
- [ ] PLAN.md contains populated FUEL-LEDGER-SUMMARY block
- [ ] Local + remote CI all green
- [ ] At least one example entry per existing M* unlocks correctly when its milestone is flipped to `done` in MILESTONES.json (test by manual flip + indexer rerun)

---

# Overall done criteria (matches spec §9)

- [ ] `kanad/atlas/entries/` contains ≥200 valid YAML entries
- [ ] `atlas/stats.py` reports cluster distribution within ±20% of Appendix C targets
- [ ] All entries pass `atlas/validators/schema.py` (CI green)
- [ ] No entry has a fabricated DOI or excerpt (verification log shows zero rejections in committed entries)
- [ ] `atlas/index.json` regenerates deterministically and is committed
- [ ] `atlas/by-milestone/M7.md` through `M14.md` exist and render
- [ ] At least one M7–M14 idea file has a populated FUEL-LEDGER block
- [ ] `PLAN.md` has a populated FUEL-LEDGER-SUMMARY block
- [ ] CI workflow `atlas-validate` enforces schema + freshness
- [ ] `MILESTONES.json` exists with M0–M19 + the 5 new milestones
- [ ] Spec curator (owner) signs off on a final review PR

---

## Progress checkpoints

- **End of week 1:** Phase 1 complete + Phase 2 in progress. ~30 atlas entries (P1 partial).
- **End of week 2:** Phase 2 + 3 + 6 partial complete. ~100 atlas entries, lit-mining pipeline built, fuel-ledger blocks rendering.
- **End of week 3:** Phase 4 complete. ~220 atlas entries. Curator starts Phase 5.
- **End of week 4:** Phase 5 + 6 complete. ~250 entries, all CI green, distribution within tolerance.

Wall-clock can compress to ~3 weeks with developer + curator working in parallel (Phase 1 tooling unblocks parallel Phases 2/3/6, and Phase 4 review can interleave with Phase 5 hand-curation).
