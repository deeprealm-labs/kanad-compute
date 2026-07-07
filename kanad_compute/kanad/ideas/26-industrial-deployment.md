# Idea 26 — Industrial deployment readiness

**Lands in:** M8-M14 (cross-cutting through Phase 2)
**Effort:** ~400 lines of plumbing + documentation + kanad-app coordination, 2–3 weeks
**Blocks:** the framework being usable in industrial R&D pipelines
**Prerequisites:** M0 (truth pass) + M8 (observables plate) + M9 (validation suite) + idea 25 (workflows)

---

## Problem

Academic users are happy to write Python in Jupyter notebooks, pip-install dependencies, and read tracebacks. **Industrial users are not.** A pharma R&D scientist running a 100-fragment screen wants:

- **Input from existing pipelines.** SMILES strings, SDF files, JSON manifests. Not "write a Python function."
- **Output to existing pipelines.** Pandas DataFrames, CSV/SDF/JSON, S3/Snowflake/internal-data-lake exports. Not "look at the Jupyter cell output."
- **Throughput.** Batch screens of 50-1000 molecules. Parallel execution. Progress tracking. Resume after interruption.
- **Reproducibility under audit.** Every result tagged with framework version, parameter set, git commit, run timestamp. For FDA/EMA submissions or internal IP records.
- **On-premise deployment.** Pharma can't send proprietary structures to cloud. Same for materials/catalyst IP. The framework must run inside the customer's firewall.
- **No nondeterministic surprises.** A re-run on the same input should produce the same output (up to optimizer stochasticity, which itself is seedable).
- **Documented compute cost.** "This batch of 100 fragments will take ~6 hours on 8 cores" — predictable cost, not "depends."
- **Vendor support.** When something breaks, someone responds. (That's `kanad-app`'s commercial role.)

Today Kanad satisfies maybe 2 of those 7. This idea closes the gap.

---

## What it ships

Six pieces of plumbing that make Kanad industrial-deployable. None are research-physics; all are production-engineering.

### IND1 — SMILES/SDF/PDB input pipeline

```python
from kanad.io import from_smiles, from_sdf, from_pdb

mol = from_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
mol = from_sdf("compound_library.sdf", index=5)
mol = from_pdb("protein.pdb", chain='A', residue_range=(100, 120))  # fragment of protein
```

Backed by RDKit (already a transitive dependency). 3D coordinates generated via ETKDG embedding; optional MMFF94 pre-optimization.

**Why:** industrial libraries are SMILES/SDF, not XYZ. PDB for biology. This is the input contract.

### IND2 — Batch / parallel runner

```python
from kanad.batch import run_batch

results = run_batch(
    inputs="my_library.sdf",        # or list of SMILES, or list of XYZ paths
    workflow="screen_fragments",     # the idea-25 workflow
    workflow_params={'properties': ['homo', 'lumo', 'dipole']},
    n_workers=8,
    resume_from="last_checkpoint.json",  # optional
    output="results.parquet",
)
```

- Parallel execution via `multiprocessing` (or `concurrent.futures`).
- Progress logged with ETA.
- Checkpointing every N molecules → resumable after crash.
- Output in Parquet (large libraries) or CSV (small).

**Why:** industrial users run hundreds of molecules; the cost is bound by parallelism + checkpointing.

### IND3 — Audit log + provenance

Every result carries a provenance dict:

```json
{
  "framework_version": "0.3.1",
  "git_commit": "a4cb066",
  "timestamp_utc": "2026-05-13T14:32:01Z",
  "workflow": "screen_fragments",
  "workflow_params": {...},
  "hardware": {"cpu_model": "Xeon Gold 6248", "n_cores_used": 8, "ram_gb": 64},
  "input_hash_sha256": "...",
  "output_hash_sha256": "...",
  "random_seed": 12345
}
```

Stored alongside every result file (`.kanad-provenance.json`). For audit, IP records, regulatory submissions.

**Why:** FDA / EMA / IP-team requirements; reproducibility under scrutiny.

### IND4 — Deterministic mode

```python
solver = VQESolver(molecule=mol, deterministic=True, random_seed=42)
```

When `deterministic=True`, *every* source of nondeterminism is seeded: optimizer initial parameters, sampling shots (if applicable), random shuffles in MP2/CC ranking, etc. Re-runs with same seed produce bit-identical output.

**Why:** regulated industries require this for validation studies.

### IND5 — Resource estimator (cost predictability)

```python
from kanad.estimate import estimate_workflow_cost

estimate = estimate_workflow_cost(
    workflow="screen_fragments",
    inputs="my_library.sdf",        # 200 SMILES
    hardware="laptop_8core",         # or "workstation_32core", "node_gpu"
)
print(estimate.eta_hours)            # 4.2
print(estimate.peak_memory_gb)       # 12
print(estimate.bottleneck)           # "VQE per molecule (avg 75 sec)"
```

Looks up empirical per-molecule cost from a calibration database (filled in by CI + reported user data). Says "this batch will take ~4 hours; max RAM ~12 GB."

**Why:** industrial planning. R&D directors need to budget time and hardware.

### IND6 — On-premise deployment package

```bash
# Single bundled binary install
curl -L https://kanad.deeprealm.io/install.sh | sh

# Or docker image
docker run -v $(pwd)/library.sdf:/data/library.sdf \
  kanad/kanad:v0.3.1 \
  run-batch --input /data/library.sdf --workflow screen_fragments
```

Bundled distribution: Python + PySCF + Kanad + RDKit + (Bader.x for density analysis) + (Qiskit if hardware path needed). Single docker image / single-script installer. No internet during operation.

**Why:** pharma firewalls; IP-sensitive customers; air-gapped HPC environments.

---

## Solution

```
kanad/io/
├── from_smiles.py
├── from_sdf.py
├── from_pdb.py
└── geometry.py                # ETKDG / MMFF94 embedding wrappers

kanad/batch/
├── runner.py
├── checkpoint.py
└── parallel.py

kanad/provenance/
├── audit_log.py
└── provenance_schema.py       # JSON schema for the provenance dict

kanad/estimate/
├── cost_model.py
├── calibration_db.json        # filled by CI runs + user-reported data
└── estimator.py

deployment/
├── Dockerfile
├── install.sh
└── README.md
```

Each component is small (~50-150 LOC) and well-tested. The hard work is making the components actually compose: a batch runner that produces audit-logged provenance + checkpoints + estimates ETA + reads SMILES + outputs Parquet.

---

## Done criteria

- **IND1:** aspirin from SMILES `CC(=O)Oc1ccccc1C(=O)O` builds a 3D-embedded molecule + runs VQE on its fragment + produces observable plate.
- **IND2:** 50-fragment SDF library runs through `screen_fragments` workflow in <30 minutes on 8 cores. Resumes correctly after kill -9.
- **IND3:** every result file has a sibling `.kanad-provenance.json`; provenance includes git commit + version + hashes + seed.
- **IND4:** deterministic mode with same seed produces bit-identical observable plate over 10 re-runs.
- **IND5:** estimator predicts batch ETA within ±20% on a 20-molecule sample (validated on real data).
- **IND6:** docker image runs offline; offline-installer script works on a fresh Ubuntu 22.04.

## Dependencies

- M8 (observables plate) — workflows operate on its outputs.
- M9 (validation suite) — calibrates the cost estimator.
- Idea 25 (workflows) — the things that get batched.
- Coordination with kanad-app for deployment story.

## Test files to add

- `tests/integration/test_io_smiles.py`
- `tests/integration/test_io_sdf.py`
- `tests/integration/test_batch_runner.py`
- `tests/integration/test_batch_resume.py`
- `tests/integration/test_provenance_audit.py`
- `tests/integration/test_deterministic_mode.py`
- `tests/integration/test_cost_estimator.py`
- `tests/integration/test_docker_build.sh`

## Notes for the executor

- **Industrial-grade ≠ enterprise-bloat.** No SAP integration, no SAML SSO, no Snowflake-of-the-month support. Just the 6 items above; each is needed for real industrial usability; none is needed for academic.
- **The validation suite (idea 21) is the trust contract.** Industrial buyers need to see that the framework produces correct answers on a documented benchmark before they deploy it. Coordinate: the validation suite outputs are visible to industrial evaluators (kanad-app gallery).
- **Cost estimator is the most-asked-for feature in industrial settings.** Calibrate it as soon as M9 lands by running the validation suite on a few standard hardware configs and recording the timings.
- **On-premise deployment is the most-asked-for *requirement*.** No proprietary structure leaves the customer's firewall. Build the docker image early so it can be tested by industrial pilot users.
- **kanad-app provides the commercial face.** The framework's job is to be runnable + correct + auditable; kanad-app's job is to wrap that in a UI + a billing model + support tickets. Coordinate via idea 17.
- **Don't pre-build the IP-safe story without a customer.** If/when an industrial pilot starts, the IP-safety implementation surfaces as a real requirement. Build to that requirement; don't speculate.
- **Determinism is hard to retrofit.** Wire `random_seed` plumbing through every solver / ansatz / mapper now (lots of small changes), not later.
- **Audit logs feed into the differentiation story.** When a regulator asks "how do we trust this number?", the answer is "here's the git commit, the input, the parameters, the framework version, the timestamp, and the validation-suite result for the molecule class." That's how science gets used in regulated industries.
