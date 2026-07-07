# Kanad benchmarks

Organised into execution lanes. Run from the repo root. On the GPU cluster use
`/root/miniconda3/bin/python` with `PYTHONPATH=/tmp/kanad-pkg`; cloud/QPU needs
backend tokens (see **Credentials**). Each script runs either as a module
(`python -m benchmarks.<lane>.<name>`) or by path (`python benchmarks/<lane>/<name>.py`).

```
benchmarks/
  run_benchmarks.py   # canonical Tier-1 entry: python -m benchmarks.run_benchmarks
  runner.py           # Tier-1 engine (HF / FCI / CASCI / VQE) + result writers
  local/   (21)  statevector truth + regression — laptop/CPU, no token
  scale/   ( 9)  GPU-cluster heavy (large statevector + large-CAS SQD)
  qpu/     (12)  cloud / real-QPU (BlueQubit / IonQ / IBM Heron)
  results/       generated leaderboards (results.csv, results.md, results_scale.md)
  _archive/( 9)  superseded single-molecule milestone one-offs (kept, runnable)
```

## Lane: `local/` — statevector truth + regression (no token)
- `tier_A_fci` / `tier_A_cbs_spec` / `tier_A_strong` — vs exact FCI / CBS+experiment / strong-correlation H-chains
- `tier_B_basis|nevpt2|saexc|select|frontier` — CASSCF / NEVPT2 / state-averaged excited frontier
- `tier_C_ham|props|solvers|probe|verify` — exotic Hamiltonians / properties / solver cross-validation
- `tier_W1|W1b|W2|W3|W4` — core-bug 4-wave regression
- `tier3_adme|props|spectra` — Round-3 ADME / properties / spectra

## Lane: `scale/` — GPU-cluster heavy (no token)
- `builder_sqd_crosscheck` — statevector-SQD vs spin-correct CASCI, 20–24q (reproduces N₂/C₂ anchors)
- `tier3_scale` / `tier4_scale` / `tier5_frontier` — qubit-count scale ladders
- `m10_cube_scale` / `m10_picker_scale` — .cube density/ESP (benzene 114 AO), AVAS picker (naphthalene 180 AO)
- `tier_hard` / `tier_hard2` / `tier_hard_dyn` — failure-probing: multireference, S–T gaps, reaction/dynamics

## Lane: `qpu/` — cloud / real hardware (needs tokens)
- `cloud_frontier` (BlueQubit) — 20→28q SQD vs exact FCI-in-CAS
- `cloud_sqd_demo` (BlueQubit + IonQ sim) — SamplingSQD cloud demo
- `ibm_sqd_demo` / `ibm_batch_sqd` (Givens-SD) / `ibm_lucj_batch` (LUCJ + recovery) — IBM Heron submit/poll
- `m11c_naphthalene_heron` — CAS(16,16)/32q champion (`--submit` / `--poll <job>`)
- `m11d_fe2s2_heron` — [Fe₂S₂Cl₄]²⁻ CAS(20,20)/40q champion (`--submit` / `--poll <job>`)
- `m5_heron_reactions_dynamics` — N₂ dissociation on real QPU
- `m8_naphthalene_excited` — 32q excited states (reuses m11c bitstrings)
- `tier_qpu` / `tier_qpu_max` — real-QPU SQD vs FCI; max-scale to 76–100q · `tier_A_qpu_fci`

## Credentials (qpu/ only)
Env vars (set in your shell; never commit). The code reads two naming
conventions per vendor — set both to cover the SQD and class/VQE paths:
- **BlueQubit**: `BLUEQUBIT_API_KEY` (+ `BLUE_TOKEN`)
- **IonQ**: `IONQ_API_KEY`  (SDK: `qiskit-ionq`)
- **IBM Heron (SQD)**: `IBM_QUANTUM_TOKEN` + `IBM_QUANTUM_CRN`
- **IBM (class/VQE)**: `IBM_API` (+ `IBM_CRN`)

> ⚠️ Security: `qpu/tier_qpu.py`, `qpu/tier_qpu_max.py`, `qpu/tier_A_qpu_fci.py`
> hardcode live tokens via `os.environ.setdefault(...)`. **Rotate them and read
> from the environment instead** — do not keep committed plaintext credentials.
