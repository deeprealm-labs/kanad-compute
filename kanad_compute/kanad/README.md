# Kanad — quantum-correlated chemistry framework

**Version:** 0.1.2 · **Maintainer:** DeepRealm Labs

Kanad is a **wavefunction-first quantum-chemistry framework** for small-to-medium molecules. Hand it a
molecule and it gives back a genuinely *correlated* answer — multireference energies, excited-state
spectra, and wavefunction-derived properties — on a laptop, with the option to push sampling to real
quantum hardware. The engine lives here; the user-facing product is
[kanad-app](https://github.com/deeprealm-labs/kanad-app), which consumes this framework directly.

---

## What's new (2026-05-30) — foundation remediated & validated at scale

The most recent body of work was a deep correctness pass on the core and a large-scale validation of it:

- **Core audit + full remediation.** A foundation audit found **24 confirmed correctness bugs** (active-space
  embedding, sample-based diagonalization, fermion↔qubit mappers, lattice Hamiltonians, condition handling).
  **All 24 are fixed** across four waves, each with a regression that exercises the previously-untested path.
  Full ledger: [`CORE_BUGS.md`](CORE_BUGS.md).
- **Validated on cluster + cloud + real QPU.** A rigorous diverse re-test confirmed the repaired core:
  SQD reaches chemical accuracy **20→28 qubits** on cloud statevector and **+0.17 mHa of exact FCI on a real
  24-qubit IBM Heron**; benzene AVAS == PySCF rotated-CASCI exactly; N₂ SQD == exact FCI even at strong
  stretch. Full report: [`RETEST_RESULTS.md`](RETEST_RESULTS.md).
- **Multireference accuracy layer.** Excited states via **SA-CASSCF + per-root NEVPT2** with character-based
  state selection (bright / valence / Rydberg, using a natural-orbital ⟨r²⟩ diagnostic); ground-state
  **CASSCF orbital optimization** and **NEVPT2** correction integrated into the builder.
- **Honest, demonstrated usefulness.** Genuine wins where single-reference classical methods fail: N₂
  dissociation (CCSD(T) collapses), butadiene 2¹Ag dark double-excitation (TD-DFT cannot represent it),
  cyclobutadiene barrier CASCI(8,8) ≈ TBE 8.9 kcal/mol (beats CCSD(T)).
- **Limits surfaced, not hidden.** Where the method has limits they are guarded and reported — e.g. NEVPT2
  on a strongly diradical reference now emits a warning and surfaces the robust bare-CASCI fallback.

---

## Overview — what Kanad computes

| Capability | How |
|---|---|
| **Ground-state energy** | exact FCI-in-active-space (CI), sample-based quantum diagonalization (SQD), or VQE |
| **Multireference correction** | CASSCF orbital optimization + strongly-contracted NEVPT2 |
| **Excited states / spectra** | SA-CASSCF+NEVPT2 with oscillator strengths, transition dipoles, character tagging |
| **Active spaces** | `full`, `frozen_core`, `frontier`, `mp2no` (MP2 natural orbitals), `avas` (atom-projected), `manual` |
| **Properties (from the quantum 1-RDM)** | dipole, natural-orbital occupations, M-diagnostic, Mulliken charges, HOMO-LUMO gap, density/ESP cube export |
| **Fermion↔qubit mappers** | Jordan-Wigner, Bravyi-Kitaev (verified isospectral) |
| **Backends** | statevector (local), BlueQubit cloud, IBM Quantum (real hardware) |
| **Other engines** | lattice/Hubbard & periodic Hamiltonians, dynamics (BOMD, NAMD, open-quantum), reaction scans |

For small-to-medium molecules (≈≤30 qubits / ≤15 active orbitals), benchmarked against FCI, CCSD(T),
CBS extrapolation, and experiment — **not** against weaker references.

---

## Quick start

```python
from kanad import MolecularBuilder

# 1) Ground-state energy — exact FCI in a chosen active space
qs = (MolecularBuilder.from_atoms([('N', (0, 0, 0)), ('N', (0, 0, 1.10))])
      .basis('cc-pvdz').active_space('frontier', n_occ=4, n_virt=4)
      .solver('ci').build())
print(qs.solve()['energy'])

# 2) Multireference — CASSCF orbital optimization + NEVPT2
qs = (MolecularBuilder.from_atoms([('O', (0, 0, 0)), ('O', (1.089, 0.681, 0)), ('O', (-1.089, 0.681, 0))])
      .basis('cc-pvdz').active_space('frontier', n_occ=6, n_virt=6)
      .solver('ci', orbital_optimization=True, pt2_correction='nevpt2').build())
print(qs.solve()['energy'])

# 3) Excited states — SA-CASSCF+NEVPT2, keep the bright valence states
spec = qs.excited_states(n_states=4, orbital_optimization=True,
                         pt2_correction='nevpt2', select='bright')
print(spec['excitation_energies_ev'], spec['oscillator_strengths'])

# 4) Sample-based quantum diagonalization on real hardware
qs = (MolecularBuilder.from_atoms([('N', (0, 0, 0)), ('N', (0, 0, 1.10))])
      .basis('cc-pvdz').active_space('manual', frozen=[0, 1], active=list(range(2, 14)))
      .solver('sqd', backend='ibm', n_samples=12000).build())
print(qs.solve()['energy'])

# 5) Wavefunction-derived observables
print(qs.observables('core'))   # dipole, NOONs, M-diagnostic, Mulliken charges
```

The fluent `MolecularBuilder` is the single programmatic entry point: `from_atoms(...)` → `.basis(...)`
→ `.active_space(...)` → `.solver(...)` → `.backend(...)` → `.build()` → `.solve()` / `.excited_states()`
/ `.observables()`.

---

## Validation

**Tier-1 (STO-3G, frozen-core where applicable) — 8/8 at chemical accuracy.** Active-space Hamiltonian +
the particle-conserving Givens-SD (UCCSD-like) ansatz both reach FCI:

| Molecule | n_qb | (e,o) | CASCI Δ vs FCI | VQE Δ vs FCI |
|---|---:|---|---:|---:|
| H₂ | 4 | (2,2) | +0.000 | +0.000 |
| LiH | 10 | (2,5) | +0.228 | +0.228 |
| H₂O | 12 | (8,6) | +0.078 | +0.196 |
| BeH₂ | 12 | (4,6) | +0.340 | +0.720 |
| NH₃ | 14 | (8,7) | +0.187 | +0.388 |
| CH₄ | 16 | (8,8) | +0.418 | +0.617 |

(Δ in mHa; HeH⁺ and HF also pass.) **Scale & hardware** (re-test, [`RETEST_RESULTS.md`](RETEST_RESULTS.md)):
SQD == exact FCI-in-CAS within chemical accuracy at 20q / 24q / 28q on cloud statevector, and +0.17 mHa on a
real 24-qubit IBM Heron; benzene π via AVAS matches PySCF rotated-CASCI exactly. The full diverse re-test
matrix (51 experiments, 5 campaigns) is in [`RETEST_MATRIX.json`](RETEST_MATRIX.json); the deep-audit trail
is in [`HARD_AUDIT.md`](HARD_AUDIT.md).

---

## Honest framing

- **SQD is a faithful reimplementation** of the published sample-based quantum-diagonalization method — its
  value is faithful circuit sampling **plus classical configuration recovery**, with recovery carrying the
  load against hardware noise (on noise-free statevector the sample alone is already ~0.1 mHa; on real Heron
  the raw sample is ~4 mHa and recovery is essential). It is not a novel quantum-sample advantage.
- **VQE is small-scale only.** The particle-conserving Givens-SD ansatz reaches FCI; the generic
  hardware-efficient ansatz is *not* chemistry-grade beyond ~4 qubits and is kept for validation only.
- **Method limits are explicit.** Single-state NEVPT2 is unbalanced for strongly diradical references
  (guarded with a warning + bare-CASCI fallback); the interacting Hubbard model lives in the fermionic/qubit
  path, not the single-particle matrix.

Where Kanad aims to be useful is the **workflow ecosystem on a correct multireference core** — spectra,
forces, reaction energetics, and strong-correlation problems where DFT / CCSD(T) / TD-DFT break down.

---

## Repository layout

| Path | What it is |
|---|---|
| `builder/` | `MolecularBuilder` fluent facade + `QuantumSystem` (solve / excited_states / observables / energy_fn) |
| `core/` | Hamiltonians (molecular, ionic, metallic, periodic), active-space selection + integral transform, mappers, integrals, density |
| `solvers/` | CI, **SamplingSQD**, VQE, CASSCF/NEVPT2 routing, SCF, subspace diagonalizers |
| `ansatze/` | Givens-SD (UCCSD-like), hardware-efficient, LUCJ (for SQD), Givens |
| `analysis/` | property / spectroscopy / thermochemistry / vibrational / bonding calculators |
| `dynamics/` | BOMD, nonadiabatic (NAMD), photodynamics, open-quantum (Lindblad) |
| `bonds/`, `reactions/`, `backends/`, `environment/`, `governance/`, `io/`, `cache/`, `optimization/` | supporting subsystems |
| `tests/`, `benchmarks/` | regression suite (validation / unit / probes) + benchmark & re-test harnesses |
| [`CORE_BUGS.md`](CORE_BUGS.md), [`RETEST_RESULTS.md`](RETEST_RESULTS.md), [`HARD_AUDIT.md`](HARD_AUDIT.md) | remediation ledger, scale re-test, and audit trail |

---

## Distribution & license

- **No PyPI publication.** End users reach the framework through [kanad-app](https://github.com/deeprealm-labs/kanad-app)
  (API routes + workers), not `pip install kanad`. The framework is consumed via submodule / CI-vendoring.
- **License:** see `LICENSE` when added; until then, internal-research-use-only.
