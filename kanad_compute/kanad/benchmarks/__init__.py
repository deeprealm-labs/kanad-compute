"""Tier-1 benchmark suite for Kanad.

Eight closed-shell molecules at STO-3G, with active-space partitions chosen for
correctness vs FCI(full):

H₂, HeH⁺, LiH, BeH₂, H₂O, HF, NH₃, CH₄.

For each, we report:

- **PySCF HF / FCI(full)** — gold-standard references.
- **Kanad active-space CASCI** — direct ED of the active Hamiltonian in the
  correct N sector. This proves the chemistry (active-space integral
  transform + Hamiltonian construction) is bit-correct independent of the
  variational optimizer.
- **Kanad VQE** — `mode='standard'` + L-BFGS-B + parameter-shift gradient.
  This exposes which molecules the optimizer actually reaches chemical
  accuracy on.

Run: ``python -m benchmarks.run_benchmarks``. Results go to
``benchmarks/results.csv`` (per-row, appended) and ``benchmarks/results.md``
(regenerated, used as the README leader).
"""

from benchmarks.runner import (
    BenchmarkCase,
    BenchmarkResult,
    TIER1_CASES,
    run_case,
    run_all,
    write_results_csv,
    write_results_markdown,
)

__all__ = [
    'BenchmarkCase',
    'BenchmarkResult',
    'TIER1_CASES',
    'run_case',
    'run_all',
    'write_results_csv',
    'write_results_markdown',
]
