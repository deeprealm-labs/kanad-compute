"""CLI entry point for the Tier-1 benchmark.

Usage:
    python -m benchmarks.run_benchmarks

Outputs:
    benchmarks/results.csv     — one row per case (machine-readable).
    benchmarks/results.md      — Markdown table; this is the README leader.

Logs go to stderr.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

# Suppress .pyc generation for this process. Iterative benchmark development
# was repeatedly catching stale bytecode (__pycache__ shadowing edits made
# moments before). The runtime cost is negligible vs the VQE solves below.
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True


# Same trick as tests/conftest.py: ensure `import kanad` resolves to *this*
# framework repo, not any pip-installed sibling. The benchmark must run
# against the M1/M2 framework code, not the older kanad-app version.
_FRAMEWORK_ROOT = Path(__file__).resolve().parent.parent
_tmp_pkg_dir = Path(tempfile.gettempdir()) / "kanad-fw-bench-pkg"
_tmp_pkg_dir.mkdir(exist_ok=True)
_kanad_link = _tmp_pkg_dir / "kanad"
if _kanad_link.exists() and _kanad_link.is_symlink():
    _kanad_link.unlink()
if not _kanad_link.exists():
    _kanad_link.symlink_to(_FRAMEWORK_ROOT)
for _cached in list(sys.modules):
    if _cached == "kanad" or _cached.startswith("kanad."):
        del sys.modules[_cached]
if str(_tmp_pkg_dir) in sys.path:
    sys.path.remove(str(_tmp_pkg_dir))
sys.path.insert(0, str(_tmp_pkg_dir))


def main():
    parser = argparse.ArgumentParser(description='Run the Tier-1 Kanad benchmark.')
    parser.add_argument('--cases', nargs='*', default=None,
                        help='Optional subset of case names (e.g. H2 HeH+ LiH). Default: all.')
    parser.add_argument('--no-vqe', action='store_true',
                        help='Skip the VQE solve, only report HF/FCI/CASCI.')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).parent / 'results',
                        help='Directory to write results.csv + results.md (default: benchmarks/results/).')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip cases already present in results.csv; useful '
                             'for resuming after a timeout-killed run.')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(message)s',
        stream=sys.stderr,
    )

    from benchmarks.runner import (
        TIER1_CASES, run_all, write_results_csv, write_results_markdown,
    )

    if args.cases:
        cases = [c for c in TIER1_CASES if c.name in args.cases]
        if not cases:
            print(f"No cases match {args.cases}; available: "
                  f"{[c.name for c in TIER1_CASES]}", file=sys.stderr)
            return 1
    else:
        cases = list(TIER1_CASES)

    # Resume support: load prior results.csv (if any) and skip cases that
    # already have a row. This lets multi-batch runs survive timeout-kills.
    prior_results = []
    if args.resume:
        csv_path = args.out_dir / 'results.csv'
        if csv_path.exists():
            import csv as _csv
            from dataclasses import fields
            with open(csv_path) as f:
                reader = _csv.DictReader(f)
                from benchmarks.runner import BenchmarkResult
                field_names = {f.name for f in fields(BenchmarkResult)}
                for row in reader:
                    coerced = {}
                    for k, v in row.items():
                        if k not in field_names:
                            continue
                        if v == '' or v is None:
                            coerced[k] = None
                        elif v == 'True':
                            coerced[k] = True
                        elif v == 'False':
                            coerced[k] = False
                        else:
                            for cast in (int, float, str):
                                try:
                                    coerced[k] = cast(v)
                                    break
                                except (ValueError, TypeError):
                                    continue
                    prior_results.append(BenchmarkResult(**coerced))
            done_names = {r.name for r in prior_results}
            print(f"Resume: skipping {sorted(done_names)} already in CSV", file=sys.stderr)
            cases = [c for c in cases if c.name not in done_names]

    print(f"Running {len(cases)} cases: {[c.name for c in cases]}", file=sys.stderr)
    # Run case-by-case, writing the COMBINED (prior + so-far) results after
    # each so partial progress survives interruption.
    from benchmarks.runner import run_case
    order = {c.name: i for i, c in enumerate(TIER1_CASES)}
    new_results = []
    for case in cases:
        print(f"[bench] {case.name}...", file=sys.stderr, flush=True)
        result = run_case(case, run_vqe=not args.no_vqe)
        new_results.append(result)
        print(
            f"  HF={result.e_hf:.6f}, FCI={result.e_fci_full:.6f}, "
            f"CASCI={result.e_casci_active}, VQE={result.e_vqe_standard}",
            file=sys.stderr, flush=True,
        )
        # Write combined so far
        combined_partial = prior_results + new_results
        combined_partial.sort(key=lambda r: order.get(r.name, 999))
        try:
            write_results_csv(combined_partial, args.out_dir / 'results.csv')
            write_results_markdown(combined_partial, args.out_dir / 'results.md')
        except Exception as exc:
            print(f"[warn] incremental write failed: {exc}", file=sys.stderr)

    # Final write with everything sorted in TIER1 order
    results = prior_results + new_results
    results.sort(key=lambda r: order.get(r.name, 999))
    write_results_csv(results, args.out_dir / 'results.csv')
    write_results_markdown(results, args.out_dir / 'results.md')

    md_path = args.out_dir / 'results.md'
    print(md_path.read_text())
    return 0


if __name__ == '__main__':
    sys.exit(main())
