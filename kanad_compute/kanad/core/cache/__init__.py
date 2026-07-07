"""Parameter cache for Kanad solvers.

`ParameterCache` stores converged VQE parameters keyed by
`(molecule_geometry, ansatz_config, mapper_type, basis)`. A second VQE solve
on the same system warm-starts from the cached θ* instead of random init —
roughly 10× faster than cold-start.

For geometry scans (bond-length sweep, IRC, geometry optimization), the
`find_similar()` method looks up a cached θ* from a nearby geometry, giving
the same warm-start benefit across the scan.

The cache also functions as the seed corpus for any future ML-augmented
project (`ideas/20-future-kanad-ml-system.md`): every converged solve writes
a structured record of `(geometry, ansatz, mapper, basis, θ*, energy,
n_iter, walltime, init_strategy, telemetry)` to disk.

Storage: SQLite at `~/.cache/kanad/parameters.sqlite` by default. Override
with the `KANAD_CACHE_DIR` env var. Disable per-call with `use_cache=False`.
"""

from kanad.core.cache.parameter_cache import (
    CacheKey,
    CachedRun,
    ParameterCache,
    build_cache_key,
    get_default_cache,
)

__all__ = [
    'CacheKey',
    'CachedRun',
    'ParameterCache',
    'build_cache_key',
    'get_default_cache',
]
