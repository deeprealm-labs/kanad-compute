"""SQLite-backed parameter cache for VQE warm-starts.

Schema and access patterns:

- Single SQLite file at `~/.cache/kanad/parameters.sqlite` (override via
  `KANAD_CACHE_DIR` env var).
- One row per converged VQE solve. Key is a hash of the system
  configuration; `find_similar()` walks rows for a given atomic species +
  ansatz config and picks the closest geometry by RMSD.
- `ParameterCache` is process-safe via SQLite's default WAL mode for
  concurrent readers; multiple writers serialize naturally.

This module is a feature of M2 PR-3. It is also the seed corpus for any
future ML-augmented project — see `ideas/20-future-kanad-ml-system.md`.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --- Versioning -------------------------------------------------------------

# Bump on schema changes. Old cache entries below this version are ignored.
CACHE_SCHEMA_VERSION = 1


# --- Keys + records ---------------------------------------------------------

@dataclass(frozen=True)
class CacheKey:
    """Composite key for a cached VQE solve.

    The hash is the SHA-256 of all fields concatenated; cache rows are
    indexed by this hash so lookup is O(1).
    """

    geometry_hash: str   # hash of (sorted atoms + rounded coords)
    ansatz_class: str    # e.g. 'HardwareEfficientAnsatz'
    ansatz_config: str   # serialized config (n_layers, entanglement, rotations)
    mapper_type: str     # e.g. 'jordan_wigner'
    basis: str           # e.g. 'sto-3g'
    # charge + spin must be keyed: distinct electronic states (e.g. singlet vs
    # triplet, or differently-charged species) of the same geometry/ansatz/mapper
    # otherwise collide on the same composite_hash.
    charge: int = 0      # total molecular charge
    spin: int = 0        # spin multiplicity surrogate (2S, i.e. n_alpha - n_beta)
    schema_version: int = CACHE_SCHEMA_VERSION

    @property
    def composite_hash(self) -> str:
        blob = '|'.join((
            self.geometry_hash,
            self.ansatz_class,
            self.ansatz_config,
            self.mapper_type,
            self.basis,
            str(self.charge),
            str(self.spin),
            str(self.schema_version),
        ))
        return hashlib.sha256(blob.encode('utf-8')).hexdigest()


@dataclass
class CachedRun:
    """A single cached VQE run.

    `parameters` is the converged θ*; downstream solvers warm-start from this
    (with optional small jitter). The other fields are telemetry — kept so
    future analysis or an ML system can train on the corpus.
    """

    parameters: np.ndarray
    final_energy: float
    n_iterations: int
    walltime_seconds: float
    init_strategy: str  # 'cached' | 'mp2' | 'random' | 'user'
    framework_version: str
    final_n_variance: Optional[float] = None
    final_s2_variance: Optional[float] = None
    final_gradient_norm: Optional[float] = None
    atom_symbols: Tuple[str, ...] = field(default_factory=tuple)
    atom_coords: Optional[np.ndarray] = None  # (n_atoms, 3) in Angstrom
    created_at: datetime = field(default_factory=datetime.utcnow)


# --- Helpers ---------------------------------------------------------------

def _hash_geometry(atom_symbols: Sequence[str], coords: np.ndarray, precision: int = 4) -> str:
    """SHA-256 of (sorted-symbol, rounded-coord) tuples.

    Rounding to 4 decimal places (0.0001 Å) means two geometries within
    sub-µÅ are treated as identical, which is the right resolution for
    cache reuse (chemical accuracy doesn't care about 1e-5 Å differences).
    """
    if len(atom_symbols) != coords.shape[0]:
        raise ValueError(
            f"atom count mismatch: {len(atom_symbols)} symbols vs {coords.shape[0]} coord rows"
        )
    # Sort atoms by (symbol, x, y, z) for canonical ordering — handles
    # cases where the user constructs the same molecule with different
    # atom orderings.
    items = sorted(zip(atom_symbols, coords.tolist()),
                   key=lambda item: (item[0], *item[1]))
    canonical = '|'.join(
        f"{sym}:{round(x, precision)},{round(y, precision)},{round(z, precision)}"
        for sym, (x, y, z) in items
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _canonical_order(
    symbols: Sequence[str], coords: Optional[np.ndarray]
) -> Tuple[Tuple[str, ...], Optional[np.ndarray]]:
    """Reorder atoms by the SAME (symbol, x, y, z) sort `_hash_geometry` uses.

    Makes ``find_similar``'s symbol-string filter and element-wise RMSD
    order-invariant — mirroring the canonical ordering that ``get()`` already
    gets for free via the geometry hash.
    """
    if coords is None:
        return tuple(symbols), coords
    items = sorted(zip(symbols, np.asarray(coords).tolist()),
                   key=lambda it: (it[0], *it[1]))
    syms = tuple(i[0] for i in items)
    crds = np.array([i[1] for i in items], dtype=float)
    return syms, crds


def _np_to_blob(arr: np.ndarray) -> bytes:
    if arr is None:
        return b''
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _blob_to_np(blob: bytes) -> Optional[np.ndarray]:
    if not blob:
        return None
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _serialize_ansatz_config(ansatz) -> str:
    """Canonical config string for an ansatz instance.

    For `HardwareEfficientAnsatz`: includes n_layers, entanglement,
    rotation_gates, n_qubits. For other ansatze: the relevant kwargs.
    Falls back to the class name if introspection fails.
    """
    if ansatz is None:
        return ''
    config = {'cls': type(ansatz).__name__}
    for attr in ('n_layers', 'entanglement', 'rotation_gates', 'n_qubits',
                 'n_electrons'):
        if hasattr(ansatz, attr):
            val = getattr(ansatz, attr)
            if isinstance(val, (list, tuple)):
                val = list(val)
            config[attr] = val
    return json.dumps(config, sort_keys=True)


# --- The cache --------------------------------------------------------------

class ParameterCache:
    """SQLite-backed parameter cache.

    All public methods are safe to call concurrently from multiple processes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = self._default_db_path()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ----- setup --------------------------------------------------------

    @staticmethod
    def _default_db_path() -> Path:
        env = os.environ.get('KANAD_CACHE_DIR')
        base = Path(env).expanduser() if env else Path.home() / '.cache' / 'kanad'
        return base / 'parameters.sqlite'

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    composite_hash TEXT PRIMARY KEY,
                    geometry_hash TEXT NOT NULL,
                    ansatz_class TEXT NOT NULL,
                    ansatz_config TEXT NOT NULL,
                    mapper_type TEXT NOT NULL,
                    basis TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    parameters BLOB NOT NULL,
                    final_energy REAL NOT NULL,
                    n_iterations INTEGER NOT NULL,
                    walltime_seconds REAL NOT NULL,
                    init_strategy TEXT NOT NULL,
                    framework_version TEXT NOT NULL,
                    final_n_variance REAL,
                    final_s2_variance REAL,
                    final_gradient_norm REAL,
                    atom_symbols TEXT,
                    atom_coords BLOB,
                    created_at TEXT NOT NULL
                )
            """)
            # Index for find_similar() lookups: same atoms + ansatz config +
            # mapper + basis, then check geometry distance in Python.
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_similar
                ON runs (atom_symbols, ansatz_class, ansatz_config, mapper_type, basis)
            """)

    # ----- public API ---------------------------------------------------

    def get(self, key: CacheKey) -> Optional[CachedRun]:
        """Exact-match lookup. Returns ``None`` if no row matches."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE composite_hash = ?",
                (key.composite_hash,),
            ).fetchone()
        return self._row_to_run(row) if row else None

    def put(self, key: CacheKey, run: CachedRun) -> None:
        """Insert or replace the cached run for this key."""
        # Canonicalize atom order so find_similar()'s symbol-string filter and
        # element-wise RMSD are order-invariant (matching get()'s behavior).
        c_syms, c_coords = _canonical_order(run.atom_symbols, run.atom_coords)
        atom_symbols_str = ','.join(c_syms)
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    composite_hash, geometry_hash, ansatz_class, ansatz_config,
                    mapper_type, basis, schema_version,
                    parameters, final_energy, n_iterations, walltime_seconds,
                    init_strategy, framework_version,
                    final_n_variance, final_s2_variance, final_gradient_norm,
                    atom_symbols, atom_coords, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key.composite_hash,
                key.geometry_hash,
                key.ansatz_class,
                key.ansatz_config,
                key.mapper_type,
                key.basis,
                key.schema_version,
                _np_to_blob(np.asarray(run.parameters)),
                float(run.final_energy),
                int(run.n_iterations),
                float(run.walltime_seconds),
                run.init_strategy,
                run.framework_version,
                run.final_n_variance,
                run.final_s2_variance,
                run.final_gradient_norm,
                atom_symbols_str,
                _np_to_blob(c_coords),
                run.created_at.isoformat(),
            ))

    def find_similar(
        self,
        key: CacheKey,
        atom_symbols: Sequence[str],
        atom_coords: np.ndarray,
        max_rmsd_angstrom: float = 0.1,
    ) -> Optional[CachedRun]:
        """Find a cached run with the same chemistry but nearby geometry.

        Useful for geometry scans (bond-length sweeps, IRC, optimization):
        the second geometry can warm-start from the first's θ* without
        needing an exact hash match.

        Returns the run with the smallest RMSD ≤ ``max_rmsd_angstrom``, or
        ``None`` if no such run exists.
        """
        # Canonicalize the query the same way put() canonicalizes stored runs,
        # so the symbol-string filter matches and RMSD compares aligned atoms.
        q_syms, target = _canonical_order(atom_symbols, atom_coords)
        atom_symbols_str = ','.join(q_syms)
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM runs
                WHERE atom_symbols = ?
                  AND ansatz_class = ?
                  AND ansatz_config = ?
                  AND mapper_type = ?
                  AND basis = ?
                  AND schema_version = ?
            """, (
                atom_symbols_str,
                key.ansatz_class,
                key.ansatz_config,
                key.mapper_type,
                key.basis,
                key.schema_version,
            )).fetchall()

        best_run = None
        best_rmsd = max_rmsd_angstrom
        for row in rows:
            run = self._row_to_run(row)
            if run is None or run.atom_coords is None:
                continue
            if run.atom_coords.shape != target.shape:
                continue
            rmsd = float(np.sqrt(np.mean((run.atom_coords - target) ** 2)))
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_run = run
        return best_run

    def clear(self) -> None:
        """Drop all cached entries — useful for testing and benchmarking."""
        with self._connect() as conn:
            conn.execute("DELETE FROM runs")

    def __len__(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    # ----- internals ----------------------------------------------------

    @staticmethod
    def _row_to_run(row) -> Optional[CachedRun]:
        if row is None:
            return None
        # sqlite3 row indices follow our INSERT column order. We use indices
        # directly rather than enabling row_factory because we want plain tuples
        # in this helper for clarity.
        (
            _composite_hash, _geometry_hash, _ansatz_class, _ansatz_config,
            _mapper_type, _basis, _schema_version,
            parameters_blob, final_energy, n_iterations, walltime_seconds,
            init_strategy, framework_version,
            n_var, s2_var, grad_norm,
            atom_symbols_str, atom_coords_blob, created_at,
        ) = row
        return CachedRun(
            parameters=_blob_to_np(parameters_blob),
            final_energy=final_energy,
            n_iterations=n_iterations,
            walltime_seconds=walltime_seconds,
            init_strategy=init_strategy,
            framework_version=framework_version,
            final_n_variance=n_var,
            final_s2_variance=s2_var,
            final_gradient_norm=grad_norm,
            atom_symbols=tuple(atom_symbols_str.split(',')) if atom_symbols_str else (),
            atom_coords=_blob_to_np(atom_coords_blob),
            created_at=datetime.fromisoformat(created_at),
        )


# --- Module-level singleton ------------------------------------------------

_DEFAULT_CACHE: Optional[ParameterCache] = None


def get_default_cache() -> ParameterCache:
    """Return the process-wide default cache. Created lazily on first call."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = ParameterCache()
    return _DEFAULT_CACHE


# --- Convenience: build a CacheKey from a (molecule, ansatz, mapper, basis) ---

def build_cache_key(
    atom_symbols: Sequence[str],
    atom_coords: np.ndarray,
    ansatz,
    mapper_type: str,
    basis: str,
    charge: int = 0,
    spin: int = 0,
) -> CacheKey:
    """Build a `CacheKey` from solver objects.

    `ansatz` is the ansatz instance (any object with a few common
    attributes); `mapper_type` is the string identifier; `basis` is the
    basis-set name string. `charge` and `spin` key the electronic state so
    differently-charged or differently-spin-polarized species don't collide.
    """
    geom_hash = _hash_geometry(atom_symbols, np.asarray(atom_coords))
    ansatz_class = type(ansatz).__name__ if ansatz is not None else 'None'
    ansatz_config = _serialize_ansatz_config(ansatz)
    return CacheKey(
        geometry_hash=geom_hash,
        ansatz_class=ansatz_class,
        ansatz_config=ansatz_config,
        mapper_type=str(mapper_type).lower(),
        basis=str(basis).lower(),
        charge=int(charge),
        spin=int(spin),
    )
