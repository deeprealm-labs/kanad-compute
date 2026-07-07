"""Unified return type for all Kanad solvers (solver-protocol refactor, 2026-06-12).

Every solver's ``solve()`` returns a ``SolverResult``. The canonical energy is
always ``result.energy`` (resolving the old ``ground_energy`` vs ``['energy']``
divergence). Solver-specific fields (eigenvectors, determinants, excitations,
telemetry, ...) live in ``result.extra``. ``to_dict()`` yields a JSON-safe dict
for the API serialization layer and external consumers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from kanad.solvers.capabilities import ExcitedStateData

# Core fields promoted out of a legacy result dict by ``from_mapping``; every
# other key falls through into ``extra``.
_CORE_KEYS = {
    "energy", "converged", "solver", "backend", "iterations",
    "hf_energy", "correlation_energy", "energy_history", "states", "analysis",
}


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / nested containers into JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        # Covers np.bool_, np.float64, np.int64, np.complex128, ... in one shot.
        return _jsonable(obj.item())
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


@dataclass(frozen=True)
class SolverResult:
    """Unified solver return value.

    Attributes:
        energy: Ground-state energy in Hartree (canonical).
        converged: Whether the solver converged.
        solver: Short solver tag, e.g. ``"vqe"``, ``"physics_vqe"``.
        backend: Backend name, e.g. ``"statevector"``.
        iterations: Iteration / evaluation count, if applicable.
        hf_energy: Hartree-Fock reference energy, if computed.
        correlation_energy: ``energy - hf_energy``, if computed.
        energy_history: Per-iteration energy trace, if recorded.
        states: Excited-state energies (Hartree), if a multi-state solver.
        analysis: Optional analysis payload (energy decomposition, bonding, ...).
        extra: Solver-specific fields not part of the stable core.
    """

    energy: float
    converged: bool
    solver: str
    backend: str
    iterations: int | None = None
    hf_energy: float | None = None
    correlation_energy: float | None = None
    energy_history: list[float] | None = None
    states: list[float] | None = None
    analysis: dict | None = None
    extra: dict = field(default_factory=dict)

    # Optional in-memory convenience slots for capability outputs. The capability
    # accessors (``solver.get_one_rdm()`` / ``nuclear_gradient()`` /
    # ``get_excited_state_data()``) are the PRIMARY API; these are None unless a
    # solver fills them and are EXCLUDED from ``to_dict()`` (ndarrays /
    # ExcitedStateData are not the JSON serialization contract).
    one_rdm_mo: Optional[np.ndarray] = None
    gradient: Optional[np.ndarray] = None
    excited: "Optional[ExcitedStateData]" = None

    @classmethod
    def from_mapping(cls, data, *, solver: str, backend: str,
                     energy_key: str = "energy") -> "SolverResult":
        """Build a ``SolverResult`` from a legacy result dict.

        Promotes the known core keys; everything else lands in ``extra``. The
        legacy dict is preserved field-for-field (nothing is dropped).

        Args:
            data: Mapping produced by a solver's internal solve routine.
            solver: Solver tag to stamp onto the result.
            backend: Backend name to stamp onto the result.
            energy_key: Key under which the canonical energy lives (e.g.
                ``"ground_energy"`` for qEOM).
        """
        d = dict(data)
        energy = d.pop(energy_key)
        core = {k: d.pop(k) for k in list(d.keys()) if k in _CORE_KEYS}
        return cls(
            energy=float(energy),
            converged=bool(core.pop("converged", True)),
            solver=solver,
            backend=backend,
            iterations=core.pop("iterations", None),
            hf_energy=core.pop("hf_energy", None),
            correlation_energy=core.pop("correlation_energy", None),
            energy_history=core.pop("energy_history", None),
            states=core.pop("states", None),
            analysis=core.pop("analysis", None),
            extra=d,
        )

    def to_dict(self) -> dict:
        """Return a JSON-serializable flat dict of the full result.

        ``extra`` is merged into the top level so legacy/external consumers that
        called ``solver.solve()['parameters']`` keep working via
        ``solver.solve().to_dict()['parameters']``. The stable core keys take
        precedence over any same-named key (``from_mapping`` already pops core
        keys out of ``extra``, so collisions don't occur in practice).
        """
        d = asdict(self)
        # Convenience capability slots are not part of the serialization contract.
        for _slot in ("one_rdm_mo", "gradient", "excited"):
            d.pop(_slot, None)
        extra = d.pop("extra", {}) or {}
        merged = {**extra, **d}  # core wins on the (non-occurring) collision
        return _jsonable(merged)
