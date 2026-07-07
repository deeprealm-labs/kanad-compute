"""Solver registry — discover solvers by name / domain / capability.

The discovery layer of the capability + domain solver protocol (Stage 3). Reference
solvers are auto-registered (see ``solvers/__init__.py``); user/community solvers
register via :func:`register_solver`. Consumers — the reactions/dynamics drivers, the
app's lab routing, the workshop — resolve a solver by NAME (:func:`get_solver`) or
query the available set by domain/capability (:func:`list_solvers`) instead of
hard-importing concrete solver classes.

    from kanad.solvers import register_solver, get_solver, list_solvers

    @register_solver                      # uses cls.META.name
    class MySolver(BaseSolver): ...

    cls   = get_solver("physics_vqe")
    metas = list_solvers(domain="md", capability="nuclear_gradient")
"""
from __future__ import annotations

from typing import List, Optional, Type

from kanad.solvers.meta import SolverMeta

# name -> BaseSolver subclass. Single source of truth for solver discovery.
_REGISTRY: dict = {}


def register_solver(cls: Type) -> Type:
    """Register a ``BaseSolver`` subclass under its ``META.name``.

    Returns the class so it can be used as a decorator. Idempotent for the same
    class; raises if a *different* class tries to claim an already-used name.
    """
    from kanad.solvers.base_solver import BaseSolver
    if not (isinstance(cls, type) and issubclass(cls, BaseSolver)):
        raise TypeError(f"register_solver expects a BaseSolver subclass, got {cls!r}")
    name = cls.META.name
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"solver name {name!r} already registered to {existing.__name__}; "
            f"cannot re-register to {cls.__name__} (pick a unique META.name)"
        )
    _REGISTRY[name] = cls
    return cls


def unregister_solver(name: str) -> None:
    """Remove a solver by name (for plugin reloads / tests). No-op if absent."""
    _REGISTRY.pop(name, None)


def get_solver(name: str) -> Type:
    """Return the registered solver class for ``name`` (raises ``KeyError`` if unknown)."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no solver registered as {name!r}; known: {sorted(_REGISTRY)}"
        ) from None


def has_solver(name: str) -> bool:
    return name in _REGISTRY


def _matches(meta: SolverMeta, domain: Optional[str], capability: Optional[str]) -> bool:
    if domain is not None and domain not in meta.domains:
        return False
    if capability is not None and capability not in meta.capabilities:
        return False
    return True


def list_solvers(*, domain: Optional[str] = None,
                 capability: Optional[str] = None) -> List[SolverMeta]:
    """SolverMeta for every registered solver, optionally filtered by domain/capability."""
    return [cls.META for cls in _REGISTRY.values()
            if _matches(cls.META, domain, capability)]


def list_solver_classes(*, domain: Optional[str] = None,
                         capability: Optional[str] = None) -> List[Type]:
    """Registered solver classes, optionally filtered by domain/capability."""
    return [cls for cls in _REGISTRY.values()
            if _matches(cls.META, domain, capability)]


def registered_names() -> List[str]:
    """Sorted list of all registered solver names."""
    return sorted(_REGISTRY)
