"""Single construction point for backends.

Replaces the string-dispatch ``if/elif`` formerly in ``BaseSolver._init_backend``.
Cloud-backend imports are lazy so the framework imports cleanly without their
optional SDKs installed; ``planck`` falls back to statevector when the package
is absent (matching legacy behavior).
"""
from __future__ import annotations

from kanad.backends.base_backend import BaseBackend
from kanad.backends.statevector_backend import StatevectorBackend


def make_backend(name: str, **kwargs) -> BaseBackend:
    """Construct a backend by name.

    Args:
        name: One of ``statevector``, ``planck``, ``bluequbit``, ``ibm``, ``ionq``.
        **kwargs: Backend-specific construction parameters (device, shots, ...).
            The statevector backend ignores extras it does not recognize.
    """
    name = (name or "statevector").lower()

    if name == "statevector":
        return StatevectorBackend(**kwargs)

    if name == "planck":
        try:
            import planck  # noqa: F401
            from kanad.backends.planck_adapter import PlanckBackend
            return PlanckBackend(**kwargs)
        except ImportError:
            # Graceful fallback when rocm-planck isn't installed.
            return StatevectorBackend(**kwargs)

    if name == "bluequbit":
        from kanad.backends.bluequbit import BlueQubitBackend
        return BlueQubitBackend(**kwargs)

    if name == "ibm":
        from kanad.backends.ibm import IBMBackend
        return IBMBackend(**kwargs)

    if name == "ionq":
        from kanad.backends.ionq import IonQBackend
        return IonQBackend(**kwargs)

    raise ValueError(f"Unknown backend: {name!r}")
