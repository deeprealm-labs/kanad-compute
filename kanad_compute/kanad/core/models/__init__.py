"""Phenomenological / pedagogical model Hamiltonians.

These are NOT on the `BondFactory` dispatch path — they're explicit-construction
objects for users who want to study a specific model system (e.g. ionic Hubbard,
metallic tight-binding).

The ab-initio molecular Hamiltonians live in `kanad.core.hamiltonians`.
"""

from kanad.core.models.ionic_hubbard import IonicHubbardModel

__all__ = ['IonicHubbardModel']
