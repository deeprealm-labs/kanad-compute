"""Active-space machinery for VQE on >12-qubit molecules.

The framework's `MolecularHamiltonian` builds the Hamiltonian over the full
basis-set orbital count; that's tractable for H₂/LiH STO-3G but explodes by
LiH 6-31G or H₂O STO-3G. Active-space methods carve the variational manifold
down to the chemically meaningful subset (e.g., valence orbitals only) and
fold the frozen-core contribution into a constant energy + an effective
one-body operator.

This package is the M1 rebuild of the deleted `core/active_space.py` module
(commit `eb3274d`), which gave LiH energy 173 mHa BELOW FCI — a variational
violation caused by applying the frozen-core formula to AO-basis integrals
instead of MO-basis integrals.

The new module is split:

- `selector.ActiveSpace` — frozen-dataclass selection result (which orbitals
  are frozen, active, virtual).
- `selector.ActiveSpaceSelector` — pure orbital-selection logic operating on
  a converged PySCF mean-field; methods `manual`, `frozen_core`, `frontier`.
- `integral_transform.build_active_space_hamiltonian` — the canonical MO-basis
  integral transform that produces the effective `(h_eff, eri_eff, E_inactive)`.
- `integral_transform.ActiveHamiltonian` — a `MolecularHamiltonian` duck-type
  that solvers consume without modification.

The two non-negotiable regression tests
(`tests/validation/test_active_space.py`) enforce:

1. `E_HF(active) + E_inactive == E_HF(full)` to ≤1e-10 Ha (HF round-trip).
2. `E_CASCI(active) >= E_FCI(full) - 1e-10` Ha (variational bound).

Either failure means the integral transform is wrong — fail loudly. The
deleted module failed both.
"""

from kanad.core.active_space.selector import ActiveSpace, ActiveSpaceSelector
from kanad.core.active_space.integral_transform import (
    ActiveHamiltonian,
    build_active_space_hamiltonian,
)

__all__ = [
    'ActiveSpace',
    'ActiveSpaceSelector',
    'ActiveHamiltonian',
    'build_active_space_hamiltonian',
]
