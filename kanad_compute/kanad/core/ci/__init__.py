"""core.ci — indigenous selected-CI / Slater-Condon engine (reorg Phase B1).

Facade re-exporting the subspace diagonalizers so high-level callers
(builder, solvers, tests) import from the package, not its submodules.
"""

from kanad.core.ci.selected_ci import (
    fci_excited_states,
    s_squared_of_subspace,
    diagonalize_pyscf,
    diagonalize_custom,
)

__all__ = [
    "fci_excited_states", "s_squared_of_subspace",
    "diagonalize_pyscf", "diagonalize_custom",
]
