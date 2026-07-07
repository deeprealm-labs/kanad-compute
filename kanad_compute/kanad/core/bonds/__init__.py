"""core.bonds — bond/molecule construction (reorg Phase C).

Pure construction surface: ``BondFactory`` + the bond classes (build Molecule +
Hamiltonian + representation + governance + mapper). The ``compute_energy``/solve
dispatcher lives on these classes but imports ``kanad.solvers`` LAZILY (call-time
only) inside the method bodies, so this package has NO module-level edge to the
solver layer — core stays downward-only. The root ``kanad.bonds`` facade
re-exports these plus the analysis / optimization tools.
"""

from kanad.core.bonds.base_bond import BaseBond
from kanad.core.bonds.bond_factory import BondFactory, BondType
from kanad.core.bonds.ionic_bond import IonicBond
from kanad.core.bonds.covalent_bond import CovalentBond
from kanad.core.bonds.metallic_bond import MetallicBond

__all__ = [
    'BaseBond',
    'BondFactory',
    'BondType',
    'IonicBond',
    'CovalentBond',
    'MetallicBond',
]
