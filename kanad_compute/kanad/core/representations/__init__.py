"""
Quantum representations for different bonding types.

Each bonding type requires different quantum circuit strategies:
- Ionic: Localized, minimal entanglement
- Covalent: Hybrid orbitals, paired entanglement
- Metallic: Delocalized k-space, collective entanglement
"""

from kanad.core.representations.base_representation import (
    BaseRepresentation,
    BondMolecule,
    Molecule,  # deprecated alias of BondMolecule
)
from kanad.core.representations.lcao_representation import (
    LCAORepresentation,
    HybridizationType
)
from kanad.core.representations.second_quantization import (
    SecondQuantizationRepresentation
)

__all__ = [
    'BaseRepresentation',
    'BondMolecule',
    'Molecule',
    'LCAORepresentation',
    'HybridizationType',
    'SecondQuantizationRepresentation',
]
