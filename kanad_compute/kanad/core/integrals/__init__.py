"""
Molecular integral computation for quantum chemistry.

Provides one-electron (kinetic, nuclear) and two-electron (repulsion)
integrals needed for Hamiltonian construction.
"""

from kanad.core.integrals.one_electron import OneElectronIntegrals
from kanad.core.integrals.two_electron import TwoElectronIntegrals
from kanad.core.integrals.overlap import OverlapIntegrals
from kanad.core.integrals.basis_sets import (
    BasisSet,
    GaussianPrimitive,
    ContractedGaussian
)
from kanad.core.integrals.basis_registry import (
    BasisSetRegistry,
    print_available_basis_sets
)
from kanad.core.integrals.one_electron import nuclear_repulsion
from kanad.core.integrals.transforms import (
    ao2mo_transform,
    ao2mo_transform_from_mol,
    one_index_transform,
    property_integral_transform,
)
from kanad.core.integrals.property_integrals import (
    compute_dipole,
    compute_rinv,
    compute_r2,
)

__all__ = [
    # Integral classes
    'OneElectronIntegrals',
    'TwoElectronIntegrals',
    'OverlapIntegrals',

    # Basis set classes
    'BasisSet',
    'GaussianPrimitive',
    'ContractedGaussian',
    'BasisSetRegistry',
    'print_available_basis_sets',

    # AO->MO transforms + nuclear repulsion (reorg Phase B3)
    'ao2mo_transform',
    'ao2mo_transform_from_mol',
    'one_index_transform',
    'property_integral_transform',
    'nuclear_repulsion',
    'compute_dipole',
    'compute_rinv',
    'compute_r2',
]
