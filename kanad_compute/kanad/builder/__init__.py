"""Kanad Molecular Builder — one fluent interface for every workflow.

    from kanad.builder import MolecularBuilder
    qs = MolecularBuilder.from_smiles("O").build()
    qs.solve()
"""

from kanad.builder.system_spec import SystemSpec
from kanad.builder.quantum_system import QuantumSystem
from kanad.builder.molecular_builder import MolecularBuilder

__all__ = ['MolecularBuilder', 'QuantumSystem', 'SystemSpec']
