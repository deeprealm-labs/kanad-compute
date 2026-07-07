"""
Quantum optimization module — supporting tooling for solvers + workflows.

Surviving classes:
- OrbitalOptimizer  — orbital localization / rotation utilities (active-space picker, M10)
- GeometryOptimizer — molecular structure optimization (scipy + PySCF gradients)

Removed in the 2026-05-28 cleanup:
- CircuitOptimizer: operated on a phantom `core.quantum_circuit.QuantumCircuit`
  gate-list model that never existed in the framework (the real circuits are
  Qiskit objects). Every pass guarded on `hasattr(circuit, 'gates')`, so it was
  inert. Zero functional consumers.
- QuantumOptimizer / AdaptiveOptimizer (M0): depended on deleted active_space
  machinery and raised TypeError on construction.
"""

from kanad.core.optimization.orbital_optimizer import OrbitalOptimizer
from kanad.core.optimization.geometry_optimizer import GeometryOptimizer

__all__ = [
    'OrbitalOptimizer',
    'GeometryOptimizer',
]
