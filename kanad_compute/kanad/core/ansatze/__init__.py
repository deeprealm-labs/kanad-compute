"""
Variational quantum ansatze for VQE.

CURRENT ANSATZE (see CLAUDE.md for the verified status of each):

- HardwareEfficientAnsatz / RealAmplitudesAnsatz / EfficientSU2Ansatz
    Device-native gates. Good for ≤4-qubit systems; linear-CNOT entanglement
    breaks N-conservation on ≥10-qubit chemistry (use GivensSD there).

- GivensRotationAnsatz / GivensSDAnsatz (M2.5)
    Particle-conserving. GivensSD is the chemistry workhorse (H₂ → FCI exactly).

- LUCJAnsatz (M4)
    Local Unitary Cluster Jastrow — hardware-efficient sampling for SQD.

- PhysicsDrivenAnsatz
    MP2-ranked minimal-excitation circuit. Driven by the PhysicsVQE solver only.

- CustomCircuitAnsatz
    User-defined gate sequences. Workshop API for researcher-built ansatze.

REMOVED:
- 2026-05-12: UCC ansatze (params didn't affect energy), governance ansatze
  (returned HF), TwoLocal (never validated).
- 2026-05-28: GovernanceMinimalAnsatz / create_minimal_ansatz (retired
  'governance' branding; hard-coded the disproven ΔEN>1.7 heuristic; unused).
"""

# Core
from kanad.core.ansatze.base_ansatz import BaseAnsatz, QuantumCircuit, Parameter

# Hardware-efficient (device-native)
from kanad.core.ansatze.hardware_efficient_ansatz import (
    HardwareEfficientAnsatz,
    RealAmplitudesAnsatz,
    EfficientSU2Ansatz,
)

# M2.5: particle-conserving ansatze
from kanad.core.ansatze.givens_rotation_ansatz import (
    GivensRotationAnsatz,
    GivensSDAnsatz,
)

# M4-A: Local Unitary Cluster Jastrow (hardware-efficient sampling)
from kanad.core.ansatze.lucj_ansatz import LUCJAnsatz

# CCSD-initialized LUCJ via ffsim — correlated SQD seed, no VQE (IBM-SQD standard).
# Optional: requires `ffsim`. Import guarded so kanad loads without it.
try:
    from kanad.core.ansatze.lucj_ffsim_ansatz import LUCJFfsimAnsatz
except Exception:  # pragma: no cover - ffsim optional
    LUCJFfsimAnsatz = None

# Physics-driven minimal ansatz
from kanad.core.ansatze.physics_driven_ansatz import PhysicsDrivenAnsatz

# Custom circuit ansatz (workshop API)
from kanad.core.ansatze.custom_circuit_ansatz import CustomCircuitAnsatz

__all__ = [
    # Core
    'BaseAnsatz',
    'QuantumCircuit',
    'Parameter',
    # Hardware-efficient
    'HardwareEfficientAnsatz',
    'RealAmplitudesAnsatz',
    'EfficientSU2Ansatz',
    # M2.5 — particle-conserving
    'GivensRotationAnsatz',
    'GivensSDAnsatz',
    # M4 — LUCJ (hardware-efficient sampling)
    'LUCJAnsatz',
    'LUCJFfsimAnsatz',
    # Physics-driven
    'PhysicsDrivenAnsatz',
    # Custom circuit
    'CustomCircuitAnsatz',
]
