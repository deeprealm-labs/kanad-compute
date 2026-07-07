"""
Open Quantum Systems for Molecular Dynamics

This module provides tools for simulating molecules interacting with their
environment (solvent, phonon bath, etc.) using quantum-mechanical treatments.

Key Components:
--------------
- LindbladEvolver: Master equation evolution for open systems
- QuantumBath: Models for solvent and phonon baths
- DecoherenceModel: Bond-type specific decoherence rates

Theory:
------
Lindblad Master Equation:
    dρ/dt = -i[H,ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

    Where:
    - H: System Hamiltonian
    - L_k: Lindblad (jump) operators
    - γ_k: Decay rates

Applications:
- Decoherence in molecular qubits
- Vibrational relaxation
- Energy transfer in photosynthesis
- Solvated molecular dynamics

References:
----------
1. Lindblad (1976) Commun. Math. Phys. 48, 119 - Original derivation
2. Schlimgen et al. (2021) PRL 127, 270503 - Quantum simulation of Lindblad
3. Hu et al. (2020) Chem. Rev. 120, 2879 - Open system chemistry
"""

from kanad.dynamics.open_quantum.lindblad import (
    LindbladEvolver,
    LindbladResult,
    create_dephasing_operator,
    create_amplitude_damping_operator,
    create_thermal_operators
)

from kanad.dynamics.open_quantum.quantum_bath import (
    QuantumBath,
    SpinBosonBath,
    DruideLorenzBath,
    create_bath_from_solvent,
    SOLVENT_PROPERTIES
)

from kanad.dynamics.open_quantum.decoherence import (
    DecoherenceModel,
    get_decoherence_rates,
    estimate_T1_T2
)

__all__ = [
    # Lindblad
    'LindbladEvolver',
    'LindbladResult',
    'create_dephasing_operator',
    'create_amplitude_damping_operator',
    'create_thermal_operators',

    # Quantum Bath
    'QuantumBath',
    'SpinBosonBath',
    'DruideLorenzBath',
    'create_bath_from_solvent',
    'SOLVENT_PROPERTIES',

    # Decoherence
    'DecoherenceModel',
    'get_decoherence_rates',
    'estimate_T1_T2',
]
