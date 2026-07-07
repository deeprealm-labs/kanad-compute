"""
Kanad Chemical Reactions Module

Provides tools for simulating chemical reactions with quantum advantage:
- Transition state finding (NEB, Dimer method)
- Intrinsic Reaction Coordinate (IRC) following
- Reaction rate calculations (Eyring-Polanyi)
- Reactive molecular dynamics
- Governance-aware bond breaking/forming

Key Features:
- Quantum transition state search using VQE/SQD
- Governance protocol integration for reaction mechanisms
- Support for covalent, ionic, and metallic reactions

Module Organization:
-------------------
- ReactionSimulator: Classical force-field based (Morse/LJ)
- QuantumReactionSimulator: VQE-based potential energy surface (TRUE QUANTUM)
"""

from kanad.reactions.reaction_dynamics import (
    ReactionSimulator,
    TransitionState,
    ReactionPath,
    ReactionResult,
    ReactionType,
    create_reaction_simulator
)

from kanad.reactions.quantum_reaction import (
    QuantumReactionSimulator,
    QuantumTransitionState,
    QuantumReactionPath,
    create_quantum_reaction_simulator
)

__all__ = [
    # Classical (force-field based)
    'ReactionSimulator',
    'TransitionState',
    'ReactionPath',
    'ReactionResult',
    'ReactionType',
    'create_reaction_simulator',
    # Quantum (VQE-based) - TRUE QUANTUM ADVANTAGE
    'QuantumReactionSimulator',
    'QuantumTransitionState',
    'QuantumReactionPath',
    'create_quantum_reaction_simulator',
]
