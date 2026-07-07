"""
Governance Protocols for Kanad Framework.

Governance protocols encode bonding physics into quantum circuits:
- IonicGovernanceProtocol: Minimal entanglement for electron transfer bonds
- CovalentGovernanceProtocol: Paired entanglement for shared electrons
- MetallicGovernanceProtocol: High entanglement for delocalized electrons

Each protocol defines:
- Valid quantum operators for the bond type
- Circuit topology constraints
- Bond-specific Hamiltonian terms
"""

from kanad.core.governance.protocols import (
    BaseGovernanceProtocol,
    IonicGovernanceProtocol,
    CovalentGovernanceProtocol,
    MetallicGovernanceProtocol,
)

__all__ = [
    'BaseGovernanceProtocol',
    'IonicGovernanceProtocol',
    'CovalentGovernanceProtocol',
    'MetallicGovernanceProtocol',
]
