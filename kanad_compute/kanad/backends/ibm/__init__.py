"""
IBM Quantum Backend for Kanad Framework

Provides access to IBM Quantum hardware and simulators via Qiskit Runtime.

Features:
- Batch mode job submission (for non-premium users)
- Quantum hardware access
- Cloud simulators
- Qiskit Runtime primitives (Sampler, Estimator)

Authentication:
    Set environment variables:
    - IBM_API: Your IBM Quantum API token
    - IBM_CRN: Cloud Resource Name (for IBM Cloud channel)

    Channels: 'ibm_quantum_platform' (default) or 'ibm_cloud'

Usage:
    from kanad.backends.ibm import IBMBackend

    # Execute on IBM (solvers build circuits + observables themselves)
    backend = IBMBackend(backend_name='ibm_brisbane')
"""

from kanad.backends.ibm.backend import IBMBackend
from kanad.backends.ibm.error_mitigation import ErrorMitigationStrategy

__all__ = [
    'IBMBackend',
    'ErrorMitigationStrategy',
]
