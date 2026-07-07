"""
IonQ Backend for Kanad.

Provides access to IonQ quantum simulators and QPU hardware.

Free tier: Simulator up to 29 qubits
Hardware: Aria (#AQ 25), Forte (#AQ 36)

Usage:
    from kanad.backends.ionq import IonQBackend

    backend = IonQBackend(device='simulator')
    result = backend.run_circuit(circuit)
"""

from kanad.backends.ionq.backend import IonQBackend

__all__ = ['IonQBackend']
