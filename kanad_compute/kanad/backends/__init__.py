"""
Kanad Quantum Backends.

Provides interfaces to various quantum simulators and hardware:
- BlueQubit: GPU-accelerated cloud simulators (up to 36 qubits)
- IonQ: Trapped-ion simulators and hardware (up to 29 qubits free)
- IBM Quantum: Superconducting hardware via qiskit-ibm-runtime

Usage:
    from kanad.backends import BlueQubitBackend, IonQBackend
    from kanad.backends.ibm import IBMBackend  # IBM requires separate import
"""

from kanad.backends.bluequbit import BlueQubitBackend
from kanad.backends.ionq import IonQBackend
from kanad.backends.ibm import IBMBackend

__all__ = ['BlueQubitBackend', 'IonQBackend', 'IBMBackend']
