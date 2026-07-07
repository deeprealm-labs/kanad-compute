"""
BlueQubit Cloud Backend for Kanad Framework

Provides access to BlueQubit's GPU and CPU quantum simulators.

Features:
- Free GPU simulators (36 qubits)
- CPU simulators (34 qubits)
- MPS tensor network simulators (40+ qubits)
- NVIDIA cuQuantum acceleration

Authentication:
    Set environment variable BLUEQUBIT_API_TOKEN or pass token directly

Usage:
    from kanad.backends.bluequbit import BlueQubitBackend

    backend = BlueQubitBackend(device='gpu')  # or 'cpu', 'mps.gpu'
"""

from kanad.backends.bluequbit.backend import BlueQubitBackend

__all__ = ['BlueQubitBackend']
