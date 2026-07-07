"""
Qiskit Backend wrapper for Kanad Framework.

Provides a simple interface to Qiskit simulators.
Supports both qiskit-aer (if available) and built-in Qiskit primitives.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Check if qiskit-aer is available
try:
    from qiskit_aer import Aer
    AER_AVAILABLE = True
except ImportError:
    AER_AVAILABLE = False
    logger.info("qiskit-aer not available, using Qiskit built-in primitives")


class QiskitBackend:
    """
    Wrapper for Qiskit backends.

    Supports:
    - aer_simulator: General purpose simulator (requires qiskit-aer)
    - aer_simulator_statevector: Exact statevector simulator (requires qiskit-aer)
    - statevector: Built-in Qiskit statevector (no qiskit-aer needed)
    """

    def __init__(
        self,
        backend_name: str = 'statevector',
        shots: Optional[int] = 1024,
        **kwargs
    ):
        """
        Initialize Qiskit backend.

        Args:
            backend_name: Name of backend ('statevector', 'aer_simulator', etc.)
            shots: Number of shots (None for statevector)
            **kwargs: Additional backend options
        """
        self.backend_name = backend_name
        self.shots = None if 'statevector' in backend_name else shots
        self.backend_options = kwargs
        self.backend = None
        self._use_aer = False

        # Try to use Aer if available and requested
        if AER_AVAILABLE and backend_name.startswith('aer_'):
            try:
                self.backend = Aer.get_backend(backend_name)
                self._use_aer = True
                logger.info(f"Initialized Qiskit Aer backend: {backend_name}")
            except Exception as e:
                logger.warning(f"Failed to get Aer backend '{backend_name}': {e}")
                logger.info("Falling back to built-in statevector")
                self.backend_name = 'statevector'

        if not self._use_aer:
            # Use built-in Qiskit primitives (no Aer dependency)
            logger.info(f"Using Qiskit built-in statevector simulator")

    def get_estimator(self):
        """
        Get Qiskit Estimator primitive.

        Returns:
            Estimator instance
        """
        if self._use_aer and self.backend is not None:
            try:
                # Use V2 Aer estimator so every return honors the V2 pub-based contract
                from qiskit_aer.primitives import EstimatorV2
                return EstimatorV2()
            except ImportError:
                pass

        # Fallback to built-in V2 statevector estimator (no Aer needed)
        from qiskit.primitives import StatevectorEstimator
        return StatevectorEstimator()

    def get_sampler(self):
        """
        Get Qiskit Sampler primitive.

        Returns:
            Sampler instance
        """
        if self._use_aer and self.backend is not None:
            try:
                # Use V2 Aer sampler so every return honors the V2 pub-based contract
                from qiskit_aer.primitives import SamplerV2
                return SamplerV2()
            except ImportError:
                pass

        # Fallback to built-in V2 statevector sampler (no Aer needed)
        from qiskit.primitives import StatevectorSampler
        return StatevectorSampler()

    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get backend information.

        Returns:
            Dictionary with backend info
        """
        return {
            'name': self.backend_name,
            'shots': self.shots,
            'backend': str(self.backend),
        }

    def run(self, circuit, **kwargs):
        """
        Run circuit on backend.

        Args:
            circuit: Qiskit QuantumCircuit
            **kwargs: Additional run options

        Returns:
            Job result
        """
        if self._use_aer and self.backend is not None:
            run_kwargs = {'shots': self.shots} if self.shots else {}
            run_kwargs.update(kwargs)
            job = self.backend.run(circuit, **run_kwargs)
            # Normalize to a plain dict so both branches honor one return contract
            return {'counts': job.result().get_counts()}

        # Use statevector for exact simulation
        from qiskit.quantum_info import Statevector
        sv = Statevector(circuit)
        return {'statevector': sv}
