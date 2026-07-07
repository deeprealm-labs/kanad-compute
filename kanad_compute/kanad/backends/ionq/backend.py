"""
IonQ Backend Implementation.

Provides interface to IonQ quantum simulators and trapped-ion hardware.
"""

import os
import logging
from typing import Dict, Any, Optional
import numpy as np

from kanad.backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)


class IonQBackend(BaseBackend):
    """
    IonQ cloud backend for quantum simulations and hardware.

    Supports:
    - Simulator: Free, up to 29 qubits, ideal simulation
    - QPU (Aria): #AQ 25, real trapped-ion hardware
    - QPU (Forte): #AQ 36, real trapped-ion hardware

    Usage:
        backend = IonQBackend(device='simulator', api_key='your_key')
        result = backend.run_circuit(circuit, shots=1024)
    """

    SUPPORTED_DEVICES = ['simulator', 'qpu', 'qpu.aria-1', 'qpu.forte-1']

    # Audit H8: framework backend identifier. BaseSolver.__init__ reads
    # self.backend.name and solvers dispatch on name == 'ionq', so this must be
    # the requested factory string (distinct from self.device / self.backend.name,
    # which is the IonQ device target like 'ionq_simulator').
    name = "ionq"

    def __init__(
        self,
        device: str = 'simulator',
        api_key: Optional[str] = None,
        **options
    ):
        """
        Initialize IonQ backend.

        Args:
            device: Device type ('simulator', 'qpu', 'qpu.aria-1', 'qpu.forte-1')
            api_key: IonQ API key (or set IONQ_API_KEY env var)
            **options: Additional options (e.g., noise_model for simulator)
        """
        self.device = device
        self.options = options

        # Get API key
        self.api_key = api_key or os.getenv('IONQ_API_KEY')

        if not self.api_key:
            raise ValueError(
                "IonQ API key required. Set IONQ_API_KEY environment variable "
                "or pass api_key parameter. Get key from https://cloud.ionq.com"
            )

        # Initialize provider
        self._init_provider()

        logger.info(f"IonQ backend initialized: device={device}")

    def _init_provider(self):
        """Initialize IonQ Qiskit provider."""
        try:
            from qiskit_ionq import IonQProvider
            # Pass token explicitly — qiskit_ionq's resolve_credentials() calls
            # dotenv_values() which reads .env BEFORE os.environ, so setting
            # os.environ['IONQ_API_KEY'] is not enough when .env has a different key.
            self.provider = IonQProvider(token=self.api_key)

            # Get backend
            if self.device == 'simulator':
                self.backend = self.provider.get_backend('ionq_simulator')
            elif self.device == 'qpu':
                self.backend = self.provider.get_backend('ionq_qpu')
            else:
                self.backend = self.provider.get_backend(f'ionq_{self.device}')

            logger.info(f"IonQ backend ready: {self.backend.name}")

        except ImportError:
            raise ImportError(
                "qiskit-ionq package required. Install with: pip install qiskit-ionq"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IonQ provider: {e}")

    def run_circuit(
        self,
        circuit,
        shots: int = 1024,
        job_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a quantum circuit on IonQ.

        Args:
            circuit: Qiskit QuantumCircuit
            shots: Number of measurement shots
            job_name: Optional job name for tracking

        Returns:
            Result dictionary with counts and optionally statevector
        """
        from qiskit import transpile, QuantumCircuit

        logger.info(f"Running on IonQ {self.device}")
        print(f"🔷 IonQ device: {self.device}")
        print(f"   Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")

        # IonQ always requires measurements (API rejects circuits without them)
        if circuit.num_clbits == 0:
            circuit = circuit.copy()
            circuit.measure_all()

        # Transpile for IonQ (use low optimization to preserve structure)
        transpiled = transpile(
            circuit,
            backend=self.backend,
            optimization_level=1  # IonQ recommends 0-1
        )

        try:
            # Submit job
            job = self.backend.run(transpiled, shots=shots)
            logger.info(f"IonQ job submitted: {job.job_id()}")

            # Wait for result
            result = job.result()

            output = {
                'job_id': job.job_id()
            }

            # Try to get statevector (simulator without measurements)
            if self.device == 'simulator':
                try:
                    sv = result.get_statevector()
                    output['statevector'] = np.array(sv)
                    logger.info(f"IonQ simulator returned statevector ({len(output['statevector'])} amplitudes)")
                except Exception as e:
                    logger.debug(f"IonQ statevector unavailable (expected for IonQ): {e}")

            # Get counts if available (circuits with measurements)
            try:
                output['counts'] = result.get_counts()
            except Exception:
                pass

            # If no statevector and no counts, try running with measurements as fallback
            if 'statevector' not in output and 'counts' not in output:
                logger.info("Retrying IonQ with measurements for counts...")
                meas_circuit = circuit.copy()
                if meas_circuit.num_clbits == 0:
                    meas_circuit.measure_all()
                meas_transpiled = transpile(meas_circuit, backend=self.backend, optimization_level=1)
                meas_job = self.backend.run(meas_transpiled, shots=shots)
                meas_result = meas_job.result()
                output['counts'] = meas_result.get_counts()
                output['job_id'] = meas_job.job_id()

            logger.info(f"IonQ job completed: {output['job_id']} (has_sv={'statevector' in output}, has_counts={'counts' in output})")
            return output

        except Exception as e:
            err_str = str(e)
            status_code = getattr(e, 'status_code', None)
            # Extract clean error message from IonQAPIError
            if status_code == 403 or 'Insufficient scope' in err_str or 'Forbidden' in err_str:
                clean_msg = ("IonQ API key is invalid, expired, or revoked (403 Forbidden). "
                             "Generate a new API key at cloud.ionq.com and update it in Profile > Credentials.")
            elif status_code == 401 or '401' in err_str or 'Unauthorized' in err_str:
                clean_msg = ("IonQ API key is invalid or not recognized (401). "
                             "Generate a new API key at cloud.ionq.com and update it in Profile > Credentials.")
            elif '429' in err_str or 'Too Many' in err_str:
                clean_msg = "IonQ rate limit exceeded. Wait a moment and try again."
            else:
                clean_msg = f"IonQ execution failed: {getattr(e, 'message', err_str)[:200]}"
            logger.error(clean_msg)
            raise RuntimeError(clean_msg) from e

    def compute_expectation_value(
        self,
        circuit,
        observable,
        shots: int = 8192
    ) -> float:
        """
        Compute expectation value using sampling.

        For each Pauli term in the observable, measures in the
        appropriate basis and estimates the expectation value.

        Args:
            circuit: State preparation circuit
            observable: SparsePauliOp observable
            shots: Number of shots per Pauli term

        Returns:
            Estimated expectation value
        """
        from qiskit import QuantumCircuit, transpile

        total_expval = 0.0

        # Group Pauli terms by commuting groups for efficiency
        for pauli_str, coeff in zip(observable.paulis.to_labels(), observable.coeffs):
            if abs(coeff) < 1e-10:
                continue

            # Create measurement circuit
            meas_circuit = circuit.copy()

            # Change basis for non-Z measurements
            for q, pauli in enumerate(reversed(pauli_str)):
                if pauli == 'X':
                    meas_circuit.h(q)
                elif pauli == 'Y':
                    meas_circuit.sdg(q)
                    meas_circuit.h(q)
                # Z and I need no basis change

            # Add measurements
            meas_circuit.measure_all()

            # Run and get counts
            transpiled = transpile(meas_circuit, backend=self.backend, optimization_level=1)
            job = self.backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()

            # Compute expectation value for this Pauli
            expval = 0.0
            total = sum(counts.values())
            for bitstring, count in counts.items():
                # Compute parity for non-I Paulis
                parity = 1
                for q, pauli in enumerate(reversed(pauli_str)):
                    if pauli != 'I':
                        bit = int(bitstring[-(q+1)])  # Get bit at position q
                        parity *= (-1) ** bit
                expval += parity * count / total

            total_expval += coeff.real * expval

        return total_expval

    def get_expectation_value(
        self,
        circuit,
        observable,
        shots: int = 8192
    ) -> float:
        """
        Compute expectation value of an observable.

        Args:
            circuit: State preparation circuit
            observable: SparsePauliOp observable
            shots: Number of shots for sampling

        Returns:
            Expectation value as float
        """
        from qiskit.quantum_info import Statevector

        # For simulator, use statevector if it is ever available.
        # NOTE: IonQ (simulator and QPU) never returns a statevector — the
        # IonQ REST API and qiskit_ionq only emit counts/probabilities — so
        # this branch is effectively dead for IonQ but kept defensively.
        if self.device == 'simulator':
            result = self.run_circuit(circuit, shots=shots)
            if 'statevector' in result:
                sv = Statevector(result['statevector'])
                return sv.expectation_value(observable).real

        # Sampling is the only physically possible mode on IonQ: delegate to
        # the per-Pauli basis-rotation sampling estimator implemented above.
        return self.compute_expectation_value(circuit, observable, shots=shots)

    # --- BaseBackend protocol (audit H8) -------------------------------------

    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        """Return <circuit| observable |circuit> via per-Pauli basis-rotation sampling."""
        return float(self.get_expectation_value(circuit, observable, shots=shots or 8192))

    def sample(self, circuit, shots: int) -> dict[str, int]:
        """Return {bitstring: count} from measuring ``circuit``."""
        result = self.run_circuit(circuit, shots=shots)
        return dict(result.get('counts', {}))

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about current device."""
        return {
            'device': self.device,
            'name': self.backend.name,
            'max_qubits': self._get_max_qubits(),
            'is_simulator': self.device == 'simulator',
            'is_hardware': 'qpu' in self.device
        }

    def _get_max_qubits(self) -> int:
        """Get maximum qubits for device."""
        limits = {
            'simulator': 29,
            'qpu': 25,  # Aria default
            'qpu.aria-1': 25,
            'qpu.forte-1': 36
        }
        return limits.get(self.device, 25)

    def __repr__(self):
        return f"IonQBackend(device='{self.device}')"
