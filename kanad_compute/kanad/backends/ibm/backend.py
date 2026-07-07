"""
IBM Quantum Backend Implementation

Provides interface to IBM Quantum hardware and cloud simulators.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np

from kanad.backends.base_backend import BaseBackend, expectation_from_counts
from kanad.backends.ibm.error_mitigation import ErrorMitigationStrategy

logger = logging.getLogger(__name__)


class _SPSAResult:
    """Minimal optimizer result mirroring the qiskit-algorithms SPSA result API."""

    def __init__(self, x: np.ndarray, fun: float, nit: int):
        self.x = x      # optimal parameters
        self.fun = fun  # final objective value
        self.nit = nit  # number of iterations performed


class _LocalSPSA:
    """
    Self-contained SPSA (Simultaneous Perturbation Stochastic Approximation).

    Drop-in replacement for ``qiskit_algorithms.optimizers.SPSA`` for the subset
    of behavior used here: construction with hyperparameters and a
    ``minimize(objective, x0)`` call returning an object with ``.x``, ``.fun``,
    ``.nit``. Implements the standard SPSA recurrence with a two-function-eval
    (+/- perturbation) gradient estimate per iteration.
    """

    def __init__(
        self,
        maxiter: int = 200,
        c0: float = 0.2,
        c1: float = 0.1,
        stability_constant: float = 0,
        alpha: float = 0.602,
        gamma: float = 0.101,
    ):
        self.maxiter = maxiter
        self.c0 = c0  # step-size scale (a)
        self.c1 = c1  # perturbation scale (c)
        self.A = stability_constant
        self.alpha = alpha
        self.gamma = gamma

    def minimize(self, objective, x0: np.ndarray) -> '_SPSAResult':
        theta = np.array(x0, dtype=float)
        rng = np.random.default_rng()
        last_value = float(objective(theta))

        for k in range(self.maxiter):
            a_k = self.c0 / (k + 1 + self.A) ** self.alpha
            c_k = self.c1 / (k + 1) ** self.gamma

            # Rademacher perturbation vector (+/-1 per component).
            delta = rng.integers(0, 2, size=theta.shape) * 2 - 1
            delta = delta.astype(float)

            f_plus = float(objective(theta + c_k * delta))
            f_minus = float(objective(theta - c_k * delta))

            # SPSA gradient estimate: g_k = (f+ - f-) / (2 c_k delta)
            grad = (f_plus - f_minus) / (2.0 * c_k) / delta
            theta = theta - a_k * grad
            last_value = float(objective(theta))

        return _SPSAResult(x=theta, fun=last_value, nit=self.maxiter)


class IBMBackend(BaseBackend):
    """
    IBM Quantum backend for quantum hardware and simulators.

    Supports:
    - Real quantum hardware (127+ qubits)
    - Cloud simulators
    - Batch mode (parallel independent jobs)
    - Session mode (reserved hardware for iterative algorithms)
    - Qiskit Runtime primitives (SamplerV2, EstimatorV2)

    Usage:
        # Batch mode (for independent jobs)
        backend = IBMBackend(backend_name='ibm_brisbane')
        results = backend.run_batch(circuits, observables)

        # Session mode (for Hi-VQE and iterative algorithms)
        results = backend.run_session(circuits, observables, max_time='1h')
    """

    # Audit H8: framework backend identifier (distinct from self.backend_name,
    # which is the IBM hardware target like 'ibm_brisbane'). BaseSolver.__init__
    # reads self.backend.name and solvers dispatch on name == 'ibm', so this must
    # be the requested factory string, not the hardware backend name.
    name = "ibm"

    def __init__(
        self,
        backend_name: Optional[str] = None,
        api_token: Optional[str] = None,
        channel: str = 'ibm_quantum_platform',
        crn: Optional[str] = None,
        instance: Optional[str] = None
    ):
        """
        Initialize IBM Quantum backend.

        Args:
            backend_name: Backend name (e.g., 'ibm_brisbane', 'ibmq_qasm_simulator')
                         If None, uses least busy backend
            api_token: IBM Quantum API token (or set IBM_API env var)
            channel: 'ibm_quantum_platform' or 'ibm_cloud'
            crn: Cloud Resource Name (required for ibm_cloud)
            instance: IBM Quantum instance (optional)
        """
        self.backend_name = backend_name
        self.channel = channel
        self.crn = crn or os.getenv('IBM_CRN')
        self.instance = instance

        # Get API token - try multiple sources
        self.api_token = api_token or os.getenv('IBM_API')
        self._use_saved_credentials = False

        # If no token provided, try to use saved credentials
        if not self.api_token:
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                # Try to load saved credentials
                QiskitRuntimeService()
                self._use_saved_credentials = True
                logger.info("Using saved IBM Quantum credentials")
            except Exception:
                raise ValueError(
                    "IBM Quantum API token required. Either:\n"
                    "1. Set IBM_API environment variable\n"
                    "2. Pass api_token parameter\n"
                    "3. Save credentials with QiskitRuntimeService.save_account()\n"
                    "Get token from https://quantum.ibm.com"
                )

        if channel == 'ibm_cloud' and not self.crn and not self._use_saved_credentials:
            raise ValueError(
                "IBM_CRN (Cloud Resource Name) required for ibm_cloud channel"
            )

        # Initialize service and backend
        self._init_service()
        self._init_backend()

        logger.info(f"IBM backend initialized: {self.backend.name}")

    def _init_service(self):
        """Initialize Qiskit Runtime service."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            if self._use_saved_credentials:
                # Use saved credentials directly
                self.service = QiskitRuntimeService()
                logger.info("IBM Quantum service initialized (saved credentials)")
            else:
                # Save account if not already saved
                try:
                    if self.channel == 'ibm_cloud':
                        QiskitRuntimeService.save_account(
                            channel=self.channel,
                            token=self.api_token,
                            instance=self.crn,
                            overwrite=True
                        )
                    else:
                        QiskitRuntimeService.save_account(
                            channel=self.channel,
                            token=self.api_token,
                            instance=self.instance,
                            overwrite=True
                        )
                except:
                    pass  # Account already exists

                # Initialize service
                self.service = QiskitRuntimeService(channel=self.channel)
                logger.info(f"IBM Quantum service initialized ({self.channel})")

        except ImportError:
            raise ImportError(
                "qiskit-ibm-runtime required. Install with: pip install qiskit-ibm-runtime"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize IBM Quantum service: {e}")

    def _init_backend(self):
        """Initialize quantum backend."""
        if self.backend_name:
            self.backend = self.service.backend(self.backend_name)
        else:
            # Get least busy backend
            self.backend = self.service.least_busy(operational=True, simulator=False)
            self.backend_name = self.backend.name

        logger.info(f"Using backend: {self.backend.name}")
        logger.info(f"  Qubits: {self.backend.num_qubits}")
        logger.info(f"  Quantum: {not self.backend.simulator}")

    def run_session(
        self,
        circuits: Union[List, 'QuantumCircuit'],
        observables: Optional[List] = None,
        shots: int = 1024,
        optimization_level: int = 1,
        resilience_level: int = 1,
        max_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run circuits in session mode (reserved hardware access).

        Session mode is ideal for Hi-VQE and other iterative algorithms that
        require sequential job execution with priority queue access.

        Args:
            circuits: Quantum circuit(s) to run
            observables: Observable(s) for Estimator (optional)
            shots: Number of measurement shots
            optimization_level: Transpilation optimization (0-3)
            resilience_level: Error mitigation level (0-2)
            max_time: Maximum session time (e.g., '1h', '30m', '2h')
                     If None, uses default session timeout

        Returns:
            Results dictionary with job ID and session context
        """
        from qiskit_ibm_runtime import Session, SamplerV2 as Sampler, EstimatorV2 as Estimator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Ensure circuits is a list
        if not isinstance(circuits, list):
            circuits = [circuits]

        logger.info(f"Running session on {self.backend.name}")
        logger.info(f"  Circuits: {len(circuits)}")
        logger.info(f"  Shots: {shots}")
        logger.info(f"  Max time: {max_time or 'default'}")

        # Transpile circuits for target hardware
        logger.info(f"  Transpiling circuits (optimization_level={optimization_level})...")
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level
        )
        transpiled_circuits = pm.run(circuits)
        logger.info(f"  Transpilation complete")

        # Apply layout to observables (correct way to handle transpiled circuits)
        if observables is not None:
            transpiled_observables = []
            for i, (trans_circuit, observable) in enumerate(zip(transpiled_circuits, observables)):
                # Use apply_layout to correctly map observable to transpiled circuit
                if trans_circuit.layout is not None:
                    mapped_observable = observable.apply_layout(trans_circuit.layout)
                    transpiled_observables.append(mapped_observable)
                    logger.info(f"  Observable {i}: applied layout ({observable.num_qubits} -> {mapped_observable.num_qubits} qubits)")
                else:
                    transpiled_observables.append(observable)

            observables = transpiled_observables

        try:
            # Build session parameters
            session_params = {'backend': self.backend}
            if max_time:
                session_params['max_time'] = max_time

            with Session(**session_params) as session:
                if observables is not None:
                    # Use Estimator for energy expectation values
                    estimator = Estimator(mode=session)

                    # Set options (V2 primitives)
                    estimator.options.default_shots = shots

                    # CRITICAL FIX (Issue #8): Use auto_configure() for error mitigation
                    error_mitigation = ErrorMitigationStrategy.auto_configure(self.backend.name)

                    # Apply auto-configured error mitigation strategy
                    estimator.options.resilience_level = error_mitigation.resilience_level
                    logger.info(f"  Resilience level: {error_mitigation.resilience_level}")

                    # Explicitly wire dynamical decoupling (mirrors run_batch); DD is not
                    # under estimator.options.resilience, so it must be set on its own namespace.
                    if error_mitigation.dynamical_decoupling:
                        try:
                            estimator.options.dynamical_decoupling.enable = True
                            estimator.options.dynamical_decoupling.sequence_type = error_mitigation.dynamical_decoupling
                            logger.info(f"  Dynamical decoupling: {error_mitigation.dynamical_decoupling}")
                        except Exception as e:
                            logger.warning(f"  DD not available: {e}")

                    # Explicitly wire gate/measurement twirling when requested.
                    if error_mitigation.twirling:
                        try:
                            estimator.options.twirling.enable_gates = True
                            estimator.options.twirling.enable_measure = True
                            logger.info(f"  Twirling: gates+measure enabled")
                        except Exception as e:
                            logger.warning(f"  Twirling not available: {e}")

                    logger.info("Using Estimator primitive (session mode)")

                    # Build pub (circuit, observable) tuples with transpiled circuits
                    pubs = [(transpiled_circuits[i], observables[i]) for i in range(len(transpiled_circuits))]

                    job = estimator.run(pubs)

                    # Return job and session info
                    return {
                        'job_id': job.job_id(),
                        'status': str(job.status()),
                        'backend': self.backend.name,
                        'session_id': session.session_id,
                        'mode': 'session'
                    }

                else:
                    # Use Sampler for measurement counts
                    sampler = Sampler(mode=session)

                    # Set options
                    sampler.options.default_shots = shots

                    logger.info("Using Sampler primitive (session mode)")

                    job = sampler.run(transpiled_circuits)

                    # Return job and session info
                    return {
                        'job_id': job.job_id(),
                        'status': str(job.status()),
                        'backend': self.backend.name,
                        'session_id': session.session_id,
                        'mode': 'session'
                    }

        except Exception as e:
            logger.error(f"IBM session execution failed: {e}")
            raise

    def run_batch(
        self,
        circuits: Union[List, 'QuantumCircuit'],
        observables: Optional[List] = None,
        shots: int = 1024,
        optimization_level: int = 3,  # Use max optimization for hardware
        resilience_level: int = 2,  # Enable ZNE + readout mitigation
        full_mitigation: bool = True  # Enable full mitigation stack
    ) -> Dict[str, Any]:
        """
        Run circuits in batch mode (for non-premium users).

        Args:
            circuits: Quantum circuit(s) to run
            observables: Observable(s) for Estimator (optional)
            shots: Number of measurement shots
            optimization_level: Transpilation optimization (0-3)
            resilience_level: Error mitigation level (0-2)

        Returns:
            Results dictionary
        """
        from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler, EstimatorV2 as Estimator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Ensure circuits is a list
        if not isinstance(circuits, list):
            circuits = [circuits]

        logger.info(f"Running batch on {self.backend.name}")
        logger.info(f"  Circuits: {len(circuits)}")
        logger.info(f"  Shots: {shots}")

        # Log input circuit stats
        for i, circ in enumerate(circuits):
            cx_count = circ.count_ops().get('cx', 0) + circ.count_ops().get('ecr', 0)
            logger.info(f"  Input circuit {i}: depth={circ.depth()}, 2Q-gates={cx_count}")

        # Transpile circuits for target hardware with max optimization
        logger.info(f"  Transpiling (optimization_level={optimization_level})...")
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=optimization_level
        )
        transpiled_circuits = pm.run(circuits)

        # Log transpiled circuit stats
        for i, circ in enumerate(transpiled_circuits):
            two_q = sum(circ.count_ops().get(g, 0) for g in ['cx', 'ecr', 'cz', 'rzz'])
            logger.info(f"  Transpiled circuit {i}: depth={circ.depth()}, 2Q-gates={two_q}")

        # Apply layout to observables (correct way to handle transpiled circuits)
        if observables is not None:
            transpiled_observables = []
            for i, (trans_circuit, observable) in enumerate(zip(transpiled_circuits, observables)):
                # Use apply_layout to correctly map observable to transpiled circuit
                if trans_circuit.layout is not None:
                    mapped_observable = observable.apply_layout(trans_circuit.layout)
                    transpiled_observables.append(mapped_observable)
                    logger.info(f"  Observable {i}: applied layout ({observable.num_qubits} -> {mapped_observable.num_qubits} qubits)")
                else:
                    transpiled_observables.append(observable)

            observables = transpiled_observables

        try:
            with Batch(backend=self.backend) as batch:
                if observables is not None:
                    # Use Estimator for energy expectation values
                    estimator = Estimator(mode=batch)

                    # Set options (V2 primitives)
                    estimator.options.default_shots = shots

                    # Configure error mitigation following IBM Quantum best practices
                    # Reference: https://quantum.cloud.ibm.com/docs/guides/configure-error-mitigation

                    # Set resilience level (2 = ZNE + readout mitigation)
                    estimator.options.resilience_level = resilience_level
                    logger.info(f"  Resilience level: {resilience_level}")

                    # Enable dynamical decoupling for coherence protection
                    try:
                        estimator.options.dynamical_decoupling.enable = True
                        estimator.options.dynamical_decoupling.sequence_type = 'XY4'
                        logger.info(f"  Dynamical decoupling: XY4")
                    except Exception as e:
                        logger.warning(f"  DD not available: {e}")

                    # Enable gate twirling to convert coherent errors to stochastic
                    try:
                        estimator.options.twirling.enable_gates = True
                        estimator.options.twirling.enable_measure = True
                        logger.info(f"  Twirling: gates+measure enabled")
                    except Exception as e:
                        logger.warning(f"  Twirling not available: {e}")

                    logger.info("  Using Estimator primitive (batch mode)")

                    # Build pub (circuit, observable) tuples with transpiled circuits
                    pubs = [(transpiled_circuits[i], observables[i]) for i in range(len(transpiled_circuits))]

                    job = estimator.run(pubs)

                    # Return job immediately (non-blocking)
                    return {
                        'job_id': job.job_id(),
                        'status': str(job.status()),
                        'backend': self.backend.name,
                        'mode': 'batch'
                    }

                else:
                    # Use Sampler for measurement counts
                    sampler = Sampler(mode=batch)

                    # Set options
                    sampler.options.default_shots = shots

                    logger.info("Using Sampler primitive")

                    job = sampler.run(transpiled_circuits)

                    # Return job immediately (non-blocking)
                    return {
                        'job_id': job.job_id(),
                        'status': str(job.status()),
                        'backend': self.backend.name,
                        'mode': 'batch'
                    }

        except Exception as e:
            logger.error(f"IBM batch execution failed: {e}")
            raise

    def run_single(
        self,
        circuit: 'QuantumCircuit',
        observable: Optional[Any] = None,
        shots: int = 1024
    ) -> Dict[str, Any]:
        """Run a single circuit (convenience method)."""
        return self.run_batch([circuit], [observable] if observable else None, shots=shots)

    # --- BaseBackend protocol (audit H8) -------------------------------------
    # The VQE/SQD solvers reach IBM execution through run_batch/run_session and
    # the legacy _ibm_backend alias, but the BaseBackend contract still requires
    # synchronous estimate_expectation/sample so make_backend('ibm') yields a
    # protocol-conformant object (BaseSolver.__init__ reads self.backend.name).

    def estimate_expectation(self, circuit, observable, shots: int | None = None) -> float:
        """Return <circuit| observable |circuit> by submitting an Estimator job and waiting."""
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        transpiled = pm.run(circuit)
        mapped_obs = observable.apply_layout(transpiled.layout) if transpiled.layout is not None else observable

        with Batch(backend=self.backend) as batch:
            estimator = Estimator(mode=batch)
            estimator.options.default_shots = shots or 4096
            job = estimator.run([(transpiled, mapped_obs)])
            result = job.result()
        return float(result[0].data.evs)

    def sample(self, circuit, shots: int) -> dict[str, int]:
        """Return {bitstring: count} by submitting a Sampler job and waiting."""
        from qiskit_ibm_runtime import SamplerV2 as Sampler, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        meas_circuit = circuit
        if meas_circuit.num_clbits == 0:
            meas_circuit = meas_circuit.copy()
            meas_circuit.measure_all()

        pm = generate_preset_pass_manager(backend=self.backend, optimization_level=1)
        transpiled = pm.run(meas_circuit)

        with Batch(backend=self.backend) as batch:
            sampler = Sampler(mode=batch)
            sampler.options.default_shots = shots
            job = sampler.run([transpiled])
            result = job.result()
        # V2 result: counts live under the (single) classical register's BitArray.
        pub_result = result[0]
        creg = next(iter(pub_result.data.__dict__)) if hasattr(pub_result.data, '__dict__') else 'meas'
        return dict(getattr(pub_result.data, creg).get_counts())

    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about current backend."""
        config = self.backend.configuration()

        info = {
            'name': self.backend.name,
            'num_qubits': self.backend.num_qubits,
            'is_simulator': self.backend.simulator,
            'is_operational': self.backend.status().operational,
            'pending_jobs': self.backend.status().pending_jobs,
            'basis_gates': config.basis_gates if hasattr(config, 'basis_gates') else None,
            'coupling_map': self.backend.coupling_map,
            'max_shots': config.max_shots if hasattr(config, 'max_shots') else 'unlimited'
        }

        return info

    def get_job_status(self, job_id: str) -> str:
        """Get status of a submitted job."""
        job = self.service.job(job_id)
        status = job.status()
        # Handle different Qiskit versions - status might be string or enum
        if isinstance(status, str):
            return status
        elif hasattr(status, 'name'):
            return status.name
        else:
            return str(status)

    def get_job_result(self, job_id: str) -> Any:
        """Retrieve results for a completed job."""
        job = self.service.job(job_id)
        return job.result()

    def list_backends(self, simulator: bool = False, operational: bool = True) -> List[str]:
        """List available backends."""
        backends = self.service.backends(simulator=simulator, operational=operational)
        return [b.name for b in backends]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running or queued job.

        Args:
            job_id: IBM job ID to cancel

        Returns:
            Dictionary with cancellation status
        """
        try:
            job = self.service.job(job_id)
            job.cancel()
            logger.info(f"IBM job {job_id} cancelled successfully")

            return {
                'status': 'cancelled',
                'job_id': job_id,
                'message': 'Job cancellation requested'
            }
        except Exception as e:
            logger.error(f"Failed to cancel IBM job {job_id}: {e}")
            raise RuntimeError(f"Failed to cancel job: {e}")

    def run_vqe_spsa(
        self,
        circuit_builder,
        observable: 'SparsePauliOp',
        initial_params: np.ndarray,
        n_iterations: int = 200,
        shots: int = 4096,
        callback=None
    ) -> Dict[str, Any]:
        """
        Run VQE optimization on real hardware using SPSA optimizer.

        SPSA (Simultaneous Perturbation Stochastic Approximation) is
        gradient-free and robust to noise - ideal for NISQ devices.

        This implements the approach from Belaloui et al. (2025):
        - Optimize parameters ON noisy hardware
        - Parameters found on noisy hardware give good results on ideal simulator

        Args:
            circuit_builder: Function(params) -> QuantumCircuit
            observable: SparsePauliOp Hamiltonian
            initial_params: Starting parameter values
            n_iterations: Number of SPSA iterations (200-400 recommended)
            shots: Shots per evaluation
            callback: Optional callback(iteration, params, energy)

        Returns:
            Dict with optimal_params, final_energy, history
        """
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        logger.info(f"Running VQE with SPSA on {self.backend.name}")
        logger.info(f"  Parameters: {len(initial_params)}")
        logger.info(f"  Iterations: {n_iterations}")
        logger.info(f"  Shots: {shots}")

        # SPSA hyperparameters (tuned for quantum hardware)
        # qiskit-algorithms is not a dependency in this environment, so use a
        # self-contained SPSA implementing the standard recurrence:
        #   a_k = c0 / (k + 1 + A)^alpha   (step size)
        #   c_k = c1 / (k + 1)^gamma       (perturbation size)
        # with a two-evaluation (+/- perturbation) gradient estimate per iteration.
        spsa = _LocalSPSA(
            maxiter=n_iterations,
            c0=0.2,  # Initial step size (learning rate)
            c1=0.1,  # Initial perturbation magnitude
            stability_constant=0,  # A in the step-size denominator
            alpha=0.602,  # Standard SPSA step-size decay
            gamma=0.101,  # Standard SPSA perturbation decay
        )

        energy_history = []
        param_history = []

        def objective(params):
            """Evaluate energy on hardware."""
            circuit = circuit_builder(params)

            # Transpile for target hardware
            pm = generate_preset_pass_manager(
                backend=self.backend,
                optimization_level=3
            )
            transpiled = pm.run(circuit)

            # Apply layout to observable
            if transpiled.layout is not None:
                mapped_obs = observable.apply_layout(transpiled.layout)
            else:
                mapped_obs = observable

            # Run on hardware in batch mode
            with Batch(backend=self.backend) as batch:
                estimator = Estimator(mode=batch)
                estimator.options.default_shots = shots
                estimator.options.resilience_level = 1  # Light mitigation during optimization

                job = estimator.run([(transpiled, mapped_obs)])
                result = job.result()

            energy = float(result[0].data.evs)
            energy_history.append(energy)
            param_history.append(params.copy())

            if callback:
                callback(len(energy_history), params, energy)

            return energy

        # Run SPSA optimization
        logger.info("Starting SPSA optimization on hardware...")
        result = spsa.minimize(objective, initial_params)

        optimal_params = result.x
        final_energy = result.fun

        logger.info(f"SPSA completed: final energy = {final_energy:.6f} Ha")

        return {
            'optimal_params': optimal_params,
            'final_energy': final_energy,
            'energy_history': energy_history,
            'param_history': param_history,
            'n_evaluations': len(energy_history),
            'converged': result.nit >= n_iterations
        }

    def run_with_zne(
        self,
        circuit: 'QuantumCircuit',
        observable: 'SparsePauliOp',
        shots: int = 4096,
        noise_factors: List[float] = None
    ) -> Dict[str, Any]:
        """
        Run circuit with Zero-Noise Extrapolation post-processing.

        ZNE amplifies noise at multiple levels and extrapolates to zero noise.
        This is the KEY technique for achieving accuracy on NISQ hardware.

        Research shows ZNE should be applied POST-optimization (to final result),
        not during VQE iterations.

        Args:
            circuit: Quantum circuit to execute
            observable: Observable to measure
            shots: Shots per noise factor
            noise_factors: Noise amplification factors [1.0, 1.5, 2.0, 3.0]

        Returns:
            Dict with extrapolated_energy, raw_energies, noise_factors
        """
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        if noise_factors is None:
            noise_factors = [1.0, 1.5, 2.0, 3.0]

        logger.info(f"Running ZNE with noise factors: {noise_factors}")

        # Transpile circuit
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=3
        )
        transpiled = pm.run(circuit)

        # Apply layout to observable
        if transpiled.layout is not None:
            mapped_obs = observable.apply_layout(transpiled.layout)
        else:
            mapped_obs = observable

        # Use the primitive's native ZNE: resilience_level=2 enables ZNE, and the
        # noise factors / extrapolator are configured explicitly so the requested
        # amplification levels are actually applied. The previous manual loop set
        # resilience_level=0 vs 2 per "noise factor" without ever amplifying noise,
        # then fit those already-mitigated values against an unrelated x-axis.
        with Batch(backend=self.backend) as batch:
            estimator = Estimator(mode=batch)
            estimator.options.default_shots = shots
            estimator.options.resilience_level = 2  # Enable ZNE

            try:
                estimator.options.resilience.zne.noise_factors = tuple(noise_factors)
                estimator.options.resilience.zne.extrapolator = 'polynomial_degree_2'
                estimator.options.resilience.zne.amplifier = 'gate_folding'
            except Exception as e:
                logger.warning(f"  ZNE options not fully configurable: {e}")

            job = estimator.run([(transpiled, mapped_obs)])
            result = job.result()
            extrapolated = float(result[0].data.evs)

        logger.info(f"ZNE extrapolated energy: {extrapolated:.6f} Ha")

        return {
            'extrapolated_energy': extrapolated,
            'noise_factors': noise_factors,
        }

    def run_with_rem(
        self,
        circuit_builder,
        observable: 'SparsePauliOp',
        hf_params: np.ndarray,
        vqe_params: np.ndarray,
        hf_energy_exact: float,
        shots: int = 8192
    ) -> Dict[str, Any]:
        """
        Run with Reference Error Mitigation (REM).

        REM uses a known reference state (Hartree-Fock) to estimate
        systematic errors and correct the VQE result.

        From JCTC 2022: Up to 100x improvement in accuracy demonstrated.

        Algorithm:
        1. Measure E_HF^noisy = <HF|H|HF> on hardware
        2. Compute error: ΔE = E_HF^noisy - E_HF^exact
        3. Measure E_VQE^noisy on hardware
        4. Correct: E_VQE^corrected = E_VQE^noisy - ΔE

        Args:
            circuit_builder: Function(params) -> QuantumCircuit
            observable: Hamiltonian as SparsePauliOp
            hf_params: Parameters that give HF state (usually zeros)
            vqe_params: Optimized VQE parameters
            hf_energy_exact: Exact classical HF energy
            shots: Number of shots

        Returns:
            Dict with corrected energy and error estimates
        """
        from qiskit_ibm_runtime import EstimatorV2 as Estimator, Batch
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        logger.info(f"Running REM on {self.backend.name}")
        logger.info(f"  HF exact energy: {hf_energy_exact:.6f} Ha")

        # Build circuits
        hf_circuit = circuit_builder(hf_params)
        vqe_circuit = circuit_builder(vqe_params)

        # Transpile both circuits
        pm = generate_preset_pass_manager(
            backend=self.backend,
            optimization_level=3
        )
        transpiled_hf = pm.run(hf_circuit)
        transpiled_vqe = pm.run(vqe_circuit)

        # Apply layout to observable
        if transpiled_hf.layout is not None:
            mapped_obs = observable.apply_layout(transpiled_hf.layout)
        else:
            mapped_obs = observable

        # Run both circuits on hardware
        with Batch(backend=self.backend) as batch:
            estimator = Estimator(mode=batch)
            estimator.options.default_shots = shots
            estimator.options.resilience_level = 1  # Light mitigation

            # Enable dynamical decoupling
            try:
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.dynamical_decoupling.sequence_type = 'XY4'
            except:
                pass

            pubs = [
                (transpiled_hf, mapped_obs),
                (transpiled_vqe, mapped_obs)
            ]
            job = estimator.run(pubs)
            result = job.result()

        hf_energy_noisy = float(result[0].data.evs)
        vqe_energy_noisy = float(result[1].data.evs)

        # Apply REM correction
        delta_e = hf_energy_noisy - hf_energy_exact
        vqe_energy_corrected = vqe_energy_noisy - delta_e

        logger.info(f"  HF noisy: {hf_energy_noisy:.6f} Ha")
        logger.info(f"  VQE noisy: {vqe_energy_noisy:.6f} Ha")
        logger.info(f"  Error estimate (ΔE): {delta_e:.6f} Ha")
        logger.info(f"  VQE corrected: {vqe_energy_corrected:.6f} Ha")

        return {
            'corrected_energy': vqe_energy_corrected,
            'raw_vqe_energy': vqe_energy_noisy,
            'raw_hf_energy': hf_energy_noisy,
            'error_estimate': delta_e,
            'improvement': abs(vqe_energy_noisy - vqe_energy_corrected)
        }

    # _richardson_extrapolation removed in reorg B5 (dead: run_with_zne uses
    # native Qiskit primitive ZNE; the math now lives in
    # kanad.core.error_mitigation.zne.richardson_extrapolation if a manual path
    # is ever needed again).

    def __repr__(self):
        return f"IBMBackend(backend='{self.backend.name}', qubits={self.backend.num_qubits})"
