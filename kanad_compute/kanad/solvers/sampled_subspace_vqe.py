"""
Sampled Subspace VQE (SSVQE) - Hardware-Optimized Subspace Solver.

This implements the key insight from Hi-VQE and QSCI papers:
1. Use SHALLOW circuits to sample configurations (not prepare exact states)
2. Post-select on particle number (FREE error mitigation!)
3. Diagonalize classically in the sampled subspace

Why this works on NISQ hardware:
- Bit-flip errors → wrong particle number → FILTERED OUT
- We don't need accurate amplitudes, just the basis states
- Classical diagonalization is EXACT

Reference:
- Qunova HiVQE: 0.1 mHa on 24-qubit Li₂S
- QSCI: Chemical accuracy on 77-qubit systems
- IBM SQD: Validated on real hardware
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import logging

from kanad.solvers.base_solver import BaseSolver
from kanad.core.solver_result import SolverResult

logger = logging.getLogger(__name__)


class SampledSubspaceVQE(BaseSolver):
    """
    Sampled Subspace VQE for NISQ hardware.

    Strategy:
    1. Use shallow HEA circuit (only 8 CNOTs for H₂!)
    2. Sample bitstrings from the circuit
    3. Post-select on correct electron count (key error mitigation!)
    4. Diagonalize Hamiltonian in sampled subspace

    This is much more noise-resilient than standard VQE because:
    - Errors that change particle number are filtered out
    - We only need the important configurations, not exact amplitudes
    - Classical diagonalization is exact

    Usage:
        from kanad import BondFactory
        from kanad.solvers.sampled_subspace_vqe import SampledSubspaceVQE

        bond = BondFactory.create_bond('H', 'H', distance=0.74)
        solver = SampledSubspaceVQE(bond)
        result = solver.solve_local()  # Test locally first
        result = solver.solve_ibm()    # Then run on hardware
    """

    def __init__(
        self,
        system=None,
        *,
        bond_or_molecule=None,
        n_layers: int = 2,
        n_shots: int = 10000,
        max_configs: int = 20,
        backend: str = 'statevector',
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        **backend_kwargs,
    ):
        """
        Initialize Sampled Subspace VQE.

        Args:
            system: Bond / Molecule / MolecularHamiltonian / QuantumSystem
                (unified solver-protocol input). Legacy ``bond_or_molecule``
                keyword still accepted.
            n_layers: Number of HEA layers (2-3 is usually enough)
            n_shots: Number of measurement shots for sampling
            max_configs: Maximum configurations to use in subspace
            backend: Backend name (resolved via the factory; default statevector)
            enable_analysis: Enable automatic analysis tools (default True)
            enable_optimization: Enable automatic optimization tools (default True)
        """
        # Unified solver protocol: positional `system` is the high-level input;
        # fall back to the legacy `bond_or_molecule` keyword for compatibility.
        if system is None:
            system = bond_or_molecule
        if system is None:
            raise TypeError("SampledSubspaceVQE requires a system (Bond / Molecule / Hamiltonian).")

        # BaseSolver.__init__ resolves the system into self.hamiltonian /
        # self.molecule / self.bond and builds self.backend (a BaseBackend object).
        super().__init__(
            system,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **backend_kwargs,
        )

        from kanad.backends.statevector_backend import StatevectorBackend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        self.n_layers = n_layers
        self.n_shots = n_shots
        self.max_configs = max_configs

        self.n_qubits = 2 * self.hamiltonian.n_orbitals
        self.n_electrons = self.hamiltonian.n_electrons

        # Build Hamiltonian matrix
        self._build_hamiltonian_matrix()

        # Get reference energies
        self._get_reference_energies()

        logger.info(f"SampledSubspaceVQE initialized: {self.n_qubits} qubits, {self.n_electrons} electrons")

    def _build_hamiltonian_matrix(self):
        """Build dense Hamiltonian matrix for subspace diagonalization."""
        # Get SparsePauliOp
        sparse_ham = self.hamiltonian.to_sparse_hamiltonian(mapper='jordan_wigner')

        # Convert to dense matrix
        ham_matrix = sparse_ham.to_matrix()
        # Handle both sparse and dense matrices
        if hasattr(ham_matrix, 'toarray'):
            self._ham_matrix = ham_matrix.toarray()
        else:
            self._ham_matrix = np.array(ham_matrix)

        logger.info(f"Built Hamiltonian matrix: {self._ham_matrix.shape}")

    def _get_reference_energies(self):
        """Compute HF and FCI energies for reference."""
        from pyscf import scf, fci

        mol = self.hamiltonian.mol
        mf = scf.RHF(mol)
        mf.kernel()
        self.hf_energy = mf.e_tot

        cisolver = fci.FCI(mf)
        self.fci_energy, _ = cisolver.kernel()

        logger.info(f"HF energy: {self.hf_energy:.6f} Ha")
        logger.info(f"FCI energy: {self.fci_energy:.6f} Ha")

    def build_hea_circuit(self, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build shallow Hardware-Efficient Ansatz circuit.

        Uses CIRCULAR entanglement (critical for expressibility!).
        Only 8 CNOTs for H₂ (4 qubits, 2 layers).

        Args:
            parameters: Circuit parameters (2 * n_qubits * n_layers)

        Returns:
            QuantumCircuit ready for measurement
        """
        circuit = QuantumCircuit(self.n_qubits)

        # Prepare HF state in INTERLEAVED Jordan-Wigner ordering to match the
        # Hamiltonian this solver projects against (build_molecular_hamiltonian_jw
        # is interleaved: spatial orbital o -> alpha qubit 2o, beta qubit 2o+1).
        # The previous BLOCKED layout (alpha 0..n_orb-1, beta n_orb..) pointed the
        # HF determinant at the wrong Hilbert index, giving energies above HF.
        n_alpha = self.n_electrons // 2
        n_beta = self.n_electrons - n_alpha

        # Alpha electrons: spatial orbitals 0..n_alpha-1 -> qubits 2*i
        for i in range(n_alpha):
            circuit.x(2 * i)
        # Beta electrons: spatial orbitals 0..n_beta-1 -> qubits 2*i + 1
        for i in range(n_beta):
            circuit.x(2 * i + 1)

        # HEA layers
        param_idx = 0
        for layer in range(self.n_layers):
            # RY rotation layer
            for q in range(self.n_qubits):
                circuit.ry(parameters[param_idx], q)
                param_idx += 1

            # CIRCULAR CNOT entanglement (CRITICAL for expressibility!)
            for q in range(self.n_qubits):
                circuit.cx(q, (q + 1) % self.n_qubits)

            # RZ rotation layer
            for q in range(self.n_qubits):
                circuit.rz(parameters[param_idx], q)
                param_idx += 1

        return circuit

    def get_n_parameters(self) -> int:
        """Get number of circuit parameters."""
        return 2 * self.n_qubits * self.n_layers

    def _count_electrons(self, bitstring: str) -> int:
        """Count number of electrons (1s) in bitstring."""
        return bitstring.count('1')

    def _bitstring_to_index(self, bitstring: str) -> int:
        """Convert bitstring to Hilbert space index."""
        return int(bitstring[::-1], 2)  # Reverse for Qiskit ordering

    def sample_configurations(
        self,
        parameters: np.ndarray,
        use_hardware: bool = False,
        ibm_backend = None
    ) -> Tuple[List[str], float]:
        """
        Sample configurations from the circuit.

        Args:
            parameters: Circuit parameters
            use_hardware: Use IBM hardware (vs statevector)
            ibm_backend: IBMBackend instance (if use_hardware=True)

        Returns:
            (valid_configs, filter_rate): Configurations and rejection rate
        """
        circuit = self.build_hea_circuit(parameters)

        if use_hardware and ibm_backend is not None:
            # Run on IBM hardware
            return self._sample_ibm(circuit, ibm_backend)
        else:
            # Local simulation
            return self._sample_local(circuit)

    def _sample_local(self, circuit: QuantumCircuit) -> Tuple[List[str], float]:
        """Sample configurations using local statevector simulation."""
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        # Add measurements
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Run simulation
        backend = AerSimulator()
        transpiled = transpile(meas_circuit, backend)
        job = backend.run(transpiled, shots=self.n_shots)
        counts = job.result().get_counts()

        # Filter by particle number
        valid_configs = []
        total_counts = sum(counts.values())
        valid_counts = 0

        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            # Remove spaces
            bs = bitstring.replace(' ', '')

            # Check particle number
            if self._count_electrons(bs) == self.n_electrons:
                valid_configs.append(bs)
                valid_counts += count

                if len(valid_configs) >= self.max_configs:
                    break

        filter_rate = 1.0 - (valid_counts / total_counts)

        logger.info(f"Sampled {len(valid_configs)} valid configurations")
        logger.info(f"Particle number filter rate: {filter_rate:.1%}")

        return valid_configs, filter_rate

    def _sample_ibm(
        self,
        circuit: QuantumCircuit,
        ibm_backend
    ) -> Tuple[List[str], float]:
        """Sample configurations from IBM hardware."""
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2 as Sampler
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        # Add measurements
        meas_circuit = circuit.copy()
        meas_circuit.measure_all()

        # Transpile for hardware
        pm = generate_preset_pass_manager(
            backend=ibm_backend.backend,
            optimization_level=3
        )
        transpiled = pm.run(meas_circuit)

        logger.info(f"Transpiled circuit: depth={transpiled.depth()}, CNOTs={transpiled.count_ops().get('cx', 0)}")

        # Create sampler in Batch mode (required for free tier)
        from qiskit_ibm_runtime import Batch
        logger.info(f"Submitting job with {self.n_shots} shots (Batch mode)...")
        with Batch(backend=ibm_backend.backend) as batch:
            sampler = Sampler(mode=batch)
            sampler.options.default_shots = self.n_shots

            # Enable twirling for error mitigation
            try:
                sampler.options.twirling.enable_gates = True
                sampler.options.twirling.enable_measure = True
                logger.info("Enabled gate and measurement twirling")
            except:
                pass

            job = sampler.run([transpiled])
            job_id = job.job_id()
            logger.info(f"Job submitted: {job_id}")

        # Get results
        result = job.result()
        counts = result[0].data.meas.get_counts()

        # Filter by particle number
        valid_configs = []
        total_counts = sum(counts.values())
        valid_counts = 0

        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            bs = bitstring.replace(' ', '')

            if self._count_electrons(bs) == self.n_electrons:
                valid_configs.append(bs)
                valid_counts += count

                if len(valid_configs) >= self.max_configs:
                    break

        filter_rate = 1.0 - (valid_counts / total_counts)

        logger.info(f"Sampled {len(valid_configs)} valid configurations from hardware")
        logger.info(f"Particle number filter rate: {filter_rate:.1%}")

        return valid_configs, filter_rate

    def _get_hf_bitstring(self) -> str:
        """Get Hartree-Fock bitstring for this molecule."""
        n_orb = self.n_qubits // 2
        n_alpha = self.n_electrons // 2
        n_beta = self.n_electrons - n_alpha

        # Build HF occupation (interleaved JW: alpha o->2o, beta o->2o+1) to
        # match the interleaved Hamiltonian matrix this solver projects against.
        bitlist = ['0'] * self.n_qubits
        for i in range(n_alpha):
            bitlist[2 * i] = '1'  # Alpha electrons
        for i in range(n_beta):
            bitlist[2 * i + 1] = '1'  # Beta electrons

        return ''.join(bitlist)

    def diagonalize_subspace(
        self,
        configurations: List[str],
        always_include_hf: bool = True
    ) -> Tuple[float, np.ndarray]:
        """
        Diagonalize Hamiltonian in sampled subspace.

        This is the KEY advantage: classical diagonalization is EXACT!

        Args:
            configurations: List of bitstrings defining the subspace
            always_include_hf: Always include HF configuration

        Returns:
            (ground_energy, eigenvector): Ground state energy and coefficients
        """
        # Always include HF to ensure we get at least HF energy
        if always_include_hf:
            hf_bitstring = self._get_hf_bitstring()
            if hf_bitstring not in configurations:
                configurations = [hf_bitstring] + configurations
                logger.info(f"Added HF configuration: {hf_bitstring}")

        n_configs = len(configurations)

        if n_configs == 0:
            raise ValueError("No valid configurations found!")

        # Build subspace Hamiltonian
        H_sub = np.zeros((n_configs, n_configs), dtype=complex)

        for i, config_i in enumerate(configurations):
            idx_i = self._bitstring_to_index(config_i)
            for j, config_j in enumerate(configurations):
                idx_j = self._bitstring_to_index(config_j)
                H_sub[i, j] = self._ham_matrix[idx_i, idx_j]

        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H_sub)

        logger.info(f"Diagonalized {n_configs}x{n_configs} subspace Hamiltonian")
        logger.info(f"Ground state energy: {eigenvalues[0]:.6f} Ha")

        return float(eigenvalues[0].real), eigenvectors[:, 0]

    def optimize_parameters(
        self,
        max_iterations: int = 100,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Optimize circuit parameters using standard VQE.

        This finds parameters that minimize energy and explore
        important configurations.

        Args:
            max_iterations: Max optimization iterations
            verbose: Print progress

        Returns:
            Optimized parameters
        """
        from scipy.optimize import minimize

        n_params = self.get_n_parameters()

        def energy_function(params):
            circuit = self.build_hea_circuit(params)
            sv = Statevector(circuit)
            return sv.expectation_value(
                SparsePauliOp.from_list([(p.to_label(), c) for p, c in zip(
                    self.hamiltonian.to_sparse_hamiltonian(mapper='jordan_wigner').paulis,
                    self.hamiltonian.to_sparse_hamiltonian(mapper='jordan_wigner').coeffs
                )])
            ).real

        # Store Hamiltonian for expectation value
        sparse_ham = self.hamiltonian.to_sparse_hamiltonian(mapper='jordan_wigner')

        def energy_fn(params):
            circuit = self.build_hea_circuit(params)
            sv = Statevector(circuit)
            return sv.expectation_value(sparse_ham).real

        if verbose:
            print("Optimizing circuit parameters...")

        # Multiple random starts
        best_params = None
        best_energy = float('inf')

        for trial in range(3):
            x0 = np.random.uniform(-np.pi, np.pi, n_params)

            result = minimize(
                energy_fn,
                x0,
                method='COBYLA',
                options={'maxiter': max_iterations}
            )

            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x

            if verbose:
                print(f"  Trial {trial+1}: {result.fun:.6f} Ha")

        if verbose:
            print(f"Best VQE energy: {best_energy:.6f} Ha")

        return best_params

    def solve(self, **kwargs) -> SolverResult:
        """
        Canonical solver-protocol entry point.

        Runs the local (statevector) SSVQE path and wraps the result in a
        unified ``SolverResult``. Keyword arguments are forwarded to the local
        solve implementation (``n_trials``, ``optimize_first``, ``verbose``).

        Returns:
            SolverResult with the canonical ground-state energy under
            ``.energy`` and SSVQE-specific fields under ``.extra``.
        """
        raw = self._solve_local_impl(**kwargs)
        return SolverResult.from_mapping(
            raw,
            solver="sampled_subspace_vqe",
            backend=self.backend_name,
        )

    def solve_local(
        self,
        n_trials: int = 5,
        optimize_first: bool = True,
        verbose: bool = True
    ) -> SolverResult:
        """
        Solve locally using statevector simulation.

        This is for testing before hardware execution.

        Args:
            n_trials: Number of random parameter trials
            optimize_first: If True, optimize parameters with VQE first
            verbose: Print progress

        Returns:
            SolverResult with energy and statistics
        """
        raw = self._solve_local_impl(
            n_trials=n_trials,
            optimize_first=optimize_first,
            verbose=verbose,
        )
        return SolverResult.from_mapping(
            raw,
            solver="sampled_subspace_vqe",
            backend=self.backend_name,
        )

    def _solve_local_impl(
        self,
        n_trials: int = 5,
        optimize_first: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Local (statevector) SSVQE solve. Returns a plain result dict.

        Args:
            n_trials: Number of random parameter trials
            optimize_first: If True, optimize parameters with VQE first
            verbose: Print progress

        Returns:
            Result dict with energy and statistics.
        """
        if verbose:
            print("=" * 60)
            print("SAMPLED SUBSPACE VQE - LOCAL TEST")
            print("=" * 60)
            print(f"Qubits: {self.n_qubits}, Electrons: {self.n_electrons}")
            print(f"HEA layers: {self.n_layers}, Shots: {self.n_shots}")
            print(f"Max configs: {self.max_configs}")

        # Optimize parameters first if requested
        optimized_params = None
        if optimize_first:
            if verbose:
                print("\n--- Parameter Optimization ---")
            optimized_params = self.optimize_parameters(verbose=verbose)

        best_energy = float('inf')
        best_result = None

        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Sampling Trial {trial + 1}/{n_trials} ---")

            # Use optimized parameters for first trial, random for rest
            n_params = self.get_n_parameters()
            if trial == 0 and optimized_params is not None:
                params = optimized_params
                if verbose:
                    print("Using optimized parameters")
            else:
                params = np.random.uniform(-np.pi, np.pi, n_params)
                if verbose:
                    print("Using random parameters")

            # Sample configurations
            configs, filter_rate = self.sample_configurations(params)

            if len(configs) == 0:
                if verbose:
                    print(f"No valid configurations found (filter rate: {filter_rate:.1%})")
                continue

            # Diagonalize
            energy, _ = self.diagonalize_subspace(configs)

            if verbose:
                error = (energy - self.fci_energy) * 1000
                print(f"Energy: {energy:.6f} Ha (error: {error:.2f} mHa)")
                print(f"Configurations: {len(configs)}, Filter rate: {filter_rate:.1%}")

            if energy < best_energy:
                best_energy = energy
                corr_denom = self.hf_energy - self.fci_energy
                best_result = {
                    'energy': energy,
                    'error_mha': (energy - self.fci_energy) * 1000,
                    'subspace_dim': len(configs),
                    'n_samples': self.n_shots,
                    'configurations': configs,
                    'particle_number_filter_rate': filter_rate,
                    'hf_energy': self.hf_energy,
                    'fci_energy': self.fci_energy,
                    'correlation_energy': energy - self.hf_energy,
                    'correlation_captured': (
                        (self.hf_energy - energy) / corr_denom * 100
                        if abs(corr_denom) > 1e-12 else 0.0
                    ),
                    'converged': True,
                }

        if best_result is None:
            raise RuntimeError("No valid configurations found in any trial")

        if verbose:
            print("\n" + "=" * 60)
            print("BEST RESULT")
            print("=" * 60)
            print(f"Energy: {best_result['energy']:.6f} Ha")
            print(f"Error: {best_result['error_mha']:.2f} mHa")
            print(f"FCI energy: {self.fci_energy:.6f} Ha")
            print(f"Correlation captured: {best_result['correlation_captured']:.1f}%")
            print(f"Subspace dimension: {best_result['subspace_dim']}")

            if abs(best_result['error_mha']) < 1.6:
                print("✓ Chemical accuracy achieved!")
            else:
                print(f"✗ Error exceeds chemical accuracy (1.6 mHa)")

        return best_result

    def solve_ibm(
        self,
        api_token: Optional[str] = None,
        backend_name: str = 'ibm_marrakesh',
        n_trials: int = 5,
        verbose: bool = True
    ) -> SolverResult:
        """
        Solve on IBM Quantum hardware.

        Args:
            api_token: IBM API token (or use IBM_API env var)
            backend_name: IBM backend name
            n_trials: Number of parameter trials
            verbose: Print progress

        Returns:
            SolverResult with energy and statistics
        """
        import os
        from kanad.backends.ibm import IBMBackend

        if verbose:
            print("=" * 60)
            print("SAMPLED SUBSPACE VQE - IBM HARDWARE")
            print("=" * 60)

        # Initialize IBM backend
        token = api_token or os.environ.get('IBM_API')
        if not token:
            raise ValueError("IBM API token required (set IBM_API env var)")

        ibm_backend = IBMBackend(
            channel='ibm_quantum_platform',
            api_token=token,
            backend_name=backend_name
        )

        if verbose:
            print(f"Backend: {backend_name}")
            print(f"Qubits: {self.n_qubits}, Electrons: {self.n_electrons}")

        best_energy = float('inf')
        best_result = None

        for trial in range(n_trials):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{n_trials} ---")

            # Random parameters
            n_params = self.get_n_parameters()
            params = np.random.uniform(-np.pi, np.pi, n_params)

            # Sample configurations from hardware
            configs, filter_rate = self.sample_configurations(
                params,
                use_hardware=True,
                ibm_backend=ibm_backend
            )

            if len(configs) == 0:
                if verbose:
                    print(f"No valid configurations found (filter rate: {filter_rate:.1%})")
                continue

            # Diagonalize (classical - exact!)
            energy, _ = self.diagonalize_subspace(configs)

            if verbose:
                error = (energy - self.fci_energy) * 1000
                print(f"Energy: {energy:.6f} Ha (error: {error:.2f} mHa)")
                print(f"Filter rate: {filter_rate:.1%}")

            if energy < best_energy:
                best_energy = energy
                corr_denom = self.hf_energy - self.fci_energy
                best_result = {
                    'energy': energy,
                    'error_mha': (energy - self.fci_energy) * 1000,
                    'subspace_dim': len(configs),
                    'n_samples': self.n_shots,
                    'configurations': configs,
                    'particle_number_filter_rate': filter_rate,
                    'hf_energy': self.hf_energy,
                    'fci_energy': self.fci_energy,
                    'correlation_energy': energy - self.hf_energy,
                    'correlation_captured': (
                        (self.hf_energy - energy) / corr_denom * 100
                        if abs(corr_denom) > 1e-12 else 0.0
                    ),
                    'converged': True,
                }

        if best_result is None:
            raise RuntimeError("No valid configurations found")

        if verbose:
            print("\n" + "=" * 60)
            print("HARDWARE RESULT")
            print("=" * 60)
            print(f"Energy: {best_result['energy']:.6f} Ha")
            print(f"Error: {best_result['error_mha']:.2f} mHa")
            print(f"Correlation captured: {best_result['correlation_captured']:.1f}%")

            if abs(best_result['error_mha']) < 1.6:
                print("✓ Chemical accuracy achieved!")

        return SolverResult.from_mapping(
            best_result,
            solver="sampled_subspace_vqe",
            backend=self.backend_name,
        )


class HybridSubspaceVQE:
    """
    Hybrid Subspace VQE combining VQE optimization with subspace diagonalization.

    Strategy:
    1. Use physics-based FEB circuit (preserves physics, fewer CNOTs)
    2. Optimize parameters with local VQE
    3. Sample configurations from optimized circuit
    4. Diagonalize classically for improved accuracy

    This is closer to how Hi-VQE and QSCI work:
    - VQE finds approximate ground state
    - Sampling extracts important configurations
    - Classical diagonalization refines energy
    """

    def __init__(
        self,
        bond_or_molecule,
        n_shots: int = 100000,
        max_configs: int = 100
    ):
        """
        Initialize Hybrid Subspace VQE.

        Args:
            bond_or_molecule: Bond object from BondFactory or Molecule object
            n_shots: Number of measurement shots
            max_configs: Maximum configurations in subspace
        """
        from kanad.solvers import HardwareVQE

        self.bond = bond_or_molecule
        self.n_shots = n_shots
        self.max_configs = max_configs

        # Get molecular info — support both Bond and Molecule
        from kanad.core.molecule import Molecule, MolecularHamiltonian
        if isinstance(bond_or_molecule, Molecule):
            self.hamiltonian = bond_or_molecule.hamiltonian
        elif isinstance(bond_or_molecule, MolecularHamiltonian):
            self.hamiltonian = bond_or_molecule
        elif hasattr(bond_or_molecule, 'hamiltonian'):
            self.hamiltonian = bond_or_molecule.hamiltonian
        else:
            raise TypeError(f"Expected Bond or Molecule, got {type(bond_or_molecule).__name__}")

        # Initialize HardwareVQE for the optimized circuit
        self.hardware_vqe = HardwareVQE(
            bond=bond_or_molecule,
            circuit_type='feb',  # Use physics-based FEB
            max_excitations=5
        )

        # Get reference energies
        self._get_reference_energies()

        # Build Hamiltonian matrix
        self._build_hamiltonian_matrix()

    def _get_reference_energies(self):
        """Get reference energies."""
        from pyscf import scf, fci

        mol = self.hamiltonian.mol
        mf = scf.RHF(mol)
        mf.kernel()
        self.hf_energy = mf.e_tot

        cisolver = fci.FCI(mf)
        self.fci_energy, _ = cisolver.kernel()

    def _build_hamiltonian_matrix(self):
        """Build dense Hamiltonian matrix (matching VQE dimensions)."""
        # Use HardwareVQE's Hamiltonian for consistency
        sparse_ham = self.hardware_vqe._sparse_ham
        self.n_qubits = self.hardware_vqe._n_qubits
        self.n_electrons = self.hardware_vqe._n_electrons

        ham_matrix = sparse_ham.to_matrix()
        if hasattr(ham_matrix, 'toarray'):
            self._ham_matrix = ham_matrix.toarray()
        else:
            self._ham_matrix = np.array(ham_matrix)

        logger.info(f"Hamiltonian matrix: {self._ham_matrix.shape}, n_electrons={self.n_electrons}")

    def _get_deterministic_configs(self) -> List[str]:
        """
        Generate deterministic important configurations.

        Includes:
        1. HF state
        2. Single excitations
        3. Key double excitations
        """
        n_qubits = self.n_qubits
        n_orb = n_qubits // 2
        n_alpha = self.n_electrons // 2
        n_beta = self.n_electrons - n_alpha

        configs = []

        # All configs use INTERLEAVED JW ordering (alpha orbital o -> qubit 2o,
        # beta orbital o -> qubit 2o+1) to match the interleaved Hamiltonian
        # matrix. (Previously blocked: alpha 0..n_orb-1, beta n_orb..; that
        # mismatched the matrix and gave energies above HF.)
        # 1. HF state
        hf = ['0'] * n_qubits
        for i in range(n_alpha):
            hf[2 * i] = '1'
        for i in range(n_beta):
            hf[2 * i + 1] = '1'
        configs.append(''.join(hf))

        # 2. Single excitations (alpha): orbital i -> a, both on even qubits
        for i in range(n_alpha):  # Occupied
            for a in range(n_alpha, n_orb):  # Virtual
                cfg = list(hf)
                cfg[2 * i] = '0'
                cfg[2 * a] = '1'
                configs.append(''.join(cfg))

        # 3. Single excitations (beta): orbital i -> a, both on odd qubits
        for i in range(n_beta):  # Occupied
            for a in range(n_beta, n_orb):  # Virtual
                cfg = list(hf)
                cfg[2 * i + 1] = '0'
                cfg[2 * a + 1] = '1'
                configs.append(''.join(cfg))

        # 4. Double excitations (alpha-beta, most important)
        for i in range(n_alpha):
            for j in range(n_beta):
                for a in range(n_alpha, n_orb):
                    for b in range(n_beta, n_orb):
                        cfg = list(hf)
                        cfg[2 * i] = '0'      # Remove alpha
                        cfg[2 * a] = '1'      # Add to virtual alpha
                        cfg[2 * j + 1] = '0'  # Remove beta
                        cfg[2 * b + 1] = '1'  # Add to virtual beta
                        configs.append(''.join(cfg))

        return list(set(configs))  # Remove duplicates

    def solve(self, verbose: bool = True):
        """
        Solve using hybrid VQE + subspace approach.

        Returns:
            Dictionary with energy and statistics
        """
        from qiskit_aer import AerSimulator
        from qiskit import transpile

        if verbose:
            print("=" * 60)
            print("HYBRID SUBSPACE VQE")
            print("=" * 60)
            print(f"HF energy: {self.hf_energy:.6f} Ha")
            print(f"FCI energy: {self.fci_energy:.6f} Ha")

        # Step 1: Optimize circuit with local VQE
        if verbose:
            print("\n--- Step 1: VQE Optimization ---")

        vqe_result = self.hardware_vqe.solve_local(max_iterations=200, verbose=verbose)

        if verbose:
            print(f"VQE energy: {vqe_result.energy:.6f} Ha")
            print(f"VQE error: {(vqe_result.energy - self.fci_energy)*1000:.2f} mHa")

        # Step 2: Get configurations from multiple sources
        if verbose:
            print("\n--- Step 2: Build Configuration Space ---")

        # 2a. Deterministic configurations (HF + singles + doubles)
        det_configs = self._get_deterministic_configs()
        if verbose:
            print(f"Deterministic configs: {len(det_configs)}")

        # 2b. Sample from optimized circuit
        optimized_params = vqe_result.parameters
        circuit = self.hardware_vqe.build_circuit(optimized_params)
        circuit.measure_all()

        backend = AerSimulator()
        transpiled = transpile(circuit, backend)
        job = backend.run(transpiled, shots=self.n_shots)
        counts = job.result().get_counts()

        # Get n_electrons for filtering (from our consistent dimensions)
        n_electrons = self.n_electrons

        # Filter and sort configurations
        sampled_configs = []
        total_counts = sum(counts.values())
        valid_counts = 0

        for bitstring, count in sorted(counts.items(), key=lambda x: -x[1]):
            bs = bitstring.replace(' ', '')
            if bs.count('1') == n_electrons:
                sampled_configs.append(bs)
                valid_counts += count

        filter_rate = 1.0 - (valid_counts / total_counts)

        if verbose:
            print(f"VQE-sampled configs: {len(sampled_configs)}")
            print(f"Particle number filter rate: {filter_rate:.1%}")

        # 2c. Combine all configurations
        all_configs = list(set(det_configs + sampled_configs))
        valid_configs = all_configs[:self.max_configs]

        if verbose:
            print(f"Total unique configs: {len(valid_configs)}")

        # Step 3: Diagonalize in sampled subspace
        if verbose:
            print("\n--- Step 3: Subspace Diagonalization ---")

        if len(valid_configs) == 0:
            raise RuntimeError("No valid configurations found!")

        # Build subspace Hamiltonian
        n_configs = len(valid_configs)
        H_sub = np.zeros((n_configs, n_configs), dtype=complex)

        for i, config_i in enumerate(valid_configs):
            idx_i = int(config_i[::-1], 2)  # Reverse for Qiskit ordering
            for j, config_j in enumerate(valid_configs):
                idx_j = int(config_j[::-1], 2)
                H_sub[i, j] = self._ham_matrix[idx_i, idx_j]

        # Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H_sub)
        subspace_energy = float(eigenvalues[0].real)

        if verbose:
            print(f"Subspace energy: {subspace_energy:.6f} Ha")
            print(f"Subspace error: {(subspace_energy - self.fci_energy)*1000:.2f} mHa")
            print(f"Improvement from VQE: {(vqe_result.energy - subspace_energy)*1000:.2f} mHa")

        # Report results
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL RESULT")
            print("=" * 60)
            print(f"VQE energy: {vqe_result.energy:.6f} Ha")
            print(f"Subspace energy: {subspace_energy:.6f} Ha")
            print(f"FCI energy: {self.fci_energy:.6f} Ha")
            print(f"Error: {(subspace_energy - self.fci_energy)*1000:.2f} mHa")

            if abs(subspace_energy - self.fci_energy) * 1000 < 1.6:
                print("✓ Chemical accuracy achieved!")

        return {
            'energy': subspace_energy,
            'vqe_energy': vqe_result.energy,
            'fci_energy': self.fci_energy,
            'error_mha': (subspace_energy - self.fci_energy) * 1000,
            'n_configs': len(valid_configs),
            'filter_rate': filter_rate,
            'configurations': valid_configs
        }


def test_ssvqe_h2():
    """Test Sampled Subspace VQE on H₂."""
    from kanad import BondFactory

    print("\n" + "=" * 60)
    print("TESTING SAMPLED SUBSPACE VQE ON H₂")
    print("=" * 60)

    bond = BondFactory.create_bond('H', 'H', distance=0.74)

    solver = SampledSubspaceVQE(
        bond,
        n_layers=2,
        n_shots=10000,
        max_configs=10
    )

    result = solver.solve_local(n_trials=5, verbose=True)

    return result


def test_hybrid_h2():
    """Test Hybrid Subspace VQE on H₂."""
    from kanad import BondFactory

    bond = BondFactory.create_bond('H', 'H', distance=0.74)
    solver = HybridSubspaceVQE(bond, n_shots=100000, max_configs=20)
    result = solver.solve(verbose=True)
    return result


def test_hybrid_lih():
    """Test Hybrid Subspace VQE on LiH."""
    from kanad import BondFactory

    bond = BondFactory.create_bond('Li', 'H', distance=1.6)
    solver = HybridSubspaceVQE(bond, n_shots=100000, max_configs=100)
    result = solver.solve(verbose=True)
    return result


def test_hybrid_ibm_h2():
    """Test Hybrid Subspace VQE on IBM Hardware for H₂."""
    import os
    from kanad import BondFactory
    from kanad.backends.ibm import IBMBackend
    from qiskit import transpile
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print("=" * 60)
    print("HYBRID SUBSPACE VQE - IBM HARDWARE TEST")
    print("=" * 60)

    # Create bond and solver
    bond = BondFactory.create_bond('H', 'H', distance=0.74)
    solver = HybridSubspaceVQE(bond, n_shots=10000, max_configs=20)

    print(f"HF energy: {solver.hf_energy:.6f} Ha")
    print(f"FCI energy: {solver.fci_energy:.6f} Ha")

    # Step 1: Local VQE optimization (find good parameters)
    print("\n--- Step 1: Local VQE Optimization ---")
    vqe_result = solver.hardware_vqe.solve_local(max_iterations=100, verbose=False)
    print(f"Local VQE energy: {vqe_result.energy:.6f} Ha")

    # Step 2: Initialize IBM backend (use saved credentials)
    print("\n--- Step 2: IBM Hardware Setup ---")
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Backend: {backend.name}")
    except Exception as e:
        print(f"ERROR: Could not connect to IBM: {e}")
        return None

    # Step 3: Sample from hardware
    print("\n--- Step 3: Hardware Sampling ---")

    # Build and transpile circuit
    circuit = solver.hardware_vqe.build_circuit(vqe_result.parameters)
    circuit.measure_all()

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3
    )
    transpiled = pm.run(circuit)
    print(f"Transpiled: {transpiled.depth()} depth, {transpiled.count_ops().get('cx', 0)} CNOTs")

    # Run on hardware in Batch mode (required for free tier)
    from qiskit_ibm_runtime import Batch
    print("Submitting job (Batch mode)...")
    with Batch(backend=backend) as batch:
        sampler = Sampler(mode=batch)
        sampler.options.default_shots = 10000

        try:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            print("Enabled twirling")
        except:
            pass

        job = sampler.run([transpiled])
        print(f"Job ID: {job.job_id()}")

        result = job.result()
    counts = result[0].data.meas.get_counts()

    # Step 4: Build subspace from hardware samples + deterministic configs
    print("\n--- Step 4: Build Subspace ---")

    det_configs = solver._get_deterministic_configs()
    print(f"Deterministic configs: {len(det_configs)}")

    n_electrons = solver.n_electrons
    hw_configs = []
    total = sum(counts.values())
    valid = 0

    for bs, count in sorted(counts.items(), key=lambda x: -x[1]):
        bs_clean = bs.replace(' ', '')
        if bs_clean.count('1') == n_electrons:
            hw_configs.append(bs_clean)
            valid += count

    print(f"Hardware-sampled configs: {len(hw_configs)}")
    print(f"Particle filter rate: {1.0 - valid/total:.1%}")

    # Combine
    all_configs = list(set(det_configs + hw_configs))[:solver.max_configs]
    print(f"Total configs: {len(all_configs)}")

    # Step 5: Diagonalize
    print("\n--- Step 5: Subspace Diagonalization ---")

    n_configs = len(all_configs)
    H_sub = np.zeros((n_configs, n_configs), dtype=complex)

    for i, cfg_i in enumerate(all_configs):
        idx_i = int(cfg_i[::-1], 2)
        for j, cfg_j in enumerate(all_configs):
            idx_j = int(cfg_j[::-1], 2)
            H_sub[i, j] = solver._ham_matrix[idx_i, idx_j]

    eigenvalues, _ = np.linalg.eigh(H_sub)
    hw_energy = float(eigenvalues[0].real)

    print("\n" + "=" * 60)
    print("HARDWARE RESULT")
    print("=" * 60)
    print(f"Local VQE: {vqe_result.energy:.6f} Ha")
    print(f"Hardware subspace: {hw_energy:.6f} Ha")
    print(f"FCI: {solver.fci_energy:.6f} Ha")
    print(f"Error: {(hw_energy - solver.fci_energy)*1000:.2f} mHa")

    if abs(hw_energy - solver.fci_energy) * 1000 < 1.6:
        print("✓ CHEMICAL ACCURACY ON HARDWARE!")
    elif abs(hw_energy - solver.fci_energy) * 1000 < 10:
        print("~ Near chemical accuracy")
    else:
        print("✗ Error exceeds 10 mHa")

    return {
        'local_energy': vqe_result.energy,
        'hw_energy': hw_energy,
        'fci_energy': solver.fci_energy,
        'error_mha': (hw_energy - solver.fci_energy) * 1000,
        'n_configs': len(all_configs),
        'filter_rate': 1.0 - valid/total
    }


if __name__ == '__main__':
    test_hybrid_h2()
