"""
Quantum Equation-of-Motion VQE (qEOM-VQE)

Computes excited states using the equation-of-motion formalism on top of
VQE ground state. This gives TRUE quantum excited states, unlike penalty-based
VQE which finds orthogonal states that may not be physical excited states.

Theory:
------
1. Ground state |ψ₀⟩ from VQE with parameters θ*
2. Excitation operators E_μ = a†_a a_i (particle-hole)
3. Build EOM matrices:
   - H_μν = ⟨ψ₀|E†_μ H E_ν|ψ₀⟩  (Hamiltonian in excitation basis)
   - S_μν = ⟨ψ₀|E†_μ E_ν|ψ₀⟩     (Overlap matrix)
4. Solve generalized eigenvalue: H·c = ω·S·c
5. Excited energies: E_k = E_0 + ω_k

Advantages over penalty-based VQE:
---------------------------------
- Gives correct excitation energies (not just orthogonal states)
- Single VQE optimization (ground state only)
- Linear response theory connection
- Systematic improvement with more excitation operators

References:
----------
1. Ollitrault et al. (2020) Chem. Sci. 11, 6842 - qEOM-VQE
2. McClean et al. (2017) New J. Phys. 19, 023023 - Theory of VQE
3. Stanton & Bartlett (1993) J. Chem. Phys. 98, 7029 - EOM-CC theory
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from scipy.linalg import eigh

from kanad.solvers.base_solver import BaseSolver
from kanad.core.solver_result import SolverResult

logger = logging.getLogger(__name__)


class qEOMVQE(BaseSolver):
    """
    Quantum Equation-of-Motion VQE for excited states.

    This computes TRUE quantum excited states by building the EOM matrix
    on top of the VQE ground state and diagonalizing classically.

    Example:
    -------
    >>> from kanad import BondFactory
    >>> from kanad.solvers import qEOMVQE

    >>> bond = BondFactory.create_bond('H', 'H', distance=0.74)
    >>> solver = qEOMVQE(bond, n_states=3)
    >>> result = solver.solve()

    >>> print(f"Ground state: {result.ground_energy:.6f} Ha")
    >>> for i, omega in enumerate(result.excitation_energies):
    ...     print(f"Excitation {i+1}: {omega:.2f} eV")
    """

    def __init__(
        self,
        system=None,
        *,
        bond_or_molecule=None,
        n_states: int = 3,
        include_singles: bool = True,
        include_doubles: bool = True,
        backend: str = 'statevector',
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        vqe_max_iterations: int = 500,
        **backend_kwargs,
    ):
        """
        Initialize qEOM-VQE solver (unified solver protocol).

        Args:
            system: Bond / Molecule / Hamiltonian / QuantumSystem exposing
                ``.hamiltonian`` (canonical first positional argument).
            bond_or_molecule: Legacy alias for ``system``.
            n_states: Number of excited states to compute
            include_singles: Include single excitations (a†_a a_i)
            include_doubles: Include double excitations (a†_a a†_b a_j a_i)
            backend: Quantum backend name for VQE
            enable_analysis: Enable automatic analysis (default: True)
            enable_optimization: Enable automatic optimization (default: True)
            vqe_max_iterations: Max VQE iterations for ground state
            **backend_kwargs: Backend construction params (device, shots, ...).
        """
        # Accept the legacy first positional name as an alias for ``system``.
        if system is None and bond_or_molecule is not None:
            system = bond_or_molecule

        # BaseSolver resolves self.hamiltonian / self.molecule / self.bond and
        # builds self.backend (a BaseBackend object); self.backend_name is the string.
        super().__init__(
            system,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **backend_kwargs,
        )

        from kanad.backends.statevector_backend import StatevectorBackend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        self.n_states = n_states
        self.include_singles = include_singles
        self.include_doubles = include_doubles
        self.vqe_max_iterations = vqe_max_iterations

        # Will be computed
        self._ground_energy = None
        self._ground_params = None
        self._ground_state = None  # Statevector
        self._n_qubits = None
        self._n_electrons = None
        self._hamiltonian = None

        logger.info(f"qEOM-VQE initialized")
        logger.info(f"  States to compute: {n_states}")
        logger.info(f"  Singles: {include_singles}, Doubles: {include_doubles}")

    def _run_ground_state_vqe(self, callback=None):
        """Run VQE to get ground state."""
        from kanad.solvers import PhysicsVQE

        logger.info(f"Computing VQE ground state (backend={self.backend_name})...")

        # Use PhysicsVQE for accurate ground state
        vqe_kwargs = {'max_excitations': 10}
        if self.backend_name and self.backend_name != 'statevector':
            vqe_kwargs['backend'] = self.backend_name

        if self.molecule is not None and self.bond is None:
            # Multi-atom builder QuantumSystem / Molecule
            solver = PhysicsVQE(molecule=self.molecule, **vqe_kwargs)
        elif self.bond is not None and hasattr(self.bond, 'atom_1'):
            # It's a Bond
            solver = PhysicsVQE(bond=self.bond, **vqe_kwargs)
        elif self.bond is not None:
            # Builder QuantumSystem exposing .hamiltonian but not a 2-atom Bond
            solver = PhysicsVQE(molecule=self.bond, **vqe_kwargs)
        else:
            # Bare Hamiltonian
            solver = PhysicsVQE(hamiltonian=self.hamiltonian, **vqe_kwargs)

        # PhysicsVQE returns a SolverResult; ``parameters`` lives in .extra.
        result = solver.solve(callback=callback)
        params = result.extra['parameters']

        self._ground_energy = result.energy
        self._ground_params = params
        self._n_qubits = solver._n_qubits
        self._n_electrons = solver._n_electrons
        self._hamiltonian = solver._sparse_ham  # SparsePauliOp
        self._vqe_solver = solver  # Keep reference for circuit building

        # Get ground state vector
        self._ground_state = self._get_statevector(solver, params)

        logger.info(f"  Ground state energy: {self._ground_energy:.6f} Ha")
        logger.info(f"  Qubits: {self._n_qubits}, Electrons: {self._n_electrons}")

    def _get_statevector(self, solver, params) -> np.ndarray:
        """Get statevector from optimized circuit."""
        from qiskit.quantum_info import Statevector

        # Build circuit with optimized parameters
        circuit = solver.build_circuit(params)

        # Get statevector
        sv = Statevector(circuit)
        return sv.data

    def _generate_excitation_operators(self) -> List[Tuple[str, List[int], List[int]]]:
        """
        Generate excitation operators for EOM.

        Returns list of (type, occupied_indices, virtual_indices) tuples.
        Single: ('S', [i], [a]) represents a†_a a_i
        Double: ('D', [i,j], [a,b]) represents a†_a a†_b a_j a_i
        """
        excitations = []

        n_occ = self._n_electrons // 2  # Spatial occupied orbitals
        n_virt = self._n_qubits // 2 - n_occ  # Spatial virtual orbitals

        logger.debug(f"Generating excitations: {n_occ} occupied, {n_virt} virtual orbitals")

        # Single excitations: i -> a
        if self.include_singles:
            for i in range(n_occ):
                for a in range(n_occ, n_occ + n_virt):
                    # Alpha spin
                    excitations.append(('S', [2*i], [2*a]))
                    # Beta spin
                    excitations.append(('S', [2*i+1], [2*a+1]))

        # Double excitations: ij -> ab
        if self.include_doubles:
            for i in range(n_occ):
                for j in range(i, n_occ):
                    for a in range(n_occ, n_occ + n_virt):
                        for b in range(a, n_occ + n_virt):
                            if i == j and a == b:
                                continue  # Skip trivial
                            # Same-spin doubles
                            excitations.append(('D', [2*i, 2*j], [2*a, 2*b]))
                            excitations.append(('D', [2*i+1, 2*j+1], [2*a+1, 2*b+1]))
                            # Opposite-spin doubles
                            if i != j or a != b:
                                excitations.append(('D', [2*i, 2*j+1], [2*a, 2*b+1]))

        logger.info(f"Generated {len(excitations)} excitation operators")
        return excitations

    def _build_excitation_operator(
        self,
        exc_type: str,
        occ_idx: List[int],
        virt_idx: List[int]
    ) -> np.ndarray:
        """
        Build excitation operator matrix in the full Hilbert space.

        E = a†_a a_i (single) or a†_a a†_b a_j a_i (double)
        """
        dim = 2 ** self._n_qubits
        E = np.zeros((dim, dim), dtype=complex)

        if exc_type == 'S':
            # Single excitation: a†_a a_i
            i = occ_idx[0]
            a = virt_idx[0]
            E = self._creation_annihilation_matrix(a, i)

        elif exc_type == 'D':
            # Double excitation: a†_a a†_b a_j a_i
            i, j = occ_idx
            a, b = virt_idx
            # Build as product of operators
            E = self._double_excitation_matrix(i, j, a, b)

        return E

    def _creation_annihilation_matrix(self, create_idx: int, annihilate_idx: int) -> np.ndarray:
        """Build matrix for a†_create a_annihilate in occupation number basis."""
        dim = 2 ** self._n_qubits
        E = np.zeros((dim, dim), dtype=complex)

        for state in range(dim):
            # Check if annihilate_idx is occupied
            if not (state & (1 << annihilate_idx)):
                continue  # Orbital not occupied, a_i|state⟩ = 0

            # Check if create_idx is empty
            if state & (1 << create_idx):
                continue  # Orbital occupied, a†_a|state⟩ = 0

            # Apply operators
            new_state = state ^ (1 << annihilate_idx)  # Remove electron from i
            new_state = new_state ^ (1 << create_idx)   # Add electron to a

            # Fermionic sign from Jordan-Wigner
            sign = self._jw_sign(state, annihilate_idx, create_idx)

            E[new_state, state] = sign

        return E

    def _double_excitation_matrix(self, i: int, j: int, a: int, b: int) -> np.ndarray:
        """Build matrix for a†_a a†_b a_j a_i."""
        dim = 2 ** self._n_qubits
        E = np.zeros((dim, dim), dtype=complex)

        for state in range(dim):
            # Check occupations
            if not (state & (1 << i)):
                continue  # i not occupied
            if not (state & (1 << j)):
                continue  # j not occupied
            if state & (1 << a):
                continue  # a already occupied
            if state & (1 << b):
                continue  # b already occupied

            # Apply operators: a†_a a†_b a_j a_i
            new_state = state
            sign = 1

            # a_i
            sign *= self._count_parity(new_state, i)
            new_state ^= (1 << i)

            # a_j
            sign *= self._count_parity(new_state, j)
            new_state ^= (1 << j)

            # a†_b
            sign *= self._count_parity(new_state, b)
            new_state ^= (1 << b)

            # a†_a
            sign *= self._count_parity(new_state, a)
            new_state ^= (1 << a)

            E[new_state, state] = sign

        return E

    def _jw_sign(self, state: int, annihilate_idx: int, create_idx: int) -> int:
        """Compute Jordan-Wigner sign for single excitation."""
        sign = 1

        # Sign from annihilation (count occupied below annihilate_idx)
        for k in range(annihilate_idx):
            if state & (1 << k):
                sign *= -1

        # Intermediate state after annihilation
        intermediate = state ^ (1 << annihilate_idx)

        # Sign from creation (count occupied below create_idx)
        for k in range(create_idx):
            if intermediate & (1 << k):
                sign *= -1

        return sign

    def _count_parity(self, state: int, idx: int) -> int:
        """Count parity of occupied orbitals below idx."""
        count = 0
        for k in range(idx):
            if state & (1 << k):
                count += 1
        return (-1) ** count

    def _build_hamiltonian_matrix(self) -> np.ndarray:
        """Build full Hamiltonian matrix from sparse representation."""
        from qiskit.quantum_info import SparsePauliOp, Operator

        # Convert sparse Pauli to dense matrix
        H_op = Operator(self._hamiltonian)
        return H_op.data

    def _compute_eom_matrices(
        self,
        excitations: List[Tuple[str, List[int], List[int]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute EOM H and S matrices.

        For excitation energies, we use the shifted Hamiltonian:
        H'_μν = ⟨ψ₀|E†_μ H E_ν|ψ₀⟩ - E_0 * S_μν

        This gives eigenvalues ω_k directly as excitation energies.

        S_μν = ⟨ψ₀|E†_μ E_ν|ψ₀⟩
        """
        n_exc = len(excitations)
        H_matrix = np.zeros((n_exc, n_exc), dtype=complex)
        S_matrix = np.zeros((n_exc, n_exc), dtype=complex)

        logger.info(f"Building {n_exc}x{n_exc} EOM matrices...")

        # Build full Hamiltonian matrix
        H_full = self._build_hamiltonian_matrix()

        # Ground state vector
        psi0 = self._ground_state
        E0 = self._ground_energy

        # Build excitation operator matrices
        E_ops = []
        for exc_type, occ_idx, virt_idx in excitations:
            E = self._build_excitation_operator(exc_type, occ_idx, virt_idx)
            E_ops.append(E)

        # Compute matrix elements
        for mu in range(n_exc):
            E_mu = E_ops[mu]
            E_mu_dag = E_mu.conj().T

            for nu in range(mu, n_exc):
                E_nu = E_ops[nu]

                # S_μν = ⟨ψ₀|E†_μ E_ν|ψ₀⟩
                temp = E_nu @ psi0
                temp = E_mu_dag @ temp
                S_matrix[mu, nu] = np.vdot(psi0, temp)

                # H_μν = ⟨ψ₀|E†_μ H E_ν|ψ₀⟩ - E_0 * S_μν
                # This gives excitation energies directly
                temp = E_nu @ psi0
                temp = H_full @ temp
                temp = E_mu_dag @ temp
                H_matrix[mu, nu] = np.vdot(psi0, temp) - E0 * S_matrix[mu, nu]

                # Hermitian symmetry
                if mu != nu:
                    H_matrix[nu, mu] = np.conj(H_matrix[mu, nu])
                    S_matrix[nu, mu] = np.conj(S_matrix[mu, nu])

        return H_matrix.real, S_matrix.real

    def solve(self, callback=None, **kwargs) -> SolverResult:
        """
        Solve for excited states using qEOM-VQE.

        Args:
            callback: Optional progress callback forwarded to the inner
                      PhysicsVQE ground-state solve. Invoked once per energy
                      evaluation as callback(iteration, energy, parameters);
                      lets the API layer stream the ground-state convergence
                      curve. The subsequent EOM-matrix step is non-iterative.

        Returns:
            SolverResult with ground energy (canonical ``energy``) and excited
            state absolute energies under ``states``; the qEOM-specific matrices
            (H, S), eigenvectors, and excitation energies live in ``extra``.
        """
        logger.info("="*60)
        logger.info("qEOM-VQE: Quantum Equation of Motion")
        logger.info("="*60)

        # Step 1: Get ground state from VQE
        self._run_ground_state_vqe(callback=callback)

        # Step 2: Generate excitation operators
        excitations = self._generate_excitation_operators()
        n_exc = len(excitations)

        if n_exc == 0:
            logger.warning("No excitation operators generated!")
            raw = {
                'ground_energy': self._ground_energy,
                'converged': True,
                'states': [],
                'excited_energies': np.array([]),
                'excitation_energies': np.array([]),
                'eigenvectors': np.array([]),
                'h_matrix': np.array([]),
                's_matrix': np.array([]),
                'n_excitations': 0,
                'method': 'qeom-vqe',
            }
            result = SolverResult.from_mapping(
                raw, solver='qeom_vqe', backend=self.backend_name,
                energy_key='ground_energy',
            )
            self._last_result = result
            return result

        # Step 3: Build EOM matrices
        H_matrix, S_matrix = self._compute_eom_matrices(excitations)

        # Step 4: Solve generalized eigenvalue problem
        # H·c = ω·S·c
        # Need to regularize S if singular
        S_reg = S_matrix + 1e-10 * np.eye(n_exc)

        try:
            eigenvalues, eigenvectors = eigh(H_matrix, S_reg)
        except np.linalg.LinAlgError:
            logger.warning("Eigenvalue problem failed, using pseudo-inverse")
            S_pinv = np.linalg.pinv(S_matrix)
            M = S_pinv @ H_matrix
            eigenvalues, eigenvectors = np.linalg.eigh(M)

        # Sort by energy (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Filter to positive excitation energies (physical excited states)
        # Use a small threshold to avoid numerical noise
        positive_mask = eigenvalues > 0.001  # ~0.03 eV threshold
        excitation_energies_ha = eigenvalues[positive_mask]
        eigenvectors_filtered = eigenvectors[:, positive_mask]

        # Take requested number of states (excluding ground)
        n_found = min(self.n_states - 1, len(excitation_energies_ha))
        excitation_energies_ha = excitation_energies_ha[:n_found]
        eigenvectors = eigenvectors_filtered[:, :n_found] if n_found > 0 else np.array([])

        # Convert to eV
        HA_TO_EV = 27.2114
        excitation_energies_ev = excitation_energies_ha * HA_TO_EV

        # Excited state absolute energies
        excited_energies = self._ground_energy + excitation_energies_ha

        # Log results
        logger.info(f"\nqEOM-VQE Results:")
        logger.info(f"  Ground state: {self._ground_energy:.6f} Ha")
        for i, (e_ha, e_ev) in enumerate(zip(excitation_energies_ha, excitation_energies_ev)):
            logger.info(f"  Excitation {i+1}: {e_ev:.2f} eV ({e_ha:.4f} Ha)")
        logger.info("="*60)

        raw = {
            'ground_energy': self._ground_energy,
            'converged': True,
            'states': list(excited_energies),  # excited-state absolute energies (Ha)
            'excited_energies': excited_energies,
            'excitation_energies': excitation_energies_ev,
            'eigenvectors': eigenvectors,
            'h_matrix': H_matrix,
            's_matrix': S_matrix,
            'n_excitations': n_exc,
            'method': 'qeom-vqe',
        }
        result = SolverResult.from_mapping(
            raw, solver='qeom_vqe', backend=self.backend_name,
            energy_key='ground_energy',
        )
        self._last_result = result
        return result

    # ── ExcitedStatesProvider capability (Stage 2) ──────────────────────────
    def solve_excited_states(self, n_states, *, spin=None, warm_state=None):
        """Compute ``n_states`` lowest states (ground + excited). Protocol wrapper."""
        if spin is not None:
            raise NotImplementedError("qEOMVQE: spin-targeted excited states not implemented")
        self.n_states = int(n_states)
        return self.solve()

    def get_excited_state_data(self):
        """Normalized excited-state payload (capability ``"excited_states"``).

        qEOM ground energy is ``result.energy``; ``result.states`` are the excited
        ABSOLUTE energies (Ha). qEOM computes no transition properties, so oscillator
        strengths / transition dipoles stay None (honesty — do not fabricate zeros).
        """
        from kanad.solvers.capabilities import ExcitedStateData
        if not self.has_capability('excited_states'):
            raise NotImplementedError(
                f"{type(self).__name__} does not declare the 'excited_states' capability"
            )
        r = getattr(self, '_last_result', None)
        if r is None:
            r = self.solve()
        ground = float(r.energy)
        excited_abs = [float(x) for x in (r.states or [])]
        state_energies_ha = np.array([ground] + excited_abs, dtype=float)
        order = np.argsort(state_energies_ha)
        state_energies_ha = state_energies_ha[order]
        HA_TO_EV = 27.2114
        excitation_energies_ev = (state_energies_ha[1:] - state_energies_ha[0]) * HA_TO_EV
        eig = r.extra.get('eigenvectors') if isinstance(r.extra, dict) else None
        eigvec_list = None
        if eig is not None:
            eig = np.asarray(eig)
            if eig.size > 0 and eig.ndim == 2:
                eigvec_list = [eig[:, i] for i in range(eig.shape[1])]
        return ExcitedStateData(
            state_energies_ha=state_energies_ha,
            excitation_energies_ev=excitation_energies_ev,
            oscillator_strengths=None, transition_dipoles=None,
            eigenvectors=eigvec_list, spin_multiplicities=None,
        )


def create_qeom_solver(bond_or_molecule, n_states: int = 3, **kwargs) -> qEOMVQE:
    """Factory function to create qEOM-VQE solver."""
    return qEOMVQE(bond_or_molecule, n_states=n_states, **kwargs)


# ``qEOMResult`` was a @dataclass return type; solvers now return ``SolverResult``
# (solver-protocol refactor, 2026-06-12). Alias kept so existing imports of the
# name resolve to the unified result type.
qEOMResult = SolverResult
