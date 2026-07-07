"""
Excited States Solver - Bonds Module Integration.

Computes excited electronic states for molecular systems.
Integrated with analysis tools for spectroscopy and photochemistry.
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import logging

from kanad.solvers.base_solver import BaseSolver

logger = logging.getLogger(__name__)


class ExcitedStatesSolver(BaseSolver):
    """
    Solver for molecular excited states.

    Methods:
    - CIS (Configuration Interaction Singles): Fast, approximate
    - TDDFT (Time-Dependent DFT): Accurate for many systems
    - EOM-CCSD (Equation-of-Motion Coupled Cluster): High accuracy
    - Quantum methods (QPE, VQE with state-averaged ansatz)

    Usage:
        from kanad.bonds import BondFactory
        from kanad.solvers import ExcitedStatesSolver

        bond = BondFactory.create_bond('H', 'H', distance=0.74)
        solver = ExcitedStatesSolver(bond, method='cis', n_states=5)
        result = solver.solve()

        print(f"Ground State: {result['energies'][0]:.6f} Ha")
        for i, E in enumerate(result['excitation_energies'], 1):
            print(f"Excitation {i}: {E:.4f} eV")
    """

    def __init__(
        self,
        system=None,
        *,
        method: str = 'cis',
        n_states: int = 5,
        backend: str = 'statevector',
        enable_analysis: bool = True,
        enable_optimization: bool = False,
        experiment_id: Optional[str] = None,
        vqe_callback: Optional[Callable] = None,  # NEW: callback for VQE progress
        **kwargs
    ):
        """
        Initialize excited states solver (unified solver protocol).

        Args:
            system: Bond (from BondFactory), Molecule, MolecularHamiltonian, or any
                object exposing a ``.hamiltonian`` (e.g. a builder QuantumSystem).
                ``bond=`` / ``molecule=`` are accepted as aliases.
            method: Excited state method ('cis', 'tddft', 'qpe', 'vqe', 'sqd')
            n_states: Number of excited states to compute
            backend: Backend name resolved via the unified backend factory
                (``statevector`` is the only path the classical CIS route needs).
            enable_analysis: Enable spectroscopy analysis
            enable_optimization: Enable geometry optimization of excited states
            experiment_id: Experiment ID for WebSocket broadcasting (optional)
            vqe_callback: Optional callback function for VQE progress (iteration, energy, parameters)
            **kwargs: Method-specific options (subspace_dim, circuit_depth for SQD)
        """
        # Accept bond=/molecule= as aliases for the positional system
        # (several callers — spectroscopy, dos, nonadiabatic, quantum_nac — pass bond=,
        # which previously fell into **kwargs and left the system arg unset -> TypeError).
        system = system or kwargs.pop('bond', None) or kwargs.pop('molecule', None)
        if system is None:
            raise TypeError("ExcitedStatesSolver requires a bond/molecule "
                            "(positional, or bond=/molecule=)")

        # Pull solver-specific kwargs out before forwarding the rest to the
        # backend factory (so e.g. subspace_dim never reaches a cloud backend).
        self._ansatz = kwargs.pop('ansatz', 'hardware_efficient')  # 'governance'/'uccsd' were removed
        self._optimizer = kwargs.pop('optimizer', 'COBYLA')
        self._max_iterations = kwargs.pop('max_iterations', 100)  # Should be set from frontend, not hardcoded
        self._penalty_weight = kwargs.pop('penalty_weight', 1.0)
        # Backend-specific kwargs (API tokens, etc.) passed through to sub-solvers.
        self._backend_kwargs = kwargs.pop('backend_kwargs', {})
        # SQD-specific kwargs
        self._subspace_dim = kwargs.pop('subspace_dim', 10)
        self._circuit_depth = kwargs.pop('circuit_depth', 3)

        super().__init__(
            system,
            backend=backend,
            enable_analysis=enable_analysis,
            enable_optimization=enable_optimization,
            **kwargs,
        )

        self.method = method.lower()
        self.n_states = n_states
        self.experiment_id = experiment_id  # Store for VQE broadcasting
        self.vqe_callback = vqe_callback  # Store callback for VQE progress

        # Not a correlation method for ground state
        self._is_correlated = False

        # The backend *object* lives on self.backend (set by BaseSolver.__init__);
        # self.backend_name is the string. The VQE/SQD sub-solver entry points still
        # take a backend *name* string, so keep `_backend` as that string.
        self._backend = self.backend_name
        from kanad.backends.statevector_backend import StatevectorBackend
        self._use_statevector = isinstance(self.backend, StatevectorBackend)

        # Initialize spectroscopy analyzer if analysis enabled
        if enable_analysis:
            from kanad.analysis import UVVisCalculator
            try:
                self.uvvis_calculator = UVVisCalculator(self.molecule)
            except (AttributeError, TypeError):
                logger.debug("UVVisCalculator initialization skipped")
                self.uvvis_calculator = None

        logger.info(f"Excited States Solver initialized: {method}, {n_states} states")

    def solve(self, **kwargs) -> 'SolverResult':
        """Solve for excited states and return a unified :class:`SolverResult`.

        The canonical ``result.energy`` is the ground-state energy; the
        excited-state energies (Hartree) are surfaced under ``result.states``.
        The full legacy result dict is preserved on ``self.results`` and merged
        into the result's ``extra`` (excitation_energies, oscillator_strengths,
        transition_dipoles, eigenvectors, ...), so legacy consumers keep working
        via ``result.to_dict()``.
        """
        from kanad.core.solver_result import SolverResult
        raw = self._solve_raw()
        # Surface excited-state energies (Hartree) under the canonical "states" key.
        if 'states' not in raw:
            raw['states'] = list(raw.get('excitation_energies_ha', []))
        return SolverResult.from_mapping(raw, solver="excited_states",
                                         backend=self.backend_name)

    # ── ExcitedStatesProvider capability (Stage 2) ──────────────────────────
    def solve_excited_states(self, n_states, *, spin=None, warm_state=None):
        """Compute ``n_states`` lowest states (ground + excited). Protocol wrapper."""
        if spin is not None:
            raise NotImplementedError(
                "ExcitedStatesSolver: spin-targeted excited states not implemented"
            )
        self.n_states = int(n_states)
        return self.solve()

    def get_excited_state_data(self):
        """Normalized excited-state payload (capability ``"excited_states"``).

        Reads the ABSOLUTE energies from ``self.results['energies']`` ([ground, excited…]).
        NB: ``result.states`` for this solver is the *excitation* energies (Ha), not
        absolute — so it must NOT be used here. Oscillator strengths / transition
        dipoles are real only on the CIS/TDA path; other methods leave them None, and
        the qeom sub-path's fabricated zero array is coerced to None (honesty).
        """
        import numpy as np
        from kanad.solvers.capabilities import ExcitedStateData
        if not self.has_capability('excited_states'):
            raise NotImplementedError(
                f"{type(self).__name__} does not declare the 'excited_states' capability"
            )
        R = self.results
        if not R or 'energies' not in R:
            raise RuntimeError(
                "ExcitedStatesSolver.get_excited_state_data: call solve() first"
            )
        energies = np.asarray(R['energies'], dtype=float)  # absolute [ground, excited…]
        order = np.argsort(energies)
        state_energies_ha = energies[order]
        HA_TO_EV = 27.2114
        exc_ev = R.get('excitation_energies_ev')
        n_exc = len(state_energies_ha) - 1
        if exc_ev is None or len(np.asarray(exc_ev)) != n_exc:
            excitation_energies_ev = (state_energies_ha[1:] - state_energies_ha[0]) * HA_TO_EV
        else:
            excitation_energies_ev = np.asarray(exc_ev, dtype=float)
        method = str(R.get('method', '')).lower()
        osc = R.get('oscillator_strengths')
        if osc is not None:
            osc = np.asarray(osc, dtype=float)
            # honesty: an all-zero osc array (qeom sub-path) is not a real measurement.
            if osc.size == 0 or ('qeom' in method and not np.any(osc)):
                osc = None
        td = R.get('transition_dipoles')
        if td is not None:
            td = np.asarray(td, dtype=float)
            if td.size == 0:
                td = None
        ev = R.get('eigenvectors')
        eigvec_list = None
        if ev is not None:
            ev = np.asarray(ev)
            if ev.size > 0 and ev.ndim == 2:
                eigvec_list = [ev[:, i] for i in range(ev.shape[1])]
        return ExcitedStateData(
            state_energies_ha=state_energies_ha,
            excitation_energies_ev=excitation_energies_ev,
            oscillator_strengths=osc, transition_dipoles=td,
            eigenvectors=eigvec_list, spin_multiplicities=None,
        )

    def _solve_raw(self) -> Dict[str, Any]:
        """
        Dispatch to the method-specific solver, returning the legacy result dict.

        Returns:
            Dictionary with comprehensive results:
                - energies: State energies [Hartree] (n_states,)
                - excitation_energies: Excitation energies [eV] (n_states-1,)
                - oscillator_strengths: Transition strengths (n_states-1,)
                - transition_dipoles: Transition dipole moments (n_states-1, 3)
                - dominant_transitions: Orbital transitions (n_states-1,)
                - uv_vis_spectrum: UV-Vis absorption spectrum (if analysis enabled)
                - analysis: Detailed photochemistry analysis
        """
        logger.info(f"Computing {self.n_states} excited states using {self.method}...")

        if self.method == 'cis':
            return self._solve_cis()
        elif self.method == 'tddft':
            return self._solve_tddft()
        elif self.method == 'qpe':
            return self._solve_qpe()
        elif self.method == 'vqe':
            return self._solve_vqe_excited()
        elif self.method == 'sqd':
            return self._solve_sqd()
        elif self.method == 'qeom' or self.method == 'qeom-vqe':
            return self._solve_qeom_vqe()
        else:
            raise ValueError(f"Unknown method: {self.method}. Available: cis, tddft, qpe, vqe, sqd, qeom")

    def _solve_cis(self) -> Dict[str, Any]:
        """
        Solve using Configuration Interaction Singles (CIS).

        CIS matrix: A[ia,jb] = δ_ij δ_ab (ε_a - ε_i) + 2(ia|jb) - (ij|ab)

        where i,j are occupied and a,b are virtual orbitals.
        """
        logger.info("Running CIS calculation...")

        # Bug #4 (planck audit): Hamiltonians that don't expose the hand-rolled SCF
        # surface but DO carry a converged PySCF mean-field (notably ActiveHamiltonian
        # from the builder — no solve_scf, and its compute_molecular_orbitals returns
        # ZERO mo-energies which would break the CIS diagonal) are routed through
        # PySCF's validated TDA(mf) (= CIS for an HF reference). Backend-independent;
        # leaves the working CovalentHamiltonian (H₂/LiH) hand-rolled path untouched.
        _mf = getattr(self.hamiltonian, 'mf', None)
        if (not hasattr(self.hamiltonian, 'solve_scf')
                and _mf is not None
                and getattr(_mf, 'mo_energy', None) is not None):
            return self._solve_cis_via_pyscf_tda(_mf)

        # Get HF reference
        density_matrix, hf_energy = self.hamiltonian.solve_scf(
            max_iterations=100,
            conv_tol=1e-8,
            use_diis=True
        )

        # Get MO energies
        mo_energies, mo_coeffs = self.hamiltonian.compute_molecular_orbitals()

        n_orb = len(mo_energies)
        n_occ = self.molecule.n_electrons // 2  # Closed shell
        n_virt = n_orb - n_occ

        logger.info(f"System: {n_occ} occupied, {n_virt} virtual orbitals")

        # Build CIS matrix
        cis_dim = n_occ * n_virt
        if cis_dim == 0:
            logger.warning("No virtual orbitals - cannot compute excited states")
            return {
                'energies': np.array([hf_energy]),
                'excitation_energies': np.array([]),
                'converged': False
            }

        logger.info(f"Building CIS matrix ({cis_dim} x {cis_dim})...")

        A = np.zeros((cis_dim, cis_dim))

        # Map (i,a) to single index
        idx_map = {}
        idx = 0
        for i in range(n_occ):
            for a in range(n_occ, n_orb):
                idx_map[(i, a)] = idx
                idx += 1

        # Get electron repulsion integrals (ERI) in MO basis
        # CRITICAL FIX: Was using placeholders (0.1, 0.05) - now using real ERIs
        logger.info("Computing two-electron integrals in MO basis...")
        try:
            from pyscf import ao2mo

            # Get PySCF molecule object
            if hasattr(self.hamiltonian, 'mol'):
                mol_pyscf = self.hamiltonian.mol
            else:
                # Fallback: construct from atoms
                from pyscf import gto
                mol_pyscf = gto.Mole()
                mol_pyscf.atom = [[atom.symbol, atom.position] for atom in self.molecule.atoms]
                mol_pyscf.basis = 'sto-3g'
                mol_pyscf.build()

            # Transform ERI from AO to MO basis via the indigenous core transform
            # (replaces inline ao2mo.kernel + restore). Returns the full 4-index
            # tensor eri_mo[i,j,k,l] = (ij|kl), chemist notation. (reorg B-audit #12)
            from kanad.core.integrals.transforms import ao2mo_transform_from_mol
            eri_mo = ao2mo_transform_from_mol(mol_pyscf, mo_coeffs)

            logger.info("  ERI tensor computed successfully")
            use_exact_eri = True

        except Exception as e:
            logger.warning(f"Could not compute exact ERI: {e}")
            logger.warning("Falling back to approximate CIS (will be less accurate)")
            use_exact_eri = False

        # Fill CIS matrix
        for i in range(n_occ):
            for a in range(n_occ, n_orb):
                ia = idx_map[(i, a)]

                for j in range(n_occ):
                    for b in range(n_occ, n_orb):
                        jb = idx_map[(j, b)]

                        # Diagonal: orbital energy difference
                        if ia == jb:
                            A[ia, jb] = mo_energies[a] - mo_energies[i]

                        # Off-diagonal: two-electron integrals
                        if use_exact_eri:
                            # EXACT CIS matrix elements using real ERIs
                            # A[ia,jb] = δ_ij δ_ab (ε_a - ε_i) + 2(ia|jb) - (ij|ab)

                            # Coulomb integral: 2(ia|jb)
                            A[ia, jb] += 2.0 * eri_mo[i, a, j, b]

                            # Exchange integral: -(ij|ab)
                            A[ia, jb] -= eri_mo[i, j, a, b]
                        else:
                            # Fallback: approximate (less accurate)
                            if i == j and a == b:
                                # Coulomb integral (approximate)
                                A[ia, jb] += 0.1
                            if i == j or a == b:
                                # Exchange integral (approximate)
                                A[ia, jb] -= 0.05

        # Diagonalize CIS matrix
        logger.info("Diagonalizing CIS matrix...")
        excitation_energies_ha, eigenvectors = np.linalg.eigh(A)

        # Take lowest n_states-1 excitations (ground state is HF)
        n_ex = min(self.n_states - 1, len(excitation_energies_ha))
        excitation_energies_ha = excitation_energies_ha[:n_ex]
        eigenvectors = eigenvectors[:, :n_ex]

        # Convert to eV
        excitation_energies_ev = excitation_energies_ha * 27.2114

        # Total energies
        excited_energies = hf_energy + excitation_energies_ha
        all_energies = np.concatenate([[hf_energy], excited_energies])

        logger.info(f"Found {n_ex} excitations:")
        for i, E in enumerate(excitation_energies_ev):
            logger.info(f"  Excitation {i+1}: {E:.4f} eV")

        # Transition dipoles + oscillator strengths (proper CIS length-gauge formula).
        #   μ_0e = √2 Σ_{ia} C_ia^(e) ⟨φ_i| r̂ |φ_a⟩   (√2 = closed-shell singlet spin adaptation)
        #   f_e  = (2/3) ΔE_e |μ_0e|²
        # Replaces the previous f ≈ 0.67·ΔE·c_max² heuristic (which used only the
        # dominant CI coefficient and a fudge prefactor). Transition dipoles between
        # orthogonal states are origin-independent, so no gauge-origin choice is needed.
        dominant_transitions = []
        oscillator_strengths = []
        transition_dipoles = []

        from kanad.core.integrals.property_integrals import compute_dipole
        from kanad.core.integrals.transforms import property_integral_transform
        ao_dip = compute_dipole(mol_pyscf)                       # (3, nao, nao), ⟨p|r|q⟩
        C = np.asarray(mo_coeffs)
        dip_mo = property_integral_transform(ao_dip, C)          # (3, n_orb, n_orb)

        for ex_idx in range(n_ex):
            coeffs = eigenvectors[:, ex_idx]
            # Dominant single-excitation label (for human-readable reporting)
            max_idx = int(np.argmax(np.abs(coeffs)))
            label = "?"
            for (i, a), idx in idx_map.items():
                if idx == max_idx:
                    label = f"HOMO-{n_occ-1-i} → LUMO+{a-n_occ}"
                    break
            dominant_transitions.append(label)
            # Full transition dipole from ALL CI amplitudes
            mu = np.zeros(3)
            for (i, a), idx in idx_map.items():
                mu = mu + np.sqrt(2.0) * coeffs[idx] * dip_mo[:, i, a]
            transition_dipoles.append(mu)
            f = (2.0 / 3.0) * excitation_energies_ha[ex_idx] * float(np.dot(mu, mu))
            oscillator_strengths.append(abs(f))
        transition_dipoles = np.array(transition_dipoles)

        # Store results
        self.results = {
            'method': 'CIS',
            'energies': all_energies,
            'ground_state_energy': hf_energy,
            'excited_state_energies': excited_energies,
            'excitation_energies_ha': excitation_energies_ha,
            'excitation_energies_ev': excitation_energies_ev,
            'excitation_energies': excitation_energies_ev,  # For compatibility
            'oscillator_strengths': np.array(oscillator_strengths),
            'transition_dipoles': transition_dipoles,
            'dominant_transitions': dominant_transitions,
            'eigenvectors': eigenvectors,
            'converged': True,
            'iterations': 1,
            'energy': hf_energy  # Ground state for base class
        }

        # UV-Vis spectrum if analysis enabled
        if self.enable_analysis and hasattr(self, 'uvvis_calculator') and self.uvvis_calculator is not None:
            try:
                spectrum = self.uvvis_calculator.compute_spectrum(
                    excitation_energies_ev,
                    oscillator_strengths,
                    broadening=0.3  # eV
                )
                self.results['uv_vis_spectrum'] = spectrum
                self.results['analysis'] = {
                    'spectroscopy': {
                        'absorption_max': excitation_energies_ev[np.argmax(oscillator_strengths)] if len(oscillator_strengths) > 0 else None,
                        'strongest_transition': dominant_transitions[np.argmax(oscillator_strengths)] if len(oscillator_strengths) > 0 else None
                    }
                }
            except Exception as e:
                logger.warning(f"UV-Vis spectrum calculation failed: {e}")

        logger.info("CIS calculation complete")

        return self.results

    def _solve_cis_via_pyscf_tda(self, mf) -> Dict[str, Any]:
        """CIS excitation energies via PySCF's TDA on an existing mean-field.

        Used for Hamiltonians (e.g. ActiveHamiltonian) that carry a converged
        PySCF ``mf`` but not the hand-rolled SCF surface. ``TDA(mf)`` is exactly
        CIS for an RHF reference, so this is the validated equivalent of the
        hand-rolled path; it returns the same result contract. (Bug #4.)
        """
        from pyscf import tdscf

        logger.info("Running CIS via PySCF TDA on the Hamiltonian's mean-field "
                    f"({type(mf).__name__}, e_tot={float(mf.e_tot):.6f} Ha)...")
        n_ex = max(0, self.n_states - 1)
        hf_energy = float(mf.e_tot)
        if n_ex == 0:
            self.results = {
                'method': 'CIS (PySCF TDA)', 'energies': np.array([hf_energy]),
                'ground_state_energy': hf_energy, 'excited_state_energies': np.array([]),
                'excitation_energies_ha': np.array([]), 'excitation_energies_ev': np.array([]),
                'excitation_energies': np.array([]), 'oscillator_strengths': np.array([]),
                'transition_dipoles': np.zeros((0, 3)), 'dominant_transitions': [],
                'converged': True, 'iterations': 1, 'energy': hf_energy,
            }
            return self.results

        td = tdscf.TDA(mf)
        td.nstates = n_ex
        td.verbose = 0
        td.kernel()

        exc_ha = np.asarray(td.e, dtype=float)              # excitation energies (Ha)
        exc_ev = exc_ha * 27.2114
        # PySCF provides validated oscillator strengths + transition dipoles.
        try:
            osc = np.asarray(td.oscillator_strength(), dtype=float)
        except Exception as exc:                            # pragma: no cover - robustness
            logger.warning(f"TDA oscillator_strength failed ({exc}); reporting zeros.")
            osc = np.zeros(len(exc_ha))
        try:
            tdip = np.asarray(td.transition_dipole(), dtype=float)
        except Exception:                                   # pragma: no cover
            tdip = np.zeros((len(exc_ha), 3))

        excited_energies = hf_energy + exc_ha
        self.results = {
            'method': 'CIS (PySCF TDA)',
            'energies': np.concatenate([[hf_energy], excited_energies]),
            'ground_state_energy': hf_energy,
            'excited_state_energies': excited_energies,
            'excitation_energies_ha': exc_ha,
            'excitation_energies_ev': exc_ev,
            'excitation_energies': exc_ev,                  # eV, for compatibility
            'oscillator_strengths': np.abs(osc),
            'transition_dipoles': tdip,
            'dominant_transitions': [f"excitation {i+1}" for i in range(len(exc_ha))],
            'converged': True,
            'iterations': 1,
            'energy': hf_energy,
        }
        logger.info(f"PySCF TDA: {len(exc_ha)} excitations, "
                    f"E1 = {exc_ev[0]:.4f} eV" if len(exc_ev) else "no excitations")
        return self.results

    def _solve_tddft(self) -> Dict[str, Any]:
        """TDDFT is not implemented in this solver — fail loud, don't alias CIS.

        Previously this silently returned ``_solve_cis()`` while the public docstrings
        advertised TDDFT as a distinct, more-accurate method. CIS == TDA-on-HF with no
        XC response kernel, so it cannot capture the long-range / charge-transfer /
        Rydberg physics a user picks TDDFT for. Returning CIS numbers under a 'tddft'
        request is silently wrong, so we raise instead and point at the real PySCF path.
        """
        raise NotImplementedError(
            "method='tddft' is not implemented in ExcitedStatesSolver (it would have "
            "returned plain CIS numbers, which lack the XC response kernel). Use "
            "method='cis' here for the CIS/TDA spectrum, or "
            "kanad.analysis.UVVisCalculator(...).compute_excited_states(method='TDDFT', "
            "functional=...) for genuine PySCF TDDFT.")

    def _compute_oscillator_strengths_sqd(
        self,
        energies: np.ndarray,
        eigenvectors: np.ndarray,
        excitation_energies_ha: np.ndarray
    ) -> np.ndarray:
        """
        Compute oscillator strengths for SQD excited states.

        NOT IMPLEMENTED. A physically correct oscillator strength is
            f = (2/3) * ΔE * |⟨ψ_0|μ|ψ_i⟩|²
        which requires the MO-basis dipole operator
        μ = Σ_pq ⟨p|r|q⟩ a_p† a_q (from mol.intor('int1e_r') transformed by the
        Hamiltonian MO coefficients) evaluated between the SQD subspace
        eigenvectors in their Slater-determinant basis.

        The previous implementation fabricated the transition dipole from the
        eigenvector overlap |⟨ψ_0|ψ_i⟩| with no dipole integral — a value with
        no physical meaning. Rather than surface a fabricated metric, this
        raises until the real second-quantized dipole evaluation is wired in.
        """
        # FIX: removed fabricated overlap-based transition dipole; no real
        # dipole integral existed. Honest failure instead of fake physics.
        raise NotImplementedError(
            "SQD oscillator strengths require the MO-basis second-quantized "
            "dipole operator μ=Σ_pq⟨p|r|q⟩a_p†a_q evaluated between SQD "
            "eigenvectors; the overlap-based heuristic was fabricated and has "
            "been removed."
        )

    def _solve_sqd(self) -> Dict[str, Any]:
        """
        Solve using Subspace Quantum Diagonalization (SQD).

        SQD is particularly well-suited for excited states because:
        1. It naturally returns multiple eigenvalues (ground + excited)
        2. Lower circuit depth than VQE
        3. More noise-resistant
        4. No optimization needed - direct diagonalization
        """
        logger.info("Running SQD excited states calculation...")

        from kanad.solvers.deterministic_ci import DeterministicCI

        # Broadcast initial status
        if self.experiment_id:
            try:
                from api.utils import broadcast_log_sync
                broadcast_log_sync(self.experiment_id, f"🔬 Starting SQD excited states calculation...")
                broadcast_log_sync(self.experiment_id, f"📊 Computing {self.n_states} states with subspace_dim={self._subspace_dim}")
            except Exception:
                pass

        # Create SQD solver with user-specified parameters
        backend_kwargs = getattr(self, '_backend_kwargs', {})
        sqd_solver = DeterministicCI(
            self.bond or self.molecule or self.hamiltonian,
            subspace_dim=self._subspace_dim,
            circuit_depth=self._circuit_depth,
            backend=self._backend,
            enable_analysis=False,  # We'll do our own analysis
            enable_optimization=False,
            experiment_id=self.experiment_id,
            **backend_kwargs
        )

        # Define callback for progress updates
        def sqd_callback(stage: int, energy: float, message: str):
            """Callback for SQD progress updates."""
            logger.info(f"SQD Stage {stage}: {message} (E={energy:.8f})")
            if self.experiment_id:
                try:
                    from api.utils import broadcast_log_sync
                    broadcast_log_sync(self.experiment_id, f"📊 {message}")
                except Exception:
                    pass

        # Solve for all states (DeterministicCI returns a SolverResult; flatten
        # to the legacy dict so the dict-subscript logic below is unchanged).
        sqd_result = sqd_solver.solve(n_states=self.n_states, callback=sqd_callback).to_dict()

        # Extract energies
        # Audit H7: sqd_result is DeterministicCI's SolverResult flattened via
        # .to_dict(), whose _jsonable() turns the 'energies' ndarray into a Python
        # list. Re-cast to ndarray so the excited-minus-ground arithmetic below
        # is array math, not list-minus-float (TypeError).
        energies = np.asarray(sqd_result['energies'], dtype=float)
        ground_energy = energies[0]
        excited_energies = energies[1:] if len(energies) > 1 else np.array([])

        # Compute excitation energies
        excitation_energies_ha = excited_energies - ground_energy if len(excited_energies) > 0 else np.array([])
        excitation_energies_ev = excitation_energies_ha * 27.2114

        logger.info(f"SQD found {len(energies)} states:")
        logger.info(f"  Ground state: {ground_energy:.8f} Ha")
        for i, (E_ex_ha, E_ex_ev) in enumerate(zip(excitation_energies_ha, excitation_energies_ev), 1):
            logger.info(f"  Excited state {i}: ΔE = {E_ex_ev:.4f} eV ({E_ex_ha:.8f} Ha)")

        # Broadcast completion
        if self.experiment_id:
            try:
                from api.utils import broadcast_log_sync
                broadcast_log_sync(self.experiment_id, f"✅ SQD complete: Found {len(excited_energies)} excited states")
            except Exception:
                pass

        # Oscillator strengths require a real second-quantized dipole operator
        # (see _compute_oscillator_strengths_sqd). Not implemented for SQD yet,
        # so we report None rather than a fabricated overlap-derived value.
        oscillator_strengths = None
        logger.warning(
            "SQD oscillator strengths not implemented (no real transition "
            "dipole); reporting None instead of a fabricated value."
        )

        # Build results dictionary compatible with other excited state methods
        self.results = {
            'method': 'SQD (Subspace Quantum Diagonalization)',
            'energies': energies,
            'ground_state_energy': ground_energy,
            'excited_state_energies': excited_energies,
            'excitation_energies_ha': excitation_energies_ha,
            'excitation_energies_ev': excitation_energies_ev,
            'excitation_energies': excitation_energies_ev,  # For compatibility
            'oscillator_strengths': oscillator_strengths,  # None: no real transition dipole for SQD yet
            'dominant_transitions': ['SQD State'] * len(excitation_energies_ev),
            'converged': True,
            'iterations': 1,  # SQD is direct diagonalization
            'energy': ground_energy,  # Ground state for base class
            'subspace_dim': self._subspace_dim,
            'circuit_depth': self._circuit_depth,
            # Include additional SQD-specific data
            'eigenvectors': sqd_result.get('eigenvectors'),
            'hf_energy': sqd_result.get('hf_energy'),
            'correlation_energy': sqd_result.get('correlation_energy')
        }

        logger.info(f"SQD excited states complete: {len(excitation_energies_ev)} excited states found")

        return self.results

    def _solve_qpe(self) -> Dict[str, Any]:
        """Solve using Quantum Phase Estimation (placeholder)."""
        logger.warning("Quantum excited states not fully implemented")
        raise NotImplementedError("QPE for excited states not yet implemented")

    def _compute_oscillator_strengths_vqe(
        self,
        all_states: list,
        ansatz_circuit,
        n_qubits: int,
        backend: str
    ) -> np.ndarray:
        """
        Compute oscillator strengths for VQE excited states.

        NOT IMPLEMENTED. A physically correct oscillator strength is
            f = (2/3) * ΔE * |⟨ψ_0|μ|ψ_i⟩|²
        where μ is the dipole operator. Computing the transition dipole
        ⟨ψ_0|μ|ψ_i⟩ requires mapping the MO-basis second-quantized dipole
        operator μ = Σ_pq ⟨p|r|q⟩ a_p† a_q (from mol.intor('int1e_r')
        transformed by the Hamiltonian MO coefficients) into the same qubit
        representation as the VQE ansatz, then taking its matrix element
        between the two statevectors.

        The previous implementation fabricated |μ|² ≈ (1 - |⟨ψ_0|ψ_i⟩|²)·n_qubits
        — literally scaling by the qubit count — plus an ad-hoc 0.1·Δ⟨X_j⟩²
        Pauli-X fudge. None of that is a transition dipole. Rather than surface
        a fabricated metric, this raises until the real dipole evaluation is
        wired in.
        """
        # FIX: removed fabricated qubit-count / Pauli-X transition dipole; no
        # real dipole integral existed. Honest failure instead of fake physics.
        raise NotImplementedError(
            "VQE oscillator strengths require the MO-basis second-quantized "
            "dipole operator μ=Σ_pq⟨p|r|q⟩a_p†a_q mapped to qubits and "
            "evaluated between the VQE statevectors; the qubit-count/Pauli-X "
            "heuristic was fabricated and has been removed."
        )

    def _solve_vqe_excited(self) -> Dict[str, Any]:
        """
        Solve using orthogonally-constrained VQE.

        Uses VQE iteratively to find excited states by adding
        orthogonality penalty to avoid previously found states.

        IMPORTANT LIMITATIONS:
        - This method works best for systems where excited states are
          close in energy to the ground state (< 5-10 eV).
        - For molecules with large HOMO-LUMO gaps (like H2 ~ 35 eV),
          the ansatz cannot reach high-energy excited states.
        - In such cases, use CIS/TDDFT methods instead.
        - This is a fundamental limitation of variational ansatze,
          not an implementation issue.

        For H2 and similar small molecules, prefer:
        - method='cis' for fast, reliable excited states
        - method='tddft' for more accurate results
        """
        logger.info("Running VQE excited states calculation...")

        import types
        from kanad.solvers.vqe_solver import VQESolver
        from qiskit.quantum_info import Statevector

        # Get quantum backend settings from kwargs (use stored values, NO hardcoded defaults!)
        backend = getattr(self, '_backend', 'statevector')
        ansatz_type = getattr(self, '_ansatz', 'hardware_efficient')  # 'governance'/'uccsd' were removed
        optimizer = getattr(self, '_optimizer', 'COBYLA')
        max_iterations = getattr(self, '_max_iterations', None)
        penalty_weight = getattr(self, '_penalty_weight', 1.0)

        # CRITICAL: If max_iterations not set, something is wrong - fail loudly
        if max_iterations is None:
            raise ValueError("max_iterations must be explicitly set for VQE excited states!")

        print(f"🔧 VQE Excited States - Using max_iterations={max_iterations} (from config, NOT hardcoded)")

        # Store results for each state
        all_states = []

        # Find ground state first
        logger.info("Finding ground state...")
        # Get backend kwargs (API tokens, etc.)
        backend_kwargs = getattr(self, '_backend_kwargs', {})

        print(f"🔍 ExcitedStatesSolver: self.vqe_callback = {self.vqe_callback}")
        vqe_ground = VQESolver(
            bond=self.bond,
            backend=backend,
            ansatz_type=ansatz_type,  # Fixed: was 'ansatz=' which is low-level API
            optimizer=optimizer,
            max_iterations=max_iterations,
            enable_analysis=False,
            experiment_id=self.experiment_id,  # Pass for WebSocket broadcasting
            callback=self.vqe_callback,  # Pass progress callback from API layer
            **backend_kwargs  # Pass credentials for IBM/BlueQubit
        )
        print(f"🔍 VQESolver created, checking _callback: {hasattr(vqe_ground, '_callback')}, value: {getattr(vqe_ground, '_callback', 'NOT SET')}")

        # VQESolver.solve() returns a SolverResult; flatten to the legacy dict so
        # the dict-subscript logic below keeps working unchanged.
        ground_result = vqe_ground.solve().to_dict()

        if not ground_result.get('converged', False):
            logger.warning("Ground state VQE did not converge")

        ground_energy = ground_result['energy']
        ground_params = ground_result.get('parameters', np.array([]))  # FIX: use 'parameters' not 'optimal_params'

        all_states.append({
            'energy': ground_energy,
            'params': ground_params,
            'iterations': ground_result.get('iterations', 0)
        })

        logger.info(f"Ground state: {ground_energy:.8f} Ha")

        # Broadcast ground state completion
        if self.experiment_id:
            try:
                from api.utils import broadcast_log_sync
                broadcast_log_sync(self.experiment_id, f"✅ Ground state: E = {ground_energy:.8f} Ha")
            except Exception:
                pass

        # Find excited states iteratively
        for state_idx in range(1, self.n_states):
            logger.info(f"Finding excited state {state_idx}...")

            # Broadcast state progress
            if self.experiment_id:
                try:
                    from api.utils import broadcast_log_sync
                    broadcast_log_sync(self.experiment_id, f"🔬 Optimizing excited state {state_idx}/{self.n_states - 1}...")
                except Exception:
                    pass

            # Create VQE solver for this state
            vqe = VQESolver(
                bond=self.bond,
                backend=backend,
                ansatz_type=ansatz_type,  # Fixed: was 'ansatz=' which is low-level API
                optimizer=optimizer,
                max_iterations=max_iterations,
                enable_analysis=False,
                experiment_id=self.experiment_id,  # Pass for WebSocket broadcasting
                callback=self.vqe_callback,  # Pass progress callback from API layer
                **backend_kwargs  # Pass credentials for IBM/BlueQubit
            )

            # Patch _compute_energy instead of _objective_function
            # This is more reliable as _objective_function is a method that calls _compute_energy
            ansatz_circuit = vqe.ansatz
            original_compute_energy = vqe._compute_energy

            # Debug counter
            call_counter = [0]

            def penalized_compute_energy(params):
                """Compute energy with orthogonality penalty."""
                # Base energy from Hamiltonian
                base_energy = original_compute_energy(params)

                # Add penalty for overlap with all previous states
                penalty = 0.0
                for prev_state in all_states:
                    prev_params = prev_state['params']

                    # Skip if params are empty or wrong shape
                    if len(prev_params) == 0 or prev_params.shape != params.shape:
                        continue

                    # Compute exact overlap using statevector
                    if backend == 'statevector':
                        try:
                            circuit1 = ansatz_circuit.assign_parameters(params)
                            circuit2 = ansatz_circuit.assign_parameters(prev_params)

                            sv1 = Statevector(circuit1)
                            sv2 = Statevector(circuit2)

                            overlap_sq = abs(sv1.inner(sv2)) ** 2
                            penalty += overlap_sq
                        except Exception as e:
                            # Fall back to parameter distance
                            param_dist = np.linalg.norm(params - prev_params)
                            penalty += np.exp(-param_dist / np.sqrt(len(params)))
                    else:
                        # Use parameter distance for hardware backends
                        param_dist = np.linalg.norm(params - prev_params)
                        penalty += np.exp(-param_dist / np.sqrt(len(params)))

                # Debug logging
                call_counter[0] += 1
                if call_counter[0] == 1:
                    print(f"🔍 Penalty function called! Base E={base_energy:.6f}, penalty={penalty:.6f}, weight={penalty_weight}, total={base_energy + penalty_weight * penalty:.6f}")

                return base_energy + penalty_weight * penalty

            # Replace the _compute_energy method using types.MethodType to properly bind it
            vqe._compute_energy = types.MethodType(lambda self, params: penalized_compute_energy(params), vqe)

            # Use random initial parameters far from ground state
            n_params = vqe.n_parameters
            initial_params = np.random.randn(n_params) * 0.5  # Larger random init

            # Solve for this excited state (SolverResult -> legacy dict)
            result = vqe.solve(initial_parameters=initial_params).to_dict()

            excited_params = result.get('parameters', np.array([]))

            # CRITICAL FIX: Compute TRUE energy without penalty
            # result['energy'] is the penalized energy, we need the actual energy
            excited_energy = original_compute_energy(excited_params)
            logger.debug(f"Penalized E={result['energy']:.6f}, True E={excited_energy:.6f}")  # FIX: use 'parameters' not 'optimal_params'

            all_states.append({
                'energy': excited_energy,
                'params': excited_params,
                'iterations': result.get('iterations', 0)
            })

            excitation_ev = (excited_energy - ground_energy) * 27.2114
            logger.info(f"Excited state {state_idx}: {excited_energy:.8f} Ha "
                       f"(ΔE = {excitation_ev:.4f} eV)")

            # Broadcast state completion
            if self.experiment_id:
                try:
                    from api.utils import broadcast_log_sync
                    broadcast_log_sync(self.experiment_id,
                                     f"✅ State {state_idx}: E = {excited_energy:.8f} Ha, ΔE = {excitation_ev:.4f} eV")
                except Exception:
                    pass

        # Extract energies
        energies = np.array([s['energy'] for s in all_states])
        excitation_energies_ha = energies[1:] - energies[0]
        excitation_energies_ev = excitation_energies_ha * 27.2114

        # Total iterations
        total_iterations = sum(s['iterations'] for s in all_states)

        # Oscillator strengths require a real second-quantized dipole operator
        # (see _compute_oscillator_strengths_vqe). Not implemented for VQE yet,
        # so we report None rather than a fabricated qubit-count-derived value.
        oscillator_strengths = None
        logger.warning(
            "VQE oscillator strengths not implemented (no real transition "
            "dipole); reporting None instead of a fabricated value."
        )

        # Build result dictionary
        self.results = {
            'method': 'VQE (Orthogonally-Constrained)',
            'energies': energies,
            'ground_state_energy': energies[0],
            'excited_state_energies': energies[1:],
            'excitation_energies_ha': excitation_energies_ha,
            'excitation_energies_ev': excitation_energies_ev,
            'excitation_energies': excitation_energies_ev,  # For compatibility
            'oscillator_strengths': oscillator_strengths,  # None: no real transition dipole for VQE yet
            'dominant_transitions': ['VQE State'] * len(excitation_energies_ev),
            'converged': True,
            'iterations': total_iterations,
            'energy': energies[0],  # Ground state for base class
            'penalty_weight': penalty_weight
        }

        logger.info(f"VQE excited states complete: {len(excitation_energies_ev)} excited states found")

        return self.results

    def _solve_qeom_vqe(self) -> Dict[str, Any]:
        """
        Solve using quantum Equation of Motion VQE (qEOM-VQE).

        This gives TRUE quantum excited states by building the EOM matrix
        on top of the VQE ground state. Unlike penalty-based VQE, this
        correctly captures excitation energies.

        Reference: Ollitrault et al. (2020) Chem. Sci. 11, 6842
        """
        logger.info("Running qEOM-VQE calculation...")
        logger.info("  This uses VQE ground state + EOM for true excited states")

        from kanad.solvers.qeom_vqe import qEOMVQE

        # Create qEOM-VQE solver
        solver = qEOMVQE(
            self.bond,
            n_states=self.n_states,
            include_singles=True,
            include_doubles=True,
            backend=self._backend
        )

        # Solve for excited states
        result = solver.solve()

        # Extract energies
        ground_energy = result.ground_energy
        excited_energies = result.excited_energies
        excitation_energies_ev = result.excitation_energies

        # Build all energies array (ground + excited)
        all_energies = np.concatenate([[ground_energy], excited_energies])

        # Compute oscillator strengths (zeros for now - would need transition dipoles)
        oscillator_strengths = np.zeros(len(excitation_energies_ev))

        # Build result dictionary
        self.results = {
            'method': 'qEOM-VQE (Quantum Equation of Motion)',
            'energies': all_energies,
            'ground_state_energy': ground_energy,
            'excited_state_energies': excited_energies,
            'excitation_energies_ha': excitation_energies_ev / 27.2114,
            'excitation_energies_ev': excitation_energies_ev,
            'excitation_energies': excitation_energies_ev,  # For compatibility
            'oscillator_strengths': oscillator_strengths,
            'dominant_transitions': ['qEOM State'] * len(excitation_energies_ev),
            'converged': True,
            'energy': ground_energy,  # Ground state for base class
            'n_excitation_operators': result.n_excitations,
            'h_matrix': result.h_matrix,
            's_matrix': result.s_matrix
        }

        logger.info(f"qEOM-VQE complete: {len(excitation_energies_ev)} excited states found")
        for i, e_ev in enumerate(excitation_energies_ev):
            logger.info(f"  S{i+1}: {e_ev:.2f} eV")

        return self.results

    def print_summary(self):
        """Print excited states summary."""
        print("=" * 80)
        print("EXCITED STATES SOLVER RESULTS")
        print("=" * 80)

        print(f"\nSystem: {'-'.join([a.symbol for a in self.atoms])}")
        print(f"Method: {self.results.get('method', self.method.upper())}")

        if 'ground_state_energy' in self.results:
            print(f"\nGround State: {self.results['ground_state_energy']:.8f} Hartree")

        if 'excitation_energies_ev' in self.results:
            n_ex = len(self.results['excitation_energies_ev'])
            print(f"\nExcited States ({n_ex} found):")
            print("-" * 80)
            # oscillator_strengths may be None (no real transition dipole computed)
            osc = self.results.get('oscillator_strengths')
            if osc is None:
                osc = [None] * n_ex
            trans_list = self.results.get('dominant_transitions', ['?'] * n_ex)
            for i, (E_ev, f, trans) in enumerate(zip(
                self.results['excitation_energies_ev'], osc, trans_list
            ), 1):
                f_str = "  n/a" if f is None else f"{f:.4f}"
                print(f"  State {i}: {E_ev:8.4f} eV  (f={f_str})  {trans}")

        if 'analysis' in self.results and 'spectroscopy' in self.results['analysis']:
            spec = self.results['analysis']['spectroscopy']
            if spec.get('absorption_max'):
                print(f"\nAbsorption Maximum: {spec['absorption_max']:.2f} eV ({1240/spec['absorption_max']:.1f} nm)")
                print(f"Strongest Transition: {spec['strongest_transition']}")

        print("=" * 80)
