"""
Base Solver Class for Kanad Framework.

All solvers inherit from this class and work with the bonds module interface.
This provides consistent API and automatic integration with analysis and optimization.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import numpy as np

from kanad.solvers.meta import SolverMeta

logger = logging.getLogger(__name__)


class BaseSolver(ABC):
    """
    Base class for all quantum chemistry solvers.

    Design Philosophy:
    1. Solvers take a BOND or MOLECULE as input
    2. Solvers automatically integrate analysis tools
    3. Solvers automatically integrate optimization tools
    4. Solvers provide rich, comprehensive results

    This makes solvers the "one-stop-shop" for users.
    """

    # ── Capability + domain protocol (Stage 1, additive) ────────────────────
    # Every solver class overrides META with its REAL, verified capabilities.
    # The default (energy / ground_state) keeps any solver that has not yet
    # declared META instantly conformant — this layer never breaks existing code.
    META: SolverMeta = SolverMeta(
        name="base",
        domains=frozenset({"ground_state"}),
        capabilities=frozenset({"energy"}),
    )

    @classmethod
    def capabilities(cls) -> frozenset:
        """The capability strings this solver declares (see ``solvers/meta.py``)."""
        return cls.META.capabilities

    @classmethod
    def has_capability(cls, capability: str) -> bool:
        """Whether this solver declares ``capability`` (e.g. ``"one_rdm"``)."""
        return capability in cls.META.capabilities

    @classmethod
    def supports_domain(cls, domain: str) -> bool:
        """Whether this solver serves ``domain`` (e.g. ``"md"``) — drives lab routing."""
        return domain in cls.META.domains

    def get_one_rdm(self, *, basis: str = "mo") -> np.ndarray:
        """Real 1-particle reduced density matrix (capability ``"one_rdm"``).

        The universal observables channel: dipole, charges, bond orders, energy
        decomposition all derive from this. Honesty rule — raises if the solver
        does not declare ``one_rdm``, if no density is available, or if the trace
        disagrees with the electron count; it never silently substitutes the HF
        density and calls it quantum.

        Call ``solve()`` first so the converged density is available.
        """
        if not self.has_capability("one_rdm"):
            raise NotImplementedError(
                f"{type(self).__name__} does not declare the 'one_rdm' capability"
            )
        rdm = None
        if hasattr(self, "get_1rdm_active_mo"):
            rdm = self.get_1rdm_active_mo()
        elif hasattr(self.hamiltonian, "get_density_matrix"):
            try:
                rdm = self.hamiltonian.get_density_matrix(basis)
            except TypeError:
                rdm = self.hamiltonian.get_density_matrix()
        if rdm is None:
            raise NotImplementedError(
                f"{type(self).__name__} declares 'one_rdm' but exposes no density "
                f"accessor; call solve() first or override get_one_rdm()."
            )
        rdm = np.asarray(rdm)
        n_e = getattr(self.hamiltonian, "n_electrons", None)
        if n_e is not None:
            tr = float(np.real(np.trace(rdm)))
            if abs(tr - n_e) > 1e-3 * max(1, n_e):
                raise ValueError(
                    f"1-RDM trace {tr:.6f} != n_electrons {n_e}: refusing an "
                    f"inconsistent / HF-fallback density (honesty rule)."
                )
        return rdm

    def get_dipole(self, *, origin=None) -> np.ndarray:
        """Electric dipole moment vector (a.u.) from the correlated 1-RDM (capability
        ``"dipole"``): ``μ = Σ_A Z_A R_A − Tr(P_AO · r_AO)``.

        Flows through the same honest 1-RDM channel as :meth:`get_one_rdm` — it embeds
        the (active-)MO density onto the Hamiltonian via ``set_quantum_density_matrix``
        and evaluates with ``PropertyCalculator(method='vqe')``, which raises rather than
        silently substituting HF. Declared only by solvers whose default construction
        yields a genuinely correlated 1-RDM (VQESolver, DeterministicCI); NOT by
        SamplingSQD, whose default non-entangling circuit collapses to HF. Call
        ``solve()`` first."""
        if not self.has_capability("dipole"):
            raise NotImplementedError(
                f"{type(self).__name__} does not declare the 'dipole' capability"
            )
        rdm_mo = self.get_one_rdm(basis="mo")  # honest: raises on trace mismatch
        ham = self.hamiltonian
        if getattr(ham, "_quantum_density_matrix_ao", None) is None \
                and hasattr(ham, "set_quantum_density_matrix"):
            ham.set_quantum_density_matrix(rdm_mo)
        from kanad.analysis.property_calculator import PropertyCalculator
        pc = PropertyCalculator(ham)
        res = pc.compute_dipole_moment(origin=origin, method="vqe")
        return np.asarray(res["dipole_au"], dtype=float)

    def _hessian_masses_amu(self):
        """Ordered atomic masses (amu) for the harmonic analysis in the ``hessian``
        capability, in the SAME atom order as the geometry passed to ``energy_fn`` /
        ``hessian`` (i.e. the PySCF-mol rebuild order). Returns ``None`` when no mass
        source is available, so the Hessian mixin returns a raw matrix rather than a
        fabricated spectrum. Prefer the PySCF mol (guaranteed to match ``energy_fn``'s
        symbol order); fall back to the resolved ``self.atoms``."""
        mol = getattr(self, "pyscf_mol", None)
        if mol is None:
            mol = getattr(self.hamiltonian, "mol", None)
        if mol is not None:
            try:
                return np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)
            except Exception:
                try:
                    return np.asarray(mol.atom_mass_list(), dtype=float)
                except Exception:
                    pass
        atoms = getattr(self, "atoms", None)
        if atoms:
            masses = [getattr(a, "atomic_mass", None) for a in atoms]
            if all(m is not None for m in masses):
                return np.asarray(masses, dtype=float)
        return None

    def __init__(
        self,
        system,
        *,
        backend: str = "statevector",
        enable_analysis: bool = True,
        enable_optimization: bool = True,
        **backend_kwargs,
    ):
        """
        Initialize base solver (unified solver protocol).

        Args:
            system: Bond (from BondFactory), Molecule, MolecularHamiltonian, or any
                object exposing a ``.hamiltonian`` (e.g. a builder QuantumSystem).
            backend: Backend name resolved via ``kanad.backends.factory.make_backend``
                (``statevector`` | ``planck`` | ``bluequbit`` | ``ibm`` | ``ionq``).
            enable_analysis: Enable automatic analysis (default: True).
            enable_optimization: Enable automatic optimization (default: True).
            **backend_kwargs: Backend construction params (device, shots, ...). The
                statevector backend ignores extras it does not recognize.
        """
        self.enable_analysis = enable_analysis
        self.enable_optimization = enable_optimization

        # Normalize the system argument into hamiltonian / molecule / bond.
        self._resolve_system(system)

        # Build the backend object (replaces the legacy _use_statevector flag +
        # _init_backend string dispatch).
        from kanad.backends.factory import make_backend
        self.backend = make_backend(backend, **backend_kwargs)
        self.backend_name = self.backend.name

        # Initialize analysis tools if enabled
        if enable_analysis:
            self._init_analysis_tools()

        # Initialize optimization tools if enabled
        if enable_optimization:
            self._init_optimization_tools()

        # Storage for results
        self.results = {}

        logger.info(f"Initialized {self.__class__.__name__} for {self._bond_type} system")

    def _resolve_system(self, system):
        """Normalize Bond | Molecule | MolecularHamiltonian | QuantumSystem.

        Sets ``self.hamiltonian``, ``self.molecule``, ``self.bond``, ``self.atoms``,
        and ``self._bond_type``. ``molecule`` / ``bond`` are ``None`` when not
        derivable from the input.
        """
        from kanad.core.molecule import Molecule, MolecularHamiltonian as _ConcreteHam
        from kanad.core.hamiltonians.molecular_hamiltonian import (
            MolecularHamiltonian as _AbstractHam,
        )
        if isinstance(system, Molecule):
            self.bond = None
            self.molecule = system
            self.hamiltonian = system.hamiltonian
            self.atoms = system.atoms
            self._bond_type = 'molecular'
        elif isinstance(system, (_AbstractHam, _ConcreteHam)) or (
            hasattr(system, 'to_sparse_hamiltonian')
            and hasattr(system, 'n_electrons')
            and not hasattr(system, 'hamiltonian')
        ):
            # Bare Hamiltonian (concrete multi-atom, a Covalent/Ionic subclass of
            # the abstract base, or a duck-typed Hamiltonian that does not subclass
            # MolecularHamiltonian). The duck-type branch is what makes the
            # ActiveHamiltonian contract real: it exposes the MolecularHamiltonian
            # surface (to_sparse_hamiltonian / n_electrons / n_orbitals) but neither
            # subclasses the base nor carries a `.hamiltonian` attribute, so without
            # it solvers raised TypeError (audit H15).
            self.bond = None
            self.molecule = getattr(system, 'molecule', None)
            self.hamiltonian = system
            self.atoms = getattr(system, 'atoms', [])
            self._bond_type = 'molecular'
        elif hasattr(system, 'hamiltonian'):
            # Bond object or builder QuantumSystem (anything exposing .hamiltonian).
            self.bond = system
            self.hamiltonian = system.hamiltonian
            self.molecule = getattr(system, 'molecule', None)
            self.atoms = getattr(system, 'atoms', [])
            self._bond_type = getattr(system, 'bond_type', 'unknown')
        else:
            raise TypeError(
                f"Expected Bond, Molecule, or MolecularHamiltonian, got {type(system).__name__}"
            )

    @classmethod
    def from_hamiltonian(cls, hamiltonian, **kw):
        """Construct the solver from a bare MolecularHamiltonian."""
        return cls(hamiltonian, **kw)

    @classmethod
    def from_bond(cls, bond, **kw):
        """Construct the solver from a Bond / QuantumSystem object."""
        return cls(bond, **kw)

    def _init_analysis_tools(self):
        """Initialize analysis tools (imported directly from kanad.analysis, not
        laundered through the bonds facade — reorg Phase C, breaks solvers->bonds)."""
        from kanad.analysis import (
            EnergyAnalyzer,
            BondingAnalyzer,
            PropertyCalculator
        )

        self.energy_analyzer = EnergyAnalyzer(self.hamiltonian)
        self.bonding_analyzer = BondingAnalyzer(self.hamiltonian)

        # PropertyCalculator may require PySCF molecule - initialize if available
        try:
            self.property_calculator = PropertyCalculator(self.hamiltonian)
        except (AttributeError, TypeError):
            logger.debug("PropertyCalculator initialization skipped (requires PySCF molecule)")
            self.property_calculator = None

        logger.debug("Analysis tools initialized")

    def _init_optimization_tools(self):
        """Initialize optimization tools (lazy / deferred)."""
        # CircuitOptimizer was removed in the 2026-05-28 cleanup (phantom gate
        # model, always None here). OrbitalOptimizer requires MO coefficients —
        # deferred until a solver wires it in.
        self.circuit_optimizer = None
        self.orbital_optimizer = None

        logger.debug("Optimization tools initialized")

    @abstractmethod
    def solve(self, **kwargs) -> "SolverResult":
        """
        Solve for the ground-state energy and properties.

        Must be implemented by the subclass.

        Returns:
            ``SolverResult`` — the unified frozen result (``.energy`` is the
            canonical energy in Hartree). Call ``.to_dict()`` for the legacy flat
            dict. (The previous ``-> Dict[str, Any]`` annotation was a leftover
            from before the 2026-06 envelope migration and mislead implementers.)
        """
        pass

    def _add_analysis_to_results(self, energy: float, density_matrix: Optional[np.ndarray] = None):
        """
        Add automatic analysis to results.

        Args:
            energy: Computed energy
            density_matrix: Density matrix (if available)
        """
        if not self.enable_analysis:
            return

        analysis = {}

        # Energy decomposition
        try:
            # Honesty guard: an identity matrix is NOT a valid density matrix.
            # decompose_energy() would silently return a numerically-valid but
            # physically-meaningless decomposition (no exception), so refuse to
            # fabricate one when no real density is available.
            if density_matrix is None:
                analysis['energy_components'] = None
            else:
                analysis['energy_components'] = self.energy_analyzer.decompose_energy(
                    density_matrix
                )
        except Exception as e:
            logger.warning(f"Energy decomposition failed: {e}")
            analysis['energy_components'] = None

        # Bonding analysis
        try:
            # BondingAnalyzer: some methods need density_matrix, others don't
            bonding_info = {}

            if hasattr(self.bonding_analyzer, 'analyze_bonding_type'):
                bonding_info['bond_type'] = self.bonding_analyzer.analyze_bonding_type()

            if hasattr(self.bonding_analyzer, 'analyze_bond_orders') and density_matrix is not None:
                bonding_info['bond_orders'] = self.bonding_analyzer.analyze_bond_orders(density_matrix)

            analysis['bonding'] = bonding_info if bonding_info else None
        except Exception as e:
            logger.warning(f"Bonding analysis failed: {e}")
            analysis['bonding'] = None

        # Molecular properties
        try:
            if self.property_calculator is not None:
                analysis['properties'] = self.property_calculator.calculate_properties(
                    self.molecule,
                    self.hamiltonian,
                    density_matrix if density_matrix is not None else None
                )
            else:
                analysis['properties'] = None
        except Exception as e:
            logger.warning(f"Property calculation failed: {e}")
            analysis['properties'] = None

        self.results['analysis'] = analysis

    def _add_optimization_stats(self):
        """Add optimization statistics to results."""
        if not self.enable_optimization:
            return

        opt_stats = {}

        # Circuit optimization stats (if applicable)
        if hasattr(self, 'circuit') and hasattr(self, 'circuit_optimizer') and self.circuit_optimizer:
            opt_stats['circuit'] = {
                'gates_before': getattr(self, '_gates_before_opt', None),
                'gates_after': getattr(self, '_gates_after_opt', None),
                'depth_before': getattr(self, '_depth_before_opt', None),
                'depth_after': getattr(self, '_depth_after_opt', None),
            }

        # Orbital optimization stats
        opt_stats['orbitals'] = {
            'localization_applied': getattr(self, '_orbital_localization', False),
            'rotation_applied': getattr(self, '_orbital_rotation', False),
        }

        self.results['optimization_stats'] = opt_stats

    def get_reference_energy(self) -> float:
        """
        Get Hartree-Fock reference energy for comparison.

        Returns:
            HF energy (Hartree)
        """
        # Preferred: the Hamiltonian's own SCF (CovalentHamiltonian etc.).
        if hasattr(self.hamiltonian, 'solve_scf'):
            try:
                density_matrix, hf_energy = self.hamiltonian.solve_scf(
                    max_iterations=100,
                    conv_tol=1e-8,
                    use_diis=True
                )
                return hf_energy
            except Exception as e:
                logger.warning(f"Could not compute HF reference via solve_scf: {e}")

        # Fallback: a carried PySCF mean-field (e.g. ActiveHamiltonian has no
        # solve_scf, but holds a converged `mf`; for a properly-built active space
        # mf.e_tot IS the active-space HF reference the VQE energy should beat).
        mf = getattr(self.hamiltonian, 'mf', None)
        if mf is not None and getattr(mf, 'e_tot', None) is not None:
            return float(mf.e_tot)

        logger.warning("Could not compute HF reference (no solve_scf and no mean-field).")
        return None

    def validate_results(self) -> Dict[str, Any]:
        """
        Validate solver results against known checks.

        Returns:
            Validation report
        """
        validation = {
            'passed': True,
            'checks': []
        }

        # Check 1: Energy is real
        if 'energy' in self.results:
            energy = self.results['energy']
            is_real = np.isreal(energy) and not np.isnan(energy) and not np.isinf(energy)
            validation['checks'].append({
                'name': 'energy_is_real',
                'passed': is_real,
                'message': f"Energy is {'valid' if is_real else 'invalid'}: {energy}"
            })
            validation['passed'] = validation['passed'] and is_real

        # Check 2: Energy below HF (for correlated methods)
        if 'energy' in self.results and hasattr(self, '_is_correlated') and self._is_correlated:
            hf_energy = self.get_reference_energy()
            if hf_energy is not None:
                # Use 1e-5 Ha (10 μHa) tolerance for VQE numerical precision
                # Previous 1e-6 was too strict and caused false positives
                below_hf = self.results['energy'] <= hf_energy + 1e-5

                # Check if this is a VQE solver (stochastic optimization)
                is_vqe = hasattr(self, 'optimizer_method')

                validation['checks'].append({
                    'name': 'energy_below_hf',
                    'passed': below_hf,
                    'message': f"Energy {'≤' if below_hf else '>'} HF ({self.results['energy']:.6f} vs {hf_energy:.6f})"
                })
                if not below_hf:
                    validation['passed'] = False
                    if is_vqe:
                        # For VQE, this is expected sometimes due to stochastic optimization
                        logger.info(f"VQE did not beat HF energy ({self.results['energy']:.6f} vs {hf_energy:.6f}) - likely stuck at local minimum. Consider multi-start VQE.")
                    else:
                        # For other methods, this is unexpected
                        logger.warning(f"Correlated method energy ({self.results['energy']:.6f}) above HF ({hf_energy:.6f})!")

        # Check 3: Convergence
        if 'converged' in self.results:
            converged = self.results['converged']
            validation['checks'].append({
                'name': 'convergence',
                'passed': converged,
                'message': f"Solver {'converged' if converged else 'did not converge'}"
            })
            if not converged:
                logger.info("Solver did not fully converge (may need more iterations or different optimizer)")

        return validation

    def print_summary(self):
        """Print human-readable summary of results."""
        print("=" * 80)
        print(f"{self.__class__.__name__} RESULTS")
        print("=" * 80)

        # System info
        atom_symbols = [a.symbol for a in self.atoms] if self.atoms else ['?']
        print(f"\nSystem: {'-'.join(atom_symbols)}")
        print(f"Type: {self._bond_type}")
        print(f"Electrons: {self.hamiltonian.n_electrons}")
        print(f"Orbitals: {self.hamiltonian.n_orbitals}")

        # Energy
        if 'energy' in self.results:
            print(f"\nGround State Energy: {self.results['energy']:.8f} Hartree")

        # Convergence
        if 'converged' in self.results:
            status = "✓ Converged" if self.results['converged'] else "✗ Not Converged"
            print(f"Status: {status}")

        if 'iterations' in self.results:
            print(f"Iterations: {self.results['iterations']}")

        # Correlation energy
        if 'correlation_energy' in self.results:
            print(f"Correlation Energy: {self.results['correlation_energy']:.8f} Hartree")

        # Analysis
        if 'analysis' in self.results and self.results['analysis']:
            print("\n" + "-" * 80)
            print("ANALYSIS")
            print("-" * 80)

            if self.results['analysis'].get('bonding'):
                print("\nBonding Analysis:")
                bonding = self.results['analysis']['bonding']
                for key, value in bonding.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value:.4f}")

        # Validation
        validation = self.validate_results()
        if not validation['passed']:
            print("\n" + "-" * 80)
            print("⚠ VALIDATION WARNINGS")
            print("-" * 80)
            for check in validation['checks']:
                if not check['passed']:
                    print(f"✗ {check['name']}: {check['message']}")

        print("=" * 80)
