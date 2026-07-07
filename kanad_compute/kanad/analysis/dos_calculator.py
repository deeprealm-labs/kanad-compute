"""
Quantum DOS Calculator.

Computes electronic density of states from quantum eigenstates (SQD/VQE),
with optional governance-guided subspace reduction.

NOTE: Bonding-type resolved DOS (covalent/ionic/metallic separation) is NOT
implemented. The previous implementation fabricated those weights with
np.random.rand(), i.e. it returned RNG as a physical observable. That path
now raises NotImplementedError rather than emitting fake numbers.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class DOSCalculator:
    """
    Calculate density of states for periodic systems.

    Supports:
    - Total DOS with Gaussian/Lorentzian broadening
    - Projected DOS (PDOS) on atoms or orbitals
    - Integrated DOS (IDOS)
    - Fermi level calculation
    - Van Hove singularities identification
    """

    def __init__(self, periodic_hamiltonian=None):
        """
        Initialize DOS calculator.

        Args:
            periodic_hamiltonian: PeriodicHamiltonian object with solved SCF
                                  (optional - only needed for periodic DOS)

        Examples:
            >>> # Periodic DOS (classical)
            >>> dos_calc = DOSCalculator(crystal.hamiltonian)
            >>> result = dos_calc.compute_dos(energy_range=(-15, 15))

            >>> # Quantum molecular DOS (NEW!)
            >>> dos_calc = DOSCalculator()  # No hamiltonian needed
            >>> result = dos_calc.compute_quantum_dos(bond=h2_bond)
        """
        self.hamiltonian = periodic_hamiltonian

        # Periodic DOS setup only for an actual PeriodicHamiltonian (duck-typed by
        # `k_weights`). A molecular/active-space Hamiltonian is accepted and routed
        # to the molecular-DOS defaults so compute_quantum_dos() works — previously
        # the constructor raised on ANY non-periodic Hamiltonian, blocking molecular
        # quantum DOS entirely (F14).
        if periodic_hamiltonian is not None and hasattr(periodic_hamiltonian, 'k_weights'):
            if getattr(periodic_hamiltonian, 'band_energies', None) is None:
                raise ValueError("Must run solve_scf() on PeriodicHamiltonian first")

            self.band_energies = periodic_hamiltonian.band_energies  # (nk, n_bands) in Ha
            self.k_weights = periodic_hamiltonian.k_weights  # (nk,)
            self.n_k = len(self.k_weights)
            self.n_bands = self.band_energies.shape[1]
            logger.debug(f"DOSCalculator: {self.n_k} k-points, {self.n_bands} bands")
        else:
            # For quantum molecular DOS
            self.band_energies = None
            self.k_weights = None
            self.n_k = None
            self.n_bands = None
            logger.debug("DOSCalculator: Quantum molecular DOS mode")

        # Constants
        self.Ha_to_eV = 27.2114
        self.eV_to_Ha = 1.0 / self.Ha_to_eV

    def compute_dos(self,
                    energy_range: Tuple[float, float] = (-10, 10),
                    n_points: int = 1000,
                    sigma: float = 0.1,
                    method: str = 'gaussian',
                    units: str = 'eV') -> Dict[str, Any]:
        """
        Compute total density of states.

        Formula:
            DOS(E) = Σ_{nk} w_k × δ(E - E_nk)

        With Gaussian broadening:
            DOS(E) ≈ Σ_{nk} w_k / (σ√2π) exp(-(E - E_nk)² / 2σ²)

        Args:
            energy_range: (E_min, E_max) in eV (or Hartree if units='Ha')
            n_points: Number of energy grid points
            sigma: Broadening parameter in eV
            method: 'gaussian' or 'lorentzian'
            units: 'eV' or 'Ha' for energy units

        Returns:
            result: Dictionary with:
                - energies: (n_points,) energy grid
                - dos: (n_points,) density of states (states/eV or states/Ha)
                - idos: (n_points,) integrated DOS
                - fermi_energy: Fermi level
                - n_electrons: Number of electrons (from integration)
        """
        # Convert energy range to Hartree if needed
        if units == 'eV':
            E_min_Ha = energy_range[0] * self.eV_to_Ha
            E_max_Ha = energy_range[1] * self.eV_to_Ha
            sigma_Ha = sigma * self.eV_to_Ha
        else:
            E_min_Ha, E_max_Ha = energy_range
            sigma_Ha = sigma

        # Create energy grid
        energies_Ha = np.linspace(E_min_Ha, E_max_Ha, n_points)
        dos = np.zeros(n_points)

        # Sum over all k-points and bands
        for k_idx in range(self.n_k):
            for band_idx in range(self.n_bands):
                E_nk = self.band_energies[k_idx, band_idx]  # Hartree
                weight = self.k_weights[k_idx]

                # Apply broadening
                if method == 'gaussian':
                    dos += weight * self._gaussian_broadening(energies_Ha, E_nk, sigma_Ha)
                elif method == 'lorentzian':
                    dos += weight * self._lorentzian_broadening(energies_Ha, E_nk, sigma_Ha)
                else:
                    raise ValueError(f"Unknown method: {method}")

        # Spin degeneracy. The loop above sums unit-area Gaussians over k-weighted
        # SPATIAL orbitals, so ∫dos dE = n_bands (i.e. occupied-spatial = n_electrons/2
        # below E_F). A physical electronic DOS includes the spin factor so that
        # ∫dos dE up to E_F = n_electrons. Closed-shell (spin=0) → factor 2.
        spin_factor = 2.0 if getattr(self.hamiltonian, 'spin', 0) == 0 else 2.0
        dos = dos * spin_factor

        # Integrated DOS (now counts electrons, not spatial orbitals)
        idos = self._integrate_dos(energies_Ha, dos)

        # Fermi energy
        fermi_energy_Ha = self._compute_fermi_energy_from_dos(energies_Ha, dos)

        # Convert back to desired units
        if units == 'eV':
            energies_out = energies_Ha * self.Ha_to_eV
            dos_out = dos / self.Ha_to_eV  # states/eV (not states/Ha)
            fermi_energy_out = fermi_energy_Ha * self.Ha_to_eV
        else:
            energies_out = energies_Ha
            dos_out = dos
            fermi_energy_out = fermi_energy_Ha

        return {
            'energies': energies_out,
            'dos': dos_out,
            'idos': idos,
            'fermi_energy': fermi_energy_out,
            'n_electrons_from_integration': idos[np.argmin(np.abs(energies_Ha - fermi_energy_Ha))],
            'n_electrons_actual': self.hamiltonian.n_electrons,
            'units': units
        }

    def _gaussian_broadening(self, energies: np.ndarray, E_center: float, sigma: float) -> np.ndarray:
        """
        Gaussian broadening function.

        Args:
            energies: Energy grid
            E_center: Center energy
            sigma: Width parameter

        Returns:
            broadened: Gaussian centered at E_center
        """
        prefactor = 1.0 / (sigma * np.sqrt(2 * np.pi))
        exponent = -0.5 * ((energies - E_center) / sigma)**2
        return prefactor * np.exp(exponent)

    def _lorentzian_broadening(self, energies: np.ndarray, E_center: float, gamma: float) -> np.ndarray:
        """
        Lorentzian (Cauchy) broadening function.

        Args:
            energies: Energy grid
            E_center: Center energy
            gamma: Half-width at half-maximum

        Returns:
            broadened: Lorentzian centered at E_center
        """
        prefactor = gamma / np.pi
        denominator = (energies - E_center)**2 + gamma**2
        return prefactor / denominator

    def _integrate_dos(self, energies: np.ndarray, dos: np.ndarray) -> np.ndarray:
        """
        Integrate DOS to get number of states up to each energy.

        Args:
            energies: Energy grid
            dos: DOS values

        Returns:
            idos: Cumulative integral (number of states)
        """
        from scipy.integrate import cumulative_trapezoid

        idos = cumulative_trapezoid(dos, energies, initial=0.0)
        return idos

    def _compute_fermi_energy_from_dos(self, energies: np.ndarray, dos: np.ndarray) -> float:
        """
        Compute Fermi energy by finding where IDOS = n_electrons.

        Args:
            energies: Energy grid
            dos: DOS values

        Returns:
            E_F: Fermi energy
        """
        idos = self._integrate_dos(energies, dos)
        n_electrons = self.hamiltonian.n_electrons

        # DOS now carries the spin factor (see compute_dos), so the IDOS counts
        # electrons; E_F is where the integrated DOS reaches n_electrons.
        idx = np.argmin(np.abs(idos - n_electrons))
        E_F = energies[idx]

        return E_F

    def compute_pdos(self,
                    atom_indices: Optional[List[int]] = None,
                    energy_range: Tuple[float, float] = (-10, 10),
                    n_points: int = 1000,
                    sigma: float = 0.1,
                    units: str = 'eV') -> Dict[str, Any]:
        """
        Compute projected density of states (PDOS).

        Projects DOS onto specified atoms to identify their contributions
        to electronic states.

        Args:
            atom_indices: List of atom indices to project onto (None = all)
            energy_range: (E_min, E_max) in eV
            n_points: Energy grid points
            sigma: Broadening in eV
            units: 'eV' or 'Ha'

        Returns:
            result: Dictionary with:
                - energies: Energy grid
                - pdos: Dictionary mapping atom_idx → PDOS array
                - total_dos: Total DOS for comparison
        """
        # Honest: real PDOS needs per-atom orbital projections, not an equal
        # split of the total DOS (which is physically meaningless).
        raise NotImplementedError(
            "PDOS requires Mulliken/Lowdin projection of PySCF MOs; not implemented."
        )

    def find_band_gap(self) -> Dict[str, Any]:
        """
        Return the band gap from the Hamiltonian's eigenvalues.

        NOTE: despite living on the DOS calculator, this does NOT read the broadened
        DOS — it delegates to ``PeriodicHamiltonian.get_band_gap()``, which derives
        the VBM/CBM directly from the mean-field band energies. (A broadening-based
        gap would be smearing-width dependent and less precise than the eigenvalues.)
        The returned dict is self-describing — in particular it carries ``method``
        ('HF'), ``gamma_only``, and a ``caveat`` flagging HF gap overestimation; read
        those before treating the number as a physical band gap.
        """
        return self.hamiltonian.get_band_gap()

    def find_van_hove_singularities(self,
                                    energy_range: Tuple[float, float] = (-10, 10),
                                    threshold: float = 2.0) -> List[Dict[str, float]]:
        """
        Identify Van Hove singularities in DOS.

        Van Hove singularities are peaks in DOS where ∇_k E(k) = 0.

        Args:
            energy_range: Energy window to search (eV)
            threshold: Minimum DOS value to qualify as singularity

        Returns:
            singularities: List of dictionaries with:
                - energy: Energy of singularity
                - dos_value: DOS at singularity
                - type: 'M0' (minimum), 'M1' (saddle), 'M2' (saddle), 'M3' (maximum)
        """
        # Compute DOS
        result = self.compute_dos(energy_range, n_points=2000, sigma=0.05)
        energies = result['energies']
        dos = result['dos']

        # Find local maxima
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(dos, height=threshold, prominence=0.5)

        singularities = []
        for peak_idx in peaks:
            singularities.append({
                'energy': energies[peak_idx],
                'dos_value': dos[peak_idx],
                'type': 'unknown'  # Would need ∇²E analysis to classify
            })

        logger.info(f"Found {len(singularities)} Van Hove singularities")

        return singularities

    def plot_dos(self,
                 result: Optional[Dict[str, Any]] = None,
                 energy_range: Tuple[float, float] = (-10, 10),
                 show_fermi: bool = True,
                 show_gap: bool = True,
                 save_path: Optional[str] = None):
        """
        Plot density of states.

        Args:
            result: Pre-computed DOS result (None = compute now)
            energy_range: Energy range for plot
            show_fermi: Show Fermi level line
            show_gap: Shade band gap region
            save_path: Path to save figure (None = show)
        """
        import matplotlib.pyplot as plt

        if result is None:
            result = self.compute_dos(energy_range)

        energies = result['energies']
        dos = result['dos']
        E_F = result['fermi_energy']

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(energies, dos, 'b-', linewidth=2, label='DOS')
        ax.fill_between(energies, 0, dos, alpha=0.3)

        if show_fermi:
            ax.axvline(E_F, color='r', linestyle='--', linewidth=2,
                      label=f'E_F = {E_F:.2f} eV')

        if show_gap:
            gap_info = self.find_band_gap()
            if gap_info['gap'] > 0.1:  # Only show if significant gap
                vbm = gap_info['vbm']
                cbm = gap_info['cbm']
                ax.axvspan(vbm, cbm, alpha=0.2, color='gray',
                          label=f'Gap = {gap_info["gap"]:.2f} eV')

        ax.set_xlabel('Energy (eV)', fontsize=14)
        ax.set_ylabel('DOS (states/eV)', fontsize=14)
        ax.set_title('Density of States', fontsize=16)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(energy_range)
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"DOS plot saved to {save_path}")
        else:
            plt.show()

    def plot_band_structure_with_dos(self,
                                     band_result: Dict[str, Any],
                                     dos_result: Optional[Dict[str, Any]] = None,
                                     save_path: Optional[str] = None):
        """
        Plot band structure and DOS side-by-side.

        Args:
            band_result: Result from compute_band_structure()
            dos_result: Result from compute_dos() (None = compute)
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt

        if dos_result is None:
            # Determine energy range from bands
            E_min = np.min(band_result['band_energies']) - 2
            E_max = np.max(band_result['band_energies']) + 2
            dos_result = self.compute_dos((E_min, E_max))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                       gridspec_kw={'width_ratios': [2, 1]})

        # Band structure
        k_dist = band_result['k_distances']
        bands = band_result['band_energies']

        for band_idx in range(bands.shape[1]):
            ax1.plot(k_dist, bands[:, band_idx], 'b-', linewidth=1.5)

        # High-symmetry points
        if 'labels' in band_result and 'label_positions' in band_result:
            for pos in band_result['label_positions']:
                ax1.axvline(pos, color='gray', linestyle='--', alpha=0.5)
            ax1.set_xticks(band_result['label_positions'])
            ax1.set_xticklabels(band_result['labels'])

        ax1.axhline(dos_result['fermi_energy'], color='r', linestyle='--',
                   linewidth=2, label='E_F')
        ax1.set_xlabel('k-path', fontsize=14)
        ax1.set_ylabel('Energy (eV)', fontsize=14)
        ax1.set_title('Band Structure', fontsize=16)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # DOS
        ax2.plot(dos_result['dos'], dos_result['energies'], 'b-', linewidth=2)
        ax2.fill_betweenx(dos_result['energies'], 0, dos_result['dos'], alpha=0.3)
        ax2.axhline(dos_result['fermi_energy'], color='r', linestyle='--', linewidth=2)

        # Match y-axis
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel('DOS (states/eV)', fontsize=14)
        ax2.set_ylabel('Energy (eV)', fontsize=14)
        ax2.set_title('Density of States', fontsize=16)
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Band+DOS plot saved to {save_path}")
        else:
            plt.show()

    def compute_quantum_dos(
        self,
        bond_or_molecule,
        energy_range: Tuple[float, float] = (-10, 10),
        n_points: int = 1000,
        n_states: int = 20,
        sigma: float = 0.1,
        solver: str = 'sqd',
        backend: str = 'statevector',
        use_governance: bool = True,
        resolve_bonding: bool = False,
        units: str = 'eV',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute molecular DOS from quantum eigenstates.

        Optional governance-guided subspace reduction.

        NOTE: Bonding-type resolved DOS (covalent/ionic/metallic separation) is
        NOT implemented. The previous implementation fabricated those weights
        with np.random.rand(), so calling with resolve_bonding=True now raises
        NotImplementedError instead of returning random noise as an observable.

        Args:
            bond_or_molecule: Bond or Molecule object
            energy_range: (E_min, E_max) in eV
            n_points: Number of energy grid points
            n_states: Number of quantum eigenstates to compute
            sigma: Gaussian broadening (eV)
            solver: 'sqd', 'vqe', or 'adapt'
            backend: 'statevector', 'aer', 'ibm', 'bluequbit'
            use_governance: Enable governance-guided subspace (5-10x speedup)
            resolve_bonding: NOT IMPLEMENTED; must be False. True raises
                NotImplementedError (see note above).
            units: 'eV' or 'Ha'
            verbose: Print progress

        Returns:
            result: Dictionary with:
                - energies: Energy grid (eV or Ha)
                - dos_total: Total DOS (states/eV)
                - eigenstates: List of eigenstate energies and characters
                - fermi_energy: Fermi level
                - homo_lumo_gap: HOMO-LUMO gap

        Examples:
            >>> # Basic quantum DOS
            >>> dos_calc = DOSCalculator(None)  # No periodic hamiltonian
            >>> result = dos_calc.compute_quantum_dos(
            ...     bond=h2_bond,
            ...     n_states=20,
            ...     solver='sqd',
            ...     backend='statevector',
            ...     resolve_bonding=False
            ... )
        """
        # Bonding-resolved DOS requires projecting each eigenvector onto
        # bonding/antibonding MO subspaces (overlap weights from the
        # Hamiltonian's MO coefficients). That machinery does not exist yet, and
        # the previous implementation fabricated weights with np.random.rand(),
        # i.e. it returned random noise as a physical observable. Refuse rather
        # than emit fake observables.
        if resolve_bonding:
            raise NotImplementedError(
                "bonding-resolved DOS requires real eigenstate-vs-bonding-orbital "
                "overlaps; not yet implemented. Call with resolve_bonding=False for "
                "total quantum DOS."
            )

        if verbose:
            logger.info(f"\n{'='*70}")
            logger.info(f"QUANTUM DOS")
            logger.info(f"{'='*70}")
            logger.info(f"Solver: {solver.upper()}")
            logger.info(f"Backend: {backend}")
            logger.info(f"Governance: {'ON' if use_governance else 'OFF'}")
            logger.info(f"Bonding resolution: {'ON' if resolve_bonding else 'OFF'}")
            logger.info(f"{'='*70}")

        # Import quantum solvers
        from kanad.solvers import DeterministicCI, VQESolver

        # Get bond object
        if hasattr(bond_or_molecule, 'bonds'):
            # Molecule object - use first bond
            bond = bond_or_molecule.bonds[0] if bond_or_molecule.bonds else None
            if bond is None:
                raise ValueError("Molecule has no bonds")
        else:
            bond = bond_or_molecule

        # Select solver
        if solver.lower() == 'sqd':
            quantum_solver = DeterministicCI(
                bond=bond,
                subspace_dim=n_states,
                backend=backend,
                use_governance=use_governance
            )
        elif solver.lower() == 'vqe':
            quantum_solver = VQESolver(
                bond=bond,
                backend=backend,
                max_iter=100
            )
        else:
            raise ValueError(f"Unknown solver: {solver}. Available: 'sqd', 'vqe'")

        # Solve for eigenstates
        if verbose:
            logger.info(f"Computing {n_states} quantum eigenstates...")

        # Solvers now return a SolverResult; flatten to the legacy dict so the
        # key-membership checks below work ('eigenvalues' in SolverResult would try
        # to iterate it -> TypeError). (planck full-analysis audit.)
        # NOTE: solve() takes no n_states (DeterministicCI gets it via subspace_dim
        # in the constructor; VQESolver is ground-state only) — passing it raised
        # "unexpected keyword argument 'n_states'". (F14)
        result_solver = quantum_solver.solve()
        if hasattr(result_solver, 'to_dict'):
            result_solver = result_solver.to_dict()

        # Extract eigenvalues. The SolverResult.to_dict() spectrum lives under
        # 'energies' (DeterministicCI/SQD) — there is NO 'eigenvalues' key, so the
        # legacy first branch was dead and the code collapsed the whole subspace
        # spectrum to the single scalar 'energy' (n_states=4 -> 1 eigenstate).
        # Read 'energies' before the single-energy fallback. (audit H6)
        if 'eigenvalues' in result_solver:
            eigenvalues = np.asarray(result_solver['eigenvalues'])  # Already in Hartree
        elif result_solver.get('energies') is not None:
            eigenvalues = np.asarray(result_solver['energies'])  # Full SQD spectrum (Hartree)
        elif 'energy' in result_solver:
            eigenvalues = np.array([result_solver['energy']])
        else:
            raise ValueError("Solver did not return eigenvalues")

        # Convert to eV if needed
        Ha_to_eV = 27.2114
        if units == 'eV':
            eigenvalues_eV = eigenvalues * Ha_to_eV
            E_min, E_max = energy_range
            sigma_eV = sigma
        else:
            eigenvalues_eV = eigenvalues
            E_min = energy_range[0] * Ha_to_eV
            E_max = energy_range[1] * Ha_to_eV
            sigma_eV = sigma * Ha_to_eV

        # Create energy grid
        energies = np.linspace(E_min, E_max, n_points)
        dos_total = np.zeros(n_points)

        # Get governance protocol for bonding character
        governance = bond.governance if hasattr(bond, 'governance') else None
        bond_type = None
        if governance:
            bond_type = governance.bond_type.value if hasattr(governance.bond_type, 'value') else str(governance.bond_type)

        if verbose:
            logger.info(f"✓ Computed {len(eigenvalues)} eigenstates")
            logger.info(f"  Bond type: {bond_type if bond_type else 'Unknown'}")
            logger.info(f"  Energy range: {eigenvalues_eV[0]:.3f} to {eigenvalues_eV[-1]:.3f} eV")

        # Build DOS with Gaussian broadening
        eigenstates_info = []

        for i, E_i in enumerate(eigenvalues_eV):
            # Gaussian broadening
            gaussian = np.exp(-0.5 * ((energies - E_i) / sigma_eV)**2) / (sigma_eV * np.sqrt(2 * np.pi))
            dos_total += gaussian

            # Bonding-character resolution is not implemented (it would require
            # real eigenstate-vs-bonding-orbital overlaps). resolve_bonding=True
            # is rejected earlier, so no bonding weights are fabricated here.
            eigenstates_info.append({
                'energy': E_i,
                'covalent_character': None,
                'ionic_character': None,
                'metallic_character': None
            })

        # Compute HOMO-LUMO gap (molecular systems)
        n_electrons = bond.hamiltonian.n_electrons if hasattr(bond.hamiltonian, 'n_electrons') else 2
        homo_idx = n_electrons // 2 - 1
        lumo_idx = n_electrons // 2

        if lumo_idx < len(eigenvalues_eV):
            homo_energy = eigenvalues_eV[homo_idx]
            lumo_energy = eigenvalues_eV[lumo_idx]
            gap = lumo_energy - homo_energy
        else:
            homo_energy = eigenvalues_eV[0] if len(eigenvalues_eV) > 0 else 0.0
            lumo_energy = None
            gap = None

        # Fermi energy (middle of gap for molecules)
        fermi_energy = (homo_energy + (lumo_energy if lumo_energy else homo_energy)) / 2.0 if gap else homo_energy

        # governance_advantage removed: it was a hardcoded 7x estimate
        # (full_space_dim = n*7; advantage = full/n == 7.0), not a measured
        # subspace reduction. Omit rather than fabricate a metric.

        if verbose:
            logger.info(f"\n📊 DOS Statistics:")
            logger.info(f"  HOMO: {homo_energy:.3f} eV")
            logger.info(f"  LUMO: {lumo_energy:.3f} eV" if lumo_energy else "  LUMO: N/A")
            logger.info(f"  Gap: {gap:.3f} eV" if gap else "  Gap: N/A")
            logger.info(f"  Fermi level: {fermi_energy:.3f} eV")
            logger.info(f"{'='*70}")

        # Build result
        result = {
            'energies': energies,
            'dos_total': dos_total,
            'eigenstates': eigenstates_info,
            'fermi_energy': fermi_energy,
            'homo_energy': homo_energy,
            'lumo_energy': lumo_energy,
            'homo_lumo_gap': gap,
            'n_states': len(eigenvalues),
            'solver': solver,
            'backend': backend,
            'governance_enabled': use_governance,
            'bond_type': bond_type,
            'units': units
        }

        # Bonding-resolved DOS is not implemented (resolve_bonding=True is
        # rejected earlier), so no fabricated dos_covalent/ionic/metallic keys
        # are added to the result.

        return result

    def __repr__(self) -> str:
        """String representation."""
        # hamiltonian is None in molecular DOS mode; guard against AttributeError
        n_electrons = self.hamiltonian.n_electrons if self.hamiltonian is not None else None
        return (f"DOSCalculator(n_k={self.n_k}, n_bands={self.n_bands}, "
                f"n_electrons={n_electrons})")
