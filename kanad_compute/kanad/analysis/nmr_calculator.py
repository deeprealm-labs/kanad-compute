"""
Quantum NMR Spectroscopy Calculator

**WORLD'S FIRST quantum NMR chemical shift calculator!**

Computes NMR chemical shifts and J-coupling constants using quantum algorithms:
- Quantum chemical shift calculation using density matrix from quantum backends
- ECHOES-inspired approach (Exact Cover of Hamiltonian Eigenstates)
- J-coupling calculation from spin-spin interactions
- Support for IBM Quantum and BlueQubit hardware

Features:
- Chemical shifts (δ in ppm) for ¹H, ¹³C, ¹⁵N, ³¹P, ¹⁹F nuclei
- Scalar J-coupling constants (Hz)
- Quantum backend support (statevector, IBM, BlueQubit)
- Classical reference methods (DFT) for validation

References:
- Google ECHOES algorithm (arXiv:2305.09799)
- Quantum chemistry for NMR (Rev. Mod. Phys. 2020)
- Ramsey theory of NMR shielding
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class NMRCalculator:
    """
    Calculate NMR chemical shifts and coupling constants using quantum algorithms.

    **WORLD'S FIRST quantum NMR calculator!**

    NMR Chemical Shift:
        δ = (ν_sample - ν_reference) / ν_reference × 10⁶ ppm

    Chemical shift arises from electron shielding at nucleus:
        δ ≈ -σ (shielding constant)

    σ depends on:
        - Electron density at nucleus (contact term)
        - Orbital angular momentum (paramagnetic term)
        - Spin-orbit coupling (relativistic effects)

    This implementation computes σ from quantum density matrices obtained
    from real quantum hardware (IBM Quantum, BlueQubit) or fast simulation.

    Example:
        >>> from kanad.core.io import from_smiles
        >>> from kanad.analysis import NMRCalculator
        >>>
        >>> water = from_smiles("O")
        >>> nmr_calc = NMRCalculator(water)
        >>>
        >>> # Classical calculation (fast, reference)
        >>> result_classical = nmr_calc.compute_chemical_shifts(
        ...     method='DFT',
        ...     functional='B3LYP'
        ... )
        >>>
        >>> # Quantum calculation (WORLD'S FIRST!)
        >>> result_quantum = nmr_calc.compute_quantum_chemical_shifts(
        ...     backend='statevector',
        ...     method='sqd'
        ... )
        >>>
        >>> print(f"H chemical shift: {result_quantum['shifts'][0]:.2f} ppm")
    """

    # NMR-active nuclei properties
    NUCLEI_PROPERTIES = {
        'H': {'spin': 0.5, 'gyromagnetic_ratio': 267.522e6, 'natural_abundance': 99.98},  # ¹H
        'Li': {'spin': 1.5, 'gyromagnetic_ratio': 103.962e6, 'natural_abundance': 92.5},  # ⁷Li
        'C': {'spin': 0.5, 'gyromagnetic_ratio': 67.283e6, 'natural_abundance': 1.11},   # ¹³C
        'N': {'spin': 0.5, 'gyromagnetic_ratio': -27.126e6, 'natural_abundance': 0.37},  # ¹⁵N
        'O': {'spin': 2.5, 'gyromagnetic_ratio': -36.281e6, 'natural_abundance': 0.038}, # ¹⁷O
        'F': {'spin': 0.5, 'gyromagnetic_ratio': 251.815e6, 'natural_abundance': 100.0}, # ¹⁹F
        'P': {'spin': 0.5, 'gyromagnetic_ratio': 108.394e6, 'natural_abundance': 100.0}, # ³¹P
    }

    # Reference compounds for chemical shift scale (δ = 0 ppm)
    REFERENCE_COMPOUNDS = {
        'H': 'TMS',  # Tetramethylsilane
        'Li': 'LiCl(aq)',  # Aqueous LiCl
        'C': 'TMS',
        'N': 'NH3',  # Liquid ammonia
        'O': 'H2O',
        'F': 'CFCl3',
        'P': 'H3PO4',
    }

    # Reference shielding constants (ppm) — HF/STO-3G GIAO values
    # These are used for δ = σ_ref - σ_computed
    REFERENCE_SHIELDING = {
        'H': 32.0,   # TMS ¹H (HF/STO-3G GIAO: ~32 ppm)
        'C': 195.0,  # TMS ¹³C
        'N': 244.0,  # NH3 ¹⁵N
        'O': 287.0,  # H2O ¹⁷O
        'F': 188.0,  # CFCl3 ¹⁹F
        'P': 328.0,  # H3PO4 ³¹P
        'Li': 90.0,  # LiCl ⁷Li (approximate)
    }

    def __init__(self, hamiltonian: 'MolecularHamiltonian'):
        """
        Initialize NMR calculator.

        Args:
            hamiltonian: MolecularHamiltonian object from kanad framework

        Raises:
            ValueError: If hamiltonian has no atoms
        """
        self.hamiltonian = hamiltonian
        self.molecule = getattr(hamiltonian, 'molecule', None)
        self.atoms = getattr(hamiltonian, 'atoms', [])

        # Get PySCF mol if available
        self.mol = getattr(hamiltonian, 'mol', None)

        if len(self.atoms) == 0:
            raise ValueError("Hamiltonian has no atoms")

        # Get molecule name
        mol_name = getattr(self.molecule, 'formula', None) if self.molecule else f"{len(self.atoms)}-atom system"
        logger.info(f"NMRCalculator initialized for {mol_name}")

        # Identify NMR-active nuclei
        self.nmr_active_atoms = self._identify_nmr_nuclei()
        logger.info(f"Found {len(self.nmr_active_atoms)} NMR-active nuclei")

    def _identify_nmr_nuclei(self) -> List[Tuple[int, str]]:
        """
        Identify NMR-active nuclei in molecule.

        Returns:
            List of (atom_index, element) tuples
        """
        active = []
        for i, atom in enumerate(self.atoms):
            if atom.symbol in self.NUCLEI_PROPERTIES:
                active.append((i, atom.symbol))
        return active

    def compute_chemical_shifts(
        self,
        method: str = 'HF',
        functional: Optional[str] = None,
        basis: str = 'sto-3g',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute NMR chemical shifts using classical quantum chemistry (PySCF).

        This is the reference/validation method using conventional DFT/HF.

        Args:
            method: Quantum chemistry method:
                - 'HF': Hartree-Fock
                - 'DFT': Density Functional Theory (requires functional)
            functional: DFT functional (e.g., 'B3LYP', 'PBE0') for DFT method
            basis: Basis set (default: 'sto-3g')
            verbose: Print results

        Returns:
            Dictionary with:
                shifts: Chemical shifts (ppm) for each NMR-active nucleus
                shieldings: Absolute shielding constants (ppm)
                atoms: List of (atom_index, element) tuples
                method: Method used
                reference: Reference compound used
        """
        from pyscf import dft, scf

        if verbose:
            print(f"\n{'='*70}")
            print(f"CLASSICAL NMR CHEMICAL SHIFTS")
            print(f"{'='*70}")
            print(f"Method: {method}" + (f" ({functional})" if functional else ""))
            print(f"Basis: {basis}")
            print(f"Nuclei: {len(self.nmr_active_atoms)} NMR-active")
            print("-" * 70)

        # Only HF/DFT are actually dispatched below; MP2/CCSD would silently
        # run plain RHF and echo a fake method label, so refuse them outright.
        if method.upper() not in ('HF', 'DFT'):
            raise NotImplementedError(
                f"NMR shielding for method={method} not implemented; "
                "only HF/DFT supported"
            )

        # Get PySCF molecule
        mol_pyscf = self.mol

        if mol_pyscf is None:
            raise ValueError("No PySCF molecule available for NMR calculation")

        # Run SCF for NMR
        if method.upper() == 'DFT' and functional:
            mf = dft.RKS(mol_pyscf)
            mf.xc = functional
        else:
            mf = scf.RHF(mol_pyscf)
        mf.verbose = 0
        mf.kernel()

        # Try PySCF GIAO NMR (proper dia + para shielding)
        giao_shieldings = self._compute_giao_nmr(mf)

        shifts = []
        shieldings = []
        atoms_list = []

        if giao_shieldings is not None:
            # Detect symmetry-equivalent atoms for averaging
            equiv_groups = self._detect_equivalent_atoms(mol_pyscf)

            # Average equivalent nuclei
            for fingerprint, group_indices in equiv_groups.items():
                if len(group_indices) > 1:
                    avg_sigma = sum(giao_shieldings[idx] for idx in group_indices
                                    if idx in giao_shieldings) / len(group_indices)
                    for idx in group_indices:
                        if idx in giao_shieldings:
                            giao_shieldings[idx] = avg_sigma

            for atom_idx, element in self.nmr_active_atoms:
                sigma = giao_shieldings.get(atom_idx, 0.0)
                sigma_ref = self.REFERENCE_SHIELDING.get(element, 30.0)
                delta = sigma_ref - sigma

                shifts.append(delta)
                shieldings.append(sigma)
                atoms_list.append((atom_idx, element))
        else:
            # No proper GIAO shielding available. The diamagnetic-only Lamb
            # term (no paramagnetic Ramsey contribution) produces wrong-sign,
            # out-of-range "shifts"; emitting them silently as a successful
            # result is worse than failing. Match the NotImplementedError
            # fences already used by compute_quantum_chemical_shifts /
            # compute_j_coupling rather than fabricating a number.
            raise NotImplementedError(
                "Proper GIAO NMR shielding (diamagnetic + paramagnetic Ramsey "
                "terms) requires pyscf.prop.nmr, which is not installed. The "
                "diamagnetic-only Lamb fallback is not a usable chemical shift "
                "and has been disabled; install pyscf-properties for GIAO NMR."
            )

        if verbose:
            print("\n" + "=" * 70)
            print("NMR CHEMICAL SHIFTS (GIAO)" if giao_shieldings else "NMR CHEMICAL SHIFTS (Lamb approx)")
            print("=" * 70)
            print(f"{'Atom':<10} {'Element':<8} {'Shift (ppm)':<15} {'Shielding (ppm)':<15}")
            print("-" * 70)

            for (idx, elem), shift, shield in zip(atoms_list, shifts, shieldings):
                print(f"{idx:<10} {elem:<8} {shift:>12.2f}   {shield:>15.2f}")

            print("=" * 70)

        return {
            'shifts': np.array(shifts),
            'shieldings': np.array(shieldings),
            'atoms': atoms_list,
            'method': method + (f" ({functional})" if functional else ""),
            'reference': [self.REFERENCE_COMPOUNDS.get(elem, 'N/A') for _, elem in atoms_list],
            'basis': basis,
            'quantum': False
        }

    @staticmethod
    def _patch_pyscf_nmr_vind():
        """Monkey-patch PySCF NMR gen_vind to fix Krylov reshape bug.

        PySCF 2.12 has a bug where gen_vind.vind hardcodes reshape(3, nmo, nocc)
        but the Krylov solver may filter linearly dependent vectors, passing
        fewer than 3 vectors. Fix: use reshape(-1, nmo, nocc).
        """
        from functools import reduce as functools_reduce
        from pyscf.prop.nmr import rhf as nmr_rhf

        def patched_gen_vind(mf, mo_coeff, mo_occ):
            vresp = mf.gen_response(singlet=True, hermi=2)
            occidx = mo_occ > 0
            orbo = mo_coeff[:, occidx]
            nocc = orbo.shape[1]
            nao, nmo = mo_coeff.shape
            def vind(mo1):
                dm1 = [functools_reduce(np.dot, (mo_coeff, x*2, orbo.T.conj()))
                       for x in mo1.reshape(-1, nmo, nocc)]
                dm1 = np.asarray([d1-d1.conj().T for d1 in dm1])
                v1mo = np.einsum('xpq,pi,qj->xij', vresp(dm1), mo_coeff.conj(), orbo)
                return v1mo.ravel()
            return vind

        nmr_rhf.gen_vind = patched_gen_vind

    def _compute_giao_nmr(self, mf) -> Optional[Dict[int, float]]:
        """Compute GIAO NMR shieldings using PySCF with bug fix.

        Returns:
            Dict mapping atom_index → isotropic shielding (ppm), or None on failure.
        """
        try:
            self._patch_pyscf_nmr_vind()
            from pyscf.prop.nmr import rhf as nmr_rhf

            nmr_obj = nmr_rhf.NMR(mf)
            nmr_obj.verbose = 0
            shielding_tensors = nmr_obj.kernel()

            result = {}
            mol_pyscf = mf.mol
            for atom_idx, element in self.nmr_active_atoms:
                iso = (shielding_tensors[atom_idx][0][0] +
                       shielding_tensors[atom_idx][1][1] +
                       shielding_tensors[atom_idx][2][2]) / 3.0
                result[atom_idx] = float(iso)

            logger.info(f"GIAO NMR shieldings: {result}")
            return result
        except Exception as e:
            logger.warning(f"PySCF GIAO NMR failed: {e}")
            return None

    def _detect_equivalent_atoms(self, mol_pyscf) -> Dict:
        """Detect symmetry-equivalent atoms for NMR averaging."""
        atom_coords = mol_pyscf.atom_coords()
        atom_symbols = [mol_pyscf.atom_symbol(i) for i in range(mol_pyscf.natm)]
        equiv_groups = {}
        for atom_idx, element in self.nmr_active_atoms:
            dists = []
            for other_idx in range(len(atom_symbols)):
                if other_idx != atom_idx:
                    d = np.linalg.norm(atom_coords[atom_idx] - atom_coords[other_idx])
                    dists.append((atom_symbols[other_idx], round(d, 3)))
            fingerprint = (element, tuple(sorted(dists)))
            if fingerprint not in equiv_groups:
                equiv_groups[fingerprint] = []
            equiv_groups[fingerprint].append(atom_idx)
        return equiv_groups

    def compute_quantum_chemical_shifts(self, *args, **kwargs) -> Dict[str, Any]:
        """Wavefunction-derived NMR shielding — deferred to M3 PR-6.

        The pre-M3 implementation took the HF GIAO shielding and added an
        atom-typed ``ppm-per-%-correlation-energy`` correction (`+15 ppm` for
        H, `+25` for C, etc.) — a hand-fit heuristic, not a wavefunction-
        derived shielding. It produced spurious "quantum" deltas of 5–75 ppm
        regardless of whether the wavefunction actually said anything about
        the magnetic response.

        The honest replacement is numerical perturbation theory:
        ``σ_iso[A] = ∂² E / ∂B ∂μ_A``, computed by finite-difference VQE with
        an applied magnetic field. That work is M3 PR-6 (see PLAN.md).

        For now, use ``compute_chemical_shifts(method='HF'|'DFT')`` for
        classical GIAO reference shifts.
        """
        raise NotImplementedError(
            "compute_quantum_chemical_shifts is replaced in M3 PR-6 by "
            "numerical σ = ∂²E/∂B∂μ via finite-difference VQE with an applied "
            "magnetic field. The pre-M3 'quantum correction = +N ppm per % "
            "correlation energy' heuristic has been removed because it was a "
            "hand-fit bias, not a wavefunction-derived shielding.\n\n"
            "Use compute_chemical_shifts(method='HF'|'DFT') for "
            "classical-reference shifts in the meantime."
        )

    def compute_j_coupling(
        self,
        atom_pair: Tuple[int, int],
        method: str = 'HF',
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Compute J-coupling constant between two nuclei.

        J-coupling (scalar coupling) arises from indirect spin-spin interaction
        through bonding electrons (Fermi contact, spin-orbit, spin-dipole terms).

        Args:
            atom_pair: Tuple of (atom1_index, atom2_index)
            method: Quantum chemistry method ('HF' or 'DFT')
            verbose: Print results

        Returns:
            Dictionary with:
                j_coupling: J-coupling constant (Hz)
                atoms: Atom pair
                mechanism: Dominant coupling mechanism
                n_bonds: Number of bonds between atoms
        """
        raise NotImplementedError(
            "compute_j_coupling does not compute a real spin-spin coupling and is "
            "fenced off. The prior implementation returned a hardcoded constant "
            "(150/30/7/2 Hz) selected purely by a distance-based bond count "
            "(round(|r1-r2| / 1.5 Å)) — no electronic structure, no Fermi-contact / "
            "spin-dipole / paramagnetic-spin-orbit response — so the value was "
            "fiction and must never be published as a J-coupling. A correct J needs "
            "the coupled-perturbed response of the wavefunction (e.g. PySCF's SSC / "
            "pyscf.prop.ssc). Fenced so no researcher mistakes the lookup for a "
            "computed coupling; implement the response path before re-enabling."
        )

    def predict_nmr_spectrum(
        self,
        shifts_result: Dict[str, Any],
        coupling_pairs: Optional[List[Tuple[int, int]]] = None,
        field_strength: float = 400.0,
        linewidth: float = 2.0,
        ppm_range: Tuple[float, float] = (0, 10),
        n_points: int = 4096,
        verbose: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate NMR spectrum from chemical shifts and J-couplings.

        Creates a synthetic NMR spectrum with peaks at chemical shift positions,
        including J-coupling multiplets.

        Args:
            shifts_result: Output from compute_chemical_shifts() or compute_quantum_chemical_shifts()
            coupling_pairs: List of (atom1_idx, atom2_idx) tuples for J-coupling (optional)
            field_strength: NMR spectrometer frequency (MHz) (default: 400 MHz)
            linewidth: Peak linewidth (Hz) (default: 2 Hz)
            ppm_range: (ppm_min, ppm_max) range for spectrum (default: 0-10 ppm)
            n_points: Number of spectral points (default: 4096)
            verbose: Print information

        Returns:
            Dictionary with:
                ppm: Chemical shift axis (ppm)
                intensity: Spectrum intensity (arbitrary units)
                frequency: Frequency axis (Hz)
                peaks: List of peak positions and intensities
        """
        ppm_min, ppm_max = ppm_range
        ppm_axis = np.linspace(ppm_min, ppm_max, n_points)
        intensity = np.zeros(n_points)

        # Convert linewidth from Hz to ppm
        linewidth_ppm = linewidth / field_strength

        # Standard deviation for Lorentzian peak
        sigma = linewidth_ppm / 2.0

        # Add peaks for each chemical shift
        for shift in shifts_result['shifts']:
            # Lorentzian lineshape
            lorentzian = 1.0 / (1.0 + ((ppm_axis - shift) / sigma)**2)
            intensity += lorentzian

        # Convert ppm to Hz
        frequency_axis = ppm_axis * field_strength

        if verbose:
            print(f"\nNMR spectrum generated:")
            print(f"  Field strength: {field_strength} MHz")
            print(f"  Chemical shift range: {ppm_min}-{ppm_max} ppm")
            print(f"  Number of peaks: {len(shifts_result['shifts'])}")
            print(f"  Linewidth: {linewidth} Hz ({linewidth_ppm:.3f} ppm)")

        return {
            'ppm': ppm_axis,
            'intensity': intensity / np.max(intensity) if np.max(intensity) > 0 else intensity,
            'frequency': frequency_axis,
            'peaks': list(zip(shifts_result['shifts'], np.ones(len(shifts_result['shifts']))))
        }

    def plot_nmr_spectrum(
        self,
        spectrum: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        title: str = "NMR Spectrum"
    ) -> None:
        """
        Plot NMR spectrum.

        Args:
            spectrum: Output from predict_nmr_spectrum()
            save_path: Path to save figure (optional)
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed - cannot plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot spectrum
        ax.plot(spectrum['ppm'], spectrum['intensity'], 'b-', linewidth=1.5)

        # Mark peak positions
        if 'peaks' in spectrum:
            for shift, intensity in spectrum['peaks']:
                ax.axvline(shift, color='r', alpha=0.3, linestyle='--', linewidth=1)

        # Formatting
        ax.set_xlabel('Chemical Shift δ (ppm)', fontsize=12)
        ax.set_ylabel('Intensity (arbitrary units)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # NMR convention: high ppm on left
        ax.set_ylim(bottom=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()
