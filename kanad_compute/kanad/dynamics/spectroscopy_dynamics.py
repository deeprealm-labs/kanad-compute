"""
Spectroscopy-Dynamics Connection

Computes spectroscopic properties from molecular dynamics trajectories,
enabling dynamical spectroscopy simulations.

Key Capabilities:
- IR spectra from dipole autocorrelation
- Raman spectra from polarizability autocorrelation
- NMR relaxation from trajectory correlations
- UV-Vis from excitation along trajectory

Theory:
------
IR absorption: α(ω) ∝ ω * Re[∫ <μ(0)·μ(t)> exp(-iωt) dt]
Raman: I(ω) ∝ Re[∫ <α(0):α(t)> exp(-iωt) dt]
NMR T1: 1/T1 ∝ ∫ <B(0)·B(t)> cos(ω₀t) dt

References:
----------
- Allen & Tildesley (2017) Computer Simulation of Liquids, Ch. 11
- McQuarrie (2000) Statistical Mechanics
- Kowalewski & Maler (2019) Nuclear Spin Relaxation in Liquids
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DynamicalSpectrum:
    """
    Spectrum computed from MD trajectory.

    Attributes:
        frequencies: Frequency array in cm⁻¹
        intensities: Spectral intensities (arbitrary units)
        spectrum_type: Type of spectrum ('ir', 'raman', 'power')
        temperature: Temperature at which trajectory was run (K)
        correlation_time: Correlation time used (fs)
        n_frames: Number of trajectory frames used
    """
    frequencies: np.ndarray
    intensities: np.ndarray
    spectrum_type: str
    temperature: float
    correlation_time: float
    n_frames: int


def compute_autocorrelation(
    property_array: np.ndarray,
    max_lag: Optional[int] = None
) -> np.ndarray:
    """
    Compute time autocorrelation function.

    C(t) = <A(0) · A(t)>

    Args:
        property_array: Time series (n_frames, ...) of property
        max_lag: Maximum lag time in frames (default: n_frames // 2)

    Returns:
        acf: Autocorrelation function (max_lag,)
    """
    n_frames = len(property_array)

    if max_lag is None:
        max_lag = n_frames // 2

    # Flatten if multidimensional
    if property_array.ndim > 1:
        # For vectors (e.g., dipole), compute dot product
        flat = property_array.reshape(n_frames, -1)
    else:
        flat = property_array[:, np.newaxis]

    # Compute ACF using FFT for efficiency
    acf = np.zeros(max_lag)

    for dim in range(flat.shape[1]):
        signal = flat[:, dim]

        # Zero-pad for FFT
        n_fft = 2 * n_frames
        fft_signal = np.fft.fft(signal, n_fft)
        power_spectrum = np.abs(fft_signal)**2
        acf_full = np.fft.ifft(power_spectrum)[:n_frames].real

        acf += acf_full[:max_lag]

    # Normalize by number of samples
    normalization = np.arange(n_frames, n_frames - max_lag, -1)
    acf /= normalization

    return acf


def compute_ir_from_dipole_autocorrelation(
    trajectory,
    timestep: float = 0.5,
    temperature: float = 300.0,
    max_lag_fs: float = 2000.0,
    frequency_resolution: float = 1.0
) -> DynamicalSpectrum:
    """
    Compute IR absorption spectrum from dipole moment autocorrelation.

    The IR absorption is proportional to the Fourier transform of the
    dipole autocorrelation function:

    α(ω) ∝ ω² * ∫₀^∞ <μ(0)·μ(t)> exp(-iωt) dt

    Args:
        trajectory: MD trajectory with dipole moments
        timestep: MD timestep in fs
        temperature: Simulation temperature in K
        max_lag_fs: Maximum correlation time in fs
        frequency_resolution: Frequency resolution in cm⁻¹

    Returns:
        DynamicalSpectrum with IR spectrum
    """
    logger.info("Computing IR spectrum from dipole autocorrelation")

    # Extract dipole moments from trajectory
    if hasattr(trajectory, 'dipoles'):
        dipoles = np.array(trajectory.dipoles)
    elif hasattr(trajectory, 'frames'):
        # Extract from frames if available
        dipoles = []
        for frame in trajectory.frames:
            if hasattr(frame, 'dipole'):
                dipoles.append(frame.dipole)
            else:
                # Estimate dipole from positions and charges
                dipoles.append(_estimate_dipole_from_frame(frame))
        dipoles = np.array(dipoles)
    else:
        raise ValueError(
            "compute_ir_from_dipole_autocorrelation: trajectory frames carry no "
            "'dipole' attribute. Fabricating dipoles with np.random returns a "
            "meaningless spectrum; supply per-frame dipoles instead."
        )

    n_frames = len(dipoles)
    max_lag = int(max_lag_fs / timestep)
    max_lag = min(max_lag, n_frames // 2)

    # Compute dipole autocorrelation
    acf = compute_autocorrelation(dipoles, max_lag)

    # Apply window function to reduce spectral leakage
    window = np.hanning(2 * max_lag)[max_lag:]
    acf_windowed = acf * window

    # Fourier transform to get spectrum
    # Pad for better frequency resolution
    n_fft = int(1 / (frequency_resolution * timestep * 2.99792458e-5))  # 2.99792458e-5 = c[cm/fs], fs to cm⁻¹
    n_fft = max(n_fft, 2 * max_lag)

    spectrum_complex = np.fft.fft(acf_windowed, n_fft)
    spectrum = np.abs(spectrum_complex[:n_fft // 2])

    # Create frequency axis
    freq_max = 1 / (2 * timestep * 2.99792458e-5)  # Nyquist in cm⁻¹
    frequencies = np.linspace(0, freq_max, n_fft // 2)

    # Apply quantum correction: multiply by ω²
    spectrum *= frequencies**2

    # Normalize
    if np.max(spectrum) > 0:
        spectrum /= np.max(spectrum)

    result = DynamicalSpectrum(
        frequencies=frequencies,
        intensities=spectrum,
        spectrum_type='ir',
        temperature=temperature,
        correlation_time=max_lag_fs,
        n_frames=n_frames
    )

    logger.info(f"IR spectrum computed: {len(frequencies)} points, 0-{freq_max:.0f} cm⁻¹")

    return result


def compute_raman_from_polarizability_autocorrelation(
    trajectory,
    timestep: float = 0.5,
    temperature: float = 300.0,
    max_lag_fs: float = 2000.0,
    frequency_resolution: float = 1.0
) -> DynamicalSpectrum:
    """
    Compute Raman spectrum from polarizability autocorrelation.

    The Raman intensity is proportional to the Fourier transform of the
    polarizability autocorrelation function:

    I(ω) ∝ ∫₀^∞ <α(0):α(t)> exp(-iωt) dt

    Args:
        trajectory: MD trajectory with polarizabilities
        timestep: MD timestep in fs
        temperature: Simulation temperature in K
        max_lag_fs: Maximum correlation time in fs
        frequency_resolution: Frequency resolution in cm⁻¹

    Returns:
        DynamicalSpectrum with Raman spectrum
    """
    logger.info("Computing Raman spectrum from polarizability autocorrelation")

    # Extract polarizabilities from trajectory
    if hasattr(trajectory, 'polarizabilities'):
        polarizabilities = np.array(trajectory.polarizabilities)
    else:
        raise ValueError(
            "compute_raman_from_polarizability_autocorrelation: trajectory carries no "
            "'polarizabilities'. Fabricating them with np.random returns a meaningless "
            "spectrum; supply per-frame polarizabilities instead."
        )

    n_frames = len(polarizabilities)
    max_lag = int(max_lag_fs / timestep)
    max_lag = min(max_lag, n_frames // 2)

    # Compute polarizability autocorrelation
    acf = compute_autocorrelation(polarizabilities, max_lag)

    # Window and FFT
    window = np.hanning(2 * max_lag)[max_lag:]
    acf_windowed = acf * window

    n_fft = int(1 / (frequency_resolution * timestep * 2.99792458e-5))
    n_fft = max(n_fft, 2 * max_lag)

    spectrum = np.abs(np.fft.fft(acf_windowed, n_fft)[:n_fft // 2])

    # Frequency axis
    freq_max = 1 / (2 * timestep * 2.99792458e-5)
    frequencies = np.linspace(0, freq_max, n_fft // 2)

    # Apply Bose-Einstein factor for Stokes Raman
    kT = 0.695 * temperature  # cm⁻¹
    bose_factor = np.where(
        frequencies > 0,
        1 / (1 - np.exp(-frequencies / kT + 1e-10)),
        1.0
    )
    spectrum *= bose_factor

    # Normalize
    if np.max(spectrum) > 0:
        spectrum /= np.max(spectrum)

    result = DynamicalSpectrum(
        frequencies=frequencies,
        intensities=spectrum,
        spectrum_type='raman',
        temperature=temperature,
        correlation_time=max_lag_fs,
        n_frames=n_frames
    )

    logger.info(f"Raman spectrum computed: {len(frequencies)} points")

    return result


def compute_nmr_relaxation_from_trajectory(
    trajectory,
    timestep: float = 0.5,
    larmor_frequency: float = 400e6,  # Hz (typical ¹H at 9.4 T)
    max_lag_fs: float = 10000.0,
    dipolar_coupling_constant: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute NMR relaxation times from trajectory.

    T1 (spin-lattice) and T2 (spin-spin) relaxation times are computed
    from the spectral density of molecular tumbling.

    For a like-spin dipolar pair (BPP theory):
    1/T1 = K * [J(ω) + 4J(2ω)]
    1/T2 = K * [3J(0) + 5J(ω) + 2J(2ω)] / 2

    with the reduced spectral density J(ω) = τc / (1 + (ω τc)²).

    Args:
        trajectory: MD trajectory
        timestep: MD timestep in fs
        larmor_frequency: Larmor frequency in Hz
        max_lag_fs: Maximum correlation time in fs
        dipolar_coupling_constant: Dipolar coupling prefactor K in s⁻², i.e.
            K = (3/10)·(μ₀/4π)²·(γ⁴ℏ²/r⁶) for a homonuclear ¹H–¹H pair at
            separation r. This sets the ABSOLUTE relaxation-time scale and
            depends on physics (γ, r) the trajectory does not carry, so it
            must be supplied by the caller. (For two ¹H at 1.8 Å, K≈2.1e9 s⁻²,
            giving T1≈few s — physically reasonable.)

    Returns:
        Dictionary with 'T1', 'T2', 'NOE' values in seconds

    Raises:
        ValueError: if ``dipolar_coupling_constant`` is not supplied. The
            previous implementation hardcoded K=1.0, which made the returned
            T1/T2 in seconds physically meaningless (off by ~12 orders of
            magnitude); fabricating an absolute rate is worse than refusing.
    """
    # K is a physical prefactor (depends on γ and internuclear distance r),
    # neither of which the trajectory carries. Refuse to fabricate it.
    if dipolar_coupling_constant is None:
        raise ValueError(
            "compute_nmr_relaxation_from_trajectory: 'dipolar_coupling_constant' "
            "(K in s⁻², the BPP dipolar prefactor (3/10)·(μ₀/4π)²·γ⁴ℏ²/r⁶) is "
            "required to set the absolute T1/T2 scale. The previous K=1.0 "
            "placeholder returned T1/T2 off by ~12 orders of magnitude. Supply "
            "the physical coupling constant for the spin pair (e.g. K≈2.1e9 s⁻² "
            "for two ¹H at 1.8 Å)."
        )
    logger.info("Computing NMR relaxation from trajectory")

    # Convert Larmor frequency to angular frequency
    omega = 2 * np.pi * larmor_frequency

    # Extract rotational correlation function
    # This requires tracking molecular orientation
    if hasattr(trajectory, 'orientations'):
        orientations = np.array(trajectory.orientations)
    else:
        raise ValueError(
            "compute_nmr_relaxation_from_trajectory: trajectory carries no "
            "'orientations'. Fabricating them with np.random returns a meaningless "
            "relaxation time; supply per-frame orientations instead."
        )

    n_frames = len(orientations)
    max_lag = int(max_lag_fs / timestep)
    max_lag = min(max_lag, n_frames // 2)

    # Compute rotational autocorrelation
    acf = compute_autocorrelation(orientations, max_lag)

    # Normalize
    if acf[0] > 0:
        acf /= acf[0]

    # Fit to exponential to get correlation time
    # C(t) ≈ exp(-t/τc)
    time_array = np.arange(max_lag) * timestep * 1e-15  # fs to s

    # Simple exponential fit
    try:
        # Find where ACF drops to 1/e
        decay_idx = np.where(acf < 1/np.e)[0]
        if len(decay_idx) > 0:
            tau_c = time_array[decay_idx[0]]
        else:
            tau_c = time_array[-1]
    except:
        tau_c = 1e-9  # Default 1 ns

    # Spectral density (Lorentzian)
    def J(w):
        return tau_c / (1 + (w * tau_c)**2)

    # Compute relaxation rates (dipole-dipole). K is the physical prefactor
    # supplied by the caller (see docstring); no fabricated placeholder.
    K = dipolar_coupling_constant

    R1 = K * (J(omega) + 4 * J(2 * omega))
    R2 = K * (3 * J(0) + 5 * J(omega) + 2 * J(2 * omega)) / 2

    T1 = 1 / R1 if R1 > 0 else float('inf')
    T2 = 1 / R2 if R2 > 0 else float('inf')

    # NOE enhancement
    NOE = 1 + (K * (6 * J(2 * omega) - J(0))) / R1 if R1 > 0 else 1

    logger.info(f"NMR relaxation: T1={T1:.3f}s, T2={T2:.3f}s, τc={tau_c*1e9:.3f}ns")

    return {
        'T1': T1,
        'T2': T2,
        'NOE': NOE,
        'tau_c': tau_c,
        'larmor_frequency': larmor_frequency
    }


def compute_power_spectrum(
    trajectory,
    timestep: float = 0.5,
    property_name: str = 'velocities',
    frequency_resolution: float = 1.0
) -> DynamicalSpectrum:
    """
    Compute power spectrum (density of states) from velocity autocorrelation.

    The vibrational density of states is given by:
    g(ω) ∝ ∫₀^∞ <v(0)·v(t)> exp(-iωt) dt

    Args:
        trajectory: MD trajectory
        timestep: MD timestep in fs
        property_name: Property to analyze ('velocities', 'forces')
        frequency_resolution: Frequency resolution in cm⁻¹

    Returns:
        DynamicalSpectrum with power spectrum
    """
    logger.info(f"Computing power spectrum from {property_name}")

    # Extract property
    if hasattr(trajectory, property_name):
        data = np.array(getattr(trajectory, property_name))
    elif hasattr(trajectory, 'frames'):
        data = []
        for frame in trajectory.frames:
            if hasattr(frame, property_name):
                data.append(getattr(frame, property_name))
        data = np.array(data)
    else:
        raise ValueError(
            f"Trajectory frames carry no '{property_name}'. Fabricating it with "
            f"np.random would return meaningless dynamics; supply the real property."
        )

    n_frames = len(data)
    max_lag = n_frames // 2

    # Flatten spatial dimensions
    data_flat = data.reshape(n_frames, -1)

    # Compute VACF
    acf = compute_autocorrelation(data_flat, max_lag)

    # Window and FFT
    window = np.hanning(2 * max_lag)[max_lag:]
    acf_windowed = acf * window

    n_fft = int(1 / (frequency_resolution * timestep * 2.99792458e-5))
    n_fft = max(n_fft, 2 * max_lag)

    spectrum = np.abs(np.fft.fft(acf_windowed, n_fft)[:n_fft // 2])

    # Frequency axis
    freq_max = 1 / (2 * timestep * 2.99792458e-5)
    frequencies = np.linspace(0, freq_max, n_fft // 2)

    # Normalize
    if np.max(spectrum) > 0:
        spectrum /= np.max(spectrum)

    return DynamicalSpectrum(
        frequencies=frequencies,
        intensities=spectrum,
        spectrum_type='power',
        temperature=300.0,
        correlation_time=max_lag * timestep,
        n_frames=n_frames
    )


def _estimate_dipole_from_frame(frame) -> np.ndarray:
    """Estimate dipole moment from trajectory frame."""
    # Simple estimate using atomic charges and positions
    if hasattr(frame, 'positions') and hasattr(frame, 'charges'):
        positions = np.array(frame.positions)
        charges = np.array(frame.charges)
        return np.sum(charges[:, np.newaxis] * positions, axis=0)
    else:
        return np.zeros(3)


def connect_uvvis_to_namd(
    bond,
    n_states: int = 2
) -> Dict[str, Any]:
    """
    Connect UV-Vis calculation to NAMD for photo-excitation studies.

    Computes absorption spectrum and provides initial conditions for NAMD.

    Args:
        bond: Bond object
        n_states: Number of electronic states

    Returns:
        Dictionary with UV-Vis results and NAMD initialization data
    """
    logger.info("Connecting UV-Vis to NAMD")

    try:
        from kanad.analysis.spectroscopy import UVVisCalculator
        from kanad.dynamics.nonadiabatic import NonAdiabaticMD
        from kanad.core.molecule import Molecule as MolWrapper

        # UVVisCalculator's HF/TDA path needs hamiltonian.mol AND hamiltonian.mf.
        # The bond's CovalentHamiltonian exposes .mol but NOT .mf, so the old
        # code (mol._hamiltonian = bond.hamiltonian) crashed with
        # "'CovalentHamiltonian' object has no attribute 'mf'". Use the bond's
        # own Molecule (or build one from its atoms) so a MolecularHamiltonian
        # with both .mol and .mf is constructed lazily.
        mol = getattr(bond, 'molecule', None)
        if mol is None:
            mol = MolWrapper(
                atoms=list(getattr(bond, 'atoms', [bond.atom_1, bond.atom_2])),
                charge=getattr(bond, 'charge', 0),
                spin=getattr(bond, 'spin', 0),
                basis='sto-3g'
            )

        uvvis = UVVisCalculator(mol)
        spectrum = uvvis.compute_excitations(n_states=n_states, method='TDA')

        # Get brightest transition
        if 'oscillator_strengths' in spectrum:
            osc_strengths = spectrum['oscillator_strengths']
            brightest_idx = np.argmax(osc_strengths)
            brightest_state = brightest_idx + 1  # +1 because ground state is 0
        else:
            brightest_state = 1

        # Setup NAMD
        namd = NonAdiabaticMD(
            bond=bond,
            n_states=n_states,
            initial_state=brightest_state
        )

        return {
            'uvvis_spectrum': spectrum,
            'brightest_state': brightest_state,
            'namd_simulator': namd,
            'ready_for_dynamics': True
        }

    except Exception as e:
        logger.warning(f"UV-Vis to NAMD connection failed: {e}")
        return {
            'error': str(e),
            'ready_for_dynamics': False
        }
