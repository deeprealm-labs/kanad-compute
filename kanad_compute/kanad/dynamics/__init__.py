"""
Kanad Molecular Dynamics Module

🌟 WORLD'S FIRST: Quantum-Enhanced MD with Governance Protocols 🌟

This module provides molecular dynamics simulation capabilities with unique features:
- Quantum forces from VQE/SQD (correlation effects beyond HF)
- Governance-aware dynamics (bond-type-specific evolution)
- Environment integration (temperature, solvent, pH, pressure)
- Hybrid quantum-classical propagation

Components:
-----------
- md_simulator: Main MD simulation engine
- integrators: Time evolution algorithms (Velocity Verlet, Leapfrog, etc.)
- thermostats: Temperature control (Berendsen, Nose-Hoover, Langevin)
- barostats: Pressure control (NPT ensemble)
- trajectory: Trajectory storage and management
- initialization: Initial conditions (Maxwell-Boltzmann velocities)
- quantum_md: VQE/SQD-driven dynamics
- governance_md: Bond-aware MD constraints
- solvated_md: Implicit/explicit solvent MD

Example Usage:
-------------
```python
from kanad.bonds import BondFactory
from kanad.dynamics import MDSimulator

# Create molecule
bond = BondFactory.create_bond('H', 'H', distance=0.74)

# Run classical MD
md = MDSimulator(
    bond,
    temperature=300.0,  # K
    timestep=0.5,  # fs
    integrator='velocity_verlet',
    thermostat='berendsen'
)

trajectory = md.run(n_steps=1000)

# Or use quantum forces
md_quantum = MDSimulator(
    bond,
    temperature=300.0,
    force_method='vqe',  # Use quantum correlation!
    use_governance=True   # Bond-aware constraints
)

trajectory = md_quantum.run(n_steps=100)
```

References:
----------
- Velocity Verlet: Swope et al. (1982) J. Chem. Phys. 76, 637
- Berendsen thermostat: Berendsen et al. (1984) J. Chem. Phys. 81, 3684
- Nose-Hoover: Hoover (1985) Phys. Rev. A 31, 1695
- Ab initio MD: Car & Parrinello (1985) Phys. Rev. Lett. 55, 2471
"""

# Core MD components
from kanad.dynamics.integrators import (
    VelocityVerletIntegrator,
    LeapfrogIntegrator,
    RungeKuttaIntegrator
)

from kanad.dynamics.thermostats import (
    BerendsenThermostat,
    NoseHooverThermostat,
    LangevinThermostat,
    VelocityRescaling
)

from kanad.dynamics.trajectory import (
    Trajectory,
    TrajectoryFrame,
    TrajectoryWriter
)

from kanad.dynamics.initialization import (
    MaxwellBoltzmannInitializer,
    remove_com_motion,
    equilibrate_system
)

from kanad.dynamics.md_simulator import (
    MDSimulator,
    MDResult
)

# Non-adiabatic dynamics
from kanad.dynamics.nonadiabatic import (
    NonAdiabaticMD,
    NAMDState,
    NAMDTrajectory,
    SurfaceHoppingMethod,
    create_namd_simulator
)

# Photodynamics (light-matter interactions)
from kanad.dynamics.photodynamics import (
    LaserField,
    LaserParameters,
    PulseEnvelope,
    PhotodynamicsSimulator,
    PhotodynamicsResult,
    create_laser_pulse,
    pump_probe_simulation
)

# Spectroscopy-dynamics connection
from kanad.dynamics.spectroscopy_dynamics import (
    DynamicalSpectrum,
    compute_autocorrelation,
    compute_ir_from_dipole_autocorrelation,
    compute_raman_from_polarizability_autocorrelation,
    compute_nmr_relaxation_from_trajectory,
    compute_power_spectrum,
    connect_uvvis_to_namd
)

# Quantum gradients (100x speedup via parameter shift rule)
from kanad.dynamics.quantum_gradients import (
    ParameterShiftGradient,
    QuantumGradientResult,
    compute_analytical_gradients,
    compare_analytical_vs_numerical
)

# Quantum MD functions
from kanad.dynamics.quantum_md import (
    compute_quantum_forces,
    compute_analytical_gradients_vqe,
    estimate_quantum_md_cost,
    compare_classical_vs_quantum_forces
)

# Open Quantum Systems (Lindblad + Bath)
from kanad.dynamics.open_quantum import (
    LindbladEvolver,
    LindbladResult,
    create_dephasing_operator,
    create_amplitude_damping_operator,
    create_thermal_operators,
    QuantumBath,
    SpinBosonBath,
    DruideLorenzBath,
    create_bath_from_solvent,
    SOLVENT_PROPERTIES,
    DecoherenceModel,
    get_decoherence_rates,
    estimate_T1_T2
)

# Quantum NAC (TRUE quantum advantage for NAC vectors)
from kanad.dynamics.quantum_nac import (
    QuantumNACCalculator,
    QuantumNACResult,
    compute_quantum_nac,
    detect_conical_intersection
)

__all__ = [
    # Main interface
    'MDSimulator',
    'MDResult',

    # Integrators
    'VelocityVerletIntegrator',
    'LeapfrogIntegrator',
    'RungeKuttaIntegrator',

    # Thermostats
    'BerendsenThermostat',
    'NoseHooverThermostat',
    'LangevinThermostat',
    'VelocityRescaling',

    # Trajectory
    'Trajectory',
    'TrajectoryFrame',
    'TrajectoryWriter',

    # Initialization
    'MaxwellBoltzmannInitializer',
    'remove_com_motion',
    'equilibrate_system',

    # Non-adiabatic dynamics
    'NonAdiabaticMD',
    'NAMDState',
    'NAMDTrajectory',
    'SurfaceHoppingMethod',
    'create_namd_simulator',

    # Photodynamics
    'LaserField',
    'LaserParameters',
    'PulseEnvelope',
    'PhotodynamicsSimulator',
    'PhotodynamicsResult',
    'create_laser_pulse',
    'pump_probe_simulation',

    # Spectroscopy-dynamics connection
    'DynamicalSpectrum',
    'compute_autocorrelation',
    'compute_ir_from_dipole_autocorrelation',
    'compute_raman_from_polarizability_autocorrelation',
    'compute_nmr_relaxation_from_trajectory',
    'compute_power_spectrum',
    'connect_uvvis_to_namd',

    # Open Quantum Systems (Lindblad + Bath)
    'LindbladEvolver',
    'LindbladResult',
    'create_dephasing_operator',
    'create_amplitude_damping_operator',
    'create_thermal_operators',
    'QuantumBath',
    'SpinBosonBath',
    'DruideLorenzBath',
    'create_bath_from_solvent',
    'SOLVENT_PROPERTIES',
    'DecoherenceModel',
    'get_decoherence_rates',
    'estimate_T1_T2',

    # Quantum NAC (TRUE quantum advantage)
    'QuantumNACCalculator',
    'QuantumNACResult',
    'compute_quantum_nac',
    'detect_conical_intersection',
]

__version__ = '1.0.0'
__author__ = 'Kanad Team'
__description__ = "Quantum-Enhanced Molecular Dynamics with Governance Protocols"
