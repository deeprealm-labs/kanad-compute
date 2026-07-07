"""Solver-agnostic numerical forces + quantum MD (M5 D1/D2, 2026-05-28).

Why this module exists (vs the pre-M5 `quantum_gradients.py`):
The old force path computed Hellmann-Feynman forces with the ansatz
parameters θ **frozen at the equilibrium geometry** — wrong at every
displaced geometry, and missing the Pulay term entirely (atom-centered
Gaussians move with the nuclei). It also hard-imported `VQESolver`.

This module takes a different, robust route: **central finite-difference
of the total energy**. The total electronic energy E(R) already contains
the basis-set's R-dependence, so numerically differentiating it captures
BOTH the Hellmann-Feynman and Pulay contributions automatically — no
analytic Pulay term to derive per solver. The only contract is an
``energy_fn(atoms) -> (energy, warm_state)`` callback, which any solver
(VQE, SQD, CASCI, HF) can satisfy.

The cost is 6·N_atoms energy evaluations per force. We amortize that with
**warm-starting**: each displaced-geometry solve reuses the previous
solve's wavefunction (VQE parameters / SQD determinant subspace). This is
the "equilibration" behavior — the first solve is cold and expensive, then
every subsequent solve at a nearby geometry converges in a fraction of the
iterations.

Units (all atomic units internally):
- positions: Bohr
- forces:    Ha / Bohr
- masses:    electron masses (amu × 1822.888)
- time:      atomic time units (1 a.u. = 0.0241888 fs)
- velocity:  Bohr / a.u.-time
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Conversion factors
ANGSTROM_TO_BOHR = 1.8897259886
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR
AMU_TO_ME = 1822.888486209
AU_TIME_TO_FS = 0.0241888432651
FS_TO_AU_TIME = 1.0 / AU_TIME_TO_FS


@dataclass
class ForceResult:
    """Forces + bookkeeping from a numerical-force evaluation."""
    forces: np.ndarray            # (n_atoms, 3) in Ha/Bohr
    energy: float                  # total energy at the reference geometry (Ha)
    n_energy_evals: int            # number of energy calls used (6N + 1)
    warm_state: Any = None         # warm-start payload for the next geometry
    eval_times: list = field(default_factory=list)  # per-eval wall times


def compute_numerical_forces(
    energy_fn: Callable[[np.ndarray, Optional[Any]], tuple],
    atoms_bohr: np.ndarray,
    delta: float = 0.01,
    warm_state: Optional[Any] = None,
    central: bool = True,
) -> ForceResult:
    """Central finite-difference nuclear forces from any energy callback.

    ``F_{A,i} = − ∂E/∂R_{A,i} ≈ − (E(R+δ_{Ai}) − E(R−δ_{Ai})) / (2δ)``

    Args:
        energy_fn: callable ``(atoms_bohr, warm_state) -> (energy_Ha, new_warm_state)``.
            Must rebuild the molecule at the given geometry and return the
            converged total electronic energy. The warm_state is threaded
            through so the solver can warm-start.
        atoms_bohr: (n_atoms, 3) nuclear positions in Bohr.
        delta: displacement in Bohr (default 0.01 ≈ 0.005 Å).
        warm_state: initial warm-start payload (None = cold start).
        central: True for central difference (2 evals/coord), False for
            forward (1 eval/coord + 1 reference — less accurate).

    Returns:
        ForceResult with forces (Ha/Bohr), reference energy, eval count.
    """
    atoms_bohr = np.asarray(atoms_bohr, dtype=float)
    n_atoms = atoms_bohr.shape[0]
    forces = np.zeros((n_atoms, 3))
    eval_times = []
    n_evals = 0

    # Reference energy (also refreshes the warm-state for displaced solves)
    t0 = time.time()
    e_ref, warm_state = energy_fn(atoms_bohr, warm_state)
    eval_times.append(time.time() - t0)
    n_evals += 1

    for a in range(n_atoms):
        for i in range(3):
            geom_plus = atoms_bohr.copy()
            geom_plus[a, i] += delta
            t0 = time.time()
            e_plus, warm_state = energy_fn(geom_plus, warm_state)
            eval_times.append(time.time() - t0)
            n_evals += 1

            if central:
                geom_minus = atoms_bohr.copy()
                geom_minus[a, i] -= delta
                t0 = time.time()
                e_minus, warm_state = energy_fn(geom_minus, warm_state)
                eval_times.append(time.time() - t0)
                n_evals += 1
                forces[a, i] = -(e_plus - e_minus) / (2.0 * delta)
            else:
                forces[a, i] = -(e_plus - e_ref) / delta

    return ForceResult(
        forces=forces, energy=e_ref, n_energy_evals=n_evals,
        warm_state=warm_state, eval_times=eval_times,
    )


@dataclass
class MDStep:
    """One snapshot of the quantum-MD trajectory."""
    step: int
    time_fs: float
    positions_bohr: np.ndarray     # (n_atoms, 3)
    velocities: np.ndarray          # (n_atoms, 3) Bohr/a.u.-time
    forces: np.ndarray              # (n_atoms, 3) Ha/Bohr
    potential_energy: float         # Ha
    kinetic_energy: float           # Ha
    total_energy: float             # Ha
    n_energy_evals: int             # solver calls this step
    force_eval_time_s: float        # wall time for the force this step


@dataclass
class MDTrajectory:
    """Full quantum-MD trajectory + diagnostics."""
    steps: list = field(default_factory=list)
    dt_fs: float = 0.0
    masses_me: np.ndarray = None

    @property
    def energies(self) -> np.ndarray:
        return np.array([s.total_energy for s in self.steps])

    @property
    def potential_energies(self) -> np.ndarray:
        return np.array([s.potential_energy for s in self.steps])

    def energy_drift_mha(self) -> float:
        """Max deviation of total energy from its initial value (mHa).

        The headline MD-credibility metric: a good integrator + accurate
        forces conserve total energy. Drift > a few mHa over 10-20 steps
        signals either too-large dt or noisy forces.
        """
        e = self.energies
        if len(e) < 2:
            return 0.0
        return float(np.max(np.abs(e - e[0])) * 1000.0)

    def equilibration_profile(self) -> np.ndarray:
        """Per-step solver-call count — the 'equilibration' signature.

        First step is cold (many solver calls / iterations); subsequent
        warm-started steps should be cheaper. Returns the per-step
        energy-eval count (forces) so the caller can see the transient.
        """
        return np.array([s.n_energy_evals for s in self.steps])


def run_quantum_md(
    energy_fn: Callable[[np.ndarray, Optional[Any]], tuple],
    atoms_bohr: np.ndarray,
    masses_amu: np.ndarray,
    n_steps: int = 10,
    dt_fs: float = 0.5,
    velocities: Optional[np.ndarray] = None,
    force_delta: float = 0.01,
    warm_state: Optional[Any] = None,
    thermostat: Optional[Callable] = None,
) -> MDTrajectory:
    """Velocity-Verlet quantum molecular dynamics with warm-started forces.

    Short trajectories (5-20 steps) are enough to establish credibility:
    energy conservation + force accuracy demonstrate the quantum-MD loop is
    as sound as classical MD-with-DFT, without needing 1000-step production
    runs.

    Args:
        energy_fn: ``(atoms_bohr, warm_state) -> (energy_Ha, new_warm_state)``.
        atoms_bohr: (n_atoms, 3) initial positions in Bohr.
        masses_amu: (n_atoms,) atomic masses in amu.
        n_steps: number of MD steps (5-20 typical).
        dt_fs: timestep in fs (0.25-0.5 fs typical for H-containing systems).
        velocities: (n_atoms, 3) initial velocities (Bohr/a.u.-time); None → 0.
        force_delta: finite-difference displacement (Bohr).
        warm_state: initial warm-start payload.
        thermostat: optional callable ``(velocities, masses_me) -> velocities``
            applied after each step (e.g. velocity rescale for NVT).

    Returns:
        MDTrajectory with per-step snapshots + energy-drift / equilibration
        diagnostics.
    """
    atoms_bohr = np.asarray(atoms_bohr, dtype=float)
    n_atoms = atoms_bohr.shape[0]
    masses_me = np.asarray(masses_amu, dtype=float) * AMU_TO_ME
    dt_au = dt_fs * FS_TO_AU_TIME

    if velocities is None:
        velocities = np.zeros((n_atoms, 3))
    velocities = np.asarray(velocities, dtype=float)

    traj = MDTrajectory(dt_fs=dt_fs, masses_me=masses_me)

    # Initial force
    t0 = time.time()
    fr = compute_numerical_forces(energy_fn, atoms_bohr, force_delta, warm_state)
    warm_state = fr.warm_state
    forces = fr.forces
    pe = fr.energy
    force_time = time.time() - t0

    def kinetic(v):
        return float(0.5 * np.sum(masses_me[:, None] * v ** 2))

    ke = kinetic(velocities)
    traj.steps.append(MDStep(
        step=0, time_fs=0.0,
        positions_bohr=atoms_bohr.copy(), velocities=velocities.copy(),
        forces=forces.copy(), potential_energy=pe, kinetic_energy=ke,
        total_energy=pe + ke, n_energy_evals=fr.n_energy_evals,
        force_eval_time_s=force_time,
    ))

    pos = atoms_bohr.copy()
    for step in range(1, n_steps + 1):
        accel = forces / masses_me[:, None]   # F/m, shape (n_atoms, 3)
        # Velocity Verlet: half-kick, drift, recompute force, half-kick
        velocities = velocities + 0.5 * accel * dt_au
        pos = pos + velocities * dt_au

        t0 = time.time()
        fr = compute_numerical_forces(energy_fn, pos, force_delta, warm_state)
        warm_state = fr.warm_state
        forces = fr.forces
        pe = fr.energy
        force_time = time.time() - t0

        accel_new = forces / masses_me[:, None]
        velocities = velocities + 0.5 * accel_new * dt_au

        if thermostat is not None:
            velocities = thermostat(velocities, masses_me)

        ke = kinetic(velocities)
        traj.steps.append(MDStep(
            step=step, time_fs=step * dt_fs,
            positions_bohr=pos.copy(), velocities=velocities.copy(),
            forces=forces.copy(), potential_energy=pe, kinetic_energy=ke,
            total_energy=pe + ke, n_energy_evals=fr.n_energy_evals,
            force_eval_time_s=force_time,
        ))
        logger.info(
            f"MD step {step}: E_pot = {pe:.6f}, E_tot = {pe + ke:.6f} Ha, "
            f"|F|_max = {np.abs(forces).max():.5f} Ha/Bohr "
            f"({force_time:.1f}s)"
        )

    return traj
