"""
Quantum Chemistry Solvers - Rebuilt with Bonds Module Integration.

New Architecture:
- All solvers work with bonds module as primary interface
- Automatic integration with analysis tools
- Automatic integration with optimization tools
- Rich, comprehensive results
- Unified API across all solvers

Available Solvers:
- VQESolver: Variational Quantum Eigensolver (ground state)
- SQDSolver: Subspace Quantum Diagonalization (ground + excited)
- ExcitedStatesSolver: Molecular excited states (CIS, TDDFT - classical)

Usage Example:
    from kanad.bonds import BondFactory
    from kanad.solvers import VQESolver

    # Create molecule
    bond = BondFactory.create_bond('H', 'H', distance=0.74)

    # Run VQE
    solver = VQESolver(bond, ansatz_type='ucc')
    result = solver.solve()

    # Print comprehensive results
    solver.print_summary()

    # Access results
    print(f"Energy: {result['energy']:.6f} Hartree")
    print(f"Correlation: {result['correlation_energy']:.6f} Hartree")
    print(f"Analysis: {result['analysis']}")
"""

# New Solvers (Bonds Module Integration)
from kanad.solvers.base_solver import BaseSolver
from kanad.solvers.vqe_solver import VQESolver
from kanad.solvers.ci_solver import CISolver
# DeterministicCI = the legacy deterministic HF+S+D classical CI (was named
# `SQDSolver`). SamplingSQDSolver = the real circuit-sampling SQD. `SQDSolver`
# is kept as a deprecated alias of DeterministicCI for back-compat.
from kanad.solvers.deterministic_ci import DeterministicCI, SQDSolver
from kanad.solvers.sampling_sqd import SamplingSQDSolver
# LanczosSolver = classical Lanczos/Krylov eigensolver (was misnamed
# `KrylovSQDSolver`, kept as a deprecated alias).
from kanad.solvers.lanczos_solver import LanczosSolver, KrylovSQDSolver
from kanad.solvers.excited_states_solver import ExcitedStatesSolver
from kanad.solvers.smart_solver import SmartSolver, solve_smart
from kanad.solvers.physics_vqe import PhysicsVQE, solve_physics_vqe
from kanad.solvers.hardware_vqe import HardwareVQE, HardwareVQEResult
# HybridSubspaceVQE retired (audit H12) — not re-exported; superseded by
# SamplingSQDSolver. The class still lives in sampled_subspace_vqe for now.
from kanad.solvers.sampled_subspace_vqe import SampledSubspaceVQE
from kanad.solvers.varqite_solver import VarQITESolver, VarQITEResult, VarQRTEResult, create_varqite_solver
from kanad.solvers.qeom_vqe import qEOMVQE, qEOMResult, create_qeom_solver
from kanad.solvers.periodic_solver import PeriodicSolver

__all__ = [
    # Base Class
    'BaseSolver',

    # New Solvers
    'VQESolver',
    'CISolver',
    'DeterministicCI',      # honest name for the legacy deterministic CI
    'SQDSolver',            # deprecated alias of DeterministicCI
    'SamplingSQDSolver',    # real circuit-sampling SQD (M4)
    'LanczosSolver',        # classical Lanczos/Krylov eigensolver
    'KrylovSQDSolver',      # deprecated alias of LanczosSolver
    'ExcitedStatesSolver',
    'SmartSolver',
    'solve_smart',
    'PhysicsVQE',
    'solve_physics_vqe',
    'HardwareVQE',
    'HardwareVQEResult',
    'SampledSubspaceVQE',
    # HybridSubspaceVQE retired (audit H12): it never inherited BaseSolver,
    # returned a bare dict, and crashed reading `SolverResult.parameters`.
    # Superseded by SamplingSQDSolver (see solvers/CLAUDE.md). No longer a
    # public exported solver.

    # VarQITE (imaginary time evolution - no barren plateaus)
    'VarQITESolver',
    'VarQITEResult',
    'VarQRTEResult',
    'create_varqite_solver',

    # qEOM-VQE (TRUE quantum excited states)
    'qEOMVQE',
    'qEOMResult',
    'create_qeom_solver',

    # Periodic / materials (band structure + DOS)
    'PeriodicSolver',
]


# ───────────────────── Capability + domain protocol (Stage 1) ─────────────────────
# Declare each reference solver's REAL, verified capabilities and the lab(s) it serves.
# Consumers route on these (see kanad/solvers/meta.py, docs/design/SOLVER_PROTOCOL_PLAN.md).
# CONSERVATIVE + HONEST: only capabilities/domains that work and are numerically honest
# TODAY (verified by tests/unit/test_capability_conformance.py). Deferred to Stage 2 as
# they get wired + value-checked: md/reaction domains (need forces/energy_fn), the
# `nuclear_gradient`/`hessian`/`dipole`/`field_response` capabilities, and richer
# excited-state solvers. (Set centrally for a single reviewable diff; may be inlined
# onto the classes in Stage 2.)
from kanad.solvers.meta import SolverMeta as _SolverMeta

VQESolver.META = _SolverMeta(
    name="vqe", domains={"ground_state", "reaction"}, capabilities={"energy", "one_rdm"},
    max_qubits=14,
    description="Variational quantum eigensolver (HEA / Givens-SD ansätze). "
                "Energy-capable across geometries → usable for reaction PES (forces "
                "for MD require a ForceProvider; see PhysicsVQE / SamplingSQDSolver).",
)
PhysicsVQE.META = _SolverMeta(
    name="physics_vqe", domains={"ground_state", "md", "reaction"},
    capabilities={"energy", "nuclear_gradient"}, analytic_gradient=False,
    description="MP2-ranked excitation VQE; chemical accuracy on Tier-1 mains. "
                "Force-capable (FD over a geometry-rebuild energy closure) for MD/reactions.",
)
HardwareVQE.META = _SolverMeta(
    name="hardware_vqe", domains={"ground_state", "reaction"}, capabilities={"energy"},
    backends={"statevector", "ibm"}, max_qubits=12,
    description="Hardware-targeting hardware-efficient ansatz + Jordan-Wigner. "
                "Energy-capable across geometries → usable for reaction PES.",
)
CISolver.META = _SolverMeta(
    name="ci", domains={"ground_state"}, capabilities={"energy"},
    description="Classical CI in a sampled subspace.",
)
DeterministicCI.META = _SolverMeta(
    name="deterministic_ci", domains={"ground_state"}, capabilities={"energy", "one_rdm"},
    description="Deterministic HF + singles/doubles CI on explicit statevectors.",
)
SamplingSQDSolver.META = _SolverMeta(
    name="sampling_sqd", domains={"ground_state", "md", "reaction"},
    capabilities={"energy", "nuclear_gradient"}, analytic_gradient=False,
    description="Sample-based quantum diagonalization (circuit sampling + recovery). "
                "Force-capable (FD over a geometry-rebuild + LUCJ re-solve) for MD/reactions.",
)
LanczosSolver.META = _SolverMeta(
    name="lanczos", domains={"ground_state", "photochemistry"},
    capabilities={"energy", "excited_states"},
    description="Classical Lanczos/Krylov subspace eigensolver (low-lying spectrum).",
)
qEOMVQE.META = _SolverMeta(
    name="qeom", domains={"photochemistry"}, capabilities={"energy", "excited_states"},
    consistent_state_tracking=False, max_qubits=8,
    description="qEOM-VQE excited states (single-geometry; no state tracking yet).",
)
ExcitedStatesSolver.META = _SolverMeta(
    name="excited_states", domains={"photochemistry"},
    capabilities={"energy", "excited_states", "transition_properties"},
    description="Classical CIS/TDDFT excited states + oscillator strengths.",
)
VarQITESolver.META = _SolverMeta(
    name="varqite", domains={"ground_state"}, capabilities={"energy"},
    description="Variational imaginary-time evolution (experimental).",
)
SampledSubspaceVQE.META = _SolverMeta(
    name="sampled_subspace_vqe", domains={"ground_state"}, capabilities={"energy"},
    description="Subspace VQE (superseded by SamplingSQDSolver).",
)
SmartSolver.META = _SolverMeta(
    name="smart", domains={"ground_state"}, capabilities={"energy"},
    description="Auto-routing meta-solver (defers to FCI for small systems).",
)
PeriodicSolver.META = _SolverMeta(
    name="periodic", domains={"materials"}, capabilities={"energy", "band_structure"},
    description="Periodic HF (KRHF) band structure + DOS via PySCF PBC.",
)


# ───────────────────── Solver registry (Stage 3) ─────────────────────
# Discovery layer: register every reference solver so consumers (reactions/dynamics
# drivers, the app's lab routing, the workshop) resolve solvers by name + domain +
# capability instead of hard-importing concrete classes. User/community solvers add
# themselves via register_solver(). See kanad/solvers/registry.py.
from kanad.solvers.registry import (  # noqa: E402
    register_solver, unregister_solver, get_solver, has_solver,
    list_solvers, list_solver_classes, registered_names,
)

for _ref_solver in (
    VQESolver, PhysicsVQE, HardwareVQE, CISolver, DeterministicCI,
    SamplingSQDSolver, LanczosSolver, qEOMVQE, ExcitedStatesSolver,
    VarQITESolver, SampledSubspaceVQE, SmartSolver, PeriodicSolver,
):
    register_solver(_ref_solver)
del _ref_solver

__all__ += [
    'register_solver', 'unregister_solver', 'get_solver', 'has_solver',
    'list_solvers', 'list_solver_classes', 'registered_names',
]
