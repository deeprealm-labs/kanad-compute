"""Quantum reduced density matrices from VQE wavefunctions.

This is the M3 foundation: every "quantum observable" (dipole, polarizability,
NMR shielding, IR/Raman intensities) is derived from the 1-RDM and 2-RDM.
The pre-M3 framework computed VQE energies but threw the wavefunction away
and reported HF densities dressed up as quantum — see
`inspection/15-audit-observables.md` for the audit trail.

`QuantumRDMExtractor` reads a converged statevector and a fermion→qubit
mapper, builds the JW image of every `a†_p a_q` excitation operator, and
returns the spatial 1-RDM in MO basis. The result is trace-validated:
``|trace(ρ) − n_electrons| < tol`` else `RuntimeError`. The pre-M3
implementation had `trace = 2.71` on HeH⁺ (should be 2.0) — this module
catches that class of bug at the source.
"""

from kanad.core.density.quantum_rdm import (
    QuantumRDMExtractor,
    extract_1rdm_from_statevector,
    energy_from_rdms,
    spin_squared_from_statevector,
    compute_natural_orbital_occupations,
    compute_m_diagnostic,
    compute_n_unpaired_electrons,
)
from kanad.core.density.density_storage import (
    embed_active_to_full_mo,
    mo_to_ao_1rdm,
    validate_trace,
)
# Selected-CI / sampled RDMs (from a determinant-list eigenvector, e.g. SQD).
from kanad.core.density.sampled_rdm import (
    embed_ci_vector,
    rdm1_from_ci_vector,
    rdm12_from_ci_vector,
)

__all__ = [
    'QuantumRDMExtractor',
    'extract_1rdm_from_statevector',
    'energy_from_rdms',
    'spin_squared_from_statevector',
    'compute_natural_orbital_occupations',
    'compute_m_diagnostic',
    'compute_n_unpaired_electrons',
    'embed_active_to_full_mo',
    'mo_to_ao_1rdm',
    'validate_trace',
    'embed_ci_vector',
    'rdm1_from_ci_vector',
    'rdm12_from_ci_vector',
]
