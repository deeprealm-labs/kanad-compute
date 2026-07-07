"""Analytical VQE gradients via the adjoint-state method.

For a parameterized circuit U(θ) acting on |0⟩ with Hamiltonian H, the
gradient of the energy ``E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩`` w.r.t. each parameter is

::

    ∂E/∂θ_k = −2 Σ_j (∂α_j/∂θ_k) · Im⟨ψ_j|G_j|R_j⟩

where the sum runs over gates ``j`` whose angle ``α_j`` depends on ``θ_k``,
``G_j`` is the Hermitian generator of gate ``j`` (e.g. ``Z/2`` for ``RZ``),
``|ψ_j⟩`` is the state right after gate ``j``, and
``|R_j⟩ = U_{j+1}^† ⋯ U_N^† H|ψ⟩`` is the "right state" propagated backwards.

The adjoint method computes all parameter gradients in one forward + one
backward pass — total cost O(N) statevector ops vs O(N²) for finite
differences. On a 92-parameter Givens-SD circuit (BeH₂), the gradient call
drops from ~50 s (scipy finite-diff) to ~0.5 s.
"""

from kanad.core.vqe_gradients.adjoint_gradient import (
    adjoint_energy_gradient,
    AdjointGradientCalculator,
)

__all__ = ['adjoint_energy_gradient', 'AdjointGradientCalculator']
