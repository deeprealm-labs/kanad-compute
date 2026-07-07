"""Mass-weighted IRC step (Fukui 1970).

The intrinsic reaction coordinate is the steepest-descent path in **mass-weighted
Cartesians** ``q_A = √m_A · x_A``, not raw Cartesians. Stepping along the raw
gradient (the pre-M7 behavior) gives the wrong path whenever atomic masses
differ — light atoms move too far, heavy atoms too little — so an H-transfer or
any heavy-atom rearrangement traces an unphysical coordinate.

In mass-weighted coords the gradient is ``g^q_A = g_A / √m_A`` and a steepest
descent step of arc length ``ds`` is ``Δq_A = ∓ ds · g^q_A / |g^q|``. Converting
back to Cartesians (``Δx_A = Δq_A / √m_A``):

    Δx_A = ∓ ds · g_A / (m_A · |g/√m|),   |g/√m| = sqrt(Σ_A |g_A|² / m_A)
"""

from __future__ import annotations

import numpy as np


def mass_weighted_irc_step(grad, masses, step_size, descend=True):
    """One mass-weighted steepest-descent IRC step.

    Args:
        grad: (n_atoms, 3) energy gradient ∂E/∂x in Ha/Bohr.
        masses: (n_atoms,) atomic masses (amu; only ratios matter).
        step_size: arc length ``ds`` in mass-weighted Bohr·√amu.
        descend: True steps downhill (toward a minimum); False steps uphill
            (the reverse IRC branch).

    Returns:
        (n_atoms, 3) Cartesian displacement Δx in Bohr.
    """
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(masses, dtype=float)[:, None]
    g_mw_norm = float(np.sqrt(np.sum(grad ** 2 / m)))
    if g_mw_norm < 1e-14:
        return np.zeros_like(grad)
    sign = -1.0 if descend else 1.0
    return sign * step_size * (grad / m) / g_mw_norm
