"""Zero-noise extrapolation (core.error_mitigation.zne).

Indigenous home for Richardson / polynomial zero-noise extrapolation. Promoted
from the (orphaned) IBMBackend._richardson_extrapolation (reorg Phase B5) — the
live IBM path now uses Qiskit primitive ZNE, so this is the reusable indigenous
helper for any manual ZNE post-processing. numpy only; no kanad imports.
"""

from __future__ import annotations

from typing import List

import numpy as np


def richardson_extrapolation(noise_factors: List[float], energies: List[float]) -> float:
    """Extrapolate ``energies`` measured at ``noise_factors`` to zero noise.

    2 points -> linear (``y0 - slope*x0``); >2 points -> polynomial fit of degree
    ``min(n-1, 3)`` evaluated at 0.
    """
    x = np.asarray(noise_factors, dtype=float)
    y = np.asarray(energies, dtype=float)
    if len(x) < 2:
        raise ValueError("richardson_extrapolation needs >=2 (noise, energy) points.")
    if len(x) == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return float(y[0] - slope * x[0])
    degree = min(len(x) - 1, 3)  # cap at cubic
    return float(np.poly1d(np.polyfit(x, y, degree))(0))


# Public alias.
zero_noise_extrapolation = richardson_extrapolation
