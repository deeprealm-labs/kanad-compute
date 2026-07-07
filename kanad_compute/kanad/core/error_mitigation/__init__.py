"""core.error_mitigation — indigenous error-mitigation home (reorg Phase B5).

Owns count->expectation conversion (with the X/Y honesty guard), zero-noise
extrapolation, and the IBM error-mitigation strategy config. Downward-only: no
kanad.backends / kanad.solvers imports — solvers and backends consume from here.
"""

from kanad.core.error_mitigation.expectation import expectation_from_counts
from kanad.core.error_mitigation.zne import (
    richardson_extrapolation,
    zero_noise_extrapolation,
)
from kanad.core.error_mitigation.strategy import ErrorMitigationStrategy

__all__ = [
    'expectation_from_counts',
    'richardson_extrapolation',
    'zero_noise_extrapolation',
    'ErrorMitigationStrategy',
]
