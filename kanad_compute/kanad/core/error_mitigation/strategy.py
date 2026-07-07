"""Error-mitigation strategy config (core.error_mitigation.strategy).

Promoted verbatim from backends/ibm/error_mitigation.py (reorg Phase B5). Holds
the resilience/readout/ZNE/DD/twirling knobs consumed by IBMBackend, plus the
simulator-vs-hardware auto_configure policy. The three dead stub methods
(get_resilience_options / get_transpiler_options / estimate_mitigation_overhead)
were dropped — they had zero callers and always returned empty/heuristic values.
stdlib + logging only; no kanad.backends / kanad.solvers imports.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ErrorMitigationStrategy:
    """Configurable error-mitigation strategy for IBM backends.

    The DD/ZNE/twirling/resilience attributes are read directly by IBMBackend
    (M11 QPU scaffolding). ``auto_configure`` picks no mitigation for simulators
    and the full stack (resilience_level=2 + readout + XY4 DD) for real hardware.
    """

    def __init__(
        self,
        resilience_level: int = 1,
        readout_mitigation: bool = True,
        zne_extrapolation: Optional[str] = None,
        zne_noise_factors: Optional[List[float]] = None,
        dynamical_decoupling: Optional[str] = None,
        twirling: bool = False,
        measure_mitigation: bool = True,
    ):
        self.resilience_level = resilience_level
        self.readout_mitigation = readout_mitigation
        self.zne_extrapolation = zne_extrapolation
        self.zne_noise_factors = zne_noise_factors or [1.0, 1.5, 2.0]
        self.dynamical_decoupling = dynamical_decoupling
        self.twirling = twirling
        self.measure_mitigation = measure_mitigation

        logger.info("Error mitigation strategy initialized:")
        logger.info(f"  Resilience level: {resilience_level}")
        logger.info(f"  Readout mitigation: {readout_mitigation}")
        logger.info(f"  ZNE extrapolation: {zne_extrapolation or 'disabled'}")
        logger.info(f"  Dynamical decoupling: {dynamical_decoupling or 'disabled'}")
        logger.info(f"  Twirling: {twirling}")

    @staticmethod
    def auto_configure(backend_name: str) -> 'ErrorMitigationStrategy':
        """Configure mitigation by backend type: simulators -> none; real hardware
        -> resilience_level=2 + readout + XY4 dynamical decoupling."""
        backend_lower = backend_name.lower()
        is_simulator = any(sim in backend_lower for sim in [
            'simulator', 'statevector', 'aer', 'fake'
        ])
        if is_simulator:
            logger.info(f"Auto-config: Simulator detected ({backend_name}) - disabling error mitigation")
            return ErrorMitigationStrategy(
                resilience_level=0,
                readout_mitigation=False,
                zne_extrapolation=None,
                dynamical_decoupling=None,
                twirling=False,
                measure_mitigation=False,
            )
        logger.info(f"Auto-config: Real hardware detected ({backend_name}) - enabling full mitigation")
        return ErrorMitigationStrategy(
            resilience_level=2,
            readout_mitigation=True,
            zne_extrapolation=None,
            zne_noise_factors=None,
            dynamical_decoupling='XY4',
            twirling=False,
            measure_mitigation=True,
        )
