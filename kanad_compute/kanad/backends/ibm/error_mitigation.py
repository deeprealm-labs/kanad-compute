"""DEPRECATED location — ErrorMitigationStrategy moved to core.error_mitigation.

Reorg Phase B5 (2026-05-31): the strategy config + ``auto_configure`` were
promoted to :mod:`kanad.core.error_mitigation.strategy` (the dead
``get_resilience_options`` / ``get_transpiler_options`` /
``estimate_mitigation_overhead`` stubs were dropped — zero callers). This module
is a thin re-export so existing imports keep working:

    from kanad.backends.ibm.error_mitigation import ErrorMitigationStrategy
"""

from kanad.core.error_mitigation.strategy import ErrorMitigationStrategy  # noqa: F401

__all__ = ['ErrorMitigationStrategy']
