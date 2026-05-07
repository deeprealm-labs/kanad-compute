"""Local credential vault.

Wraps the OS keyring (Keychain on macOS, Credential Manager on Windows,
Secret Service on Linux) so user-supplied API tokens for IBM Quantum, IonQ,
and friends never leave the machine.

Phase 2 design — credentials are stored locally only. The kanad-app cloud no
longer needs `User.kanad_compute_key`-style plaintext columns: the compute
node holds the secrets and uses them when running an experiment.

For environments without a usable keyring backend (headless Linux, some CI
boxes), ``Vault`` exposes the failure as a structured error so the CLI can
guide the user to install ``secret-tool`` / ``gnome-keyring`` rather than
silently fall through to plaintext storage.

Storage layout: every key lives under service ``kanad-compute`` with the
canonical name (e.g. ``ibm_api_token``). ``Hello.vault`` reports presence
under shorter logical names (``ibm``, ``ionq``, ``bluequbit``) by mapping
each logical name to one or more canonical key names.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

SERVICE = "kanad-compute"

# Canonical storage keys — what the worker reads when invoking solvers.
CANONICAL_KEYS: tuple[str, ...] = (
    "ibm_api_token",
    "ibm_crn",
    "ionq_api_key",
    "bluequbit_api_key",
)

# Logical → canonical mapping for Hello.vault
LOGICAL_TO_CANONICAL: dict[str, tuple[str, ...]] = {
    "ibm": ("ibm_api_token",),  # IBM works without CRN; CRN is optional
    "ionq": ("ionq_api_key",),
    "bluequbit": ("bluequbit_api_key",),
}


class VaultError(RuntimeError):
    pass


class Vault:
    """Thin wrapper around ``keyring``.

    Constructor is cheap; the underlying backend is resolved lazily on first
    access so importing the module never raises.
    """

    def __init__(self, service: str = SERVICE):
        self.service = service

    # ── primitive ops ──────────────────────────────────────────────────────

    def set(self, key: str, value: str) -> None:
        if key not in CANONICAL_KEYS:
            raise VaultError(f"unknown vault key: {key!r}; allowed: {CANONICAL_KEYS}")
        try:
            import keyring
            keyring.set_password(self.service, key, value)
        except Exception as e:
            raise VaultError(f"keyring set failed for {key}: {e}") from e

    def get(self, key: str) -> Optional[str]:
        try:
            import keyring
            return keyring.get_password(self.service, key)
        except Exception as e:
            logger.debug(f"keyring get failed for {key}: {e}")
            return None

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self, key: str) -> bool:
        try:
            import keyring
            keyring.delete_password(self.service, key)
            return True
        except Exception as e:
            logger.debug(f"keyring delete failed for {key}: {e}")
            return False

    # ── higher-level ───────────────────────────────────────────────────────

    def status(self) -> dict[str, bool]:
        """Return ``Hello.vault``-shaped presence map keyed by logical names."""
        return {
            logical: all(self.has(c) for c in canonicals)
            for logical, canonicals in LOGICAL_TO_CANONICAL.items()
        }

    def all(self) -> dict[str, Optional[str]]:
        """Snapshot of every canonical key. Values are full secrets — only
        intended for the worker passing creds into a solver call. Never log."""
        return {k: self.get(k) for k in CANONICAL_KEYS}

    def list_present(self) -> list[str]:
        return [k for k in CANONICAL_KEYS if self.has(k)]
