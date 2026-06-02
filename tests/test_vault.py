"""Vault tests using an in-memory keyring backend.

The real OS keyring is unavailable in CI / headless environments. We swap
in `keyring.backends.fail.Keyring` is the wrong choice (would error). Use a
tiny stub backend that behaves like an in-memory dict.
"""

from __future__ import annotations

from typing import Any

import pytest

import keyring
import keyring.backend

from kanad_compute.vault import (
    CANONICAL_KEYS,
    LOGICAL_TO_CANONICAL,
    Vault,
    VaultError,
)


class _MemBackend(keyring.backend.KeyringBackend):
    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) in self._store:
            del self._store[(service, username)]
        else:
            raise keyring.errors.PasswordDeleteError("not found")


@pytest.fixture(autouse=True)
def mem_backend(monkeypatch):
    backend = _MemBackend()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    keyring.set_keyring(backend)
    yield backend


def test_set_get_roundtrip():
    v = Vault(service="test-kanad")
    v.set("ibm_api_token", "secret-token")
    assert v.get("ibm_api_token") == "secret-token"
    assert v.has("ibm_api_token")


def test_set_rejects_unknown_key():
    v = Vault(service="test-kanad")
    with pytest.raises(VaultError):
        v.set("definitely_not_a_key", "x")


def test_get_returns_none_for_missing():
    v = Vault(service="test-kanad")
    assert v.get("ibm_api_token") is None
    assert not v.has("ibm_api_token")


def test_clear_removes_entry():
    v = Vault(service="test-kanad")
    v.set("ionq_api_key", "abc")
    assert v.clear("ionq_api_key") is True
    assert v.get("ionq_api_key") is None


def test_clear_missing_is_false():
    v = Vault(service="test-kanad")
    assert v.clear("ionq_api_key") is False


def test_status_maps_logical_to_canonical():
    v = Vault(service="test-kanad")
    # No creds set → all False
    assert v.status() == {"ibm": False, "ionq": False, "bluequbit": False}

    v.set("ibm_api_token", "x")
    s = v.status()
    assert s["ibm"] is True
    assert s["ionq"] is False
    assert set(s.keys()) == set(LOGICAL_TO_CANONICAL.keys())


def test_list_present_returns_canonical_names():
    v = Vault(service="test-kanad")
    v.set("ibm_api_token", "x")
    v.set("ionq_api_key", "y")
    present = sorted(v.list_present())
    assert present == ["ibm_api_token", "ionq_api_key"]


def test_canonical_keys_are_known():
    """Sanity: anything in LOGICAL_TO_CANONICAL must reference a real key."""
    for canonicals in LOGICAL_TO_CANONICAL.values():
        for c in canonicals:
            assert c in CANONICAL_KEYS, f"{c} not in CANONICAL_KEYS"
