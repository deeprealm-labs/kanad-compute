//! Local credential vault — Rust mirror of `kanad_compute/vault.py`.
//!
//! Wraps the OS keyring (Keychain on macOS, Credential Manager on Windows,
//! Secret Service on Linux). Public surface mirrors the Python `Vault`
//! class so the CLI behaviour is identical regardless of which runtime
//! is in use.

use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use thiserror::Error;
use zeroize::Zeroizing;

pub const SERVICE: &str = "kanad-compute";

/// Canonical storage keys — anything passed to `set` must match.
pub const CANONICAL_KEYS: &[&str] = &[
    "ibm_api_token",
    "ibm_crn",
    "ionq_api_key",
    "bluequbit_api_key",
    "kanad_access_token",
];

/// Logical → canonical mapping for `Hello.vault`.
pub fn logical_to_canonical() -> BTreeMap<&'static str, &'static [&'static str]> {
    let mut m = BTreeMap::new();
    m.insert("ibm", &["ibm_api_token"][..]);
    m.insert("ionq", &["ionq_api_key"][..]);
    m.insert("bluequbit", &["bluequbit_api_key"][..]);
    m
}

#[derive(Debug, Error)]
pub enum VaultError {
    #[error("unknown vault key: {0:?}")]
    UnknownKey(String),
    #[error("keyring backend error: {0}")]
    Backend(String),
}

/// Backend abstraction so tests can inject an in-memory map without
/// touching the real OS keyring.
pub trait Backend: Send + Sync {
    fn set(&self, service: &str, key: &str, value: &str) -> Result<(), VaultError>;
    fn get(&self, service: &str, key: &str) -> Result<Option<String>, VaultError>;
    fn clear(&self, service: &str, key: &str) -> Result<bool, VaultError>;
}

pub struct KeyringBackend;

impl Backend for KeyringBackend {
    fn set(&self, service: &str, key: &str, value: &str) -> Result<(), VaultError> {
        let entry =
            keyring::Entry::new(service, key).map_err(|e| VaultError::Backend(e.to_string()))?;
        entry
            .set_password(value)
            .map_err(|e| VaultError::Backend(e.to_string()))
    }

    fn get(&self, service: &str, key: &str) -> Result<Option<String>, VaultError> {
        let entry =
            keyring::Entry::new(service, key).map_err(|e| VaultError::Backend(e.to_string()))?;
        match entry.get_password() {
            Ok(s) => Ok(Some(s)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => {
                tracing::debug!("keyring get failed for {key}: {e}");
                Ok(None)
            }
        }
    }

    fn clear(&self, service: &str, key: &str) -> Result<bool, VaultError> {
        let entry =
            keyring::Entry::new(service, key).map_err(|e| VaultError::Backend(e.to_string()))?;
        match entry.delete_credential() {
            Ok(()) => Ok(true),
            Err(keyring::Error::NoEntry) => Ok(false),
            Err(e) => {
                tracing::debug!("keyring delete failed for {key}: {e}");
                Ok(false)
            }
        }
    }
}

/// Simple in-memory backend, for tests.
#[derive(Default, Clone)]
pub struct MemoryBackend(Arc<Mutex<HashMap<(String, String), String>>>);

impl Backend for MemoryBackend {
    fn set(&self, service: &str, key: &str, value: &str) -> Result<(), VaultError> {
        self.0
            .lock()
            .insert((service.into(), key.into()), value.into());
        Ok(())
    }
    fn get(&self, service: &str, key: &str) -> Result<Option<String>, VaultError> {
        Ok(self.0.lock().get(&(service.into(), key.into())).cloned())
    }
    fn clear(&self, service: &str, key: &str) -> Result<bool, VaultError> {
        Ok(self
            .0
            .lock()
            .remove(&(service.into(), key.into()))
            .is_some())
    }
}

pub struct Vault {
    service: String,
    backend: Box<dyn Backend>,
}

impl Vault {
    pub fn new() -> Self {
        Self::with_backend(Box::new(KeyringBackend))
    }

    pub fn with_backend(backend: Box<dyn Backend>) -> Self {
        Self {
            service: SERVICE.into(),
            backend,
        }
    }

    pub fn set(&self, key: &str, value: &str) -> Result<(), VaultError> {
        if !CANONICAL_KEYS.contains(&key) {
            return Err(VaultError::UnknownKey(key.into()));
        }
        // Hold the secret in a Zeroizing buffer until handed to the backend
        // so debug formatting / panic-unwind paths can't leak it via stack.
        let buf = Zeroizing::new(value.to_owned());
        self.backend.set(&self.service, key, &buf)
    }

    pub fn get(&self, key: &str) -> Option<String> {
        self.backend.get(&self.service, key).ok().flatten()
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn clear(&self, key: &str) -> bool {
        self.backend.clear(&self.service, key).unwrap_or(false)
    }

    /// `Hello.vault`-shaped presence map, keyed by logical names.
    pub fn status(&self) -> BTreeMap<&'static str, bool> {
        logical_to_canonical()
            .into_iter()
            .map(|(logical, canonicals)| (logical, canonicals.iter().all(|c| self.has(c))))
            .collect()
    }

    pub fn list_present(&self) -> Vec<&'static str> {
        CANONICAL_KEYS
            .iter()
            .copied()
            .filter(|k| self.has(k))
            .collect()
    }

    /// Snapshot every canonical key. Values are full secrets — only use
    /// when piping into a solver call. Never log.
    pub fn all(&self) -> HashMap<&'static str, Option<String>> {
        CANONICAL_KEYS.iter().map(|k| (*k, self.get(k))).collect()
    }
}

impl Default for Vault {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_vault() -> Vault {
        Vault::with_backend(Box::new(MemoryBackend::default()))
    }

    #[test]
    fn set_get_roundtrip() {
        let v = mem_vault();
        v.set("ibm_api_token", "secret").unwrap();
        assert_eq!(v.get("ibm_api_token").as_deref(), Some("secret"));
        assert!(v.has("ibm_api_token"));
    }

    #[test]
    fn unknown_key_rejected() {
        let v = mem_vault();
        assert!(matches!(v.set("nope", "x"), Err(VaultError::UnknownKey(_))));
    }

    #[test]
    fn missing_returns_none() {
        let v = mem_vault();
        assert_eq!(v.get("ibm_api_token"), None);
        assert!(!v.has("ibm_api_token"));
    }

    #[test]
    fn clear_returns_present_flag() {
        let v = mem_vault();
        v.set("ionq_api_key", "x").unwrap();
        assert!(v.clear("ionq_api_key"));
        assert!(!v.clear("ionq_api_key"));
    }

    #[test]
    fn status_reflects_canonical_presence() {
        let v = mem_vault();
        let s = v.status();
        assert!(!s["ibm"]);
        v.set("ibm_api_token", "x").unwrap();
        let s = v.status();
        assert!(s["ibm"]);
        assert!(!s["ionq"]);
    }

    #[test]
    fn list_present_canonical_only() {
        let v = mem_vault();
        v.set("ibm_api_token", "a").unwrap();
        v.set("kanad_access_token", "b").unwrap();
        let mut got = v.list_present();
        got.sort();
        assert_eq!(got, vec!["ibm_api_token", "kanad_access_token"]);
    }
}
