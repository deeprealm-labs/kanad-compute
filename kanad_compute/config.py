"""Configuration management for Kanad Compute server."""

import json
import secrets
import uuid
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".kanad-compute"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _default_config() -> dict:
    return {
        "node_id": str(uuid.uuid4()),
        "api_key": secrets.token_urlsafe(32),
        "host": "0.0.0.0",
        "port": 7440,
        "kanad_api_url": "https://kanad-api-640826962316.us-central1.run.app",
        "max_workers": 2,
        "max_qubits": 20,
        "gpu_enabled": False,
        "gpu_device": "auto",
        "ibm_api_token": None,
        "ibm_crn": None,
        "ionq_api_key": None,
        "log_level": "info",
    }


def load_config() -> dict:
    """Load config from disk, or return defaults if no config exists."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            saved = json.load(f)
        # Merge with defaults (in case new fields were added)
        cfg = _default_config()
        cfg.update(saved)
        return cfg
    return _default_config()


def save_config(cfg: dict) -> Path:
    """Save config to disk. Creates config dir if needed."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    CONFIG_FILE.chmod(0o600)  # Restrict permissions (contains API key)
    return CONFIG_FILE


def init_config(
    port: int = 7440,
    max_qubits: int = 20,
    gpu: bool = False,
    ibm_token: Optional[str] = None,
    ionq_key: Optional[str] = None,
) -> dict:
    """Initialize a fresh config (preserves node_id/api_key if existing)."""
    existing = load_config() if CONFIG_FILE.exists() else {}

    cfg = _default_config()
    # Preserve identity across re-inits
    if "node_id" in existing:
        cfg["node_id"] = existing["node_id"]
    if "api_key" in existing:
        cfg["api_key"] = existing["api_key"]

    cfg["port"] = port
    cfg["max_qubits"] = max_qubits
    cfg["gpu_enabled"] = gpu
    if ibm_token:
        cfg["ibm_api_token"] = ibm_token
    if ionq_key:
        cfg["ionq_api_key"] = ionq_key

    save_config(cfg)
    return cfg
