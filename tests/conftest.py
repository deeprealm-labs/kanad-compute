"""Shared pytest fixtures for kanad-compute."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure kanad_compute is importable when tests run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
