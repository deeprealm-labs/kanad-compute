"""Kanad Compute — Turn your computer into a quantum chemistry compute server."""

__version__ = "0.2.6"

# Make bundled kanad framework importable as "import kanad"
import sys as _sys
import os as _os

_bundled_kanad = _os.path.join(_os.path.dirname(__file__), "kanad")
if _os.path.isdir(_bundled_kanad):
    # Only add if kanad isn't already installed externally
    try:
        import kanad  # noqa: F401
    except ImportError:
        # Insert the parent dir of the bundled kanad package into sys.path
        _parent = _os.path.dirname(__file__)
        if _parent not in _sys.path:
            _sys.path.insert(0, _parent)
