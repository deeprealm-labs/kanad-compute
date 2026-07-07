"""Packaging for the Kanad framework.

Flat layout: this directory IS the top-level ``kanad`` package — ``kanad/__init__.py``
lives here alongside the subpackages (``core/``, ``solvers/``, ``bonds/``, ...). The
``package_dir={"kanad": "."}`` mapping tells setuptools that the ``kanad`` package's
source is this very directory, so an editable install exposes ``import kanad`` /
``import kanad.core.*`` exactly as the PYTHONPATH layout does — without moving files.
"""
from setuptools import find_packages, setup

# Sub-packages live directly under this dir (core/, solvers/, ...). find_packages
# returns them by their dotted path relative to here ("core", "core.ci", ...); we
# re-parent each under the top-level "kanad" package. Non-shipped trees are excluded.
_EXCLUDE = [
    "tests", "tests.*",
    "benchmarks", "benchmarks.*",
    "docs", "docs.*",
    "ideas", "ideas.*",
    "scripts", "scripts.*",
]
_subpackages = find_packages(where=".", exclude=_EXCLUDE)

setup(
    name="kanad",
    version="0.1.2",
    description="Kanad — governance-driven multi-representation quantum chemistry framework",
    python_requires=">=3.11",
    packages=["kanad"] + [f"kanad.{pkg}" for pkg in _subpackages],
    package_dir={"kanad": "."},
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "qiskit>=1.2.0",
        "qiskit-aer>=0.15.0",
        "pyscf>=2.4",
    ],
)
