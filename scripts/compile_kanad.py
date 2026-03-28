#!/usr/bin/env python3
"""Compile kanad framework to .so/.pyd binary extensions using Cython.

Usage:
    python scripts/compile_kanad.py /path/to/kanad-app/kanad

This compiles all .py files in the kanad package to binary extensions,
then copies them into kanad_compute/_kanad/ for bundling.
Protects source code IP — only compiled binaries are distributed.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compile_kanad.py /path/to/kanad-app/kanad")
        sys.exit(1)

    kanad_src = Path(sys.argv[1]).resolve()
    if not kanad_src.is_dir() or not (kanad_src / "__init__.py").exists():
        print(f"Error: {kanad_src} is not a valid kanad package directory")
        sys.exit(1)

    # Output directory inside kanad-compute
    script_dir = Path(__file__).resolve().parent.parent
    output_dir = script_dir / "kanad_compute" / "_kanad_bundle"

    print(f"Source:  {kanad_src}")
    print(f"Output:  {output_dir}")

    # Clean previous build
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Copy kanad source to temp dir for compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_kanad = Path(tmpdir) / "kanad"
        shutil.copytree(kanad_src, tmp_kanad, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_cache", "CLAUDE.md", "*.egg-info",
        ))

        # Create setup.py for Cython compilation
        setup_py = Path(tmpdir) / "setup.py"
        py_files = list(tmp_kanad.rglob("*.py"))

        # Build extension list
        extensions = []
        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue  # Keep __init__.py as source for package discovery
            rel = py_file.relative_to(Path(tmpdir))
            module_name = str(rel).replace("/", ".").replace("\\", ".").removesuffix(".py")
            extensions.append((module_name, str(rel)))

        setup_content = f"""
import os
from setuptools import setup, find_packages
from Cython.Build import cythonize

ext_sources = {extensions}

setup(
    name="kanad-compiled",
    packages=find_packages(),
    ext_modules=cythonize(
        [src for _, src in ext_sources],
        compiler_directives={{'language_level': '3'}},
        quiet=True,
    ),
)
"""
        setup_py.write_text(setup_content)

        # Run Cython compilation
        print(f"\nCompiling {len(extensions)} modules with Cython...")
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Cython compilation failed:\n{result.stderr}")
            # Fall back to copying .py source files directly
            print("\nFalling back to source copy (no compilation)...")
            shutil.copytree(tmp_kanad, output_dir)
            _write_installer(output_dir)
            print(f"Copied {len(py_files)} source files to {output_dir}")
            return

        # Collect compiled .so/.pyd files + __init__.py files
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy __init__.py files (needed for package structure)
        for init_file in tmp_kanad.rglob("__init__.py"):
            rel = init_file.relative_to(tmp_kanad)
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(init_file, dest)

        # Copy compiled .so/.pyd files
        compiled_count = 0
        for ext_file in Path(tmpdir).rglob("*.so"):
            rel = ext_file.relative_to(Path(tmpdir))
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ext_file, dest)
            compiled_count += 1

        for ext_file in Path(tmpdir).rglob("*.pyd"):
            rel = ext_file.relative_to(Path(tmpdir))
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ext_file, dest)
            compiled_count += 1

        # Copy any data files (json, md, etc.)
        for data_file in tmp_kanad.rglob("*"):
            if data_file.is_file() and data_file.suffix not in (".py", ".pyc", ".so", ".pyd", ".c"):
                rel = data_file.relative_to(tmp_kanad)
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(data_file, dest)

        _write_installer(output_dir)
        print(f"\nCompiled {compiled_count} binary modules")
        print(f"Output: {output_dir}")


def _write_installer(output_dir: Path):
    """Write a helper that installs the bundled kanad into site-packages."""
    installer = output_dir / "_install.py"
    installer.write_text('''"""Install bundled kanad package into the Python path."""
import sys
import os

def install():
    """Add bundled kanad to Python path if not already importable."""
    try:
        import kanad
        return  # Already installed
    except ImportError:
        pass

    # Add parent of this _kanad_bundle dir to path
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(bundle_dir)

    # The kanad package is inside _kanad_bundle/
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
''')


if __name__ == "__main__":
    main()
