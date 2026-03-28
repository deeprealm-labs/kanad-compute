#!/bin/bash
# Bundle kanad framework into kanad-compute for distribution.
# Usage: ./scripts/bundle_kanad.sh /path/to/kanad-app/kanad
#
# This copies the kanad package into kanad_compute/kanad/ so it's
# included in the wheel when published to PyPI.

set -e

KANAD_SRC="${1:-../kanad-app/kanad}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DEST="$REPO_DIR/kanad_compute/kanad"

if [ ! -f "$KANAD_SRC/__init__.py" ]; then
    echo "Error: $KANAD_SRC is not a valid kanad package"
    echo "Usage: $0 /path/to/kanad-app/kanad"
    exit 1
fi

echo "Bundling kanad from: $KANAD_SRC"
echo "Into: $DEST"

# Clean previous bundle
rm -rf "$DEST"

# Copy kanad package (excluding dev files)
rsync -a --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='CLAUDE.md' \
    --exclude='*.egg-info' --exclude='tests' \
    "$KANAD_SRC/" "$DEST/"

# Count files
PY_COUNT=$(find "$DEST" -name "*.py" | wc -l)
echo "Bundled $PY_COUNT Python files"
echo "Done. kanad is now included in kanad-compute package."
