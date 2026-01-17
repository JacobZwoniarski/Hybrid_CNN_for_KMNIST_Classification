#!/usr/bin/env bash
set -euo pipefail

# Config 
VENV_DIR=".venv"

echo "Python: $(python --version)"
echo "Creating venv in ${VENV_DIR} ..."

# Create venv
python -m venv "${VENV_DIR}"

# Activate (works for macOS/Linux bash/zsh)
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

# Install project package
pip install -e .

echo ""
echo "Done."
echo "To activate later, run:"
echo "  source ${VENV_DIR}/bin/activate"
