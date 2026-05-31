#!/usr/bin/env bash
#
# Create the `venv_vega` conda environment for running the Vega model.
#
# Usage:
#   ./create_vega_env.sh linux        # native conda env (CUDA/GPU available on NVIDIA)
#   ./create_vega_env.sh notlinux     # macOS / other: osx-64 (Rosetta) env, CPU-only
#
set -euo pipefail

PLATFORM="${1:-}"
ENV_NAME="venv_vega"
REQ_FILE="requirements_vega.txt"

if [[ "$PLATFORM" != "linux" && "$PLATFORM" != "notlinux" ]]; then
  echo "Usage: ./create_vega_env.sh [linux|notlinux]"
  echo "  linux     : native conda env; torch 1.5.1 ships CUDA 10.2 -> GPU on NVIDIA hosts"
  echo "  notlinux  : macOS / other; creates an osx-64 (Rosetta) env, CPU-only"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found on PATH. Install Miniconda/Anaconda first."
  exit 1
fi

# Make `conda activate` usable inside this non-interactive script.
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ">> Creating environment '$ENV_NAME' (platform: $PLATFORM)"
if [[ "$PLATFORM" == "linux" ]]; then
  conda create -n "$ENV_NAME" python=3.7 -y
else
  # Apple Silicon / non-Linux: build the env as Intel (osx-64) so the old pinned
  # wheels (torch==1.5.1, etc.) resolve. Requires Rosetta on Apple Silicon.
  CONDA_SUBDIR=osx-64 conda create -n "$ENV_NAME" python=3.7 -y
fi

conda activate "$ENV_NAME"

if [[ "$PLATFORM" == "notlinux" ]]; then
  # Pin the env to osx-64 so later installs keep using Intel wheels.
  conda config --env --set subdir osx-64
fi

echo ">> Installing $REQ_FILE into '$ENV_NAME'"
pip install --upgrade pip
pip install -r "$REQ_FILE"

echo ">> Done. Activate with: conda activate $ENV_NAME"
