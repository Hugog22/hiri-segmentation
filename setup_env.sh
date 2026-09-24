#!/bin/bash
# ==============================================================================
# Setup environment variables for HI&RI project and nnU-Net v2
# Usage: source setup_env.sh
# ==============================================================================

# Ensure script is sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "⚠️ ERROR: This script must be sourced to set environment variables."
    echo "Run: source setup_env.sh"
    exit 1
fi

PROJECT_ROOT="$(pwd)"

export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_ROOT}/nnUNet_results"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Ensure directories exist
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}" logs checkpoints results data/raw data/processed

echo "✅ Environment configured:"
echo "   nnUNet_raw          = ${nnUNet_raw}"
echo "   nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "   nnUNet_results      = ${nnUNet_results}"
echo "   PYTHONPATH          = ${PROJECT_ROOT}"
