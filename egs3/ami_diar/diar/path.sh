#!/usr/bin/env bash
# Path configuration for AMI diarization recipe

# Add ESPnet to PYTHONPATH
MAIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${MAIN_ROOT}:${PYTHONPATH:-}"

# Add TEMPLATE to PYTHONPATH
TEMPLATE_DIR="${MAIN_ROOT}/egs3/TEMPLATE/asr"
export PYTHONPATH="${TEMPLATE_DIR}:${PYTHONPATH}"

# Add recipe to PYTHONPATH
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${RECIPE_DIR}:${PYTHONPATH}"

# Activate Python environment if tools/activate_python.sh exists
if [ -f "${MAIN_ROOT}/tools/activate_python.sh" ]; then
  . "${MAIN_ROOT}/tools/activate_python.sh"
fi
