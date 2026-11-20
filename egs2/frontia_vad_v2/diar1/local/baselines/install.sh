#!/usr/bin/env bash
# Create the Python environment of the external VAD baselines:
#   local/baselines/install.sh <venv dir> [<python>]
# then: local/baselines/eval.sh --python <venv dir>/bin/python ...
set -euo pipefail
venv=$1
py=${2:-python3}
"${py}" -m venv "${venv}"
"${venv}/bin/pip" install --upgrade pip
"${venv}/bin/pip" install -r "$(dirname "$0")/requirements.txt"
