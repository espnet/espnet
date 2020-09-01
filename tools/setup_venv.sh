#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <python> [venv-path]"
    echo "e.g."
    echo "$0 \$(which python3)"
    exit 1;
elif [ $# -eq 2 ]; then
    PYTHON="$1"
    VENV="$2"
elif [ $# -eq 1 ]; then
    PYTHON="$1"
    VENV="venv"
fi

if ! "${PYTHON}" -m venv --help 2>&1 > /dev/null; then
    echo "Error: ${PYTHON} is not Python3?"
    exit 1
fi
if [ -e activate_python.sh ]; then
    echo "Warning: activate_python.sh already exists. It will be overwritten"
fi

"${PYTHON}" -m venv ${VENV}
echo ". $(cd ${VENV}; pwd)/bin/activate" > activate_python.sh

. ./activate_python.sh
python3 -m pip install -U pip wheel
