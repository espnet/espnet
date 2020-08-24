#!/bin/bash
set -euo pipefail

if [ $# -eq 0 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <python> [venv-path]"
    exit 1;
elif [ $# -eq 2 ]; then
    PYTHON="$1"
    VENV="$2"
elif [ $# -eq 1 ]; then
    PYTHON="$1"
    VENV=""
fi

if ! "${PYTHON}" --version; then
    echo "Error: ${PYTHON} is not Python?"
    exit 1
fi

if [ -n "${VENV}" ]; then
    "${PYTHON}" -m venv ${VENV}
    echo ". $(cd ${VENV}; pwd)/bin/activate" > activate_python.sh
else
    PYTHON_DIR="$(cd ${PYTHON%/*} && pwd)"
    if [ ! -x "${PYTHON_DIR}"/python3 ]; then
        echo "${PYTHON_DIR}/python3 doesn't exist."
        exit 1
    elif [ ! -x "${PYTHON_DIR}"/pip3 ]; then
        echo "${PYTHON_DIR}/pip3 doesn't exist."
        exit 1
    fi
    echo "PATH=${PYTHON_DIR}:\${PATH}" > activate_python.sh
fi

. ./activate_python.sh
python3 -m pip install -U pip wheel
