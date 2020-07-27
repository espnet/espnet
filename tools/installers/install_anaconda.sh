#!/bin/bash
set -euo pipefail

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

if [ $# != 3 ]; then
    echo "Usage: $0 <conda-env-name> <python-version> <output>"
    exit 1;
fi
name="$1"
PYTHON_VERSION="$2"
output_dir="$3"

if [ ! -e "${output_dir}/etc/profile.d/conda.sh" ]; then
    if [ ! -e miniconda.sh ]; then
        wget --tries=3 "${CONDA_URL}" -O miniconda.sh
    fi

    bash miniconda.sh -b -p "./${output_dir}"
fi

source "${output_dir}/etc/profile.d/conda.sh"
conda deactivate
# If the env already exists, skip recreation
if ! conda activate ${name}; then
    conda create -yn "${name}"
fi
conda activate ${name}
conda install -y conda "python=${PYTHON_VERSION}" pip setuptools -c anaconda
