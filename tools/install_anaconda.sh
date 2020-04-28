#!/bin/bash
set -euo pipefail 

if [ -z "${PS1:-}" ]; then
    PS1=__dummy__
fi
CONDA_URL=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <python-version> <output>"
    exit 1;
fi
PYTHON_VERSION="$1"
output_dir="$2"

if [ ! -e miniconda.sh ]; then
	wget --tries=3 "${CONDA_URL}" -O miniconda.sh
fi

if [ ! -e "$(pwd)/${output_dir}" ]; then
    bash miniconda.sh -b -p "$(pwd)/${output_dir}"
fi

source "${output_dir}/etc/profile.d/conda.sh"
conda deactivate
conda activate

conda install -y setuptools -c anaconda
conda install -y pip -c anaconda
conda update -y conda
conda install -y "python=${PYTHON_VERSION}"
