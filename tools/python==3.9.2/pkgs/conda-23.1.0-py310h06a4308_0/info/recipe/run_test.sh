#!/usr/bin/env bash
set -x

export TEST_MINOR_VER=9

unset CONDA_SHLVL
unset _CE_CONDA
unset _CE_M
unset CONDA_EXE

# Used to interactively load shell functions
eval "$(python -m conda shell.bash hook)"

conda activate base

export PYTHON_MAJOR_VERSION=$(python -c "import sys; print(sys.version_info[0])")
export TEST_PLATFORM=$(python -c "import sys; print('win' if sys.platform.startswith('win') else 'unix')")
export PYTHONHASHSEED=$(python -c "import random as r; print(r.randint(0,4294967296))") && echo "PYTHONHASHSEED=$PYTHONHASHSEED"

env | sort

conda info

# Our tests finish by creating, activating and then deactivating a conda environment
conda create -y -p ./built-conda-test-env python=3.${TEST_MINOR_VER}
conda activate ./built-conda-test-env
echo $CONDA_PREFIX
[ "$CONDA_PREFIX" = "$PWD/built-conda-test-env" ] || exit 1
[ $(python -c "import sys; print(sys.version_info[1])") = ${TEST_MINOR_VER} ] || exit 1
# conda deactivate
