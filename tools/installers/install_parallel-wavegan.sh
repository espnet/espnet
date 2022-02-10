#!/usr/bin/env bash

set -euo pipefail

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_17_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.7"):
    print("true")
else:
    print("false")
EOF
)

python_36_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import sys

if V(sys.version) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)

cuda_version=$(python3 <<EOF
import torch
if torch.cuda.is_available():
    version=torch.version.cuda.split(".")
    # 10.1.aa -> 101
    print(version[0] + version[1])
else:
    print("")
EOF
)
echo "cuda_version=${cuda_version}"

if "${torch_17_plus}" && "${python_36_plus}"; then

    rm -rf ParallelWaveGAN

    # ParallelWaveGAN  Commit id when making this PR: `commit 5bef5a0610e5e0d6153b601bbf91c78582260c8f`
    git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
    cd ParallelWaveGAN
    pip install -e .
    cd ..

    # additional requirement for jq
    wget -O jq https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64 
else
    echo "[WARNING] ParallelWaveGAN is recommanded for pytorch>1.7.0, python>3.6"

fi
