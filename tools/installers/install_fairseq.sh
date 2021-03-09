#!/usr/bin/env bash

set -euo pipefail

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_15_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.5"):
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

if "${torch_15_plus}" && "${python_36_plus}"; then

    rm -rf fairseq

    # FairSeq Commit id when making this PR: `commit 6225dccb989ebfb268274bad36a794b27e4dd43f`
    git clone https://github.com/pytorch/fairseq.git
    python3 -m pip install --editable ./fairseq
    python3 -m pip install filelock

else
    echo "[WARNING] fairseq is not prepared for pytorch<1.5.0, python<3.6 now"

fi
