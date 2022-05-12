#!/usr/bin/env bash

set -euo pipefail

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi
torch_17_plus=$(python3 <<EOF
from packaging.version import parse as V
import torch

if V(torch.__version__) >= V("1.7"):
    print("true")
else:
    print("false")
EOF
)

python_36_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys

if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.6"):
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

    rm -rf s3prl

    # S3PRL Commit id when making this PR: `commit a9b3ba906f3406c3e00aa65b08fbcefa0ea115ef`
    git clone https://github.com/s3prl/s3prl.git

else
    echo "[WARNING] s3prl is not prepared for pytorch<1.7.0, python<3.6 now"

fi
