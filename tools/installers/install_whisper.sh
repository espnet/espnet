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
if V(torch.__version__) >= V("1.7.0"):
    print("true")
else:
    print("false")
EOF
)

python_38_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys
if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.8.0"):
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

# hash of tested commit: 9e653bd0ea0f1e9493cb4939733e9de249493cfb

if "${torch_17_plus}" && "${python_38_plus}"; then
    python -m pip install openai-whisper==20230308
else
    echo "[ERROR] whisper does not work with pytorch<1.7.0, python<3.8"
fi