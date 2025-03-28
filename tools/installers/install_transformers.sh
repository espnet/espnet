#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_ver(){
    python3 <<EOF
from packaging.version import parse as L
import torch
if L(torch.__version__) < L('$1'):
    print("true")
else:
    print("false")
EOF
}

TR_VER="4.9.1"
if $(torch_ver 2.1.0); then
    TR_VER+=",<4.50.0"
fi

python3 -m pip install "transformers>=${TR_VER}"
