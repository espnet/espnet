#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_18_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.8"):
    print("true")
else:
    print("false")
EOF
)


# Install speechbrain
if [ ! -e speechbrain.done ]; then
    if "${torch_18_plus}"; then
        python3 -m pip install speechbrain==0.5.11
        touch speechbrain.done
    else
        echo "[ERROR]: speechbrain requires pytorch>=1.8.0"
        exit 1
    fi
else
    echo "speechbrain is already installed."
fi
