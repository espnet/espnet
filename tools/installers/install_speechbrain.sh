#!/usr/bin/env bash

set -euo pipefail

MAKE=make

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
    (
        if "${torch_18_plus}"; then
            set -euo pipefail
            python3 -m pip install speechbrain==0.5.11
        else
            echo "ERROR: pytorch version <1.8.0"
        fi
    )
    touch speechbrain.done
else
    echo "speechbrain is already installed."
fi
