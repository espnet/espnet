#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

python_36_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import sys

if V(sys.version) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)

pip install git+https://github.com/roshansh-cmu/longformer.git
pip install datasets bert-score
## NLG eval needs Java
conda install openjdk
pip install git+https://github.com/Maluuba/nlg-eval.git@master
