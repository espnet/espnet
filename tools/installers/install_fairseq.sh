#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_version=$(python3 -c "import torch; print(torch.__version__)")
python_36_plus=$(python3 <<EOF
from distutils.version import LooseVersion as V
import sys

if V(sys.version) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)

pt_plus(){
    python3 <<EOF
import sys
from distutils.version import LooseVersion as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"


if "$(pt_plus 1.8.0)" && "${python_36_plus}"; then

    rm -rf fairseq

    # FairSeq Commit id when making this PR: `commit 313ff0581561c7725ea9430321d6af2901573dfb`
    # git clone --depth 1 https://github.com/pytorch/fairseq.git
    # TODO(jiatong): to fix after the issue #4035 is fixed in fairseq
    git clone https://github.com/pytorch/fairseq.git
    cd fairseq
    git checkout -b sync_commit 313ff0581561c7725ea9430321d6af2901573dfb
    cd ..
    python3 -m pip install --editable ./fairseq
    python3 -m pip install filelock

else
    echo "[WARNING] fairseq requires pytorch>=1.8.0, python>=3.6"

fi
