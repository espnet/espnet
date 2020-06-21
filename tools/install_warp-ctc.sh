#!/bin/bash
set -euo pipefail

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

torch_12_plus=$(python <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.2"): 
    print("true")
else:
    print("false")
EOF
)

torch_11_plus=$(python <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.1"): 
    print("true")
else:
    print("false")
EOF
)

torch_10_plus=$(python <<EOF
from distutils.version import LooseVersion as V
import torch

if V(torch.__version__) >= V("1.0"): 
    print("true")
else:
    print("false")
EOF
)

cuda_version=$(python <<EOF
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

if "${torch_12_plus}"; then

    echo "[WARNING] warp-ctc is not prepared for pytorch>=1.2.0 now"

elif "${torch_11_plus}"; then

    if [ -z "${cuda_version}" ]; then
        pip install warpctc-pytorch11-cpu; 
    else 
        pip install warpctc-pytorch11-cuda"${cuda_version}"
    fi 

elif "${torch_10_plus}"; then

    if [ -z "${cuda_version}" ]; then
        pip install warpctc-pytorch10-cpu
    else
        pip install warpctc-pytorch10-cuda"${cuda_version}"
    fi 

else

    rm -rf warp-ctc
    git clone https://github.com/espnet/warp-ctc.git
    ( 
        set -euo pipefail

        cd warp-ctc
        git checkout -b pytorch-0.4 remotes/origin/pytorch-0.4
        mkdir build
        
        ( 
            set -euo pipefail
            cd build && cmake .. && ${MAKE}
        )

        pip install cffi
        ( 
            set -euo pipefail
            cd pytorch_binding && python setup.py installl
        )
    )

fi
