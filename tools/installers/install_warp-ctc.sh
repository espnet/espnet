#!/usr/bin/env bash
set -euo pipefail

MAKE=make

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
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

torch_11_plus=$(python3 <<EOF
from packaging.version import parse as V
import torch

if V(torch.__version__) >= V("1.1"):
    print("true")
else:
    print("false")
EOF
)

torch_10_plus=$(python3 <<EOF
from packaging.version import parse as V
import torch

if V(torch.__version__) >= V("1.0"):
    print("true")
else:
    print("false")
EOF
)

torch_version=$(python3 <<EOF
import torch
version = torch.__version__.split(".")
print(version[0] + version[1])
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

if "${torch_17_plus}"; then

    echo "[WARNING] warp-ctc is not prepared for pytorch>=1.7.0 now"

elif "${torch_11_plus}"; then

    warpctc_version=0.2.2
    release_page_url=https://github.com/espnet/warp-ctc/releases/tag/v${warpctc_version}
    if [ -z "${cuda_version}" ]; then
        python3 -m pip install warpctc-pytorch==${warpctc_version}+torch"${torch_version}".cpu -f ${release_page_url}
    else
        python3 -m pip install warpctc-pytorch==${warpctc_version}+torch"${torch_version}".cuda"${cuda_version}" -f ${release_page_url}
    fi

elif "${torch_10_plus}"; then

    if [ -z "${cuda_version}" ]; then
        python3 -m pip install warpctc-pytorch10-cpu
    else
        python3 -m pip install warpctc-pytorch10-cuda"${cuda_version}"
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

        python3 -m pip install cffi
        (
            set -euo pipefail
            cd pytorch_binding && python3 -m pip install -e .
        )
    )

fi
