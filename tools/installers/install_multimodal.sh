#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi
torch_version=$(python3 -c "import torch; print(torch.__version__)")
python_36_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys

if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.6"):
    print("true")
else:
    print("false")
EOF
)
pt_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"

if ! "${python_36_plus}"; then
    echo "[ERROR] python<3.6 is not supported"
    exit 1
else

    if $(pt_plus 1.8.0); then
        python3 -m pip install opencv-python
        python3 -m pip install dlib==19.17.0
        python3 -m pip install sk-video
        python3 -m pip install scikit-image
        if [[ "${torch_version}" == "1.8.0" ]]; then
            python3 -m pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
        elif [[ "${torch_version}" == "1.8.1" ]]; then
            python3 -m pip install torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1
        elif [[ "${torch_version}" == "1.9.0" ]]; then
            python3 -m pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0
        elif [[ "${torch_version}" == "1.9.1" ]]; then
            python3 -m pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1
        elif [[ "${torch_version}" == "1.10.0" ]]; then
            python3 -m pip install torch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0
        elif [[ "${torch_version}" == "1.10.1" ]]; then
            python3 -m pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
        elif [[ "${torch_version}" == "1.11.0" ]]; then
            python3 -m pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
        else
            python3 -m pip install torchvision torchaudio
        fi
    else
        echo "[WARNING] ESPNet Multimodal Vision requires pytorch>=1.8.*"
    fi

fi


# Check the pytorch version is not changed from the original version
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ ${torch_version} != "${current_torch_version}" ]; then
    echo "[ERROR] The torch version has been changed. Please report to espnet administrators"
    exit 1
fi

