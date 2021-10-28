#!/usr/bin/env bash

set -euo pipefail

use_pip=false

if [ $# -gt 1 ]; then
    echo "Usage: $0 [use_pip | true or false]"
    exit 1;
elif [ $# -eq 1 ]; then
    use_pip="$1"
fi


has_conda=$([[ $(conda list -e -c -f --no-pip pytorch) =~ pytorch ]] && echo true || echo false)
# Uncomment to use pip
# has_conda=false
python_version=$(python3 <<EOF
import sys
print(sys.version.split()[0])
EOF
)

cuda_version=$(python3 <<EOF
try:
    import torch
except:
    raise RuntimeError("Please install torch before running this script")

if torch.cuda.is_available():
    version=torch.version.cuda.split(".")
    # 10.1.aa -> 10.1
    print(version[0] + "." + version[1])
else:
    print("")
EOF
)
torch_version=$(python3 -c "import torch; print(torch.__version__)")

python_plus(){
    python3 <<EOF
from distutils.version import LooseVersion as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
pt_plus(){
    python3 <<EOF
from distutils.version import LooseVersion as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
install_torchaudio(){
    if "${has_conda}" && ! "${use_pip}"; then
        conda install -y -c pytorch "torchaudio=$1"
    else
        python3 -m pip install "torchaudio==$1"
    fi
}


echo "[INFO] python_version=${python_version}"
echo "[INFO] torch_version=${torch_version}"
echo "[INFO] cuda_version=${cuda_version}"


if ! "$(python_plus 3.6)" && $(pt_plus 1.6.0); then
    echo "[ERROR] torchaudio>=1.6.0 is not provide for python<3.6"
    exit 1
else

    if $(pt_plus 1.10.1); then
        echo "[ERROR] This script doesn't support pytorch>1.10.0"
        exit 1

    elif $(pt_plus 1.10.0); then
        if "${has_conda}" && ! "${use_pip}"; then
            if [ -z "${cuda_version}" ]; then
                conda install -y "torchaudio=0.10.0" "torch=1.10.0" cpuonly -c pytorch
            elif [ "${cuda_version}" = 10.2 ]; then
                conda install -y "torchaudio=0.10.0" "torch=1.10.0" cudatoolkit=10.2 -c pytorch
            elif [ "${cuda_version}" = 11.3 ]; then
                conda install -y "torchaudio=0.10.0" "torch=1.10.0" cudatoolkit=11.3 -c pytorch
            else
                echo "[ERROR] cuda=${cuda_version} is not supported"
            fi
        else
            if [ -z "${cuda_version}" ]; then
                # cpu only
                python3 -m pip install "torchaudio==0.10.0+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
            elif [ "${cuda_version}" = 10.2 ]; then
                python3 -m pip install "torchaudio==0.10.0"
            elif [ "${cuda_version}" = 11.3 ]; then
                python3 -m pip install "torchaudio==0.10.0+cu113" -f https://download.pytorch.org/whl/cu113/torch_stable.html
            else
                echo "[ERROR] cuda=${cuda_version} is not supported"
            fi
        fi

    elif $(pt_plus 1.9.1); then
        install_torchaudio 0.9.1

    elif $(pt_plus 1.9.0); then
        install_torchaudio 0.9.0

    elif $(pt_plus 1.8.1); then
        install_torchaudio 0.8.1

    elif $(pt_plus 1.8.0); then
        install_torchaudio 0.8.0

    elif $(pt_plus 1.7.1); then
        install_torchaudio 0.7.2

    elif $(pt_plus 1.7.0); then
        install_torchaudio 0.7.0

    elif $(pt_plus 1.6.0); then
        install_torchaudio 0.6.0

    elif $(pt_plus 1.5.1); then
        install_torchaudio 0.5.1

    elif $(pt_plus 1.5.0); then
        install_torchaudio 0.5.0

    elif $(pt_plus 1.4.0); then
        install_torchaudio 0.4.0

    elif $(pt_plus 1.3.1); then
        install_torchaudio 0.3.2

    elif $(pt_plus 1.3.0); then
        install_torchaudio 0.3.1

    elif $(pt_plus 1.2.0); then
        install_torchaudio 0.3.0
    fi
fi


# Check the pytorch version is not changed from the original version
current_torch_version="$(python3 -c 'import torch; print(torch.__version__)')"
if [ ${torch_version} != "${current_torch_version}" ]; then
    echo "[ERROR] The torch version has been changed. Please report to espnet developers"
    exit 1
fi
