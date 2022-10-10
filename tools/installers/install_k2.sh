#!/usr/bin/env bash
set -euo pipefail

# Please update if too old. See https://k2-fsa.org/nightly/, https://anaconda.org/k2-fsa/k2/files
pip_k2_version="1.10.dev20211112"
conda_k2_version="1.10.dev20211103"  # Empty indicates latest version

if [ $# -gt 2 ]; then
    echo "Usage: $0 [use-conda|true or false] [<k2-version>]"
    exit 1;
elif [ $# -gt 0 ]; then
    use_conda="$1"
    if [ "${use_conda}" != false ] && [ "${use_conda}" != true ]; then
        echo "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        echo "Usage: $0 [use-conda|true or false] [<k2-version>]"
        exit 1
    fi

    if [ $# -eq 2 ]; then
        k2_version="$2"
        pip_k2_version="${k2_version}"
        conda_k2_version="${k2_version}"
    fi
else
    use_conda=$([[ $(conda list -e -c -f --no-pip pytorch 2>/dev/null) =~ pytorch ]] && echo true || echo false)
fi

if [[ ! $(uname -s) =~ Linux ]]; then
    echo "Warning: This script doesn't support MacOS and Windows. Please install k2 manually."
    exit 0
fi


if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

python_36_plus=$(python3 <<EOF
from packaging.version import parse as V
import sys

if V("{}.{}.{}".format(*sys.version_info[:3])) >= V("3.6"):
    print("true")
else:
    print("false")
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
torch_version=$(python3 <<EOF
import torch
# e.g. 1.10.0+cpu -> 1.10.0
torch_version=torch.__version__.split("+")[0]
print(torch_version)
EOF
)
libc_path="$(ldd /bin/bash | grep libc.so | awk '{ print $3 }')"
libc_version="$(${libc_path} | grep "GNU C Library" | grep -oP "version [0-9]*.[0-9]*" | cut -d" " -f2)"

pytorch_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
libc_plus(){
    python3 <<EOF
import sys
from packaging.version import parse as L
if L('$libc_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

echo "[INFO] torch_version=${torch_version}"
echo "[INFO] cuda_version=${cuda_version}"
echo "[INFO] libc_version=${libc_version}"

if ! "${python_36_plus}"; then
    echo "[ERROR] k2 requires python>=3.6"
    exit 1
fi

# Check pytorch version.
# Please exit without error code for CI.
if "${use_conda}"; then
    if [ "${conda_k2_version}" = "1.10.dev20211103" ]; then
        if ! $(libc_plus 2.27); then
            echo "[WARNING] k2=${conda_k2_version} requires GLIBC_2.27, but your GLIBC is ${libc_version}. Skip k2-installation"
            exit
        fi
        if "$(pytorch_plus 1.10.1)"; then
            echo "[WARNING] k2=${conda_k2_version} doesn't provide conda package for pytorch=${torch_version}. Skip k2-installation"
            exit
        elif ! "$(pytorch_plus 1.5.0)"; then
            echo "[WARNING] k2=${conda_k2_version} doesn't provide conda package for pytorch=${torch_version}. Skip k2-installation"
            exit
        fi
        if "$(pytorch_plus 1.10.0)"; then
            if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.1" ] && [ "${cuda_version}" != "11.3" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} provides conda package for CUDA10.2, 11.1, and 11.3 only. Skip k2-installation"
                exit
            fi
        elif "$(pytorch_plus 1.9.0)"; then
            if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.1" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} provides conda package for CUDA10.2, and 11.1 only. Skip k2-installation"
                exit
            fi
        elif "$(pytorch_plus 1.8.0)"; then
            if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.1" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.1" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} provides conda package for CUDA10.1, 10.2 and 11.1 only. Skip k2-installation"
                exit
            fi
        elif "$(pytorch_plus 1.7.0)"; then
            if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.1" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.0" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} provides conda package for CUDA10.1, 10.2 and 11.0 only. Skip k2-installation"
                exit
            fi
        elif "$(pytorch_plus 1.6.0)"; then
            if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.1" ] && [ "${cuda_version}" != "10.2" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} provides conda package for CUDA10.1, 10.2 and 11.0 only. Skip k2-installation"
                exit
            fi
        else
            if [ -n "${cuda_version}" ]; then
                echo "[WARNING] k2=${conda_k2_version} for pytorch=${torch_version} doesn't provides conda package for CUDA. Skip k2-installation"
                exit
            fi
        fi
    elif [ "${conda_k2_version}" = "1.6.dev20210824" ]; then
        if "$(pytorch_plus 1.9.1)"; then
            echo "[WARNING] k2=${conda_k2_version} doesn't provide conda package for pytorch=${torch_version}. Skip k2-installation"
            exit
        elif ! "$(pytorch_plus 1.8.1)"; then
            echo "[WARNING] k2=${conda_k2_version} doesn't provide conda package for pytorch=${torch_version}. Skip k2-installation"
            exit
        fi
        if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.0" ] && [ "${cuda_version}" != "11.1" ]; then
            echo "[WARNING] k2=${conda_k2_version} provides conda package for CUDA10.2, 11.0, and 11.1 only. Skip k2-installation"
            exit
        fi
    fi
else
    if [ "${pip_k2_version}" = "1.10.dev20211112" ]; then
        if ! $(libc_plus 2.27); then
            echo "[WARNING] k2=${conda_k2_version} requires GLIBC_2.27, but your GLIBC is ${libc_version}. Skip k2-installation"
            exit
        fi
        if "$(pytorch_plus 1.10.1)"; then
            echo "[WARNING] k2=${pip_k2_version} for pip doesn't provide pytorch=${torch_version} binary. Skip k2-installation"
            exit
        elif ! "$(pytorch_plus 1.4.0)"; then
            echo "[WARNING] k2=${pip_k2_version} for pip doesn't provide pytorch=${torch_version} binary. Skip k2-installation"
            exit
        fi
        if [ -n "${cuda_version}" ] && [ "${torch_version}" != "1.7.1" ]; then
            echo "[WARNING] k2=${pip_k2_version}+cuda for pip provides pytorch=1.7.1 binary only. Skip k2-installation"
            exit
        fi
        if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.1" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.0" ]; then
            echo "[WARNING] k2=${pip_k2_version} for pip provides CUDA10.1, 10.2, and 11.0 binary only. Skip k2-installation"
            exit
        fi
    elif [ "${pip_k2_version}" = "1.6.dev20210907" ]; then
        if "$(pytorch_plus 1.9.1)"; then
            echo "[WARNING] k2=${pip_k2_version} for pip doesn't provide pytorch=${torch_version} binary. Skip k2-installation"
            exit
        elif ! "$(pytorch_plus 1.3.1)"; then
            echo "[WARNING] k2=${pip_k2_version} for pip  doesn't provide pytorch=${torch_version} binary. Skip k2-installation"
            exit
        fi
        if [ -n "${cuda_version}" ] && [ "${torch_version}" != "1.7.1" ]; then
            echo "[WARNING] k2=${pip_k2_version}+cuda for pip provides pytorch=1.7.1 binary only. Skip k2-installation"
            exit
        fi
        if [ -n "${cuda_version}" ] && [ "${cuda_version}" != "10.1" ] && [ "${cuda_version}" != "10.2" ] && [ "${cuda_version}" != "11.0" ]; then
            echo "[WARNING] k2=${pip_k2_version} for pip provides CUDA10.1, 10.2, and 11.0 binary only. Skip k2-installation"
            exit
        fi
    fi
fi



if "${use_conda}"; then
    [ -z "${conda_k2_version}" ] && k2="k2" || k2="k2=${conda_k2_version}"

    if [ -z "${cuda_version}" ]; then
        echo conda install -y -c k2-fsa -c pytorch cpuonly "${k2}" "pytorch=${torch_version}"
        conda install -y -c k2-fsa -c pytorch cpuonly "${k2}" "pytorch=${torch_version}"
    else
        # NOTE(kamo): K2 requires cudatoolkit from conda-forge channel and k2-cpu is installed if the other channel is used, e.g. anaconda, nvidia
        echo conda install -y -c k2-fsa -c pytorch -c conda-forge "${k2}" "cudatoolkit=${cuda_version}" "pytorch=${torch_version}"
        conda install -y -c k2-fsa -c pytorch -c conda-forge "${k2}" "cudatoolkit=${cuda_version}" "pytorch=${torch_version}"
    fi

else
    if [ -z "${cuda_version}" ]; then
        echo pip install "k2==${pip_k2_version}+cpu.torch${torch_version}" -f https://k2-fsa.org/nightly/
        pip install "k2==${pip_k2_version}+cpu.torch${torch_version}" -f https://k2-fsa.org/nightly/
    else
        echo pip install "k2==${pip_k2_version}+cuda${cuda_version}.torch${torch_version}" -f https://k2-fsa.org/nightly/
        pip install "k2==${pip_k2_version}+cuda${cuda_version}.torch${torch_version}" -f https://k2-fsa.org/nightly/
    fi
fi
