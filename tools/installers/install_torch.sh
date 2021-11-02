#!/usr/bin/env bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $# -ne 3 ]; then
    log "Usage: $0 <[use_conda| true or false> <torch_version> <cuda_version>"
    exit 1;
elif [ $# -eq 3 ]; then
    use_conda="$1"
    torch_version="$2"
    cuda_version="$3"
fi
if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
    cuda_version=
fi


python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
cuda_version_without_dot=$(echo "${cuda_version}" | sed "s/\.//g")


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
install_torch(){
    if "${use_conda}"; then
        if $(pt_plus 1.9.0) && ! $(pt_plus 1.10.0); then
            if [ -z "${cuda_version}" ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            elif [ "${cuda_version}" = 11.1 ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c nvidia
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c nvidia
            else
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
            fi
        elif $(pt_plus 1.8.0) && ! $(pt_plus 1.10.0); then
            if [ -z "${cuda_version}" ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            elif [ "${cuda_version}" = 11.1 ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c conda-forge
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c conda-forge
            else
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
            fi
        else
            if [ -z "${cuda_version}" ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            else
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
            fi
        fi
    else
        if $(pt_plus 1.10.0); then
            if [ -z "${cuda_version}" ]; then
                log python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
                python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html
            else
                log python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/cu${cuda_version_without_dot}/torch_stable.html"
                python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1+cu${cuda_version_without_dot}" -f "https://download.pytorch.org/whl/cu${cuda_version_without_dot}/torch_stable.html"
            fi
        else
            if [ -z "${cuda_version}" ]; then
                log python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1" -f https://download.pytorch.org/whl/cpu/torch_stable.html
                python3 -m pip install "torch==${torch_version}+cpu" "torchaudio==$1" -f https://download.pytorch.org/whl/cpu/torch_stable.html
            else
                log python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1" -f "https://download.pytorch.org/whl/cu${cuda_version_without_dot}/torch_stable.html"
                python3 -m pip install "torch==${torch_version}+cu${cuda_version_without_dot}" "torchaudio==$1" -f "https://download.pytorch.org/whl/cu${cuda_version_without_dot}/torch_stable.html"
            fi
        fi
    fi
}
check_cuda_version(){
    supported=false
    for v in "" $@; do
        [ "${cuda_version}" = "${v}" ] && supported=true
    done

    if ! "${supported}"; then
        # See https://anaconda.org/pytorch/pytorch/files to provided version
        log "[ERROR] pytorch${torch_version} doesn't provide CUDA=${cuda_version} version."
        exit 1
    fi
}


log "[INFO] python_version=${python_version}"
log "[INFO] torch_version=${torch_version}"
log "[INFO] cuda_version=${cuda_version}"


if $(pt_plus 1.10.1); then
    log "[ERROR] This script doesn't support pytorch=${torch_version}"

elif $(pt_plus 1.10.0); then
    check_cuda_version 11.3 11.1 10.2
    install_torch 0.10.0

elif $(pt_plus 1.9.1); then
    check_cuda_version 11.1 10.2
    install_torch 0.9.1

elif $(pt_plus 1.9.0); then
    check_cuda_version 11.1 10.2
    install_torch 0.9.0

elif $(pt_plus 1.8.1); then
    check_cuda_version 11.1 10.2 10.1
    install_torch 0.8.1

elif $(pt_plus 1.8.0); then
    check_cuda_version 11.1 10.2 10.1
    install_torch 0.8.0

elif $(pt_plus 1.7.1); then
    check_cuda_version 11.0 10.2 10.1 9.2
    install_torch 0.7.2

elif $(pt_plus 1.7.0); then
    check_cuda_version 11.0 10.2 10.1 9.2
    install_torch 0.7.0

elif $(pt_plus 1.6.0); then
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.6.0

elif $(pt_plus 1.5.1); then
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.5.1

elif $(pt_plus 1.5.0); then
    check_cuda_version 10.2 10.1 9.2
    install_torch 0.5.0

elif $(pt_plus 1.4.0); then
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.4.0

elif $(pt_plus 1.3.1); then
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.3.2

elif $(pt_plus 1.3.0); then
    check_cuda_version 10.1 10.0 9.2
    install_torch 0.3.1

elif $(pt_plus 1.2.0); then
    check_cuda_version 10.0 9.2
    install_torch 0.3.0
else
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
fi

