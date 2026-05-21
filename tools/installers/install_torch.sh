#!/usr/bin/env bash
set -euo pipefail
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

unames="$(uname -s)"
if [[ ${unames} =~ Linux ]]; then
    os_type=linux
elif [[ ${unames} =~ Darwin ]]; then
    os_type=macos
elif [[ ${unames} =~ MINGW || ${unames} =~ CYGWIN || ${unames} =~ MSYS ]]; then
    os_type=windows
else
    os_type=unknown
fi


if [ $# -ne 3 ]; then
    log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
    exit 1
elif [ $# -eq 3 ]; then
    use_conda="$1"
    if [ "${use_conda}" != false ] && [ "${use_conda}" != true ]; then
        log "[ERROR] <use_conda> must be true or false, but ${use_conda} is given."
        log "Usage: $0 <use_conda| true or false> <torch_version> <cuda_version>"
        exit 1
    fi
    torch_version="$2"
    cuda_version="$3"
fi

if [ "${cuda_version}" = cpu ] || [ "${cuda_version}" = CPU ]; then
    cuda_version=
fi

if [ -n "${cuda_version}" ] && [ "${os_type}" = macos ]; then
    log "Error: cuda is not supported for MacOS"
    exit 1
fi

if [ "${os_type}" == macos ]; then
    pip_cpu_module_suffix=
else
    pip_cpu_module_suffix="+cpu"
fi


python_version=$(python3 -c "import sys; print(sys.version.split()[0])")
cuda_version_without_dot="${cuda_version/\./}"


python_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$python_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
pytorch_plus(){
    python3 <<EOF
from packaging.version import parse as L
if L('$torch_version') >= L('$1'):
    print("true")
else:
    print("false")
EOF
}
install_torch(){
# Usage: install_torch <torchaudio-version>
    if [ -z "${cuda_version}" ]; then
        log python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cpu
    else
        log python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
        python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
    fi
}
check_python_version(){
    if $(python_plus $1) || ! $(python_plus 3.7); then
        log "[ERROR] pytorch=${torch_version} requires python>=$1,<=3.7, but your python is ${python_version}"
        exit 1
    fi
}
check_cuda_version(){
    supported=false
    for v in "" $@; do
        [ "${cuda_version}" = "${v}" ] && supported=true
    done
    if ! "${supported}"; then
        # See https://anaconda.org/pytorch/pytorch/files to confirm the provided versions
        log "[WARNING] Pre-built package for Pytorch=${torch_version} with CUDA=${cuda_version} is not provided."
        return 1
    fi
}


log "[INFO] python_version=${python_version}"
log "[INFO] torch_version=${torch_version}"
log "[INFO] cuda_version=${cuda_version}"

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

if $(pytorch_plus 2.11.1); then
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
    exit 1

elif $(pytorch_plus 2.11.0); then
    check_python_version 3.13  # Error if python>=<number>
    if ! check_cuda_version 12.8 13.0; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=12.8"
        cuda_version=12.8
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 2.11.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 2.10.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 2.10.0); then
    check_python_version 3.13  # Error if python>=<number>
    if ! check_cuda_version 12.8 13.0; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=12.8"
        cuda_version=12.8
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 2.10.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 2.9.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 2.9.1); then
    check_python_version 3.13  # Error if python>=<number>
    if ! check_cuda_version 12.6 12.8 13.0; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=12.8"
        cuda_version=12.8
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 2.9.1  # install_torch <torch-audio-ver>

else
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
    exit 1
fi
