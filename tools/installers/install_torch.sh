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
    if "${use_conda}"; then
        if $(pytorch_plus 1.13.0); then
            if [ -z "${cuda_version}" ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            else
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "pytorch-cuda=${cuda_version}" -c pytorch -c nvidia
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "pytorch-cuda=${cuda_version}" -c pytorch -c nvidia
            fi
        else
            if [ -z "${cuda_version}" ]; then
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" cpuonly -c pytorch
            elif [ "${cuda_version}" = "11.5" ] || [ "${cuda_version}" = "11.6" ]; then
                # NOTE(kamo): In my environment, cudatoolkit of conda-forge only could be installed, but I don't know why @ 12, May, 2022
                cudatoolkit_channel=conda-forge
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"

            elif [ "${cuda_version}" = "11.1" ] || [ "${cuda_version}" = "11.2" ]; then
                # Anaconda channel, which is default main channel, doesn't provide cudatoolkit=11.1, 11.2 now (Any pytorch version doesn't provide cuda=11.2).
                # https://anaconda.org/anaconda/cudatoolkit/files

                # https://anaconda.org/nvidia/cudatoolkit/files
                cudatoolkit_channel=nvidia

                # https://anaconda.org/conda-forge/cudatoolkit/files
                # cudatoolkit_channel=conda-forge

                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch -c "${cudatoolkit_channel}"
            else
                log conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
                conda install -y "pytorch=${torch_version}" "torchaudio=$1" "cudatoolkit=${cuda_version}" -c pytorch
            fi
        fi
    else
        if [ -z "${cuda_version}" ]; then
            log python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cpu
            python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cpu
        else
            log python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
            python3 -m pip install "torch==${torch_version}" "torchaudio==$1" --extra-index-url https://download.pytorch.org/whl/cu"${cuda_version_without_dot}"
        fi
    fi
}
check_python_version(){
    if $(python_plus $1) || ! $(python_plus 3.6); then
        log "[ERROR] pytorch=${torch_version} doesn't provide binary build for python>=$1,<3.6, but your python is ${python_version}"
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
        log "[WARNING] Pre-build package for Pytorch=${torch_version} with CUDA=${cuda_version} is not provided."
        return 1
    fi
}


log "[INFO] python_version=${python_version}"
log "[INFO] torch_version=${torch_version}"
log "[INFO] cuda_version=${cuda_version}"

if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi

if $(pytorch_plus 2.0.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 2.0.1); then
    check_python_version 3.12  # Error if python>=<number>
    if ! check_cuda_version 11.8 11.7; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.7"
        cuda_version=11.8
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 2.0.2  # install_torch <torch-audio-ver>

elif $(pytorch_plus 2.0.0); then
    check_python_version 3.12  # Error if python>=<number>
    if ! check_cuda_version 11.8 11.7; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.7"
        cuda_version=11.8
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 2.0.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.13.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.13.1); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.7 11.6; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.7"
        cuda_version=11.7
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.13.1  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.13.0); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.7 11.6; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.7"
        cuda_version=11.7
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.13.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.12.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.12.1); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.6 11.3 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.6"
        cuda_version=11.6
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.12.1  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.12.0); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.6 11.3 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.6"
        cuda_version=11.6
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.12.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.11.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.11.0); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.5 11.3 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.5"
        cuda_version=11.5
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.11.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.10.3); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.10.2); then
    check_python_version 3.10  # Error if python>=<number>
    if ! check_cuda_version 11.3 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.3"
        cuda_version=11.3
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.10.2  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.10.1); then
    check_python_version 3.10  # Error if python>=<number>
    if ! check_cuda_version 11.3 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.3"
        cuda_version=11.3
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.10.1  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.10.0); then
    check_python_version 3.11  # Error if python>=<number>
    if ! check_cuda_version 11.3 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.3"
        cuda_version=11.3
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.10.0  # install_torch <torch-audio-ver>

elif $(pytorch_plus 1.9.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.9.1); then
    check_python_version 3.10
    if ! check_cuda_version 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.1"
        cuda_version=11.7
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.9.1

elif $(pytorch_plus 1.9.0); then
    check_python_version 3.10
    if ! check_cuda_version 11.1 10.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.1"
        cuda_version=11.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.9.0

elif $(pytorch_plus 1.8.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.8.1); then
    check_python_version 3.10
    if ! check_cuda_version 11.1 10.2 10.1; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.1"
        cuda_version=11.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.8.1

elif $(pytorch_plus 1.8.0); then
    check_python_version 3.10
    if ! check_cuda_version 11.1 10.2 10.1; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.1"
        cuda_version=11.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.8.0

elif $(pytorch_plus 1.7.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.7.1); then
    check_python_version 3.10
    if ! check_cuda_version 11.0 10.2 10.1 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.0"
        cuda_version=11.0
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.7.2

elif $(pytorch_plus 1.7.0); then
    check_python_version 3.10
    if ! check_cuda_version 11.0 10.2 10.1 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=11.0"
        cuda_version=11.0
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.7.0

elif $(pytorch_plus 1.6.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.6.0); then
    check_python_version 3.9
    if ! check_cuda_version 10.2 10.1 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.2"
        cuda_version=10.2
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.6.0

elif $(pytorch_plus 1.5.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.5.1); then
    check_python_version 3.9
    if ! check_cuda_version 10.2 10.1 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.2"
        cuda_version=10.2
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.5.1

elif $(pytorch_plus 1.5.0); then
    check_python_version 3.9
    if ! check_cuda_version 10.2 10.1 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.2"
        cuda_version=10.2
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.5.0

elif $(pytorch_plus 1.4.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.4.0); then
    check_python_version 3.9
    if ! check_cuda_version 10.1 10.0 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.1"
        cuda_version=10.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.4.0

elif $(pytorch_plus 1.3.2); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.3.1); then
    check_python_version 3.8
    if ! check_cuda_version 10.1 10.0 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.1"
        cuda_version=10.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.3.2

elif $(pytorch_plus 1.3.0); then
    check_python_version 3.8
    if ! check_cuda_version 10.1 10.0 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.1"
        cuda_version=10.1
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.3.1

elif $(pytorch_plus 1.2.1); then
    log "[ERROR] pytorch=${torch_version} doesn't exist"
    exit 1

elif $(pytorch_plus 1.2.0); then
    check_python_version 3.8
    if ! check_cuda_version 10.0 9.2; then
        log "[INFO] Fallback: cuda_version=${cuda_version} -> cuda_version=10.0"
        cuda_version=10.0
        cuda_version_without_dot="${cuda_version/\./}"
    fi
    install_torch 0.3.0
else
    log "[ERROR] This script doesn't support pytorch=${torch_version}"
    exit 1
fi
