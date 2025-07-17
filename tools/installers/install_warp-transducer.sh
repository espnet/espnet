#!/usr/bin/env bash
set -euo pipefail

if [ $# -eq 0 ]; then
    with_openmp=ON
elif [ $# -eq 1 ]; then
    with_openmp=$1
elif [ $# -gt 1 ]; then
    echo "Usage: $0 [with_openmp| ON or OFF]"
    exit 1;
fi
unames="$(uname -s)"
if [[ ! ${unames} =~ Linux && ! ${unames} =~ Darwin ]]; then
    echo "Warning: This script may not work with ${unames}. Exit with doing nothing"
    exit 0
fi


if ! python -c "import packaging.version" &> /dev/null; then
    python3 -m pip install packaging
fi
# TODO(kamo): Consider clang case
# Note: Requires gcc>=4.9.2 to build extensions with pytorch>=1.0
if python3 -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
    python3 -c "from packaging.version import parse as V;assert V('$(gcc -dumpversion)') >= V('4.9.2'), 'Requires gcc>=4.9.2'"; \
fi

rm -rf warp-transducer
git clone https://github.com/ljn7/warp-transducer.git

(
    set -euo pipefail
    cd warp-transducer

    : "${CUDA_HOME:=/usr/local/cuda}"

    cuda_include="$CUDA_HOME/targets/x86_64-linux/include"
    cuda_lib="$CUDA_HOME/targets/x86_64-linux/lib"
    cuda_bin="$CUDA_HOME/bin"
    cuda_lib64="$CUDA_HOME/lib64"

    if [[ -d "$cuda_include" && -d "$cuda_lib" && -d "$cuda_bin" && -d "$cuda_lib64" ]]; then
        if [[ "${CFLAGS:-}" != *"$cuda_include"* ]]; then
            export CFLAGS="${CFLAGS:-} -I$cuda_include"
        fi

        if [[ "${CXXFLAGS:-}" != *"$cuda_include"* ]]; then
            export CXXFLAGS="${CXXFLAGS:-} -I$cuda_include"
        fi

        if [[ "${LDFLAGS:-}" != *"$cuda_lib"* ]]; then
            export LDFLAGS="${LDFLAGS:-} -L$cuda_lib"
        fi

        if [[ ":$PATH:" != *":$cuda_bin:"* ]]; then
            export PATH="$cuda_bin:$PATH"
        fi

        if [[ "${LD_LIBRARY_PATH:-}" != *"$cuda_lib64"* ]]; then
            export LD_LIBRARY_PATH="$cuda_lib64:${LD_LIBRARY_PATH:-}"
        fi
    else
        echo "Warning: One or more CUDA paths do not exist!"
        echo "Checked:"
        echo "  $cuda_include"
        echo "  $cuda_lib"
        echo "  $cuda_bin"
        echo "  $cuda_lib64"
        echo "Please verify \$CUDA_HOME is correctly set to a valid CUDA installation."
        exit 1
    fi

    mkdir build
    (
        set -euo pipefail
        cd build && cmake -DWITH_OMP="${with_openmp}" .. && make
        # cd build && cmake -DWITH_OMP="${with_openmp}" -DCMAKE_CXX_FLAGS="-std=c++1z" .. && make

    )

    (
        set -euo pipefail
        cd pytorch_binding && python3 -m pip install -e .
    )
)

if ! python -c "import ninja" &> /dev/null; then
    (
	set -euo pipefail
	echo "Installing ninja package for RWKV decoder (training only)."

	python3 -m pip install ninja
    )
fi
