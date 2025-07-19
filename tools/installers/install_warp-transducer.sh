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


if ! python3 -c "import packaging.version" &> /dev/null; then
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

 	if [[ "$unames" == "Linux" ]]; then
    	: "${CUDA_HOME:=/usr/local/cuda}"
		arch="$(uname -m)"

		if [[ "$arch" == "x86_64" ]]; then
			cuda_include="$CUDA_HOME/targets/x86_64-linux/include"
			cuda_lib="$CUDA_HOME/targets/x86_64-linux/lib"
		elif [[ "$arch" == "aarch64" ]]; then
			cuda_include="$CUDA_HOME/include"
			cuda_lib="$CUDA_HOME/lib64"
		else
			echo "Unknown Linux architecture ($arch). Skipping CUDA setup."
			cuda_include=""
			cuda_lib=""
		fi

		cuda_bin="$CUDA_HOME/bin"
	    cuda_lib64="$CUDA_HOME/lib64"

	    if [[ -n "$cuda_include" && -d "$cuda_include" && -d "$cuda_lib" ]]; then
	        export CFLAGS="${CFLAGS:-} -I$cuda_include"
			export CXXFLAGS="${CXXFLAGS:-} -I$cuda_include"
			export LDFLAGS="${LDFLAGS:-} -L$cuda_lib"
			export PATH="$cuda_bin:$PATH"
			export LD_LIBRARY_PATH="$cuda_lib64:${LD_LIBRARY_PATH:-}"
	    else
	        echo "Warning: CUDA paths not found or unsupported architecture: $arch"
	    fi
	else
        echo "Note: Skipping CUDA setup on non-Linux platform: $unames"
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

if ! python3 -c "import ninja" &> /dev/null; then
    (
		set -euo pipefail
		echo "Installing ninja package for RWKV decoder (training only)."

		python3 -m pip install ninja
    )
fi
