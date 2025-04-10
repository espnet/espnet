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
git clone --single-branch --branch update_torch2.1 https://github.com/b-flo/warp-transducer.git

(
    set -euo pipefail
    cd warp-transducer

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
