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


rm -rf chainer_ctc
git clone https://github.com/jheymann85/chainer_ctc.git
python3 -m pip install cython
(
    set -euo pipefail
    cd chainer_ctc
    (
        set -euo pipefail
        mkdir -p ext/warp-ctc
        cd ext/warp-ctc
        git clone https://github.com/jnishi/warp-ctc .
        mkdir build
        cd build
        cmake -DWITH_OMP="${with_openmp}" ../
        make
    )
    python3 -m pip install -e .
)
