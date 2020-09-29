#!/bin/bash

set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 <cuda-version>"
    echo "e.g.: $0 10.0"
    exit 1;
fi

cuda_ver=$1

if [ $cuda_ver != "10.0" ]; then
    echo "warp-rnnt was only tested with CUDA_VERSION=10.0. Skipping install."
    exit 0
fi

# TODO(kamo): Consider clang case
# Note: Requires gcc>=5.4 to build extensions with pytorch>=1.0
if python3 -c 'import torch as t;assert t.__version__[0] >= "1.0"' &> /dev/null; then \
    python3 -c "from distutils.version import LooseVersion as V;assert V('$(gcc -dumpversion)') >= V('5.4'), 'Requires gcc>=5.4'"; \
    
fi

rm -rf warp-rnnt
git clone https://github.com/1ytic/warp-rnnt

(
    set -euo pipefail
    cd warp-rnnt/pytorch_binding && python3 setup.py install
)
