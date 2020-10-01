#!/bin/bash

set -euo pipefail

if [ $# != 1 ]; then
    echo "Usage: $0 <cuda-version>"
    echo "e.g.: $0 10.0"
    exit 1;
fi

cuda_ver=$1

if ! [[ "$cuda_ver" =~ ^(10.0|10.1|10.2)$ ]]; then
    echo "warp-rnnt was not tested with CUDA_VERSION=$cuda_ver. Skipping install."
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
