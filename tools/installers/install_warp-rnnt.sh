#!/usr/bin/env bash

set -euo pipefail

cuda_version=$(python3 <<EOF
import torch
if torch.cuda.is_available():
   version=torch.version.cuda.split(".")
   print(version[0] + version[1])
else:
   print("")
EOF
)

if ! [[ "$cuda_version" =~ ^(100|101|102)$ ]]; then
    echo "warp-rnnt was not tested with CUDA_VERSION=$cuda_version. Skipping install."
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
    cd warp-rnnt/pytorch_binding && python3 -m pip install -e .
)
