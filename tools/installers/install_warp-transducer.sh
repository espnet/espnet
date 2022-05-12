#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# TODO(kamo): Consider clang case
# Note: Requires gcc>=4.9.2 to build extensions with pytorch>=1.0
if python3 -c 'import torch as t;assert t.__version__[0] == "1"' &> /dev/null; then \
    python3 -c "from packaging.version import parse as V;assert V('$(gcc -dumpversion)') >= V('4.9.2'), 'Requires gcc>=4.9.2'"; \
fi

rm -rf warp-transducer
git clone --single-branch --branch espnet_v1.1 https://github.com/b-flo/warp-transducer.git

(
    set -euo pipefail
    cd warp-transducer

    mkdir build
    (
        set -euo pipefail
        cd build && cmake .. && make
    )

    (
        set -euo pipefail
        cd pytorch_binding && python3 -m pip install -e .
    )
)
