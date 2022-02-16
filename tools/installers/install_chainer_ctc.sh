#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf chainer_ctc
git clone https://github.com/jheymann85/chainer_ctc.git
python3 -m pip install cython
(
    set -euo pipefail
    cd chainer_ctc && chmod +x install_warp-ctc.sh && ./install_warp-ctc.sh && python3 -m pip install -e .
)
