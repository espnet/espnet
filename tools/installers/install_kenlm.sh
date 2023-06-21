#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

rm -rf kenlm
git clone https://github.com/kpu/kenlm.git
(
    set -euo pipefail
    cd kenlm

    mkdir build
    (
        set -euo pipefail
        cd build && cmake .. && make
    )
    (
        set -euo pipefail
        python3 -m pip install -e .
    )
)
