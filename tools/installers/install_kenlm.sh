#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi
# dependencies 
apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev

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
