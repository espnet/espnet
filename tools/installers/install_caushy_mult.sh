#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install custom cuda kernel
if [ ! -e cauchy_mult.done ]; then

    git clone https://github.com/HazyResearch/state-spaces.git
    cd state-spaces/extensions/cauchy
    python setup.py install
    cd ../../../

    touch cauchy_mult.done
else
    echo "cauchy_mult is already installed"
fi
