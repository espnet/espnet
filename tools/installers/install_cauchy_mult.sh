#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install pykeops
if ! python3 -c "import pykeops.version" &> /dev/null; then
    (
        set -euo pipefail
        python3 -m pip install pykeops
    )
else
    echo "pykeops is already installed."
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
