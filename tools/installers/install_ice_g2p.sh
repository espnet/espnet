#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install ice-g2p
if [ ! -e ice-g2p.done ]; then
    . activate_python.sh
    pip install ice-g2p
    (
        set -euo pipefail
        fetch-models
    )
    touch ice-g2p.done
else
    echo "ice-g2p is already installed"
fi
