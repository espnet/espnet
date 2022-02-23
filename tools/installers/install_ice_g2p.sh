#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install ice-g2p
if [ ! -e ice-g2p.done ]; then
    rm -rf ice-g2p
    . activate_python.sh
    git clone https://github.com/G-Thor/ice-g2p.git
    (
        set -euo pipefail
        pip install ./ice-g2p/

    )
    touch ice-g2p.done
else
    echo "ice-g2p is already installed"
fi