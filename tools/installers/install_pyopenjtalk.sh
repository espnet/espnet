#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install pyopenjtalk
if [ ! -e pyopenjtalk.done ]; then
    (
        set -euo pipefail
        python3 -m pip install pyopenjtalk==0.1.0
        python3 -c "import pyopenjtalk; pyopenjtalk.g2p('download dict')"
    )
    touch pyopenjtalk.done
else
    echo "pyopenjtalk is already installed"
fi
