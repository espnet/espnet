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
        # Since this installer overwrite existing pyopenjtalk, remove done file.
        [ -e tdmelodic_pyopenjtalk.done ] && rm tdmelodic_pyopenjtalk.done
        python3 -m pip install pyopenjtalk==0.3.0
        python3 -c "import pyopenjtalk; pyopenjtalk.g2p('download dict')"
    )
    touch pyopenjtalk.done
else
    echo "pyopenjtalk is already installed."
fi
