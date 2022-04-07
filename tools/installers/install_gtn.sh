#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install gtn
if [ ! -e gtn.done ]; then
    (
        set -euo pipefail
        python3 -m pip install gtn==0.0.0
    )
    touch gtn.done
else
    echo "gtn is already installed"
fi
