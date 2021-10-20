#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install k2
if [ ! -e k2.done ]; then
    (
        set -euo pipefail
        # Will install pytorch==1.7.1 automatically.
        # Refer to
        # https://k2.readthedocs.io/en/latest/installation/index.html
        # for more alternatives to install k2
        python3 -m pip install k2
    )
    touch k2.done
else
    echo "k2 is already installed"
fi

