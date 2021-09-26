#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install transformers
if [ ! -e transformers.done ]; then
    (
        set -euo pipefail
        python3 -m pip install transformers>=4.9.1
    )
    touch transformers.done
else
    echo "transformers is already installed."
fi
