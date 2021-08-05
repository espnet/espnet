#!/usr/bin/env bash
set -euo pipefail


if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# Install Hugging Face Transformers
if [ ! -e transformers.done ]; then
    (
        set -euo pipefail
        python3 -m pip install transformers
    )
    touch transformers.done
else
    echo "Hugging Face Transformers is already installed"
fi
