#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# chime-utils is not on PyPI, so it cannot be an espnet extra: pip refuses a
# URL dependency in a published package's metadata. Recipes that need its
# CHiME-8 text normalizers install it here instead.
if python3 -c "import chime_utils" &> /dev/null; then
    echo "chime-utils is already installed"
    exit 0
fi

python3 -m pip install git+https://github.com/chimechallenge/chime-utils
