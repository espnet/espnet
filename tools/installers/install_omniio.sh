#!/usr/bin/env bash

set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

# omniio provides ESPnet's Kaldi ark/scp I/O (omniio.kaldi). It is not on PyPI,
# so it cannot be declared as an extra in pyproject.toml and is installed here.
# It replaced kaldiio, whose license restricts redistribution; see
# https://github.com/espnet/espnet/issues/6529
python3 -m pip install "omniio @ git+https://github.com/wavlab-speech/omniio.git"
