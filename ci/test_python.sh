#!/usr/bin/env bash

if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi

set -euo pipefail

"$(dirname $0)"/test_flake8.sh

pycodestyle -r espnet test utils --show-source --show-pep8 
if ! black --check espnet2 test/espnet2 setup.py; then
    echo "Please apply: 'black espnet2/ test/espnet2 setup.py'"
    exit 1
fi

# espnet2 follows "black" style.
pycodestyle -r espnet2 test/espnet2 setup.py --max-line-length 88 --ignore E203,W503 --show-source --show-pep8 

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest -q
