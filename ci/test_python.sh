#!/usr/bin/env bash

set -euo pipefail

if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi

flake8 espnet test utils;
autopep8 -r espnet test utils --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
test ! -s check_autopep8

pip install codecov
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/chainer_ctc/ext/warp-ctc/build" pytest--cov=codecov
codecov
