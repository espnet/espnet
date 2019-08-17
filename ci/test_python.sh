#!/usr/bin/env bash

if ${USE_CONDA:-}; then
    . tools/venv/bin/activate
fi

set -euo pipefail

flake8 espnet test utils doc;
autopep8 -r espnet test utils --global-config .pep8 --diff --max-line-length 120 | tee check_autopep8
test ! -s check_autopep8

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest
