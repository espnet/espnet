#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

check_chainer(){
    python3 -c "
import sys
try:
    import chainer
    sys.exit(0)
except ImportError:
    sys.exit(1)
"
}

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
"$(dirname $0)"/test_flake8.sh espnet
# pycodestyle
pycodestyle --exclude "${exclude}" --show-source --show-pep8

if ! check_chainer; then
    echo "WARNING: Chainer is not installed, skipping espnet1 python tests."
    echo "         Chainer is being deprecated and will be removed in a future release."
    echo "         To run these tests, install Chainer via 'make chainer.done' in the tools directory."
    exit 0
fi

pytest -q --ignore test/espnet2 --ignore test/espnetez test

echo "=== report ==="
coverage report
coverage xml
