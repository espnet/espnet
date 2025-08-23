#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

numpy_plus(){
    python3 <<EOF
from packaging.version import parse as L
import numpy as np
if L(np.__version__) >= L('$1'):
    print("true")
else:
    print("false")
EOF
}

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
"$(dirname $0)"/test_flake8.sh espnet
# pycodestyle
pycodestyle --exclude "${exclude}" --show-source --show-pep8

if $(numpy_plus 2.0.0); then
    echo "WARNING: The current numpy version is not supported by 'Chainer',"
    echo "         a dependency required for ESPnet<202509."
    echo "         Try with a different lower version of ESPnet for running these tests"
    exit 0
fi
pytest -q --ignore test/espnet2 --ignore test/espnetez test

echo "=== report ==="
coverage report
coverage xml
