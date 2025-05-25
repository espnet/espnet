#!/usr/bin/env bash

set -euo pipefail

if [ ! -f ./tools/activate_python.sh ]; then
    touch ./tools/activate_python.sh
fi

exclude="egs2/TEMPLATE/asr1/utils,"
exclude+="egs2/TEMPLATE/asr1/steps,"
exclude+="egs2/TEMPLATE/tts1/sid,doc,tools,"
exclude+="test_utils/bats-core,test_utils/bats-support,"
exclude+="test_utils/bats-assert"

# flake8
"$(dirname $0)"/test_flake8.sh
# pycodestyle
pycodestyle --exclude "${exclude}" --show-source --show-pep8

pytest -q test/espnet3/test_espnet3.py

echo "=== report ==="
coverage report
coverage xml
