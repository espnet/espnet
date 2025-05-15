#!/usr/bin/env bash

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
# "$(dirname $0)"/test_flake8.sh
# pycodestyle
# pycodestyle --exclude "${exclude}" --show-source --show-pep8

pytest -q test/espnet3/test_espnet3.py

echo "=== report ==="
coverage report
coverage xml
