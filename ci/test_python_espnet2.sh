#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
# TODO(nelson): Add documentation on espnet2 folder and uncomment this.
# echo "=== Run test flake8 ==="
# "$(dirname $0)"/test_flake8.sh espnet2

# pycodestyle
echo "=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8

echo "=== Run pytest tests ==="
pytest -q test/espnet2

echo "=== report ==="
coverage report
coverage xml
