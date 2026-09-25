#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
echo "::group::=== Run test flake8 ==="
"$(dirname $0)"/test_flake8.sh espnet3
echo "::endgroup::"

# pycodestyle
echo "::group::=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8
echo "::endgroup::"

# It will set default timeout to 10.0 seconds for each test.
# If the test is marked with @pytest.mark.execution_timeout,
# the value in the mark will be used as the timeout value.
echo "::group::=== Run pytest ==="
# test/egs3 as well as test/espnet3: a recipe's dataset and src modules are
# tested there, and nothing else in ci/ reaches that directory. The tests that
# need a corpus skip themselves when the environment does not name one.
pytest -q --execution-timeout 10.0 --timeouts-order moi test/espnet3/ test/egs3/
echo "::endgroup::"

echo "::group::=== Report ==="
coverage report
coverage xml
echo "::endgroup::"
