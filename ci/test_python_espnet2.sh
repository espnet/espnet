#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

# Use sysmon core on Python 3.12+ to avoid sys.settrace performance regression
# (CPython gh-107674: tracing overhead ~7x on 3.12 vs ~3x on 3.10)
if python3 -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)"; then
    export COVERAGE_CORE=sysmon
fi

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
# TODO(nelson): Add documentation on espnet2 folder and uncomment this.
# echo "=== Run test flake8 ==="
# "$(dirname $0)"/test_flake8.sh espnet2

# pycodestyle
echo "=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8

# It will set default timeout to 10.0 seconds for each test.
# If the test is marked with @pytest.mark.execution_timeout,
# the value in the mark will be used as the timeout value.
pytest -q --execution-timeout 10.0 --timeouts-order moi test/espnet2

echo "=== report ==="
coverage report
coverage xml
