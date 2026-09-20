#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
echo "::group::=== Run test flake8 ==="
"$(dirname $0)"/test_flake8.sh espnet3
# splet/ rides along with espnet3's job rather than having one of its own:
# it is small, it has no dependency on ESPnet, and it is meant to be
# extracted into its own repository (espnet/espnet#6760), at which point it
# takes its CI with it. Until then it is still checked on every push. This is
# the flake8-docstrings pass; test_flake8.sh is not reused because its first,
# much larger invocation would then run twice.
flake8 --show-source splet
echo "::endgroup::"

# pycodestyle
echo "::group::=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8
echo "::endgroup::"

# It will set default timeout to 10.0 seconds for each test.
# If the test is marked with @pytest.mark.execution_timeout,
# the value in the mark will be used as the timeout value.
echo "::group::=== Run pytest ==="
pytest -q --execution-timeout 10.0 --timeouts-order moi test/espnet3/ test/splet/
echo "::endgroup::"

echo "::group::=== Report ==="
coverage report
coverage xml
echo "::endgroup::"
