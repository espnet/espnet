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
echo "::group::=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8
echo "::endgroup::"

# test/espnet2/layers/test_create_adapter*.py build an S3prlFrontend at import
# time, so the hubert_base checkpoint is fetched during pytest collection. When
# huggingface.co rate limits the download, s3prl writes the HTML error page to
# the checkpoint path instead of failing, and torch.load then aborts the whole
# run with exit code 2 rather than failing a single test:
#
#   _pickle.UnpicklingError: Weights only load failed ... Unsupported operand 60
#
# 60 is 0x3C, '<'. Fetch it here instead, where a failure is visible and named,
# and delete anything under 1 MB afterwards - the error page is ~3 kB - so that
# a truncated download is not what actions/cache stores for every later run.
echo "::group::=== Warm s3prl checkpoint cache ==="
python3 - <<'PY' || echo "warm-up failed; tests will download on demand"
from espnet2.asr.frontend.s3prl import S3prlFrontend

S3prlFrontend(frontend_conf={"upstream": "hubert_base"})
PY
find "${HOME}/.cache/s3prl/download" -type f -size -1M -delete 2>/dev/null || true
echo "::endgroup::"

# It will set default timeout to 10.0 seconds for each test.
# If the test is marked with @pytest.mark.execution_timeout,
# the value in the mark will be used as the timeout value.
echo "::group::=== Run pytest ==="
pytest -q --execution-timeout 10.0 --timeouts-order moi test/espnet2
echo "::endgroup::"

echo "::group::=== Report ==="
coverage report
coverage xml
echo "::endgroup::"
