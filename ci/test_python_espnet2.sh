#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

# Use sysmon core on Python 3.12+ to avoid sys.settrace performance regression
# (CPython gh-107674: tracing overhead ~7x on 3.12 vs ~3x on 3.10)
if python3 -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)"; then
    export COVERAGE_CORE=sysmon
fi

# One BLAS/OMP thread per worker: 4 xdist workers each spawning 4 threads on a
# 4 vCPU runner oversubscribes the box and makes the suite slower, not faster.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,test_utils/bats-core,test_utils/bats-support,test_utils/bats-assert"

# flake8
# TODO(nelson): Add documentation on espnet2 folder and uncomment this.
# echo "=== Run test flake8 ==="
# "$(dirname $0)"/test_flake8.sh espnet2

# pycodestyle
echo "::group::=== Run pycodestyle tests ==="
pycodestyle --exclude "${exclude}" --show-source --show-pep8
echo "::endgroup::"

# Populate the s3prl checkpoint cache serially before forking workers.
# test/espnet2/layers/test_create_adapter*.py build an S3prlFrontend at import
# time, so without this every xdist worker would download the same checkpoint
# to the same path at once. Non-fatal: if it fails the tests report the real
# error. The size check keeps a truncated download (an HTML error page from
# huggingface.co is ~3 kB) out of the cache that actions/cache then stores.
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
# -n: ubuntu-latest runners have 4 vCPUs.
# --dist loadfile keeps every test in a file on one worker, so module-level
# fixtures and module-level model construction are not repeated or raced
# across workers.
pytest -q -n "$(nproc)" --dist loadfile --execution-timeout 10.0 --timeouts-order moi test/espnet2
echo "::endgroup::"

echo "::group::=== Report ==="
coverage report
coverage xml
echo "::endgroup::"
