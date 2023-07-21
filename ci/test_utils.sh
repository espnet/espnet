#!/usr/bin/env bash

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z "${GITHUB_PR_LABEL_ESPNET1:-}" ] && [ -z "${GITHUB_PR_LABEL_ESPNET2:-}" ]; then
    # If not Label tag ESPNET 1 or ESPNET 2 but Docker, then skip all tests.
    if [ -n "${GITHUB_PR_LABEL_DOCKER:-}" ]; then
        log Only Docker related modifications. Skipping tests.
        exit 0
    fi
fi

echo "=== run integration tests at test_utils ==="

PATH=$(pwd)/bats-core/bin:$PATH
if ! [ -x "$(command -v bats)" ]; then
    echo "=== install bats ==="
    git clone https://github.com/bats-core/bats-core.git
fi
bats test_utils/integration_test_*.bats

echo "=== report ==="

source tools/activate_python.sh
coverage combine egs/*/*/.coverage
coverage report
coverage xml
