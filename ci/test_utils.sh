#!/usr/bin/env bash

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
