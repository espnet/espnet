#!/usr/bin/env bash

echo "=== run unit tests at test_utils ==="
source tools/activate_python.sh
source tools/extra_path.sh

PATH=$(pwd)/test_utils/bats-core/bin:$PATH
if ! [ -x "$(command -v bats)" ]; then
    echo "=== install bats ==="
    git clone https://github.com/bats-core/bats-core.git "$(pwd)"/test_utils/bats-core
    git clone https://github.com/bats-core/bats-support.git "$(pwd)"/test_utils/bats-support
    git clone https://github.com/bats-core/bats-assert.git "$(pwd)"/test_utils/bats-assert
fi
bats test_utils/test_*.bats
