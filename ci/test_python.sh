#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

exclude="^(egs2/TEMPLATE/asr1/utils|egs2/TEMPLATE/asr1/steps|egs2/TEMPLATE/tts1/sid|doc)"

# black
if ! black --exclude "${exclude}" .; then
    printf '[INFO] Please apply black:\n    $ black --exclude "$s"\n' "${exclude}"
    exit 1
fi
# isort
if ! isort -c -v --exclude "${exclude}" .; then
    printf '[INFO] Please apply isort:\n    $ black --exclude "$s"\n' "${exclude}"
    exit 1
fi

# flake8
"$(dirname $0)"/test_flake8.sh
# pycodestyle
pycodestyle -r --exclude "${exclude}" . --show-source --show-pep8

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest -q

echo "=== report ==="
coverage report
coverage xml
