#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

modules="espnet espnet2 test utils setup.py egs*/*/*/local egs2/TEMPLATE/*/pyscripts tools/*.py ci/*.py"

# black
if ! black --check ${modules}; then
    printf '[INFO] Please apply black:\n    $ black %s\n' "${modules}"
    exit 1
fi
# isort
if ! isort -c -v ${modules}; then
    printf '[INFO] Please apply isort:\n    $ isort %s\n' "${modules}"
    exit 1
fi

# flake8
"$(dirname $0)"/test_flake8.sh
# pycodestyle
pycodestyle -r ${modules} --show-source --show-pep8

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" \
    PYTHONPATH="${PYTHONPATH:-}:$(pwd)/tools/s3prl" pytest -q

echo "=== report ==="
coverage report
coverage xml
