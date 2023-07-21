#!/usr/bin/env bash

. tools/activate_python.sh
. tools/extra_path.sh

set -euo pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

exclude="egs2/TEMPLATE/asr1/utils,egs2/TEMPLATE/asr1/steps,egs2/TEMPLATE/tts1/sid,doc,tools,bats-core"

# flake8
# "$(dirname $0)"/test_flake8.sh
# pycodestyle
pycodestyle --exclude "${exclude}" --show-source --show-pep8

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest -q

echo "=== report ==="
coverage report
coverage xml
