#!/usr/bin/env bash

. tools/activate_python.sh

set -euo pipefail

modules="espnet espnet2 test utils setup.py egs*/*/*/local egs2/TEMPLATE/asr1/pyscripts"

white_list="egs2/librispeech/asr1/local/make_lexicon_fst.py"

# black
if ! black --check ${modules} --exclude=${white_list}; then
    printf 'Please apply:\n    $ black %s\n' "${modules}"
    exit 1
fi

# flake8
"$(dirname $0)"/test_flake8.sh
# pycodestyle
pycodestyle -r ${modules} --exclude=${white_list} --show-source --show-pep8

LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(pwd)/tools/chainer_ctc/ext/warp-ctc/build" pytest -q
