#!/usr/bin/env bash

set -euo pipefail

if [ -z ${LD_LIBRARY_PATH+x} ]; then
    LD_LIBRARY_PATH=$(pwd)/chainer_ctc/ext/warp-ctc/build
else
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(pwd)/chainer_ctc/ext/warp-ctc/build
fi

export LD_LIBRARY_PATH

# test asr recipe
(
    cd ./egs/mini_an4/asr1 || exit 1
    ./run.sh
    ./run.sh --stage 3 --backend chainer
)
# test tts recipe
(
    cd ./egs/mini_an4/tts1 || exit 1
    ./run.sh
)

# TODO(karita): test asr_mix, mt, st?
