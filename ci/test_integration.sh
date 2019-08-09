#!/usr/bin/env bash

# test asr recipe
(
    cd ./egs/mini_an4/asr1 || exit 1
    ./run.sh
)
# test tts recipe
(
    cd ../tts1 || exit 1
    ./run.sh
)

# TODO(karita): test asr_mix, mt, st?
