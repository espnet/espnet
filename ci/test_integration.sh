#!/usr/bin/env bash

(
    cd ./egs/mini_an4/asr1 || exit 1
    ./run.sh
)

# TODO(karita): test asr_mix, tts, mt, st?
