#!/usr/bin/env bash

# test asr recipe
(
    cd ./egs/mini_an4/asr1 || exit 1
    . path.sh  # source here to avoid undefined variable errors

    set -euo pipefail

    echo "==== ASR (backend=pytorch) ==="
    ./run.sh
    echo "==== ASR (backend=pytorch, dtype=float64) ==="
    ./run.sh --stage 3 --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"
    echo "==== ASR (backend=chainer) ==="
    ./run.sh --stage 3 --backend chainer
)
# test tts recipe
(
    set -euo pipefail

    cd ./egs/mini_an4/tts1 || exit 1
    echo "==== TTS (backend=pytorch) ==="
    ./run.sh
)

# TODO(karita): test asr_mix, mt, st?
