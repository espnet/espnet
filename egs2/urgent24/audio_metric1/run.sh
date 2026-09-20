#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./audio_metric.sh \
    --train_config conf/train.yaml \
    --inference_config conf/decode.yaml \
    --train_set train \
    --valid_set dev \
    --test_sets "dev test" \
    --nbpe 500  \
    --ngpu 1 "$@"
