#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./codec.sh \
    --nj 2 \
    --inference_nj 2 \
    --train_config conf/train_soundstream_debug.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set train \
    --valid_set train_dev \
    --test_sets "train_dev test" "$@"
