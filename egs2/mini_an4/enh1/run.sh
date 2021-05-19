#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./enh.sh \
    --fs 16k \
    --lang en \
    --spk-num 1 \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --enh_config ./conf/train.yaml \
    --inference_model "valid.loss.best.pth" \
    "$@"
