#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./uni_versa.sh \
    --nj 2 \
    --inference_nj 2 \
    --train_config conf/train_universa.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test"  "$@"
