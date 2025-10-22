#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./uni_versa.sh \
    --train_config conf/train_universa.yaml \
    --inference_config conf/decode_universa.yaml \
    --train_set train \
    --valid_set dev \
    --test_sets "dev test" \
    --nbpe 500  \
    --ngpu 1 "$@"
