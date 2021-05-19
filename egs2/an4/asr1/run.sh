#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --lang en \
    --train_set train_nodev \
    --lm_config conf/train_lm.yaml \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --lm_train_text "data/train_nodev/text" "$@"
