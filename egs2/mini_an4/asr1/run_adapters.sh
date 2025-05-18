#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

#run this shell file to test out adapter aided training.

set -e
set -u
set -o pipefail

./asr.sh \
    --lang en \
    --asr_config conf/adapter_example.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --lm_train_text "data/train_nodev/text" "$@"
