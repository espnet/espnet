#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./asr.sh \
    --nj 2 \
    --inference_nj 2 \
    --lang en \
    --asr_config conf/train_asr_rnn_debug.yaml \
    --lm_config conf/train_lm_rnn_debug.yaml \
    --inference_config conf/decode_asr_debug.yaml \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test test_seg" \
    --num_ref 2 \
    --lm_train_text "data/train_nodev/text" "$@"
