#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_set="train_10h"
valid_set="dev_clean"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning.yaml
inference_config=conf/decode.yaml


./asr.sh \
    --lang en \
    --ngpu 1 \
    --nj 8 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --token_type char \
    --inference_asr_model valid.loss.ave.pth \
    --feats-normalize null "$@"
