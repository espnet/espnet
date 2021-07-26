#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_10h"
valid_set="dev_other"
test_sets="dev_clean" #"test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

. ./db.sh

#local/prepare_librilight.sh ${LIBRISPEECH}/librispeech_finetuning

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nj 4 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --token_type char \
    --lm_train_text "data/${train_set}/text" \
    --inference_asr_model valid.loss.ave.pth \
    --feats-normalize null  "$@" 

