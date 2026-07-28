#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_ta"
valid_set="dev_ta"
test_sets="dev_ta test_ta"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

./asr.sh \
    --lang ta \
    --ngpu 1 \
    --nj 16 \
    --gpu_inference true \
    --inference_nj 1 \
    --token_type bpe \
    --nbpe 1000 \
    --bpemode unigram \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --use_lm true \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
