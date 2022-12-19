#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"

asr_config=conf/tuning/train_asr_conformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --nj 8 \
    --ngpu 2 \
    --gpu_inference true \
    --inference_nj 2 \
    --feats_type raw \
    --audio_format "flac.ark" \
    --token_type bpe \
    --nbpe 500 \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
