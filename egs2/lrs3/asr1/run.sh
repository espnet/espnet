#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"


asr_tag=train_asr_transformer
asr_config=conf/train_asr_transformer.yaml
lm_config=conf/train_lm.yaml  # Not Used, as use_lm=false

./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --stage 1 \
    --lang en \
    --ngpu 1 \
    --nj 32 \
    --inference_nj 32 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "wav" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --lm_config ${lm_config} \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
