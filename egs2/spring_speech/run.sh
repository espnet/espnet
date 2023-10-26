#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="eval"

#                SPRING_INX Available languages :- 
#assamese, bengali, gujarati, hindi, marathi, kannada, malayalam, odia, punjabi, tamil
lang="assamese"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --ngpu 1 \
    --nj 16 \
    --stage 1 \
    --stop-stage 13 \
    --gpu_inference true \
    --inference_nj 1 \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw \
    --asr_tag "${lang}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
