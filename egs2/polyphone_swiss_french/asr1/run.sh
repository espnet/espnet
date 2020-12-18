#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="valid_train"
valid_set="valid_dev"
test_sets="valid_dev valid_test"

asr_config=conf/tuning/train_asr_conformer.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang fr \
    --ngpu 8 \
    --nbpe 800 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
