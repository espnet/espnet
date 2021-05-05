#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="test dev"

asr_config=conf/train_asr_transformer.yaml
lm_config=conf/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

stage=10
./asr.sh \
    --stage "${stage}" \
    --lang en \
    --ngpu 8 \
    --nbpe 500 \
    --max_wav_duration 20 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
