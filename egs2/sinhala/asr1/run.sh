#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/train_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --stage 1 \
    --use_lm false \
    --nbpe 5000 \
    --token_type word\
    --audio_format wav\
    --feats_type raw\
    --max_wav_duration 30 \
    --inference_asr_model valid.acc.ave_5best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
