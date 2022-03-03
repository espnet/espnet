#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="devel"
test_sets="test devel"

asr_config=conf/tuning/train_asr_config3.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --use_lm false \
    --ngpu 1 \
    --nbpe 5000 \
    --audio_format wav \
    --stage 11\
    --stop_stage 13 \
    --skip_upload true\
    --token_type word\
    --feats_type raw\
    --gpu_inference true \
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 4 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
