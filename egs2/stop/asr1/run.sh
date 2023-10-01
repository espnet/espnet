#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="valid"
test_sets="test valid"

asr_config=conf/train_asr_whisper_full_correct.yaml

./asr.sh \
    --lang en \
    --ngpu 1 \
    --use_lm false \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --feats_type raw\
    --max_wav_duration 30 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_asr_model valid.acc.ave_10best.pth\
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
