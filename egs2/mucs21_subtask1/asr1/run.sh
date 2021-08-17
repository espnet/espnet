#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set='train'
valid_set='test'
test_set='test'

asr_config=conf/train.yaml
lm_config_=conf/lm.yaml
./asr.sh \
    --ngpu 2 \
    --use_lm true \
    --lm_config "${lm_config_}" \
    --audio_format wav\
    --feats_type fbank_pitch\
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" "$@"
