#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
lang=hi-en
train_set="hi-en/train/transcripts"
valid_set="hi-en/test/transcripts"
test_set="hi-en/test/transcripts"

asr_config=conf/train2.yaml
lm_config_=conf/lm.yaml
./asr.sh \
    --lang hi-en \
    --ngpu 2 \
    --use_lm true \
    --lm_config "${lm_config_}" \
    --nbpe 5000 \
    --token_type char\
    --audio_format wav\
    --feats_type fbank_pitch\
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" "$@"
