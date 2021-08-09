#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail
lang=hi-en
train_set="hi-en/train/transcripts"
valid_set="hi-en/valid/transcripts"
test_set="hi-en/valid/transcripts"

asr_config=conf/train.yaml

./asr.sh \
    --lang hi-en \
    --ngpu 1 \
    --skip_data_prep true\
    --use_lm false \
    --nbpe 5000 \
    --token_type char\
    --audio_format wav\
    --feats_type fbank_pitch\
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_set}" "$@"
