#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets=test

asr_config=conf/train_asr_conformer.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

./asr.sh \
    --ngpu 8 \
    --nj 16 \
    --inference_nj 16 \
    --max_wav_duration 14 \
    --lang jp \
    --use_lm true \
    --token_type char \
    --feats_type raw \
    --audio_format flac \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --lm_config "${lm_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
