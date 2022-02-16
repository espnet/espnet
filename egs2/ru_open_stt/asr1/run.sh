#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="asr_calls_2_val buriy_audiobooks_2_val public_youtube700_val"

asr_config=conf/train_asr_conformer.yaml
lm_config=conf/train_lm_transformer.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang ru \
    --ngpu 4 \
    --nbpe 100 \
    --max_wav_duration 30 \
    --audio_format "flac.ark" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" "$@"
