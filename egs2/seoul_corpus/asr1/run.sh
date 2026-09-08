#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
valid_set="dev"
test_sets="dev test"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# Use "--local_data_opts '--tier utt.prono.'" to train on the pronounced
# (colloquial) transcription instead of the orthographic one.
local_data_opts=""

./asr.sh \
    --lang ko \
    --audio_format flac.ark \
    --nbpe 2000 \
    --use_lm false \
    --min_wav_duration 1.0 \
    --max_wav_duration 20 \
    --local_data_opts "${local_data_opts}" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train/text" \
    --lm_train_text "data/train/text" "$@"
