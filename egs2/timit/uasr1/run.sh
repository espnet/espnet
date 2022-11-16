#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Set this to one of ["phn", "char"] depending on your requirement
trans_type=phn
if [ "${trans_type}" = phn ]; then
    # If the transcription is "phn" type, the token splitting should be done in word level
    token_type=word
else
    token_type="${trans_type}"
fi

train_set="train"
valid_set="dev"
test_sets="dev"

uasr_config=conf/train_uasr.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_uasr.yaml

./uasr.sh \
    --lang en \
    --stage 16 \
    --stop_stage 17 \
    --local_data_opts "" \
    --token_type "${token_type}" \
    --ngpu 1 \
    --nj 1 \
    --silence_trim false \
    --use_lm false \
    --use_feature_clustering true \
    --write_collected_feats true \
    --audio_format "wav" \
    --max_wav_duration 30 \
    --use_ngram true \
    --uasr_config "${uasr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
