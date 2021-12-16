#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="test_clean"

feats_type=raw
local_data_opts=""

if [ "${feats_type}" = fbank_pitch ]; then
    local_data_opts="---pipe_wav true"
fi

./asr.sh \
    --token_type bpe \
    --local_data_opts "${local_data_opts}" \
    --nbpe 5000 \
    --use_lm false \
    --lang kr \
    --lm_config conf/train_lm.yaml \
    --asr_config conf/tuning/train_asr_transformer5.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train_data_01/text" \
    --lm_train_text "data/train_data_01/text" "$@"
