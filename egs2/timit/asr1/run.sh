#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train"
train_dev="dev"
test_sets="dev test"

# Set this to one of ["phn", "char"] depending on your requirement
trans_type=phn
if [ "${trans_type}" = phn ]; then
    # If the transcription is "phn" type, the token splitting should be done in word level
    token_type=word
else
    token_type="${trans_type}"
fi

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm_rnn.yaml
inference_config=conf/decode_asr.yaml


./asr.sh \
    --token_type "${token_type}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --local_data_opts "--trans_type ${trans_type}" \
    --lm_train_text "data/${train_set}/text" "$@"
