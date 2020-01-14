#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# data preparation
trans_type=char     # Transcript type: char or phn

train_set=train
dev_set=dev
eval_sets="test "

./asr.sh \
    --nbpe 5000 \
    --token_type char \
    --train_set "${train_set}" \
    --dev_set "${dev_set}" \
    --eval_sets "${eval_sets}" \
    --srctexts "data/${train_set}/text" "$@" \
    --local_data_opts "${trans_type}"
