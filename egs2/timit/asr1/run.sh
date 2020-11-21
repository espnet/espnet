#!/bin/bash
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

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --token_type "${trans_type}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --use_lm false \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --local_data_opts "--trans_type ${trans_type}" \
    --srctexts "data/${train_set}/text" "$@"
