#!/usr/bin/env bash
#  Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_nodev
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --nj 128 \
    --inference_nj 256 \
    --lang en \
    --ngpu 4 \
    --nbpe 20000 \
    --use_lm true \
    --lm_config "${lm_config}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
