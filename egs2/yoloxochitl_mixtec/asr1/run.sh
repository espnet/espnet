#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_org_sp
train_dev=dev_org
test_set=test_org

asr_config=conf/tuning/train_asr_transformer.yaml
# asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --local_data_opts "--stage 1" \
    --stage 10 \
    --stop_stage 100 \
    --ngpu 1 \
    --nj 40 \
    --inference_nj 40 \
    --use_lm true \
    --token_type bpe \
    --nbpe 500 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --asr_tag "transformer_org_data" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --inference_asr_model valid.acc.best.pth \
    --lm_train_text "data/${train_set}/text"  "$@"

