#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=sw_train
train_dev=sw_dev
test_sets="sw_dev sw_test"

asr_config=conf/train.yaml
inference_config=conf/decode.yaml

./asr.sh \
    --stage 1 \
    --stop_stage 100 \
    --ngpu 1 \
    --nj 4 \
    --inference_nj 4 \
    --gpu_inference false \
    --inference_asr_model valid.acc.ave.pth \
    --use_lm false \
    --token_type bpe \
    --nbpe 300 \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@"
