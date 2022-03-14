#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

train_set="train_ml"
train_dev="dev_ml"
test_set="test_ml"

asr_config=conf/tuning/train_asr_conformer5.yaml
inference_config=conf/decode_asr.yaml
lm_config=conf/train_lm.yaml

ngpu=4

./asr.sh \
    --stage 1 \
    --stop_stage 13 \
    --ngpu 4 \
    --lang "${lang}" \
    --local_data_opts "--lang ${lang}" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe 1000 \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"