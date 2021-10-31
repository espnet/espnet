#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_sp.es
train_dev=train_dev.es
test_set="fisher_dev.es fisher_dev2.es fisher_test.es callhome_devtest.es callhome_evltest.es"

asr_config=conf/train_asr.conf
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

nbpe=1000

./asr.sh \
    --ngpu 4 \
    --local_data_opts "--stage 0" \
    --use_lm true \
    --lm_config "${lm_config}" \
    --token_type bpe \
    --nbpe $nbpe \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --bpe_train_text "data/${train_set}/text" \
    --lm_train_text "data/${train_set}/text" "$@"
