#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=es

train_set=${lang}_train
train_dev=${lang}_dev
test_set=${lang}_test

lm_train_text=data/${lang}_lm_train.txt

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

./asr.sh \
    --local_data_opts "--lang ${lang}" \
    --stage 1 \
    --stop_stage 100 \
    --nj 40 \
    --ngpu ${ngpu} \
    --use_lm true \
    --token_type bpe \
    --nbpe 150 \
    --feats_type raw \
    --asr_tag transformer \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "${lm_train_text}" "$@"
