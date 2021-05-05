#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lid=false # whether to use language id as additional label

lang="java"

train_set="${lang}_train"
train_dev="${lang}_dev"
test_set="${lang}_test"

asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

ngpu=1

./asr.sh \
    --stage 1 \
    --stop_stage 100 \
    --local_data_opts "--lang ${lang}" \
    --ngpu ${ngpu} \
    --nj 80 \
    --inference_nj 256 \
    --use_lm false \
    --token_type bpe \
    --nbpe 1000 \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --lm_train_text "data/${train_set}/text" \
    --local_score_opts "--score_lang_id ${lid}" "$@"

