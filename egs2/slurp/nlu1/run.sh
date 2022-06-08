#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_type=en
tgt_type=ner

train_set="train"
valid_set="devel"
test_sets="test devel"

nlu_config=conf/train_nlu.yaml
inference_config=conf/decode_nlu.yaml

src_case=tc
tgt_case=tc

src_nbpe=1000
tgt_nbpe=1000   # if token_joint is true, then only tgt_nbpe is used

./nlu.sh \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 32 \
    --src_type ${src_type} \
    --tgt_type ${tgt_type} \
    --src_token_type "word" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "word" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --nlu_config "${nlu_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_type}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_type}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_type}" "$@"
