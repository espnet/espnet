#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=de
tgt_lang=en

train_set=train
train_dev=valid
test_sets="test valid"

mt_config=conf/tuning/train_mt_branchformer_lr3e-3_warmup10k_share_enc_dec_input_dropout0.3.yaml
inference_config=conf/decode_mt.yaml

src_nbpe=1000
tgt_nbpe=10000   # if token_joint is True, then only tgt_nbpe is used

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
# Note, it is best to keep tgt_case as tc to match IWSLT22 eval
src_case=tc
tgt_case=tc

./mt.sh \
    --use_lm false \
    --token_joint true \
    --ngpu 1 \
    --nj 16 \
    --inference_nj 32 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --mt_config "${mt_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@"
