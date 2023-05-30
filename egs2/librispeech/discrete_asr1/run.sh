#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


local_data_feats_type="wavlm+large+24"  # use model_type+model_version+layer_index
nclusters=2000

src_lang=$(echo "${local_data_feats_type}" | tr "+" "_")"_km${nclusters}"
tgt_lang=en

train_set="train_960"
train_dev="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_discrete_asr_e_branchformer1.yaml 
inference_config=conf/decode_ctc0.3.yaml

src_nbpe=6000   # I use src_nbpe=3000 for km1000, and src_nbpe=6000 for km2000.
tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used

# ts: true sequence
# rm: deduplicated sequence which removes duplicated tokens
src_case="rm"
tgt_case="ts"

./discrete_asr.sh \
    --local_data_feats_type "${local_data_feats_type}" \
    --nclusters "${nclusters}" \
    --use_lm false \
    --token_joint false \
    --ngpu 2 \
    --nj 32 \
    --inference_nj 32 \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --src_bpe_train_text "data/train_960_sp/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/train_960_sp/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/train_960_sp/text.${tgt_case}.${tgt_lang}" "$@"
