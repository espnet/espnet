#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_case=lc.rm
tgt_case=lc.rm
st_config=conf/train_st.yaml

./st.sh \
    --src_lang en \
    --tgt_lang en \
    --src_token_type "bpe" \
    --src_nbpe 30 \
    --tgt_token_type "bpe" \
    --tgt_nbpe 30 \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --src_bpe_train_text "data/train_nodev/text.${src_case}.en" \
    --tgt_bpe_train_text "data/train_nodev/text.${tgt_case}.en" \
    --use_lm false \
    --token_joint false \
    --st_config "${st_config}" \
    --train_set "train_nodev" \
    --valid_set "train_dev" \
    --test_sets "train_dev test test_seg" \
    --lm_train_text "data/train_nodev/text.${tgt_case}.en" "$@"
