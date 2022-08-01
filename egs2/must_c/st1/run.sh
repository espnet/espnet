#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=en
tgt_lang=de

train_set=train.en-${tgt_lang}
train_dev=dev.en-${tgt_lang}
test_set="tst-COMMON.en-${tgt_lang} tst-HE.en-${tgt_lang}"

st_config=conf/tuning/train_st_conformer.yaml
inference_config=conf/tuning/decode_st_conformer.yaml

src_nbpe=4000
tgt_nbpe=4000

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
src_case=lc.rm
tgt_case=tc

./st.sh \
    --local_data_opts "${tgt_lang}" \
    --audio_format "flac.ark" \
    --nj 40 \
    --inference_nj 40 \
    --audio_format "flac.ark" \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "bpe" \
    --src_nbpe $src_nbpe \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --feats_type raw \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@"
