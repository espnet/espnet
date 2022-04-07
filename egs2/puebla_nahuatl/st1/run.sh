#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=na
tgt_lang=es

train_set=train_st
train_dev=dev_st
test_set="dev_st test_st"

st_config=conf/train_st.yaml
inference_config=conf/decode_st.yaml

src_nbpe=500
tgt_nbpe=500

src_case=lc.rm
tgt_case=lc.rm

./st.sh \
    --local_data_opts "--stage 0" \
    --stage 11 \
    --stop_stage 11 \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --st_tag "asr_pretrained" \
    --nj 40 \
    --inference_nj 4 \
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
    --pretrained_asr "/projects/tir3/users/jiatongs/els/puebla_nahuatl/asr1/exp/asr_train_asr_transformer_specaug_raw_bpe500_sp/valid.acc.ave_10best.pth" \
    --ignore_init_mismatch true \
    --st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}"  "$@"
