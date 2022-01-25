#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=es
tgt_lang=en

train_set=train
train_dev=dev
test_set="fisher_dev fisher_dev2 fisher_test callhome_devtest callhome_evltest"

st_config=conf/train_st.yaml
inference_config=conf/decode_st.yaml

src_nbpe=500
tgt_nbpe=500

src_case=lc.rm
tgt_case=lc.rm

./st.sh \
    --local_data_opts "--stage 0" \
    --stage 1 \
    --stop_stage 13 \
    --audio_format "flac.ark" \
    --use_lm false \
    --token_joint false \
    --nj 40 \
    --inference_nj 40 \
    --st_tag "transformer_asr_pretrained" \
    --pretrained_asr "/projects/tir3/users/jiatongs/st/espnet/egs2/fisher_callhome_spanish/asr1/exp/asr_train_asr_raw_bpe1000_sp/valid.acc.ave_10best.pth" \
    --ignore_init_mismatch true \
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
