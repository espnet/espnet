#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

src_lang=en
tgt_lang=en

train_set=train
train_dev=devel
test_set="devel test"

st_config=conf/md_token_8.yaml
# inference_config=conf/decode_md.yaml

src_nbpe=500
tgt_nbpe=500

# tc: truecase
# lc: lowercase
# lc.rm: lowercase with punctuation removal
# Note, it is best to keep tgt_case as tc to match IWSLT22 eval
src_case=asr
tgt_case=ner

./st.sh \
    --use_multidecoder true \
    --use_lm false \
    --use_asrlm false \
    --use_asr false \
    --use_mt false \
    --use_asr_inference_text false \
    --use_ext_st false \
    --token_joint false \
    --ngpu 1 \
    --feats_type raw \
    --stage 1\
    --stop_stage 9\
    --gpu_inference true\
    --audio_format "flac.ark" \
    --inference_config conf/decode_asr_md_ctc_0.1.yaml\
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "word" \
    --tgt_token_type "word" \
    --src_case ${src_case} \
    --tgt_case ${tgt_case} \
    --st_config "${st_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_set}" \
    --src_bpe_train_text "data/${train_set}/text.${src_case}.${src_lang}" \
    --tgt_bpe_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" \
    --lm_train_text "data/${train_set}/text.${tgt_case}.${tgt_lang}" "$@"