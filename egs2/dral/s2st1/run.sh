#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# language related
src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
tgt_lang=en

train_set=train
train_dev=dev
test_sets="test dev"

st_config=conf/train_s2st.yaml
use_src_lang=true
use_tgt_lang=true
inference_config=conf/decode_s2st.yaml

./s2st.sh \
    --ngpu 1 \
    --stage 5 \
    --nj 1 \
    --feats_type raw \
    --audio_format "wav" \
    --use_src_lang ${use_src_lang} \
    --use_tgt_lang ${use_tgt_lang} \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --s2st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" "$@"
