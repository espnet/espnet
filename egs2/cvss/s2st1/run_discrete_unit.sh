#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# language related
src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh
version=c # c or t (please refer to cvss paper for details)

train_set=train_${src_lang}
train_dev=dev_${src_lang}
test_sets="test_${src_lang} dev_${src_lang}"

st_config=conf/train_s2st_discrete_unit.yaml
use_src_lang=true
use_tgt_lang=true
inference_config=conf/decode_s2st.yaml

CUDA_VISIBLE_DEVICES=8,9 ./s2st.sh \
    --ngpu 2 \
    --stage 7 \
    --nj 10 \
    --inference_nj 1 \
    --use_discrete_unit true \
    --local_data_opts "--stage 0 --src_lang ${src_lang} --version ${version}" \
    --feats_type raw \
    --audio_format "wav" \
    --use_src_lang ${use_src_lang} \
    --use_tgt_lang ${use_tgt_lang} \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang en \
    --feature_layer 6 \
    --s3prl_upstream_name hubert \
    --clustering_portion 0.5 \
    --feature_num_clusters 500 \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --s2st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --vocoder_file unit_pretrained_vocoder/checkpoint-350000steps.pkl \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" "$@"
