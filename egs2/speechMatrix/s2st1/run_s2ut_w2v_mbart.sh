#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# language related
src_lang=es # ar ca cy de et es fa fr id it ja lv mn nl pt ru sl sv ta tr zh

train_set=train_${src_lang}
train_dev=dev_${src_lang}
test_sets="test_${src_lang}"

st_config=conf/train_s2ut_w2v_mbart.yaml
use_src_lang=true
use_tgt_lang=true
inference_config=conf/decode_s2st.yaml
score_asr_model_tag=byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp

# This is word because the text files contain integer arrays and we have a dictionary of 0-100 integers
token_type="word"

./s2st_local.sh \
    --ngpu 2 \
    --nj 64 \
    --inference_nj 64 \
    --use_discrete_unit true \
    --feats_type raw \
    --audio_format "wav" \
    --use_src_lang ${use_src_lang} \
    --use_tgt_lang ${use_tgt_lang} \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang en \
    --s3prl_upstream_name hubert \
    --storage_save_mode false \
    --clustering_num_threads 60 \
    --src_token_type ${token_type} \
    --tgt_token_type ${token_type} \
    --s2st_config "${st_config}" \
    --inference_config "${inference_config}" \
    --score_asr_model_tag "${score_asr_model_tag}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" "$@"
    # --feature_layer ${feature_layer} \
    # --clustering_portion ${clustering_portion} \
    # --feature_num_clusters ${clustering_num_clusters}
