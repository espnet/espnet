#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# language related
src_lang=lt 
tgt_lang=en 

stage=1
stop_stage=5

# kmeans related
ssl_model=mhubert_base_vp_en_es_fr_it3
clustering_portion=1
clustering_num_clusters=1000
feature_layer=11

train_set=train_${src_lang}_${tgt_lang}
valid_set=dev_${src_lang}_${tgt_lang}
test_sets="test_fleurs_${src_lang}_${tgt_lang}"

st_config=conf/train_s2st_discrete_unit.yaml
use_src_lang=false  # TODO: 
use_tgt_lang=false  # TODO:
inference_config=conf/decode_s2st.yaml
vocoder_file=none  # TODO: Have to retrain the vocoder with discrete units
score_asr_model_tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"

./s2st.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --ngpu 1 \
    --nj 64 \
    --inference_nj 64 \
    --use_discrete_unit true \
    --kmeans_opts "--skip_train_kmeans true --km_dir dump/pretrained_kmeans" \
    --km_tag pretrained_kmeans \
    --feats_type raw \
    --audio_format "wav" \
    --use_src_lang ${use_src_lang} \
    --use_tgt_lang ${use_tgt_lang} \
    --token_joint false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --feature_layer ${feature_layer} \
    --s3prl_upstream_name ${ssl_model} \
    --storage_save_mode false \
    --clustering_num_threads 60 \
    --clustering_portion ${clustering_portion} \
    --feature_num_clusters ${clustering_num_clusters} \
    --src_token_type "char" \
    --tgt_token_type "char" \
    --inference_config "${inference_config}" \
    --vocoder_file "${vocoder_file}" \
    --score_asr_model_tag "${score_asr_model_tag}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" "$@"
