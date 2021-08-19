#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh


pretrain_start_iter=0
pretrain_stop_iter=2

n_clusters_iter0=100
n_clusters_iter1=500
n_clusters_iter2=500

feature_iter0="mfcc"
feature_iter1="HuBERT6"
feature_iter2="HuBERT9"

pretrain_train_set="train_960"
pretrain_valid_set="dev"

pretrain_config_iter0=conf/tuning/train_asr_hubert_base_960h_pretrain_it0.yaml
pretrain_config_iter1=conf/tuning/train_asr_hubert_base_960h_pretrain_it1.yaml
pretrain_config_iter2=conf/tuning/train_asr_hubert_base_960h_pretrain_it2.yaml

finetune_train_set="train_10h"
finetune_valid_set="dev"
finetune_test_sets="test_clean test_other dev_clean dev_other"

finetune_asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning.yaml
inference_config=conf/decode_asr.yaml

pretrain_configs="${pretrain_config_iter0} ${pretrain_config_iter1} ${pretrain_config_iter2}"
n_clusters="${n_clusters_iter0} ${n_clusters_iter1} ${n_clusters_iter2}"
features_km="${feature_iter0} ${feature_iter1} ${feature_iter2}"

./hubert_asr.sh \
    --lang en \
    --pretrain_ngpu 1 \
    --pretrain_num_nodes 1 \
    --pretrain_start_iter "${pretrain_start_iter}"\
    --pretrain_stop_iter "${pretrain_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30 \
    --pretrain_configs "${pretrain_configs}" \
    --n_clusters "${n_clusters}" \
    --features_km "${features_km}" \
    --use_lm false \
    --finetune_ngpu 4 \
    --pretrain_train_set "${pretrain_train_set}" \
    --pretrain_valid_set "${pretrain_valid_set}" \
    --finetune_train_set "${finetune_train_set}" \
    --finetune_valid_set "${finetune_valid_set}" \
    --finetune_test_sets "${finetune_test_sets}" \
    --finetune_config "${finetune_asr_config}" \
    --inference_config "${inference_config}" \
    --token_type char \
    --inference_asr_model valid.loss.ave.pth "$@"
