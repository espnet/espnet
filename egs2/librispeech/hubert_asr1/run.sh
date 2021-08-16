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

pretrain_train_set="dev_clean"
pretrain_valid_set="dev_clean"

pretrain_config_iter0=conf/tuning/train_asr_hubert_base_960h_pretrain_it0.yaml
pretrain_config_iter1=conf/tuning/train_asr_hubert_base_960h_pretrain_it1.yaml
pretrain_config_iter2=conf/tuning/train_asr_hubert_base_960h_pretrain_it2.yaml

finetune_train_set="train_10h"
finetune_valid_set="dev"
finetune_test_sets="test_clean test_other dev_clean dev_other"

finetune_asr_config=conf/tuning/train_asr_hubert_base_10h_finetuning.yaml
inference_config=conf/decode_asr.yaml
pretrain_config_list[0]=0
n_clusters_list[0]=0
feature_list[0]=0
for ((iter=${pretrain_start_iter}; iter<=${pretrain_stop_iter};iter++)); do
    pretrain_config_list[${iter}]=$(eval "echo \${pretrain_config_iter${iter}}")
    n_clusters_list[${iter}]=$(eval "echo \${n_clusters_iter${iter}}")
    feature_list[${iter}]=$(eval "echo \${feature_iter${iter}}")
done

./hubert_asr.sh \
    --lang en \
    --pretrain_ngpu 1 \
    --pretrain_start_iter "${pretrain_start_iter}"\
    --pretrain_stop_iter "${pretrain_stop_iter}" \
    --nj 4 \
    --max_wav_duration 30 \
    --pretrain_config_list "${pretrain_config_list}" \
    --n_clusters_list "${n_clusters_list}" \
    --feature_list "${feature_list}" \
    --use_lm false \
    --finetune_ngpu 1 \
    --pretrain_train_set "${pretrain_train_set}" \
    --pretrain_valid_set "${pretrain_valid_set}" \
    --finetune_train_set "${finetune_train_set}" \
    --finetune_valid_set "${finetune_valid_set}" \
    --finetune_test_sets "${finetune_test_sets}" \
    --finetune_config "${finetune_asr_config}" \
    --inference_config "${inference_config}" \
    --token_type char \
    --inference_asr_model valid.loss.ave.pth "$@"
