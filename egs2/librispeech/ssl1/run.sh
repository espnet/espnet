#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

pretrain_start_iter=0
pretrain_stop_iter=2

n_clusters_iter0=100
n_clusters_iter1=500
n_clusters_iter2=500

# Extract mfcc feature for k-means clustering to generate pseudo targets
feature_iter0="mfcc"
# Extract latent features from transformer layer 6 of HuBERT model pre-trained in the iteration0
feature_iter1="HuBERT6"
# Extract latent features from transformer layer 9 of HuBERT model pre-trained in the iteration1
feature_iter2="HuBERT9"

train_set="train_960"
valid_set="dev"

pretrain_config_iter0=conf/tuning/train_asr_hubert_base_960h_pretrain_it0.yaml
pretrain_config_iter1=conf/tuning/train_asr_hubert_base_960h_pretrain_it1.yaml
pretrain_config_iter2=conf/tuning/train_asr_hubert_base_960h_pretrain_it2.yaml

./hubert.sh \
    --lang en \
    --ngpu 8 \
    --num_nodes 4 \
    --pretrain_start_iter "${pretrain_start_iter}"\
    --pretrain_stop_iter "${pretrain_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30 \
    --pretrain_configs "${pretrain_config_iter0} ${pretrain_config_iter1} ${pretrain_config_iter2}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1} ${n_clusters_iter2}" \
    --features_km "${feature_iter0} ${feature_iter1} ${feature_iter2}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" "$@"
