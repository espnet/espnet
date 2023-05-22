#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./db.sh

train_start_iter=0
train_stop_iter=1  # 1 iterations is enough for base model

n_clusters_iter0=100
n_clusters_iter1=500
n_clusters_iter2=500

# Extract mfcc feature for k-means clustering to generate pseudo targets
feature_iter0="mfcc"
layer_iter0="0"
# Extract latent features from transformer layer 6 of HuBERT model pre-trained in the iteration0
feature_iter1="hubert"
layer_iter1="6"
# Extract latent features from transformer layer 9 of HuBERT model pre-trained in the iteration1
feature_iter2="hubert"
layer_iter2="9"

train_set="train_960"
valid_set="dev"

train_config_iter0=conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it0.yaml
train_config_iter1=conf/tuning/train_ssl_torchaudiohubert_base_960h_pretrain_it1.yaml
train_config_iter2=conf/tuning/train_ssl_torchaudiohubert_large_960h_pretrain_it2.yaml

./hubert.sh \
    --ngpu 8 \
    --num_nodes 1 \
    --lang "en" \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 32 \
    --max_wav_duration 30 \
    --train_configs "${train_config_iter0} ${train_config_iter1} ${train_config_iter2}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1} ${n_clusters_iter2}" \
    --features_km "${feature_iter0} ${feature_iter1} ${feature_iter2}" \
    --layers_km "${layer_iter0} ${layer_iter1} ${layer_iter2}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --portion_km 0.1 \
    --gpu_dump_feature true \
    --alignment_phoneme_dir "./data/librispeech_phoneme_alignment" "$@"
