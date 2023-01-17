#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


train_start_iter=0
train_stop_iter=1

n_clusters_iter0=10
n_clusters_iter1=10

feature_iter0="mfcc"
layer_iter0="0"
feature_iter1="hubert"
layer_iter1="6"

train_set="train_nodev"
valid_set="train_dev"

train_config_iter0=conf/train_ssl_torchaudiohubert_base_pretrain_it0.yaml
train_config_iter1=conf/train_ssl_torchaudiohubert_base_pretrain_it1.yaml

./hubert.sh \
    --train_start_iter "${train_start_iter}"\
    --train_stop_iter "${train_stop_iter}" \
    --nj 4 \
    --max_wav_duration 30 \
    --train_configs "${train_config_iter0} ${train_config_iter1}" \
    --n_clusters "${n_clusters_iter0} ${n_clusters_iter1}" \
    --features_km "${feature_iter0} ${feature_iter1}" \
    --layers_km "${layer_iter0} ${layer_iter1}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --portion_km 1.0  "$@"
